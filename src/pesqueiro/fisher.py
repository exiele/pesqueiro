from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, Protocol

from .core import BiteWatcher
from .mouse import LatestReading, PointerCalibration, move_to


class Device(Protocol):
    def move_relative(self, dx: int, dy: int) -> None: ...
    def click_right(self) -> None: ...
    def press_key(self, name: str) -> None: ...


class State(Enum):
    IDLE = "idle"
    CASTING = "casting"
    WAIT_BOBBER = "wait_bobber"
    MOVING = "moving"
    WATCHING = "watching"
    CLICKING = "clicking"


class Fisher:
    def __init__(
        self,
        device: Device,
        cursor: LatestReading,
        log: Callable[[str], None],
        cast_key: str = "4",
        start_delay: float = 3.0,
        land_delay: float = 2.0,
        bobber_timeout: float = 12.0,
        fish_timeout: float = 25.0,
        loot_delay: float = 3.0,
        stable_scans: int = 3,
        stable_radius: int = 6,
        tolerance: int = 6,
        click_tolerance: int = 20,
        park_offset: tuple[int, int] = (75, 25),
        return_window: float = 6.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._device = device
        self._cursor = cursor
        self._log = log
        self._cast_key = cast_key
        self._start_delay = start_delay
        self._armed_at: float | None = None
        self._land_delay = land_delay
        self._bobber_timeout = bobber_timeout
        self._fish_timeout = fish_timeout
        self._loot_delay = loot_delay
        self._stable_scans = stable_scans
        self._stable_radius = stable_radius
        self._tolerance = tolerance
        self._click_tolerance = click_tolerance
        self._park_offset = park_offset
        self._return_window = return_window
        self._lost_at: float | None = None
        self._returned = False
        self._clock = clock
        self._watcher = BiteWatcher()
        self._calibration = PointerCalibration()
        self.state = State.IDLE
        self._since = 0.0
        self._bobber: tuple[int, int] | None = None
        self._anchor: tuple[int, int] | None = None
        self._stable = 0
        self._watch_goal: tuple[int, int] | None = None
        self._seen = 0
        self._recent: deque[tuple[int, int]] = deque(maxlen=12)
        self._sizes: deque[int] = deque(maxlen=12)
        self._bobber_size: int | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()
        self._armed_at = None
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.state = State.IDLE

    def resume(self) -> None:
        self._cancel.clear()

    def update(self, bobber: tuple[int, int] | None, size: int | None = None) -> None:
        self._bobber = bobber
        self._bobber_size = size
        if self._lost_at is not None and bobber is not None and self._cursor.get() is not None:
            self._returned = True
        if self.busy or self._cancel.is_set():
            return
        now = self._clock()
        if self._armed_at is None:
            self._armed_at = now + self._start_delay
            if self._start_delay > 0:
                self._log(f"Starting in {self._start_delay:g} s; switch to the game window")
        if now < self._armed_at:
            return
        if self.state is State.IDLE:
            self.state = State.CASTING
            self._spawn(self._cast)
        elif self.state is State.WAIT_BOBBER:
            self._wait_for_bobber(now)
        elif self.state is State.WATCHING:
            self._watch(now)

    def _wait_for_bobber(self, now: float) -> None:
        if now - self._since > self._bobber_timeout:
            self._log("No bobber appeared; casting again")
            self.state = State.IDLE
            return
        bobber = self._bobber
        if bobber is None or now - self._since < self._land_delay:
            self._anchor, self._stable = None, 0
            return
        if self._anchor is not None and math.dist(bobber, self._anchor) <= self._stable_radius:
            self._stable += 1
        else:
            self._anchor, self._stable = bobber, 1
        if self._stable >= self._stable_scans and self._cursor.get() is not None:
            self.state = State.MOVING
            self._spawn(self._move)

    def _watch(self, now: float) -> None:
        if now - self._since > self._fish_timeout:
            self._log("No bite; casting again")
            self.state = State.IDLE
            return
        bobber = self._bobber
        if bobber is None:
            if self._seen >= 6:
                self._log("Bobber vanished; moving in to see if it bounces back")
                self._lost_at = now
                self._returned = False
                self.state = State.CLICKING
                self._spawn(self._click)
            return
        self._seen += 1
        if self._watch_goal is not None and math.dist(bobber, self._watch_goal) > 30:
            self._watch_goal = bobber
            self._recent.clear()
            self._watcher.reset(bobber)
        self._recent.append(bobber)
        self._watch_goal = (
            round(statistics.median(p[0] for p in self._recent)),
            round(statistics.median(p[1] for p in self._recent)),
        )
        dipped = self._watcher.is_bite(bobber)
        size = self._bobber_size
        if size is not None:
            if len(self._sizes) >= 6 and size < 0.6 * statistics.median(self._sizes):
                dipped = True
            else:
                self._sizes.append(size)
        if dipped and abs(bobber[0] - self._watch_goal[0]) <= 15:
            self._log("Bite detected")
            self.state = State.CLICKING
            self._spawn(self._click)

    def _spawn(self, target: Callable[[], None]) -> None:
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def _cast(self) -> None:
        self._log(f"Casting with key {self._cast_key}")
        self._device.press_key(self._cast_key)
        self._anchor, self._stable = None, 0
        self._watch_goal = None
        self._since = self._clock()
        self.state = State.WAIT_BOBBER

    def _go_to(self, goal: tuple[int, int], tolerance: int | None = None) -> bool:
        return move_to(
            self._device,
            self._cursor,
            lambda: goal,
            self._cancel.is_set,
            tolerance=self._tolerance if tolerance is None else tolerance,
            calibration=self._calibration,
            log=self._log,
        )

    def _move(self) -> None:
        goal = self._bobber
        if goal is None:
            self.state = State.IDLE
            return
        park = (goal[0] + self._park_offset[0], goal[1] + self._park_offset[1])
        self._log("Moving cursor next to the bobber")
        try:
            parked = self._go_to(park, tolerance=15)
        except Exception as error:
            self._log(f"Cursor move failed: {error}")
            parked = False
        self._log("Cursor parked; waiting for a bite" if parked else "Cursor did not reach its parking spot; waiting anyway")
        self._watch_goal = goal
        self._seen = 0
        self._recent.clear()
        self._sizes.clear()
        self._lost_at = None
        if self._cancel.is_set():
            self.state = State.IDLE
            return
        self._watcher.reset(goal)
        self._since = self._clock()
        self.state = State.WATCHING

    def _click(self) -> None:
        try:
            goal = self._watch_goal
            reading = self._cursor.get()
            on_bobber = (
                goal is not None
                and reading is not None
                and math.dist(reading.position, goal) <= self._click_tolerance
            )
            if not on_bobber and goal is not None:
                self._log("Moving onto the bobber")
                on_bobber = self._go_to(goal)
            if self._lost_at is not None and on_bobber:
                deadline = self._lost_at + self._return_window
                while not self._returned and self._clock() < deadline and not self._cancel.is_set():
                    self._cancel.wait(0.03)
                if not self._returned:
                    self._log(
                        f"Bobber did not come back (it timed out {self._lost_at - self._since:.0f} s "
                        "into the watch); not clicking"
                    )
                    self._lost_at = None
                    self._cancel.wait(0.5)
                    self.state = State.IDLE
                    return
            if on_bobber and not self._cancel.is_set():
                self._device.click_right()
                self._log("Right-clicked bobber")
            else:
                self._log("Could not reach the bobber in time; not clicking")
        except Exception as error:
            self._log(f"Click failed: {error}")
        self._lost_at = None
        self._cancel.wait(self._loot_delay)
        self.state = State.IDLE
