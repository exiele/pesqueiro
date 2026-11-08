from __future__ import annotations

import math
import random
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
    def click_left(self) -> None: ...
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
        cast_key: str = "F12",
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
        lure_key: str = "F11",
        rod_position: Callable[[], tuple[int, int] | None] = lambda: None,
    ) -> None:
        self._device = device
        self._cursor = cursor
        self._log = log
        self._cast_key = cast_key
        self._lure_key = lure_key
        self._rod_position = rod_position
        self._next_lure_at: float | None = None
        self._cast_started_at: float | None = None
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
        self._clock = clock
        self._watcher = BiteWatcher()
        self._calibration = PointerCalibration()
        self.state = State.IDLE
        self._since = 0.0
        self._bobber: tuple[int, int] | None = None
        self._anchor: tuple[int, int] | None = None
        self._stable = 0
        self._watch_goal: tuple[int, int] | None = None
        self._recent: deque[tuple[int, int]] = deque(maxlen=12)
        self._displaced_anchor: tuple[int, int] | None = None
        self._displaced: list[tuple[int, int]] = []
        self._bobber_size: int | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._cursor_on_bobber = False
        self._waiting_for_cursor = False

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()
        self._armed_at = None
        self._next_lure_at = None
        self._waiting_for_cursor = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.state = State.IDLE

    def resume(self) -> None:
        self._cancel.clear()

    def update(
        self,
        bobber: tuple[int, int] | None,
        size: int | None = None,
        cursor_on_bobber: bool = False,
    ) -> None:
        self._bobber = bobber
        self._bobber_size = size
        self._cursor_on_bobber = cursor_on_bobber
        if self.busy or self._cancel.is_set():
            return
        now = self._clock()
        if self._armed_at is None:
            self._armed_at = now + self._start_delay
            self._next_lure_at = self._armed_at + random.uniform(480.0, 600.0)
            if self._start_delay > 0:
                self._log(f"Starting in {self._start_delay:g} s; switch to the game window")
        if now < self._armed_at:
            return
        if self.state is State.IDLE:
            if self._cursor.get() is None:
                if not self._waiting_for_cursor:
                    self._log("Waiting for cursor tracking before casting")
                    self._waiting_for_cursor = True
                return
            if self._waiting_for_cursor:
                self._log("Cursor tracking available; resuming fishing")
                self._waiting_for_cursor = False
            if self._next_lure_at is not None and now >= self._next_lure_at and self._rod_position() is not None:
                if self._cast_started_at is not None and now < self._cast_started_at + self._fish_timeout:
                    return
                self.state = State.CASTING
                self._spawn(self._apply_lure)
                return
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
        if self._stable < self._stable_scans:
            return
        if self._cursor.get() is not None:
            self.state = State.MOVING
            self._spawn(self._move)

    def _watch(self, now: float) -> None:
        if now - self._since > self._fish_timeout:
            self._log("No bite; casting again")
            self.state = State.IDLE
            return
        bobber = self._bobber
        if bobber is None:
            self._displaced_anchor = None
            self._displaced.clear()
            return
        if self._watch_goal is not None and (
            abs(bobber[0] - self._watch_goal[0]) > 30
            or abs(bobber[1] - self._watch_goal[1]) > 100
        ):
            if self._displaced_anchor is not None and math.dist(bobber, self._displaced_anchor) <= self._stable_radius:
                self._displaced.append(bobber)
            else:
                self._displaced_anchor = bobber
                self._displaced = [bobber]
            if len(self._displaced) < self._stable_scans:
                return
            self._watch_goal = bobber
            self._recent.clear()
            self._watcher.reset(self._displaced[0])
            self._recent.extend(self._displaced[:-1])
            for position in self._displaced[1:-1]:
                self._watcher.is_bite(position)
        self._displaced_anchor = None
        self._displaced.clear()
        self._recent.append(bobber)
        self._watch_goal = (
            round(statistics.median(p[0] for p in self._recent)),
            round(statistics.median(p[1] for p in self._recent)),
        )
        if self._watcher.is_bite(bobber) and abs(bobber[0] - self._watch_goal[0]) <= 30:
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
        self._displaced_anchor = None
        self._displaced.clear()
        self._since = self._clock()
        self._cast_started_at = self._since
        self.state = State.WAIT_BOBBER

    def _apply_lure(self) -> None:
        try:
            goal = self._rod_position()
            if goal is None or not self._go_to(goal, tolerance=6) or self._cancel.is_set():
                self._log("Could not reach the fishing rod; retrying the lure later")
                self._next_lure_at = self._clock() + 30.0
                return
            self._log(f"Applying lure with key {self._lure_key}")
            self._device.press_key(self._lure_key)
            if self._cancel.is_set():
                return
            self._device.click_left()
            if not self._cancel.wait(random.uniform(6.0, 6.5)):
                self._next_lure_at = self._clock() + random.uniform(480.0, 600.0)
        except Exception as error:
            self._log(f"Lure application failed: {error}")
            self._next_lure_at = self._clock() + 30.0
        finally:
            self.state = State.IDLE

    def _go_to(
        self,
        goal: tuple[int, int],
        tolerance: int | None = None,
        quick: bool = False,
    ) -> bool:
        return move_to(
            self._device,
            self._cursor,
            lambda: goal,
            self._cancel.is_set,
            tolerance=self._tolerance if tolerance is None else tolerance,
            settle=0.08 if quick else 0.25,
            fresh_timeout=0.6 if quick else 2.0,
            calibration=self._calibration,
            log=self._log,
            path_duration_scale=0.5 if quick else 1.0,
            allow_unconfirmed=not quick,
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
        self._recent.clear()
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
                self._cursor_on_bobber
                or (
                    goal is not None
                    and reading is not None
                    and math.dist(reading.position, goal) <= self._click_tolerance
                )
            )
            if not on_bobber and goal is not None:
                self._log("Moving onto the bobber")
                on_bobber = self._go_to(goal, tolerance=min(10, self._click_tolerance), quick=True)
            if on_bobber and not self._cancel.is_set():
                self._device.click_right()
                self._cast_started_at = None
                self._log("Right-clicked bobber")
            else:
                self._log("Could not reach the bobber in time; not clicking")
        except Exception as error:
            self._log(f"Click failed: {error}")
        self._cancel.wait(self._loot_delay)
        self.state = State.IDLE
