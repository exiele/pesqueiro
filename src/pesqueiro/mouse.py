from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .humanize import Step, build_path


class MouseError(RuntimeError):
    pass


class Mouse(Protocol):
    def move_relative(self, dx: int, dy: int) -> None: ...


class UInputDevice:
    def __init__(self, keys: list[str] | None = None, rng: random.Random | None = None) -> None:
        try:
            from evdev import UInput, ecodes
        except ImportError as error:
            raise MouseError("python-evdev is missing; install pesqueiro with the 'move' extra") from error
        self._ecodes = ecodes
        self._rng = rng or random.Random()
        self._key_codes = {name.lower(): self._key_code(name) for name in keys or []}
        capabilities = {
            ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y],
            ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, *self._key_codes.values()],
        }
        try:
            self._device = UInput(capabilities, name="pesqueiro-input")
        except OSError as error:
            raise MouseError(
                f"cannot open /dev/uinput ({error}); grant your user write access, "
                "e.g. a udev rule with GROUP=\"input\", MODE=\"0660\" and add yourself to that group"
            ) from error

    def _key_code(self, name: str) -> int:
        code = getattr(self._ecodes, f"KEY_{name.strip().upper()}", None)
        if not isinstance(code, int):
            raise MouseError(f"unknown key '{name}'; use a name like 4, f1 or a")
        return code

    def move_relative(self, dx: int, dy: int) -> None:
        self._device.write(self._ecodes.EV_REL, self._ecodes.REL_X, dx)
        self._device.write(self._ecodes.EV_REL, self._ecodes.REL_Y, dy)
        self._device.syn()

    def click_right(self) -> None:
        self._tap(self._ecodes.BTN_RIGHT)

    def press_key(self, name: str) -> None:
        self._tap(self._key_codes[name.lower()])

    def _tap(self, code: int) -> None:
        self._device.write(self._ecodes.EV_KEY, code, 1)
        self._device.syn()
        time.sleep(self._rng.uniform(0.05, 0.12))
        self._device.write(self._ecodes.EV_KEY, code, 0)
        self._device.syn()

    def close(self) -> None:
        self._device.close()


@dataclass(frozen=True)
class Reading:
    position: tuple[int, int]
    timestamp: float


class LatestReading:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reading: Reading | None = None

    def set(self, position: tuple[int, int] | None, timestamp: float | None = None) -> None:
        with self._lock:
            self._reading = (
                None
                if position is None
                else Reading(position, time.monotonic() if timestamp is None else timestamp)
            )

    def get(self) -> Reading | None:
        with self._lock:
            return self._reading


def _play(mouse: Mouse, steps: list[Step], cancelled: Callable[[], bool]) -> None:
    for step in steps:
        if cancelled():
            return
        mouse.move_relative(step.dx, step.dy)
        time.sleep(step.delay)


class PointerMismatch(MouseError):
    pass


class PointerCalibration:
    MIN_COMMAND = 15
    PROBE = 60

    def __init__(self) -> None:
        self.gain = [1.0, 1.0]
        self.measured = [False, False]

    def observe(self, commanded: tuple[int, int], observed: tuple[float, float]) -> None:
        for axis in (0, 1):
            if abs(commanded[axis]) < self.MIN_COMMAND:
                continue
            gain = observed[axis] / commanded[axis]
            if not 0.15 <= gain <= 8:
                raise PointerMismatch(
                    f"cursor moved {observed[axis]:.0f}px on axis {'xy'[axis]} for a command of "
                    f"{commanded[axis]} (gain {gain:.2f}); input is not mapping to the captured screen"
                )
            self.gain[axis] = gain if not self.measured[axis] else 0.5 * (self.gain[axis] + gain)
            self.measured[axis] = True


def _wait_fresh(cursor: LatestReading, after: float, timeout: float, cancelled: Callable[[], bool]):
    deadline = after + timeout
    while not cancelled() and time.monotonic() < deadline:
        fresh = cursor.get()
        if fresh is not None and fresh.timestamp > after:
            return fresh
        time.sleep(0.03)
    return None


def move_to(
    mouse: Mouse,
    cursor: LatestReading,
    target: Callable[[], tuple[int, int] | None],
    cancelled: Callable[[], bool] = lambda: False,
    tolerance: int = 6,
    max_corrections: int = 5,
    settle: float = 0.25,
    fresh_timeout: float = 2.0,
    rng: random.Random | None = None,
    calibration: PointerCalibration | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> bool:
    rng = rng or random.Random()
    calibration = calibration or PointerCalibration()
    for attempt in range(max_corrections + 1):
        reading = cursor.get()
        goal = target()
        if cancelled() or reading is None or goal is None:
            return False
        gap = math.dist(reading.position, goal)
        if gap <= tolerance:
            return True

        leg_goal = list(goal)
        for axis in (0, 1):
            distance = goal[axis] - reading.position[axis]
            if not calibration.measured[axis] and abs(distance) > calibration.PROBE:
                leg_goal[axis] = reading.position[axis] + math.copysign(calibration.PROBE, distance)
        gain_x, gain_y = calibration.gain
        start = (reading.position[0] / gain_x, reading.position[1] / gain_y)
        end = (leg_goal[0] / gain_x, leg_goal[1] / gain_y)
        full_leg = tuple(leg_goal) == tuple(goal)
        steps = build_path(start, end, rng, overshoot=attempt == 0 and full_leg and all(calibration.measured))
        log(
            f"Leg {attempt + 1}: cursor {reading.position} -> {tuple(round(v) for v in leg_goal)} "
            f"(bobber {goal}, gain {gain_x:.2f},{gain_y:.2f})"
        )
        _play(mouse, steps, cancelled)

        sent_at = time.monotonic() + settle
        time.sleep(settle)
        fresh = _wait_fresh(cursor, sent_at, fresh_timeout, cancelled)
        if fresh is None:
            if math.dist(leg_goal, goal) <= 40 and not cancelled():
                log("Cursor no longer recognised at the target; assuming it arrived")
                return True
            return False
        commanded = (sum(s.dx for s in steps), sum(s.dy for s in steps))
        observed = (fresh.position[0] - reading.position[0], fresh.position[1] - reading.position[1])
        try:
            calibration.observe(commanded, observed)
        except PointerMismatch as error:
            again = _wait_fresh(cursor, fresh.timestamp, 0.8, cancelled)
            try:
                if again is None:
                    raise error
                calibration.observe(
                    commanded,
                    (again.position[0] - reading.position[0], again.position[1] - reading.position[1]),
                )
            except PointerMismatch as final:
                log(str(final))
                return False
    reading, goal = cursor.get(), target()
    return reading is not None and goal is not None and math.dist(reading.position, goal) <= tolerance
