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


_KEY_ALIASES = {
    "[": "LEFTBRACE",
    "]": "RIGHTBRACE",
    "-": "MINUS",
    "=": "EQUAL",
    ";": "SEMICOLON",
    "'": "APOSTROPHE",
    "`": "GRAVE",
    "\\": "BACKSLASH",
    ",": "COMMA",
    ".": "DOT",
    "/": "SLASH",
    "ctrl": "LEFTCTRL",
    "shift": "LEFTSHIFT",
    "alt": "LEFTALT",
    "esc": "ESC",
    "tab": "TAB",
    "enter": "ENTER",
}


class UInputDevice:
    def __init__(self, keys: list[str] | None = None, rng: random.Random | None = None) -> None:
        try:
            from evdev import UInput, ecodes
        except ImportError as error:
            raise MouseError("python-evdev is missing; install pesqueiro with the 'move' extra") from error
        self._ecodes = ecodes
        self._rng = rng or random.Random()
        self._key_codes = {name.lower(): self._key_code(name) for name in keys or []}
        mouse_caps = {
            ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y],
            ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT],
        }
        # A keyboard needs many keys or udev/libinput will not classify it as one.
        keyboard_caps = {ecodes.EV_KEY: list(range(1, 249))}
        try:
            self._device = UInput(mouse_caps, name="pesqueiro-mouse")
            self._keyboard = UInput(keyboard_caps, name="pesqueiro-keyboard")
        except OSError as error:
            raise MouseError(
                f"cannot open /dev/uinput ({error}); grant your user write access, "
                "e.g. a udev rule with GROUP=\"input\", MODE=\"0660\" and add yourself to that group"
            ) from error
        # Let the compositor register the new devices before the first event.
        time.sleep(1.0)

    def _key_code(self, name: str) -> int:
        if name.lower().startswith("code:"):
            return int(name[5:])
        key = name.strip()
        key = _KEY_ALIASES.get(key.lower(), key)
        code = getattr(self._ecodes, f"KEY_{key.upper()}", None)
        if not isinstance(code, int):
            raise MouseError(f"unknown key '{name}'; use a name like 4, f1 or a")
        return code

    def move_relative(self, dx: int, dy: int) -> None:
        self._device.write(self._ecodes.EV_REL, self._ecodes.REL_X, dx)
        self._device.write(self._ecodes.EV_REL, self._ecodes.REL_Y, dy)
        self._device.syn()

    def click_right(self) -> None:
        self._tap(self._device, self._ecodes.BTN_RIGHT)

    def click_left(self) -> None:
        self._tap(self._device, self._ecodes.BTN_LEFT)

    def press_key(self, name: str) -> None:
        self._tap(self._keyboard, self._key_codes[name.lower()])

    def _tap(self, device, code: int) -> None:
        device.write(self._ecodes.EV_KEY, code, 1)
        device.syn()
        time.sleep(self._rng.uniform(0.05, 0.12))
        device.write(self._ecodes.EV_KEY, code, 0)
        device.syn()

    def close(self) -> None:
        self._device.close()
        self._keyboard.close()


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

    def observe(self, commanded: tuple[int, int], observed: tuple[float, float]) -> tuple[int, ...]:
        ignored: list[int] = []
        for axis in (0, 1):
            if abs(commanded[axis]) < self.MIN_COMMAND:
                continue
            gain = observed[axis] / commanded[axis]
            if not 0.15 <= gain <= 8 or (
                self.measured[axis] and not 0.5 * self.gain[axis] <= gain <= 2 * self.gain[axis]
            ):
                if self.measured[axis]:
                    ignored.append(axis)
                    continue
                raise PointerMismatch(
                    f"cursor moved {observed[axis]:.0f}px on axis {'xy'[axis]} for a command of "
                    f"{commanded[axis]} (gain {gain:.2f}); input is not mapping to the captured screen"
                )
            self.gain[axis] = gain if not self.measured[axis] else 0.5 * (self.gain[axis] + gain)
            self.measured[axis] = True
        return tuple(ignored)


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
    path_duration_scale: float = 1.0,
    allow_unconfirmed: bool = True,
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
            if abs(distance) > max(40, tolerance * 2):
                travel = abs(distance) * 0.6
                if not calibration.measured[axis]:
                    travel = min(travel, calibration.PROBE / 2)
                leg_goal[axis] = reading.position[axis] + math.copysign(travel, distance)
        gain_x, gain_y = calibration.gain
        start = (reading.position[0] / gain_x, reading.position[1] / gain_y)
        end = (leg_goal[0] / gain_x, leg_goal[1] / gain_y)
        full_leg = tuple(leg_goal) == tuple(goal)
        steps = build_path(
            start,
            end,
            rng,
            overshoot=False,
            duration_scale=path_duration_scale,
        )
        log(
            f"Leg {attempt + 1}: cursor {reading.position} -> {tuple(round(v) for v in leg_goal)} "
            f"(bobber {goal}, gain {gain_x:.2f},{gain_y:.2f})"
        )
        _play(mouse, steps, cancelled)

        sent_at = time.monotonic() + settle
        time.sleep(settle)
        fresh = _wait_fresh(cursor, sent_at, fresh_timeout, cancelled)
        if fresh is None:
            if allow_unconfirmed and math.dist(leg_goal, goal) <= 40 and not cancelled():
                log("Cursor no longer recognised at the target; assuming it arrived")
                return True
            if not cancelled() and steps:
                log("Cursor lost during move; retracing the last leg")
                reverse = [Step(-step.dx, -step.dy, step.delay) for step in reversed(steps)]
                _play(mouse, reverse, cancelled)
                returned_at = time.monotonic() + settle
                time.sleep(settle)
                if _wait_fresh(cursor, returned_at, fresh_timeout, cancelled) is not None:
                    log("Cursor tracking recovered")
                elif not cancelled():
                    log("Cursor tracking still lost; move it back into the captured screen")
            return False
        commanded = (sum(s.dx for s in steps), sum(s.dy for s in steps))
        observed = (fresh.position[0] - reading.position[0], fresh.position[1] - reading.position[1])
        try:
            ignored = calibration.observe(commanded, observed)
            if (
                len(ignored) == 1
                and full_leg
                and gap <= 60
                and all(calibration.measured)
                and allow_unconfirmed
                and not cancelled()
            ):
                log("Cursor changed form near target; trusting calibrated movement")
                return True
            for axis in ignored:
                log(
                    f"Ignoring inconsistent cursor reading on axis {'xy'[axis]}; "
                    f"keeping calibrated gain {calibration.gain[axis]:.2f}"
                )
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
