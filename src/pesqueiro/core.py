from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median

import numpy as np
from PIL import Image


class ClassifierMode(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class PixelClassifier:
    mode: ClassifierMode = ClassifierMode.RED
    colour_multiplier: float = 0.5
    colour_closeness_multiplier: float = 2.0

    def is_match(self, red: int, green: int, blue: int) -> bool:
        if self.mode is ClassifierMode.RED:
            dominant, first, second = red, green, blue
        else:
            dominant, first, second = blue, green, red

        return (
            self._is_bigger(dominant, first)
            and self._is_bigger(dominant, second)
            and self._are_close(first, second)
        )

    def _is_bigger(self, colour: int, other: int) -> bool:
        return colour * self.colour_multiplier > other

    def _are_close(self, first: int, second: int) -> bool:
        maximum = max(first, second)
        minimum = min(first, second)
        return minimum * self.colour_closeness_multiplier > maximum - 20


@dataclass(frozen=True)
class BobberMatch:
    position: tuple[int, int]
    score: int
    matching_pixels: int

    def screen_position(self, frame_origin: tuple[int, int]) -> tuple[int, int]:
        return (
            self.position[0] + frame_origin[0],
            self.position[1] + frame_origin[1],
        )


def find_bobber(
    image: Image.Image,
    classifier: PixelClassifier,
    previous_position: tuple[int, int] | None = None,
    search_radius: int = 100,
    max_matching_pixels: int = 5000,
) -> BobberMatch | None:
    width, height = image.size
    if previous_position is None:
        min_x, max_x, min_y, max_y = 0, width, 0, height
    else:
        previous_x, previous_y = previous_position
        min_x = max(previous_x - search_radius, 0)
        max_x = min(previous_x + search_radius, width)
        min_y = max(previous_y - search_radius, 0)
        max_y = min(previous_y + search_radius, height)

    if min_x >= max_x or min_y >= max_y:
        return None
    search_image = image if previous_position is None else image.crop((min_x, min_y, max_x, max_y))
    area = np.asarray(search_image.convert("RGB"), dtype=np.int16)
    red, green, blue = area.transpose(2, 0, 1)
    if classifier.mode is ClassifierMode.RED:
        dominant, first, second = red, green, blue
    else:
        dominant, first, second = blue, green, red
    matches = (
        (dominant * classifier.colour_multiplier > first)
        & (dominant * classifier.colour_multiplier > second)
        & (np.minimum(first, second) * classifier.colour_closeness_multiplier > np.maximum(first, second) - 20)
    )
    ys, xs = np.nonzero(matches)
    matching_pixels = len(xs)
    if matching_pixels == 0 or matching_pixels > max_matching_pixels:
        return None

    xs += min_x
    ys += min_y
    order = np.lexsort((ys, xs))
    xs, ys = xs[order], ys[order]
    integral = np.pad(matches.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    left = np.maximum(xs - min_x - 9, 0)
    right = np.minimum(xs - min_x + 10, matches.shape[1])
    top = np.maximum(ys - min_y - 9, 0)
    bottom = np.minimum(ys - min_y + 10, matches.shape[0])
    scores = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
    best = int(np.argmax(scores))
    best_position = (int(xs[best]), int(ys[best]))
    best_score = int(scores[best])
    body_left = max(0, best_position[0] - 8)
    body_top = max(0, best_position[1] - 6)
    body_right = min(width, best_position[0] + 72)
    body_bottom = min(height, best_position[1] + 72)
    body = np.asarray(image.crop((body_left, body_top, body_right, body_bottom)).convert("RGB"), dtype=np.int16)
    body_position = _attached_float_center(body, (best_position[0] - body_left, best_position[1] - body_top))
    if body_position is not None:
        best_position = (body_position[0] + body_left, body_position[1] + body_top)

    return BobberMatch(best_position, best_score, matching_pixels)


def _attached_float_center(
    rgb: np.ndarray, anchor: tuple[int, int]
) -> tuple[int, int] | None:
    x, y = anchor
    left, right = max(0, x - 8), min(rgb.shape[1], x + 72)
    top, bottom = max(0, y - 6), min(rgb.shape[0], y + 72)
    patch = rgb[top:bottom, left:right]
    red, green, blue = patch.transpose(2, 0, 1)
    tan = (
        (red > green * 1.08)
        & (green > blue * 1.12)
        & (red > 55)
        & (green > 35)
        & (red < 200)
    )
    ys, xs = np.nonzero(tan)
    if len(xs) < 40:
        return None

    visited = np.zeros(tan.shape, dtype=bool)
    largest: list[tuple[int, int]] = []
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        component = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        for current_y, current_x in component:
            for neighbor_y, neighbor_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= neighbor_y < tan.shape[0]
                    and 0 <= neighbor_x < tan.shape[1]
                    and tan[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    component.append((neighbor_y, neighbor_x))
        if len(component) > len(largest):
            largest = component
    if len(largest) < 40:
        return None
    body_y, body_x = zip(*largest)
    return (
        left + (min(body_x) + max(body_x)) // 2,
        top + (min(body_y) + max(body_y)) // 2,
    )


class BiteWatcher:
    def __init__(self, strike_value: int = 12) -> None:
        if strike_value < 1:
            raise ValueError("strike_value must be positive")
        self.strike_value = strike_value
        self._positions: deque[int] = deque(maxlen=12)
        self._dip_baseline: float | None = None
        self._excursion_positions: deque[int] = deque(maxlen=20)

    def reset(self, initial_position: tuple[int, int]) -> None:
        self._positions.clear()
        self._positions.append(initial_position[1])
        self._dip_baseline = None
        self._excursion_positions.clear()

    def is_bite(self, current_position: tuple[int, int]) -> bool:
        current_y = current_position[1]
        if not self._positions:
            self._positions.append(current_y)
            return False
        if self._dip_baseline is not None:
            if abs(current_y - self._dip_baseline) <= self.strike_value / 2:
                self._dip_baseline = None
                self._excursion_positions.clear()
                self._positions.append(current_y)
                return True
            self._excursion_positions.append(current_y)
            if len(self._excursion_positions) == self._excursion_positions.maxlen:
                self._positions.clear()
                self._positions.extend(self._excursion_positions)
                self._excursion_positions.clear()
                self._dip_baseline = None
            return False
        baseline = median(self._positions)
        if abs(current_y - baseline) >= self.strike_value:
            self._dip_baseline = baseline
            self._excursion_positions.append(current_y)
        else:
            self._positions.append(current_y)
        return False