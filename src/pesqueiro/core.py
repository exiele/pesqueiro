from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
    search_radius: int = 40,
    max_matching_pixels: int = 1000,
) -> BobberMatch | None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if previous_position is None:
        min_x, max_x, min_y, max_y = 0, width, 0, height
    else:
        previous_x, previous_y = previous_position
        min_x = max(previous_x - search_radius, 0)
        max_x = min(previous_x + search_radius, width)
        min_y = max(previous_y - search_radius, 0)
        max_y = min(previous_y + search_radius, height)

    points: list[tuple[int, int]] = []
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            if classifier.is_match(*rgb.getpixel((x, y))):
                points.append((x, y))
                if len(points) > max_matching_pixels:
                    return None

    if not points:
        return None

    best_position = points[0]
    best_score = -1
    for point_x, point_y in points:
        score = sum(
            abs(other_x - point_x) < 10 and abs(other_y - point_y) < 10
            for other_x, other_y in points
        )
        if score > best_score:
            best_position = (point_x, point_y)
            best_score = score

    return BobberMatch(best_position, best_score, len(points))


class BiteWatcher:
    def __init__(self, strike_value: int = 7) -> None:
        if strike_value < 1:
            raise ValueError("strike_value must be positive")
        self.strike_value = strike_value
        self._positions: list[int] = []

    def reset(self, initial_position: tuple[int, int]) -> None:
        self._positions = [initial_position[1]]

    def is_bite(self, current_position: tuple[int, int]) -> bool:
        current_y = current_position[1]
        if current_y not in self._positions:
            self._positions.append(current_y)
            self._positions.sort()

        median_index = int((len(self._positions) + 0.5) / 2)
        y_difference = self._positions[median_index] - current_y
        return y_difference <= -self.strike_value