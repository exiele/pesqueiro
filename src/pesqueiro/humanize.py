from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    dx: int
    dy: int
    delay: float


def build_path(
    start: tuple[float, float],
    end: tuple[float, float],
    rng: random.Random | None = None,
    overshoot: bool = True,
) -> list[Step]:
    rng = rng or random.Random()
    distance = math.dist(start, end)
    if distance < 1:
        return []

    if overshoot and distance > 150 and rng.random() < 0.5:
        ux, uy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
        extra = rng.uniform(4, 14)
        aim = (end[0] + ux * extra, end[1] + uy * extra)
        return _leg(start, aim, rng) + _leg(aim, end, rng, fast=False)
    return _leg(start, end, rng)


def _leg(
    start: tuple[float, float],
    end: tuple[float, float],
    rng: random.Random,
    fast: bool = True,
) -> list[Step]:
    distance = math.dist(start, end)
    if distance < 1:
        return []
    nx, ny = -(end[1] - start[1]) / distance, (end[0] - start[0]) / distance

    def control(fraction: float) -> tuple[float, float]:
        bend = rng.uniform(-0.25, 0.25) * distance
        return (
            start[0] + (end[0] - start[0]) * fraction + nx * bend,
            start[1] + (end[1] - start[1]) * fraction + ny * bend,
        )

    p1, p2 = control(rng.uniform(0.2, 0.4)), control(rng.uniform(0.6, 0.8))
    duration = (0.25 + distance / 1800) * rng.uniform(0.8, 1.3) if fast else rng.uniform(0.08, 0.18)
    count = max(6, int(duration * 90))

    steps: list[Step] = []
    last = (round(start[0]), round(start[1]))
    for index in range(1, count + 1):
        t = index / count
        eased = t * t * (3 - 2 * t)
        u = 1 - eased
        x = u**3 * start[0] + 3 * u * u * eased * p1[0] + 3 * u * eased * eased * p2[0] + eased**3 * end[0]
        y = u**3 * start[1] + 3 * u * u * eased * p1[1] + 3 * u * eased * eased * p2[1] + eased**3 * end[1]
        if index < count:
            x += rng.gauss(0, 0.6)
            y += rng.gauss(0, 0.6)
        point = (round(x), round(y)) if index < count else (round(end[0]), round(end[1]))
        dx, dy = point[0] - last[0], point[1] - last[1]
        last = point
        if dx or dy:
            steps.append(Step(dx, dy, duration / count * rng.uniform(0.7, 1.3)))
    return steps
