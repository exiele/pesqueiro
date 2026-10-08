from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _data_dir() -> Path:
    override = os.environ.get("PESQUEIRO_HOME")
    if override:
        return Path(override)
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists():
        return checkout
    data_home = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(data_home) / "pesqueiro"


DATA_DIR = _data_dir()
DEFAULT_CURSOR_DIR = DATA_DIR / "cursors"
DEFAULT_SCALES = (1.0, 1.5, 2.0, 0.75)
_HOTSPOT = re.compile(r"@(\d+),(\d+)$")
_COARSE_SCALE = 4
_COARSE_CANDIDATES = 3
_COARSE_MAX_PIXELS = 24
_FINE_MAX_PIXELS = 160
_MAX_TEMPLATE_SIDE = 48
_EXTENSIONS = {".png", ".webp", ".gif", ".bmp"}


@dataclass(frozen=True)
class CursorTemplate:
    name: str
    gray: np.ndarray
    mask: np.ndarray
    hotspot: tuple[int, int]
    scale: float = 1.0


@dataclass(frozen=True)
class CursorMatch:
    position: tuple[int, int]
    score: float
    template: str
    scale: float = 1.0
    top_left: tuple[int, int] = (0, 0)
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)

    def screen_position(self, frame_origin: tuple[int, int]) -> tuple[int, int]:
        return self.position[0] + frame_origin[0], self.position[1] + frame_origin[1]


def blank_cursor(
    image: Image.Image, match: CursorMatch | None, offset: tuple[int, int] = (0, 0), margin: int = 3
) -> Image.Image:
    if match is None or match.mask is None:
        return image
    mask = Image.fromarray(np.pad(match.mask, margin).astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(2 * margin + 1))
    canvas = Image.new("L", image.size, 0)
    canvas.paste(mask, (match.top_left[0] - margin - offset[0], match.top_left[1] - margin - offset[1]))
    result = image.copy()
    result.paste((0, 0, 0), mask=canvas)
    return result


def _make_template(
    name: str, rgba: Image.Image, hotspot: tuple[int, int], scale: float
) -> CursorTemplate | None:
    box = rgba.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
    if box is None:
        return None
    rgba = rgba.crop(box)
    hotspot = (max(0, hotspot[0] - box[0]), max(0, hotspot[1] - box[1]))
    alpha = np.asarray(rgba.getchannel("A"))
    mask = alpha >= 250
    if not mask.any():
        mask = alpha >= 128
    gray = np.asarray(rgba.convert("L"), dtype=np.float32)
    if gray[mask].std() < 8:
        return None
    return CursorTemplate(name, gray, mask, hotspot, scale)


def load_cursor_templates(
    folder: Path | str = DEFAULT_CURSOR_DIR, scales: tuple[float, ...] = (1.0,)
) -> list[CursorTemplate]:
    templates: list[CursorTemplate] = []
    folder = Path(folder)
    if not folder.is_dir():
        return templates
    seen: set[bytes] = set()
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXTENSIONS)
    for path in paths:
        match = _HOTSPOT.search(path.stem)
        hotspot = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        with Image.open(path) as source:
            base = source.convert("RGBA")
        longest = max(base.size)
        if longest > _MAX_TEMPLATE_SIDE * 2:
            factor = 32 / longest
            base = base.resize(
                (max(1, round(base.width * factor)), max(1, round(base.height * factor))),
                Image.Resampling.LANCZOS,
            )
            hotspot = (round(hotspot[0] * factor), round(hotspot[1] * factor))
        for scale in scales:
            rgba = base
            scaled_hotspot = hotspot
            if scale != 1.0:
                rgba = base.resize(
                    (max(4, round(base.width * scale)), max(4, round(base.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                scaled_hotspot = (round(hotspot[0] * scale), round(hotspot[1] * scale))
            template = _make_template(path.stem, rgba, scaled_hotspot, scale)
            if template is None:
                continue
            key = template.gray[template.mask].tobytes() + template.mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            templates.append(template)
    return templates


def _zncc_map(
    frame: np.ndarray, gray: np.ndarray, mask: np.ndarray, max_pixels: int = 1 << 30
) -> np.ndarray:
    height, width = mask.shape
    out_h, out_w = frame.shape[0] - height + 1, frame.shape[1] - width + 1
    ys, xs = np.nonzero(mask)
    if len(ys) > max_pixels:
        keep = np.linspace(0, len(ys) - 1, max_pixels).astype(int)
        ys, xs = ys[keep], xs[keep]
    values = gray[ys, xs]
    count = len(values)
    sum_f = np.zeros((out_h, out_w), dtype=np.float64)
    sum_ff = np.zeros_like(sum_f)
    sum_ft = np.zeros_like(sum_f)
    for y, x, value in zip(ys, xs, values):
        window = frame[y : y + out_h, x : x + out_w]
        sum_f += window
        sum_ff += window * window
        sum_ft += window * value
    sum_t, sum_tt = float(values.sum()), float((values * values).sum())
    template_var = count * sum_tt - sum_t * sum_t
    frame_var = count * sum_ff - sum_f * sum_f
    numerator = count * sum_ft - sum_f * sum_t
    valid = (frame_var > 16 * count * count) & (template_var > 0)
    denominator = np.sqrt(np.where(valid, frame_var, 1.0) * max(template_var, 1e-6))
    return np.where(valid, numerator / denominator, -1.0).astype(np.float32)


def _downscale(template: CursorTemplate) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = template.mask.shape
    small = (max(1, width // _COARSE_SCALE), max(1, height // _COARSE_SCALE))
    premultiplied = Image.fromarray((template.gray * template.mask).astype(np.uint8))
    coverage = Image.fromarray(template.mask.astype(np.uint8) * 255)
    weight = np.asarray(coverage.resize(small, Image.Resampling.BOX), dtype=np.float32) / 255
    summed = np.asarray(premultiplied.resize(small, Image.Resampling.BOX), dtype=np.float32)
    mask = weight >= 0.5
    gray = (summed / np.maximum(weight, 1e-3)).clip(0, 255)
    return (gray, mask) if mask.sum() >= 4 else None


def _match_template(
    frame: np.ndarray, small_frame: np.ndarray, template: CursorTemplate
) -> tuple[float, tuple[int, int]] | None:
    height, width = template.mask.shape
    if frame.shape[0] < height or frame.shape[1] < width:
        return None

    coarse = _downscale(template)
    if coarse is None:
        return None
    c_gray, c_mask = coarse
    if small_frame.shape[0] < c_mask.shape[0] or small_frame.shape[1] < c_mask.shape[1]:
        return None
    scores = _zncc_map(small_frame, c_gray, c_mask, _COARSE_MAX_PIXELS)
    picks = np.argpartition(-scores.ravel(), min(_COARSE_CANDIDATES, scores.size) - 1)
    search = _COARSE_SCALE * 2

    best: tuple[float, tuple[int, int]] | None = None
    for index in picks[:_COARSE_CANDIDATES]:
        cx = int(index % scores.shape[1]) * _COARSE_SCALE
        cy = int(index // scores.shape[1]) * _COARSE_SCALE
        x0, y0 = max(0, cx - search), max(0, cy - search)
        x1 = min(frame.shape[1], cx + width + search)
        y1 = min(frame.shape[0], cy + height + search)
        patch = frame[y0:y1, x0:x1]
        if patch.shape[0] < height or patch.shape[1] < width:
            continue
        fine = _zncc_map(patch, template.gray, template.mask, _FINE_MAX_PIXELS)
        at = np.unravel_index(int(np.argmax(fine)), fine.shape)
        score = (1.0 - float(fine[at])) * 100
        if best is None or score < best[0]:
            best = (score, (x0 + int(at[1]), y0 + int(at[0])))
    return best


def find_cursor(
    image: Image.Image,
    templates: list[CursorTemplate],
    region: tuple[int, int, int, int] | None = None,
    threshold: float | None = 25.0,
) -> CursorMatch | None:
    if not templates:
        return None
    left, top = (region[0], region[1]) if region else (0, 0)
    source = (image.crop(region) if region else image).convert("L")
    frame = np.asarray(source, dtype=np.float32)
    small_frame = np.asarray(
        source.resize(
            (max(1, source.width // _COARSE_SCALE), max(1, source.height // _COARSE_SCALE)),
            Image.Resampling.BOX,
        ),
        dtype=np.float32,
    )

    best: CursorMatch | None = None
    for template in templates:
        result = _match_template(frame, small_frame, template)
        if result is None:
            continue
        score, (x, y) = result
        if (threshold is None or score <= threshold) and (best is None or score < best.score):
            best = CursorMatch(
                (left + x + template.hotspot[0], top + y + template.hotspot[1]),
                score,
                template.name,
                template.scale,
                (left + x, top + y),
                template.mask,
            )
    return best
