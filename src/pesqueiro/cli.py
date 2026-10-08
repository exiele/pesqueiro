from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from PIL import Image

from .core import ClassifierMode, PixelClassifier, find_bobber


def _point(value: str) -> tuple[int, int]:
    try:
        x_value, y_value = value.split(",", maxsplit=1)
        return int(x_value), int(y_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected X,Y integers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pesqueiro",
        description="Inspect a saved game frame or capture one through the Wayland portal.",
    )
    parser.add_argument("image", type=Path, nargs="?", help="path to a saved screenshot")
    parser.add_argument(
        "--wayland",
        action="store_true",
        help="capture one frame using the desktop ScreenCast permission portal",
    )
    parser.add_argument("--mode", choices=[mode.value for mode in ClassifierMode], default="red")
    parser.add_argument("--origin", type=_point, default=(0, 0), metavar="X,Y")
    parser.add_argument("--previous", type=_point, metavar="X,Y")
    parser.add_argument("--radius", type=int, default=40)
    parser.add_argument("--colour-multiplier", type=float, default=0.5)
    parser.add_argument("--closeness-multiplier", type=float, default=2.0)
    return parser


async def _capture_wayland():
    from .wayland import WaylandScreenCapture

    async with WaylandScreenCapture() as capture:
        frame = await capture.capture()
        return frame.image, frame.origin, frame.stream_id


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.wayland == (arguments.image is not None):
        build_parser().error("provide either an image path or --wayland")

    source = "wayland-portal" if arguments.wayland else str(arguments.image)
    stream_id = None
    try:
        if arguments.wayland:
            image, origin, stream_id = asyncio.run(_capture_wayland())
        else:
            with Image.open(arguments.image) as input_image:
                image = input_image.convert("RGB")
            origin = arguments.origin
    except (OSError, RuntimeError) as error:
        print(f"pesqueiro: {error}", file=sys.stderr)
        return 3

    classifier = PixelClassifier(
        mode=ClassifierMode(arguments.mode),
        colour_multiplier=arguments.colour_multiplier,
        colour_closeness_multiplier=arguments.closeness_multiplier,
    )
    match = find_bobber(
        image,
        classifier,
        previous_position=arguments.previous,
        search_radius=arguments.radius,
    )
    result = {
        "source": source,
        "stream_id": stream_id,
        "frame_size": list(image.size),
        "frame_origin": list(origin),
        "mode": arguments.mode,
        "bobber": None,
    }
    if match is not None:
        result["bobber"] = {
            "frame_position": list(match.position),
            "screen_position": list(match.screen_position(origin)),
            "cluster_score": match.score,
            "matching_pixels": match.matching_pixels,
        }

    print(json.dumps(result, indent=2))
    return 0 if match is not None else 2