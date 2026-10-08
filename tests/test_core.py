import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from pesqueiro.core import BiteWatcher, ClassifierMode, PixelClassifier, find_bobber
from pesqueiro.wayland import (
    WaylandCaptureError,
    _build_pipewire_command,
    _caps_dimensions,
    _decode_rgba_frame,
    _received_unix_fd,
    _stream_geometry,
)


class PixelClassifierTests(unittest.TestCase):
    def test_red_mode_matches_red_and_rejects_blue(self):
        classifier = PixelClassifier()
        self.assertTrue(classifier.is_match(255, 40, 45))
        self.assertFalse(classifier.is_match(30, 45, 240))

    def test_blue_mode_matches_blue(self):
        classifier = PixelClassifier(mode=ClassifierMode.BLUE)
        self.assertTrue(classifier.is_match(30, 45, 240))
        self.assertFalse(classifier.is_match(255, 40, 45))

    def test_equal_intensity_boundary_does_not_match(self):
        classifier = PixelClassifier()
        self.assertFalse(classifier.is_match(100, 50, 50))


class BobberFinderTests(unittest.TestCase):
    def test_returns_center_of_densest_matching_cluster(self):
        image = Image.new("RGB", (40, 30), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 13, 13), fill=(255, 20, 20))
        draw.point((30, 20), fill=(255, 20, 20))

        result = find_bobber(image, PixelClassifier())

        self.assertIsNotNone(result)
        self.assertEqual((10, 10), result.position)
        self.assertEqual(16, result.score)
        self.assertEqual(17, result.matching_pixels)
        self.assertEqual((310, 210), result.screen_position((300, 200)))

    def test_previous_position_limits_search_area(self):
        image = Image.new("RGB", (100, 60), "black")
        ImageDraw.Draw(image).point((80, 30), fill=(255, 20, 20))

        self.assertIsNone(find_bobber(image, PixelClassifier(), (10, 10)))

    def test_excess_matching_pixels_are_rejected(self):
        image = Image.new("RGB", (5, 5), (255, 20, 20))

        self.assertIsNone(
            find_bobber(image, PixelClassifier(), max_matching_pixels=10)
        )


class BiteWatcherTests(unittest.TestCase):
    def test_bite_requires_downward_strike_threshold(self):
        watcher = BiteWatcher(strike_value=7)
        watcher.reset((100, 100))

        for y_position in range(101, 107):
            self.assertFalse(watcher.is_bite((100, y_position)))
        self.assertTrue(watcher.is_bite((100, 113)))

    def test_reset_discards_previous_observations(self):
        watcher = BiteWatcher(strike_value=7)
        watcher.reset((100, 100))
        watcher.is_bite((100, 90))
        watcher.reset((100, 50))

        for y_position in range(51, 57):
            self.assertFalse(watcher.is_bite((100, y_position)))
        self.assertTrue(watcher.is_bite((100, 63)))


class CommandLineTests(unittest.TestCase):
    def test_cli_reports_frame_and_screen_coordinates(self):
        image = Image.new("RGB", (20, 20), "black")
        ImageDraw.Draw(image).rectangle((5, 5, 7, 7), fill=(255, 20, 20))
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "frame.png"
            image.save(image_path)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pesqueiro",
                    str(image_path),
                    "--origin",
                    "300,200",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(0, process.returncode, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual([5, 5], result["bobber"]["frame_position"])
        self.assertEqual([305, 205], result["bobber"]["screen_position"])


class WaylandCaptureTests(unittest.TestCase):
    def test_stream_geometry_unwraps_portal_variant_values(self):
        class VariantValue:
            def __init__(self, value):
                self.value = value

        size, origin = _stream_geometry(
            {"size": VariantValue((1920, 1080)), "position": VariantValue((120, 40))}
        )

        self.assertEqual((1920, 1080), size)
        self.assertEqual((120, 40), origin)

    def test_stream_geometry_allows_portals_that_omit_window_size(self):
        size, origin = _stream_geometry({"position": (120, 40)})

        self.assertIsNone(size)
        self.assertEqual((120, 40), origin)

    def test_rgba_frame_decodes_to_rgb(self):
        image = _decode_rgba_frame(bytes((255, 20, 10, 255, 0, 0, 255, 255)), (2, 1))

        self.assertEqual("RGB", image.mode)
        self.assertEqual((255, 20, 10), image.getpixel((0, 0)))
        self.assertEqual((0, 0, 255), image.getpixel((1, 0)))

    def test_rgba_frame_rejects_invalid_buffer_length(self):
        with self.assertRaises(WaylandCaptureError):
            _decode_rgba_frame(b"short", (2, 1))

    def test_pipewire_command_uses_portal_fd_and_stream_geometry(self):
        command = _build_pipewire_command(42, (1920, 1080), 11, 13)

        self.assertIn("fd=11", command)
        self.assertIn("fd=13", command)
        self.assertIn("path=42", command)
        self.assertIn("video/x-raw,format=RGBA,width=1920,height=1080", command)

    def test_pipewire_command_negotiates_size_for_window_stream(self):
        command = _build_pipewire_command(42, None, 11, 13)

        self.assertIn("video/x-raw,format=RGBA", command)
        self.assertNotIn("video/x-raw,format=RGBA,width=0,height=0", command)
        self.assertIn("fd=13", command)

    def test_gstreamer_caps_provide_window_dimensions(self):
        self.assertEqual(
            (2560, 1440),
            _caps_dimensions(b"caps = video/x-raw, format=(string)RGBA, width=(int)2560, height=(int)1440"),
        )
        self.assertIsNone(_caps_dimensions(b"pipeline is waiting for a stream"))

    def test_portal_handle_indexes_reply_unix_fd_array(self):
        self.assertEqual(73, _received_unix_fd([1], [41, 73]))

    def test_portal_handle_rejects_missing_descriptor(self):
        with self.assertRaises(WaylandCaptureError):
            _received_unix_fd([1], [41])


if __name__ == "__main__":
    unittest.main()