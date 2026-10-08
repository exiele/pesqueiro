import random
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from pesqueiro.cursor import blank_cursor, find_cursor, load_cursor_templates
from pesqueiro.fisher import Fisher, State
from pesqueiro.humanize import build_path
from pesqueiro.mouse import LatestReading, move_to


def _arrow() -> Image.Image:
    image = Image.new("RGBA", (12, 18), (0, 0, 0, 0))
    ImageDraw.Draw(image).polygon([(0, 0), (0, 16), (5, 12), (11, 12)], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    return image


class CursorTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        _arrow().save(Path(folder.name) / "arrow@2,1.png")
        self.templates = load_cursor_templates(folder.name)

    def test_loads_hotspot_from_filename(self):
        self.assertEqual((2, 1), self.templates[0].hotspot)

    def test_finds_cursor_hotspot_in_frame(self):
        frame = Image.new("RGB", (400, 300), (40, 90, 60))
        frame.paste(_arrow(), (203, 117), _arrow())

        match = find_cursor(frame, self.templates)

        self.assertEqual((205, 118), match.position)
        self.assertEqual((305, 218), match.screen_position((100, 100)))

    def test_blank_cursor_removes_only_the_cursor_pixels(self):
        frame = Image.new("RGB", (400, 300), (40, 90, 60))
        frame.paste(_arrow(), (203, 117), _arrow())
        match = find_cursor(frame, self.templates)

        cleaned = blank_cursor(frame, match)
        shifted = blank_cursor(frame.crop((100, 50, 400, 300)), match, (100, 50))

        self.assertEqual((0, 0, 0), cleaned.getpixel((204, 125)))
        self.assertEqual((40, 90, 60), cleaned.getpixel((300, 200)))
        self.assertEqual((0, 0, 0), shifted.getpixel((104, 75)))

    def test_no_match_on_empty_frame(self):
        self.assertIsNone(find_cursor(Image.new("RGB", (200, 150), (40, 90, 60)), self.templates))


class HumanizeTests(unittest.TestCase):
    def test_path_sums_to_displacement_and_is_curved(self):
        steps = build_path((100, 100), (700, 400), random.Random(3))

        self.assertEqual(600, sum(s.dx for s in steps))
        self.assertEqual(300, sum(s.dy for s in steps))
        x = y = 0
        offsets = []
        for step in steps:
            x, y = x + step.dx, y + step.dy
            offsets.append(abs(300 * x - 600 * y) / (600**2 + 300**2) ** 0.5)
        self.assertGreater(max(offsets), 3)

    def test_paths_differ_between_runs(self):
        first = build_path((0, 0), (500, 200), random.Random(1))
        second = build_path((0, 0), (500, 200), random.Random(2))
        self.assertNotEqual(first, second)


class FakeDevice:
    def __init__(self, slot, position, responsive=True, gain=(0.9, 0.9)):
        self.slot, self.x, self.y = slot, float(position[0]), float(position[1])
        self.responsive = responsive
        self.gain = gain
        self.sent = 0.0
        self.keys, self.clicks = [], []
        self._running = True
        threading.Thread(target=self._tick, daemon=True).start()

    def _tick(self):
        while self._running:
            self.slot.set((round(self.x), round(self.y)))
            time.sleep(0.02)

    def stop(self):
        self._running = False

    def move_relative(self, dx, dy):
        self.sent += (dx * dx + dy * dy) ** 0.5
        if self.responsive:
            self.x += dx * self.gain[0]
            self.y += dy * self.gain[1]

    def press_key(self, name):
        self.keys.append(name)

    def click_right(self):
        self.clicks.append((round(self.x), round(self.y)))


class MoveTests(unittest.TestCase):
    def _device(self, position, responsive=True, gain=(0.9, 0.9)):
        slot = LatestReading()
        device = FakeDevice(slot, position, responsive, gain)
        self.addCleanup(device.stop)
        return slot, device

    def test_closed_loop_converges_despite_acceleration_error(self):
        slot, device = self._device((50, 50))

        arrived = move_to(device, slot, lambda: (60, 55), settle=0.0, fresh_timeout=0.2, rng=random.Random(5))

        self.assertTrue(arrived)

    def test_gives_up_instead_of_running_to_the_screen_edge(self):
        slot, device = self._device((50, 50), responsive=False)

        arrived = move_to(device, slot, lambda: (800, 500), settle=0.0, fresh_timeout=0.1, rng=random.Random(5))

        self.assertFalse(arrived)
        self.assertLess(device.sent, 1200)

    def test_learns_a_pointer_that_travels_far_more_than_commanded(self):
        slot, device = self._device((100, 100), gain=(3.0, 2.5))

        arrived = move_to(device, slot, lambda: (900, 600), settle=0.0, fresh_timeout=0.2, rng=random.Random(5))

        self.assertTrue(arrived)
        self.assertLess(device.sent, 600)

    def test_refuses_an_inverted_axis_without_flinging_the_cursor(self):
        slot, device = self._device((100, 400), gain=(1.0, -1.0))
        logs = []

        arrived = move_to(
            device, slot, lambda: (900, 100), settle=0.0, fresh_timeout=0.2, rng=random.Random(5), log=logs.append
        )

        self.assertFalse(arrived)
        self.assertLess(device.sent, 200)
        self.assertTrue(any("gain" in line for line in logs))


class FisherTests(unittest.TestCase):
    def _run(self, off_bobber_at_bite, vanish=None, settles_at=None, shrink=False):
        slot = LatestReading()
        device = FakeDevice(slot, (100, 100))
        self.addCleanup(device.stop)
        clock = [0.0]
        logs = []
        fisher = Fisher(device, slot, logs.append, "4", start_delay=0.0, loot_delay=0, clock=lambda: clock[0])
        self.addCleanup(fisher.cancel)
        bobber = [300, 200]
        rest = list(settles_at) if settles_at else [300, 200]
        watching_scans = 0
        vanished_for = 0
        size = 100
        for _ in range(400):
            hidden = False
            if vanish is not None and watching_scans >= 8:
                hidden = vanish == "timeout" or vanished_for < 3
                vanished_for += 1
            fisher.update(None if hidden else tuple(bobber), size)
            if fisher.state is State.WATCHING and not device.clicks:
                watching_scans += 1
                bobber[0] = rest[0]
                if watching_scans < 6:
                    bobber[1] = rest[1] + watching_scans
                elif watching_scans == 6 and shrink:
                    size = 40
                elif watching_scans == 6 and vanish is None:
                    if off_bobber_at_bite:
                        device.x, device.y = 600.0, 500.0
                    bobber[1] = rest[1] + 15
            clock[0] += 0.5
            time.sleep(0.03)
            if device.clicks or (vanish == "timeout" and any(l.startswith("Bobber did not come back") for l in logs)):
                break
        return device, logs

    def test_casts_moves_then_clicks_on_bite(self):
        device, logs = self._run(off_bobber_at_bite=False)

        self.assertEqual(["4"], device.keys[:1])
        self.assertEqual(1, len(device.clicks))
        self.assertLessEqual(abs(device.clicks[0][0] - 300) + abs(device.clicks[0][1] - 200), 12)

    def test_clicks_when_the_bobber_shrinks_without_moving(self):
        device, logs = self._run(off_bobber_at_bite=False, shrink=True)

        self.assertEqual(1, len(device.clicks))
        self.assertIn("Bite detected", logs)

    def test_clicks_where_the_bobber_settled_not_where_it_was_first_seen(self):
        device, logs = self._run(off_bobber_at_bite=False, settles_at=(340, 270))

        self.assertEqual(1, len(device.clicks))
        self.assertLessEqual(abs(device.clicks[0][0] - 340) + abs(device.clicks[0][1] - 273), 14)

    def test_clicks_when_the_bobber_blinks_out_and_returns(self):
        device, logs = self._run(off_bobber_at_bite=False, vanish="bounce")

        self.assertIn("Bobber vanished; moving in to see if it bounces back", logs)
        self.assertEqual(1, len(device.clicks))

    def test_does_not_click_when_the_bobber_times_out(self):
        device, logs = self._run(off_bobber_at_bite=False, vanish="timeout")

        self.assertTrue(any(l.startswith("Bobber did not come back") for l in logs))
        self.assertEqual([], device.clicks)

    def test_moves_again_when_cursor_left_the_bobber(self):
        device, logs = self._run(off_bobber_at_bite=True)

        self.assertIn("Moving onto the bobber", logs)
        self.assertEqual(1, len(device.clicks))
        self.assertLessEqual(abs(device.clicks[0][0] - 300) + abs(device.clicks[0][1] - 200), 12)


if __name__ == "__main__":
    unittest.main()
