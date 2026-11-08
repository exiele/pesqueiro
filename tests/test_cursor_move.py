import random
import tempfile
import threading
import time
import unittest
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from pesqueiro.cursor import blank_cursor, find_cursor, load_cursor_templates
from pesqueiro.fisher import Fisher, State
from pesqueiro.humanize import build_path
from pesqueiro.mouse import LatestReading, PointerCalibration, move_to


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

    def test_builtin_templates_include_requested_images_and_hotspots(self):
        from pesqueiro.cursor import load_builtin_cursor_templates

        templates = load_builtin_cursor_templates()

        self.assertEqual(
            {"real_cog@27,27": (27, 27), "real_gauntlet_cast@0,0": (0, 0)},
            {template.name: template.hotspot for template in templates},
        )

    def test_builtin_templates_match_embedded_cursor_images(self):
        from pesqueiro._cursor_images import CURSOR_IMAGES
        from pesqueiro.cursor import load_builtin_cursor_templates

        templates = load_builtin_cursor_templates()
        for name, hotspot, data in CURSOR_IMAGES:
            with self.subTest(name=name):
                with Image.open(BytesIO(b64decode(data))) as source:
                    cursor = source.convert("RGBA")
                frame = Image.new("RGB", (400, 300), (40, 90, 60))
                frame.paste(cursor, (203, 117), cursor)
                template = next(template for template in templates if template.name == name)

                match = find_cursor(frame, [template])

                self.assertIsNotNone(match)
                self.assertEqual((203 + hotspot[0], 117 + hotspot[1]), match.position)

    def test_local_templates_override_same_named_builtins(self):
        from pesqueiro.cursor import load_available_cursor_templates

        with tempfile.TemporaryDirectory() as folder:
            _arrow().save(Path(folder) / "real_cog@27,27.png")
            templates = load_available_cursor_templates(folder)

        self.assertEqual(1, sum(template.name == "real_cog@27,27" for template in templates))
        self.assertEqual((17, 12), templates[0].mask.shape)
        self.assertIn("real_gauntlet_cast@0,0", {template.name for template in templates})

    def test_builtins_load_without_a_custom_folder_at_multiple_scales(self):
        from pesqueiro.cursor import load_available_cursor_templates

        with tempfile.TemporaryDirectory() as folder:
            templates = load_available_cursor_templates(Path(folder) / "missing", scales=(1.0, 1.5))

        self.assertEqual(
            {(name, scale) for name in ("real_cog@27,27", "real_gauntlet_cast@0,0") for scale in (1.0, 1.5)},
            {(template.name, template.scale) for template in templates},
        )

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

    def test_quick_path_reduces_duration_without_changing_endpoint(self):
        normal = build_path((0, 0), (120, 40), random.Random(5), overshoot=False)
        quick = build_path(
            (0, 0), (120, 40), random.Random(5), overshoot=False, duration_scale=0.5
        )

        self.assertEqual(sum(step.dx for step in normal), sum(step.dx for step in quick))
        self.assertEqual(sum(step.dy for step in normal), sum(step.dy for step in quick))
        self.assertLess(sum(step.delay for step in quick), sum(step.delay for step in normal))


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

    def test_underestimated_gain_does_not_overshoot_first_leg(self):
        slot = LatestReading()
        slot.set((100, 100))
        device = Mock()
        calibration = PointerCalibration()
        calibration.gain = [1.3, 1.3]
        calibration.measured = [True, True]
        positions = []

        def play(mouse, steps, cancelled):
            previous = slot.get().position
            position = (
                previous[0] + 2 * sum(step.dx for step in steps),
                previous[1] + 2 * sum(step.dy for step in steps),
            )
            positions.append(position)
            slot.set(position)

        with patch("pesqueiro.mouse._play", side_effect=play), patch("pesqueiro.mouse._wait_fresh", side_effect=lambda *args: slot.get()):
            arrived = move_to(device, slot, lambda: (400, 400), settle=0, calibration=calibration)

        self.assertTrue(arrived)
        self.assertLess(positions[0][0], 400)
        self.assertLess(positions[0][1], 400)

    def test_refuses_an_inverted_axis_without_flinging_the_cursor(self):
        slot, device = self._device((100, 400), gain=(1.0, -1.0))
        logs = []

        arrived = move_to(
            device, slot, lambda: (900, 100), settle=0.0, fresh_timeout=0.2, rng=random.Random(5), log=logs.append
        )

        self.assertFalse(arrived)
        self.assertLess(device.sent, 200)
        self.assertTrue(any("gain" in line for line in logs))

    def test_ignores_zero_motion_outlier_after_gain_is_calibrated(self):
        calibration = PointerCalibration()
        self.assertEqual((), calibration.observe((40, 40), (60, 60)))

        ignored = calibration.observe((-24, 23), (0, 34))

        self.assertEqual((0,), ignored)
        self.assertEqual(1.5, calibration.gain[0])
        self.assertAlmostEqual(1.49, calibration.gain[1], places=2)

    def test_ignores_implausible_gain_change_after_calibration(self):
        calibration = PointerCalibration()
        calibration.observe((40, 40), (60, 60))

        ignored = calibration.observe((40, 40), (16, 60))

        self.assertEqual((0,), ignored)
        self.assertEqual(1.5, calibration.gain[0])

    def test_retraces_unconfirmed_leg_to_recover_cursor(self):
        slot = LatestReading()
        slot.set((100, 100))
        device = Mock()
        logs = []
        played = []

        def play(mouse, steps, cancelled):
            played.append(steps)

        with patch("pesqueiro.mouse._play", side_effect=play), patch(
            "pesqueiro.mouse._wait_fresh", side_effect=[None, slot.get()]
        ):
            arrived = move_to(device, slot, lambda: (600, 400), settle=0, log=logs.append)

        self.assertFalse(arrived)
        self.assertEqual(2, len(played))
        self.assertEqual(
            [(step.dx, step.dy) for step in played[0]],
            [(-step.dx, -step.dy) for step in reversed(played[1])],
        )
        self.assertIn("Cursor tracking recovered", logs)

    def test_accepts_calibrated_near_target_move_when_cursor_form_shifts(self):
        slot, device = self._device((100, 100), gain=(1.5, 1.5))
        calibration = PointerCalibration()
        calibration.gain = [1.5, 1.5]
        calibration.measured = [True, True]
        logs = []

        with patch.object(PointerCalibration, "observe", return_value=(0,)):
            arrived = move_to(
                device,
                slot,
                lambda: (63, 134),
                settle=0.0,
                fresh_timeout=0.2,
                rng=random.Random(5),
                calibration=calibration,
                log=logs.append,
            )

        self.assertTrue(arrived)
        self.assertTrue(any("changed form near target" in line for line in logs))

    def test_refuses_unconfirmed_movement_when_requested(self):
        slot = LatestReading()
        slot.set((100, 100))
        device = Mock()
        calibration = PointerCalibration()
        calibration.gain = [1.5, 1.5]
        calibration.measured = [True, True]

        with patch("pesqueiro.mouse._play"), patch("pesqueiro.mouse._wait_fresh", return_value=None):
            arrived = move_to(
                device, slot, lambda: (130, 120), settle=0, fresh_timeout=0,
                calibration=calibration, allow_unconfirmed=False,
            )

        self.assertFalse(arrived)


class FisherTests(unittest.TestCase):
    def test_recorded_bounce_survives_single_sideways_tracking_outlier(self):
        device = Mock()
        fisher = Fisher(device, LatestReading(), lambda message: None, loot_delay=0)
        fisher.state = State.WATCHING
        fisher._watch_goal = (877, 433)
        fisher._watcher.reset((877, 433))
        fisher._spawn = lambda target: target()
        fisher._cursor_on_bobber = True

        for bobber in [(862, 463), (860, 476), (786, 491), (861, 470),
                       (871, 445), (881, 415), (881, 414), (882, 410),
                       (883, 417), (884, 426), (882, 432)]:
            fisher._bobber = bobber
            fisher._watch(1.0)

        device.click_right.assert_called_once_with()

    def test_sustained_tracking_relocation_rebases_without_clicking(self):
        device = Mock()
        fisher = Fisher(device, LatestReading(), lambda message: None, loot_delay=0)
        fisher.state = State.WATCHING
        fisher._watch_goal = (300, 200)
        fisher._watcher.reset((300, 200))
        fisher._spawn = lambda target: target()

        for bobber in [(340, 270), (341, 270), (340, 271)]:
            fisher._bobber = bobber
            fisher._watch(1.0)

        self.assertEqual((340, 270), fisher._watch_goal)
        device.click_right.assert_not_called()

    def test_click_move_uses_click_tolerance(self):
        cursor = LatestReading()
        cursor.set((375, 225))
        device = Mock()
        fisher = Fisher(device, cursor, lambda message: None, loot_delay=0)
        fisher._watch_goal = (300, 200)
        fisher._go_to = Mock(return_value=True)

        fisher._click()

        fisher._go_to.assert_called_once_with((300, 200), tolerance=10, quick=True)
        device.click_right.assert_called_once_with()

    def test_cog_during_bounce_does_not_interrupt_detection_and_clicks(self):
        cursor = LatestReading()
        cursor.set((375, 225))
        logs = []
        device = Mock()
        fisher = Fisher(device, cursor, logs.append, start_delay=0, loot_delay=0, clock=lambda: 1.0)
        fisher.state = State.WATCHING
        fisher._watch_goal = (300, 200)
        fisher._watcher.reset((300, 200))
        fisher._cursor_on_bobber = True
        fisher._spawn = lambda target: target()
        fisher._go_to = Mock(return_value=True)

        fisher._bobber = (300, 215)
        fisher._watch(1.0)
        self.assertEqual(State.WATCHING, fisher.state)
        device.click_right.assert_not_called()
        fisher._bobber = (300, 202)
        fisher._watch(1.0)

        self.assertIn("Bite detected", logs)
        device.click_right.assert_called_once_with()
        fisher._go_to.assert_not_called()

    def test_waits_for_cursor_before_casting_and_resumes_when_found(self):
        cursor = LatestReading()
        logs = []
        fisher = Fisher(Mock(), cursor, logs.append, start_delay=0)
        fisher._spawn = Mock()

        fisher.update(None)
        fisher.update(None)
        fisher._spawn.assert_not_called()
        self.assertEqual(1, logs.count("Waiting for cursor tracking before casting"))

        cursor.set((100, 100))
        fisher.update(None)
        fisher._spawn.assert_called_once_with(fisher._cast)
        self.assertIn("Cursor tracking available; resuming fishing", logs)

    def test_lure_waits_for_rod_selection_without_blocking_casts(self):
        rod = [None]
        cursor = LatestReading()
        cursor.set((100, 100))
        fisher = Fisher(Mock(), cursor, lambda message: None, start_delay=0,
                        clock=lambda: 500.0, rod_position=lambda: rod[0])
        fisher._armed_at = 0.0
        fisher._next_lure_at = 1.0
        fisher._spawn = Mock()

        fisher.update(None)
        fisher._spawn.assert_called_once_with(fisher._cast)

        rod[0] = (400, 300)
        fisher.state = State.IDLE
        fisher._spawn.reset_mock()
        fisher.update(None)
        fisher._spawn.assert_called_once_with(fisher._apply_lure)

    def test_lure_waits_for_cast_to_finish_and_reschedules(self):
        clock = [0.0]
        device = Mock()
        cursor = LatestReading()
        cursor.set((100, 100))
        fisher = Fisher(device, cursor, lambda message: None, start_delay=0, clock=lambda: clock[0],
                rod_position=lambda: (400, 300))
        fisher._spawn = lambda target: target()
        fisher._go_to = Mock(return_value=True)
        with patch("pesqueiro.fisher.random.uniform", side_effect=[480.0, 6.25, 600.0]) as interval:
            fisher.update(None)
            device.press_key.assert_called_once_with("F12")
            clock[0] = 480.0
            fisher.state = State.WATCHING
            fisher._since = clock[0]
            fisher.update((300, 200))
            device.press_key.assert_called_once_with("F12")

            fisher.state = State.IDLE
            fisher._cast_started_at = 470.0
            fisher.update(None)
            device.press_key.assert_called_once_with("F12")

            clock[0] = 495.0
            with patch.object(fisher._cancel, "wait", return_value=False) as wait:
                fisher.update(None)
            self.assertEqual(["F12", "F11"], [call.args[0] for call in device.press_key.call_args_list])
            fisher._go_to.assert_called_once_with((400, 300), tolerance=6)
            device.click_left.assert_called_once_with()
            wait.assert_called_once_with(6.25)
            self.assertEqual(1095.0, fisher._next_lure_at)
            self.assertEqual(3, interval.call_count)
            fisher.update(None)
            self.assertEqual("F12", device.press_key.call_args.args[0])

    def test_lure_presses_key_before_clicking_rod(self):
        device = Mock()
        fisher = Fisher(device, LatestReading(), lambda message: None, lure_key="code:42",
                        rod_position=lambda: (400, 300))
        fisher._go_to = Mock(return_value=True)
        with patch.object(fisher._cancel, "wait", return_value=False), patch("pesqueiro.fisher.random.uniform", return_value=6.0):
            fisher._apply_lure()
        self.assertEqual([("press_key", ("code:42",)), ("click_left", ())],
                         [(call[0], call.args) for call in device.mock_calls])

    def test_lure_does_not_press_or_click_when_rod_is_unreachable(self):
        device = Mock()
        fisher = Fisher(device, LatestReading(), lambda message: None, clock=lambda: 100.0,
                        rod_position=lambda: (400, 300))
        fisher._go_to = Mock(return_value=False)
        fisher._apply_lure()
        device.press_key.assert_not_called()
        device.click_left.assert_not_called()
        self.assertEqual(130.0, fisher._next_lure_at)

    def test_cancelling_lure_wait_prevents_next_cast(self):
        device = Mock()
        fisher = Fisher(device, LatestReading(), lambda message: None, rod_position=lambda: (400, 300))
        fisher._go_to = Mock(return_value=True)

        def cancel_during_wait(delay):
            fisher._cancel.set()
            return True

        with patch.object(fisher._cancel, "wait", side_effect=cancel_during_wait):
            fisher._apply_lure()
        fisher.update(None)
        device.press_key.assert_called_once_with("F11")
        device.click_left.assert_called_once_with()

    def _run(self, off_bobber_at_bite, vanish=None, settles_at=None, shrink=False, cog_form=False):
        slot = LatestReading()
        device = FakeDevice(slot, (300, 200) if cog_form else (100, 100))
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
            on_bobber = cog_form and abs(device.x - bobber[0]) <= 20 and abs(device.y - bobber[1]) <= 20
            fisher.update(None if hidden else tuple(bobber), size, cursor_on_bobber=on_bobber)
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
                elif watching_scans == 7 and vanish is None and not shrink:
                    bobber[1] = rest[1] + 3
            clock[0] += 0.5
            time.sleep(0.03)
            if device.clicks or (fisher.state is State.IDLE and watching_scans > 8):
                break
        return device, logs

    def test_casts_moves_then_clicks_on_bite(self):
        device, logs = self._run(off_bobber_at_bite=False)

        self.assertEqual(["4"], device.keys[:1])
        self.assertEqual(1, len(device.clicks))
        self.assertLessEqual(abs(device.clicks[0][0] - 300) + abs(device.clicks[0][1] - 200), 12)

    def test_does_not_click_when_the_bobber_shrinks_without_a_bounce(self):
        device, logs = self._run(off_bobber_at_bite=False, shrink=True)

        self.assertEqual([], device.clicks)
        self.assertNotIn("Bite detected", logs)

    def test_clicks_where_the_bobber_settled_not_where_it_was_first_seen(self):
        device, logs = self._run(off_bobber_at_bite=False, settles_at=(340, 270))

        self.assertEqual(1, len(device.clicks))
        self.assertLessEqual(abs(device.clicks[0][0] - 340) + abs(device.clicks[0][1] - 273), 14)

    def test_does_not_click_when_the_bobber_blinks_out_and_returns(self):
        device, logs = self._run(off_bobber_at_bite=False, vanish="bounce")

        self.assertNotIn("Bite detected", logs)
        self.assertEqual([], device.clicks)

    def test_does_not_click_when_the_bobber_times_out(self):
        device, logs = self._run(off_bobber_at_bite=False, vanish="timeout")

        self.assertEqual([], device.clicks)

    def test_moves_again_when_cursor_left_the_bobber(self):
        device, logs = self._run(off_bobber_at_bite=True)

        self.assertIn("Moving onto the bobber", logs)
        self.assertEqual(1, len(device.clicks))
        self.assertLessEqual(abs(device.clicks[0][0] - 300) + abs(device.clicks[0][1] - 200), 12)

    def test_cog_form_parks_cursor_then_moves_back_on_bite(self):
        device, logs = self._run(off_bobber_at_bite=False, cog_form=True)

        self.assertIn("Moving cursor next to the bobber", logs)
        self.assertIn("Moving onto the bobber", logs)
        self.assertEqual(1, len(device.clicks))
        self.assertLessEqual(abs(device.clicks[0][0] - 300) + abs(device.clicks[0][1] - 200), 12)


if __name__ == "__main__":
    unittest.main()
