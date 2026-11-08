import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from pesqueiro.gui import (
        CaptureWorker,
        MainWindow,
        _central_region,
        _compose_preview,
        _find_bobber_for_scan,
        _is_bobber_cursor,
        _next_bobber_search,
    )
except ImportError:
    QApplication = None

from PIL import Image, ImageDraw
from pesqueiro.cursor import load_cursor_templates
from pesqueiro.core import ClassifierMode, PixelClassifier
from pesqueiro.fisher import State
from pesqueiro.mouse import MouseError


@unittest.skipIf(QApplication is None, "PySide6 GUI extra is not installed")
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_fishing_acquires_new_bobber_not_static_background_before_cast(self):
        baseline = Image.new("RGB", (200, 150), "black")
        ImageDraw.Draw(baseline).rectangle((20, 20, 39, 39), fill=(255, 20, 20))
        frame = baseline.copy()
        ImageDraw.Draw(frame).rectangle((120, 100, 129, 109), fill=(255, 20, 20))
        classifier = PixelClassifier()

        for state in (State.IDLE, State.CASTING):
            self.assertIsNone(_find_bobber_for_scan(baseline, classifier, None, baseline, True, state, None, (0, 0)))
        match = _find_bobber_for_scan(frame, classifier, None, baseline, True, State.WAIT_BOBBER, None, (0, 0))
        self.assertIsNotNone(match)
        self.assertEqual((120, 100), match.position)

    def test_fishing_rejects_tiny_cluster_only_for_initial_acquisition(self):
        baseline = Image.new("RGB", (100, 100), "black")
        frame = baseline.copy()
        ImageDraw.Draw(frame).line((20, 20, 26, 20), fill=(255, 20, 20))
        classifier = PixelClassifier()

        self.assertIsNone(_find_bobber_for_scan(frame, classifier, None, baseline, True, State.WAIT_BOBBER, None, (0, 0)))
        tracked = _find_bobber_for_scan(frame, classifier, (20, 20), baseline, True, State.WATCHING, None, (0, 0))
        self.assertIsNotNone(tracked)
        self.assertEqual(7, tracked.score)

    def test_saved_trace_includes_cursor_form_and_frame_time(self):
        window = MainWindow(debug=True)
        self.addCleanup(window.close)
        worker = Mock()
        worker.trace = [(10.5, "watching", (30, 40, 100), (35, 42), ("real_cog@27,27", 4.5), 10.4)]
        window._worker = worker
        with tempfile.TemporaryDirectory() as folder, patch("pesqueiro.gui.DATA_DIR", Path(folder)):
            window._save_trace()
            files = list((Path(folder) / "debug").glob("trace-*.txt"))
            self.assertEqual(1, len(files))
            trace = files[0].read_text()
            self.assertIn("cursor_template(score) frame_time", trace)
            self.assertIn("('real_cog@27,27', 4.5) 10.4", trace)
        window._worker = None

    def test_main_window_starts_with_capture_controls_in_expected_state(self):
        window = MainWindow()
        self.addCleanup(window.close)

        self.assertTrue(window.start_button.isEnabled())
        self.assertFalse(window.fish_button.isEnabled())
        self.assertEqual("Vermelha", window.mode_box.currentText())
        self.assertEqual("F12", window.cast_key_box.value())
        self.assertEqual("F11", window.lure_key_box.value())

    def test_both_key_fields_capture_physical_keys(self):
        window = MainWindow()
        self.addCleanup(window.close)
        for field in (window.cast_key_box, window.lure_key_box):
            event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F3, Qt.KeyboardModifier.NoModifier, 69, 0, 0)
            field.keyPressEvent(event)
            self.assertEqual("code:61", field.value())
            self.assertEqual("F3", field.text())

    def test_selects_rod_from_preview_and_maps_to_frame(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window._worker = Mock()
        window._worker._rod_position = None
        window._set_fishing_availability(True, "Pronto para pescar")
        window.rod_button.setEnabled(True)
        window.rod_button.setChecked(True)
        self.assertEqual("Clique na vara na prévia", window.rod_button.text())
        window.show()
        window.preview.set_frame(_compose_preview(Image.new("RGB", (100, 80)), (25, 20, 75, 60), None))
        self.app.processEvents()

        QTest.mouseClick(window.preview, Qt.MouseButton.LeftButton, pos=QPoint(window.preview.width() // 2, window.preview.height() // 2))

        x, y = window._worker._rod_position
        self.assertAlmostEqual(0.5, x, delta=0.01)
        self.assertAlmostEqual(0.5, y, delta=0.01)
        self.assertTrue(window.fish_button.isEnabled())
        self.assertFalse(window.rod_button.isChecked())
        self.assertEqual("Vara selecionada", window.rod_button.text())
        window._worker = None

        worker = CaptureWorker(ClassifierMode.RED)
        worker._rod_position = (x, y)
        worker._frame_origin = (100, 200)
        worker._frame_size = (1000, 800)
        self.assertAlmostEqual(600, worker._rod_screen_position()[0], delta=10)
        self.assertAlmostEqual(600, worker._rod_screen_position()[1], delta=10)

    def test_fishing_can_start_without_selecting_rod(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window._worker = Mock()
        window._worker._rod_position = None
        window._set_fishing_availability(True, "Pronto para pescar")

        window.fish_button.click()
        self.assertTrue(window._countdown_timer.isActive())
        for _ in range(3):
            window._countdown_tick()

        window._worker.set_auto_move.assert_called_once_with(True)
        self.assertEqual("Parar pesca", window.fish_button.text())
        window._worker = None

    def test_fishing_unavailable_reason_is_visible(self):
        window = MainWindow()
        self.addCleanup(window.close)

        window._set_fishing_availability(False, "Entrada indisponível")

        self.assertFalse(window.fish_button.isEnabled())
        self.assertEqual("Entrada indisponível", window.fishing_status.text())

    def test_worker_registers_both_keys(self):
        worker = CaptureWorker(ClassifierMode.RED, [object()], cast_key="code:61", lure_key="code:62")
        availability = Mock()
        worker.fishing_availability.connect(availability)
        with patch("pesqueiro.gui.UInputDevice") as device:
            fisher = worker._create_mover(Mock())
        device.assert_called_once_with(["code:61", "code:62"])
        self.assertEqual("code:61", fisher._cast_key)
        self.assertEqual("code:62", fisher._lure_key)
        availability.assert_called_once_with(True, "Pronto para pescar")

    def test_worker_reports_missing_cursor_templates(self):
        worker = CaptureWorker(ClassifierMode.RED)
        availability = Mock()
        worker.fishing_availability.connect(availability)

        self.assertIsNone(worker._create_mover(Mock()))
        self.assertFalse(availability.call_args.args[0])
        self.assertIn("cursor", availability.call_args.args[1])

    def test_worker_reports_input_error(self):
        worker = CaptureWorker(ClassifierMode.RED, [object()])
        availability = Mock()
        worker.fishing_availability.connect(availability)

        with patch("pesqueiro.gui.UInputDevice", side_effect=MouseError("permission denied")):
            self.assertIsNone(worker._create_mover(Mock()))

        availability.assert_called_once_with(False, "Entrada indisponível: permission denied")

    def test_changed_only_drops_static_pixels_and_keeps_new_ones(self):
        from pesqueiro.gui import _changed_only

        baseline = Image.new("RGB", (20, 20), (10, 10, 10))
        baseline.putpixel((2, 2), (200, 30, 30))
        current = baseline.copy()
        current.putpixel((15, 15), (220, 20, 20))

        result = _changed_only(current, baseline)

        self.assertEqual((0, 0, 0), result.getpixel((2, 2)))
        self.assertEqual((220, 20, 20), result.getpixel((15, 15)))

    def test_search_region_is_centered_in_captured_frame(self):
        self.assertEqual((480, 270, 1440, 810), _central_region((1920, 1080)))

    def test_cog_cursor_template_marks_pointer_on_bobber(self):
        class Match:
            template = "real_cog@27,27"

        self.assertTrue(_is_bobber_cursor(Match()))
        self.assertFalse(_is_bobber_cursor(None))

    def test_bobber_search_keeps_center_through_brief_misses(self):
        previous = (80, 90)
        misses = 0
        for expected_misses in range(1, 5):
            previous, misses = _next_bobber_search(previous, None, misses)
            self.assertEqual((80, 90), previous)
            self.assertEqual(expected_misses, misses)

        previous, misses = _next_bobber_search(previous, None, misses)
        self.assertIsNone(previous)
        self.assertEqual(5, misses)

        previous, misses = _next_bobber_search(previous, (120, 130), misses)
        self.assertEqual((120, 130), previous)
        self.assertEqual(0, misses)

    def test_cursor_search_finds_centered_form_before_full_frame_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            template = Image.new("RGBA", (12, 18), (0, 0, 0, 0))
            ImageDraw.Draw(template).polygon(
                [(0, 0), (0, 16), (5, 12), (11, 12)],
                fill=(255, 255, 255, 255),
                outline=(0, 0, 0, 255),
            )
            template.save(Path(temporary_directory) / "arrow@2,1.png")
            templates = load_cursor_templates(temporary_directory)
            frame = Image.new("RGB", (400, 300), (40, 90, 60))
            arrow = template.convert("RGB")
            frame.paste(arrow, (203, 117), template)

        match = CaptureWorker._search_cursor(frame, templates, None)

        self.assertIsNotNone(match)
        self.assertEqual((205, 118), match.position)

    def test_cursor_search_retries_all_scales_after_locked_scale_misses(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            template = Image.new("RGBA", (32, 48), (0, 0, 0, 0))
            ImageDraw.Draw(template).polygon(
                [(0, 0), (0, 44), (14, 34), (30, 34)],
                fill=(255, 255, 255, 255),
                outline=(0, 0, 0, 255),
            )
            template.save(Path(temporary_directory) / "arrow@2,1.png")
            templates = load_cursor_templates(temporary_directory, scales=(1.0, 0.75))
            frame = Image.new("RGB", (400, 300), (40, 90, 60))
            scaled = template.resize((24, 36), Image.Resampling.LANCZOS)
            frame.paste(scaled.convert("RGB"), (203, 117), scaled)

        worker = CaptureWorker(ClassifierMode.RED, templates)
        worker._locked_scale = 1.0
        match = worker._locate_cursor(frame, None)

        self.assertIsNotNone(match)
        self.assertEqual(0.75, match.scale)
        self.assertEqual(0.75, worker._locked_scale)

    def test_cursor_scale_retry_does_not_repeat_locked_scale(self):
        locked = SimpleNamespace(scale=1.0)
        other = SimpleNamespace(scale=0.75)
        worker = CaptureWorker(ClassifierMode.RED, [locked, other])
        worker._locked_scale = 1.0
        alternate = SimpleNamespace(scale=0.75, score=2.0, template="arrow")
        image = Image.new("RGB", (40, 40))
        worker._search_cursor = Mock(side_effect=[None, alternate])

        self.assertIs(alternate, worker._locate_cursor(image, None))
        self.assertEqual([locked], worker._search_cursor.call_args_list[0].args[1])
        self.assertEqual([other], worker._search_cursor.call_args_list[1].args[1])

    def test_cursor_scale_does_not_retry_an_acceptable_match(self):
        locked = SimpleNamespace(scale=1.0)
        other = SimpleNamespace(scale=0.75)
        match = SimpleNamespace(scale=1.0, score=12.0, template="gauntlet")
        worker = CaptureWorker(ClassifierMode.RED, [locked, other])
        worker._locked_scale = 1.0
        worker._search_cursor = Mock(return_value=match)

        self.assertIs(match, worker._locate_cursor(Image.new("RGB", (40, 40)), None))
        worker._search_cursor.assert_called_once()

    def test_preview_overlay_marks_bobber_inside_search_region(self):
        image = Image.new("RGB", (100, 80), (20, 20, 20))

        preview = _compose_preview(image, (25, 20, 75, 60), (20, 15))

        self.assertEqual((100, 80), (preview.width(), preview.height()))
        self.assertEqual((255, 184, 108), preview.pixelColor(45, 35).getRgb()[:3])

    def test_live_frame_updates_preview_and_screen_coordinate_readout(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        frame = _compose_preview(
            Image.new("RGB", (100, 80), (20, 20, 20)),
            (25, 20, 75, 60),
            (5, 6),
        )

        window._show_frame(frame, (40, 50), (100, 80), (5, 6), (25, 20, 75, 60))
        self.app.processEvents()

        self.assertIsNotNone(window.preview.pixmap())
        self.assertEqual("BOIA  70, 76", window.coordinate_label.text())
        self.assertEqual("100 x 80  |  40,50", window.frame_label.text())

    def test_error_traceback_is_logged_and_copy_button_copies_it(self):
        window = MainWindow()
        self.addCleanup(window.close)
        error_text = "Traceback (most recent call last):\nRuntimeError: portal denied"

        window._capture_failed(error_text)
        window._capture_finished()
        window.copy_log_button.click()

        copied_text = QApplication.clipboard().text()
        self.assertIn("portal denied", window.fishing_status.text())
        self.assertIn("ERROR", copied_text)
        self.assertIn("RuntimeError: portal denied", copied_text)
        self.assertIn("Traceback (most recent call last):", copied_text)

    def test_clear_button_removes_log_history(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window._append_log("test message")

        window.clear_log_button.click()

        self.assertEqual("", window.log_output.toPlainText())


if __name__ == "__main__":
    unittest.main()