import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    from pesqueiro.gui import MainWindow, _central_region, _compose_preview
except ImportError:
    QApplication = None

from PIL import Image


@unittest.skipIf(QApplication is None, "PySide6 GUI extra is not installed")
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_starts_with_capture_controls_in_expected_state(self):
        window = MainWindow()
        self.addCleanup(window.close)

        self.assertTrue(window.start_button.isEnabled())
        self.assertFalse(window.fish_button.isEnabled())
        self.assertEqual("Vermelha", window.mode_box.currentText())

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
        window.copy_log_button.click()

        copied_text = QApplication.clipboard().text()
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