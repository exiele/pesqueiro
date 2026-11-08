from __future__ import annotations

import argparse
import asyncio
import math
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .core import ClassifierMode, PixelClassifier, find_bobber
from .cursor import (
    DATA_DIR,
    DEFAULT_CURSOR_DIR,
    DEFAULT_SCALES,
    CursorTemplate,
    blank_cursor,
    find_cursor,
    load_available_cursor_templates,
)
from .fisher import Fisher, State
from .mouse import LatestReading, MouseError, UInputDevice

_CURSOR_COLOUR = "#8be9fd"
_CURSOR_SEARCH_RADIUS = 160
_CURSOR_THRESHOLD = 15.0
_CHANGE_THRESHOLD = 60
_REAL_CURSOR_THRESHOLD = 20.0
_CURSOR_JUMP = 250
_SCALE_UNLOCK_MISSES = 12
_BOBBER_REACQUIRE_MISSES = 4
_BOBBER_ACQUIRE_MIN_SCORE = 12
_DETECTION_INTERVAL = 0.05
_PREVIEW_INTERVAL = 0.1


def _to_qimage(image: Image.Image) -> QImage:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    return QImage(
        rgb_image.tobytes(),
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    ).copy()


class KeyCaptureEdit(QLineEdit):
    """Records the physical key pressed instead of typed text."""

    def __init__(self, default: str = "F12") -> None:
        super().__init__(default)
        self._default = default
        self._code: int | None = None
        self.setReadOnly(True)
        self.setToolTip("Clique e pressione a tecla desejada")

    def value(self) -> str:
        return f"code:{self._code}" if self._code is not None else self._default

    def keyPressEvent(self, event) -> None:
        code = event.nativeScanCode() - 8  # xkb keycodes are evdev codes + 8
        if code <= 0:
            return
        self._code = code
        self.setText(QKeySequence(event.key()).toString() or f"tecla {code}")


def _threshold_for(match) -> float:
    return _REAL_CURSOR_THRESHOLD if match.template.startswith("real_") else _CURSOR_THRESHOLD


def _is_bobber_cursor(match) -> bool:
    return match is not None and "cog" in match.template.lower()


def _next_bobber_search(previous, current, misses: int):
    if current is not None:
        return current, 0
    misses += 1
    return (previous if misses <= _BOBBER_REACQUIRE_MISSES else None), misses


def _central_region(size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    left, top = width // 4, height // 4
    return left, top, width - left, height - top


def _changed_only(current: Image.Image, baseline: Image.Image) -> Image.Image:
    now = np.asarray(current.convert("RGB"), dtype=np.int16)
    before = np.asarray(baseline.convert("RGB"), dtype=np.int16)
    changed = np.abs(now - before).sum(axis=2) > _CHANGE_THRESHOLD
    return Image.fromarray(np.where(changed[:, :, None], now, 0).astype(np.uint8))


def _find_bobber_for_scan(image, classifier, previous, baseline, fishing, state, cursor_match, offset):
    if fishing and state in (State.IDLE, State.CASTING):
        return None
    if fishing and previous is None and baseline is not None:
        image = _changed_only(image, baseline)
    match = find_bobber(blank_cursor(image, cursor_match, offset), classifier, previous)
    if fishing and previous is None and match is not None and match.score < _BOBBER_ACQUIRE_MIN_SCORE:
        return None
    return match


def _compose_preview(
    image: Image.Image,
    region: tuple[int, int, int, int],
    bobber: tuple[int, int] | None,
    cursor: tuple[int, int] | None = None,
) -> QImage:
    preview = _to_qimage(image)
    painter = QPainter(preview)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    left, top, right, bottom = region

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(33, 34, 44, 105))
    painter.drawRect(QRectF(0, 0, preview.width(), top))
    painter.drawRect(QRectF(0, bottom, preview.width(), preview.height() - bottom))
    painter.drawRect(QRectF(0, top, left, bottom - top))
    painter.drawRect(QRectF(right, top, preview.width() - right, bottom - top))

    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor("#50fa7b"), 2))
    painter.drawRect(QRectF(left + 1, top + 1, right - left - 2, bottom - top - 2))
    if bobber is not None:
        bobber_x, bobber_y = left + bobber[0], top + bobber[1]
        painter.setPen(QPen(QColor("#ffb86c"), 2))
        painter.drawEllipse(QRectF(bobber_x - 9, bobber_y - 9, 18, 18))
        painter.drawLine(bobber_x - 15, bobber_y, bobber_x + 15, bobber_y)
        painter.drawLine(bobber_x, bobber_y - 15, bobber_x, bobber_y + 15)
    if cursor is not None:
        painter.setPen(QPen(QColor(_CURSOR_COLOUR), 2))
        painter.drawRect(QRectF(cursor[0] - 10, cursor[1] - 10, 20, 20))
        painter.drawText(cursor[0] + 14, cursor[1] - 12, "cursor")
    painter.end()
    return preview


class PreviewCanvas(QLabel):
    position_selected = Signal(float, float)

    def __init__(self) -> None:
        super().__init__("Aguardando captura")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background:#21222c;color:#6272a4;border:1px solid #44475a;"
            "font-size:14px;border-radius:8px;"
        )
        self._frame: QImage | None = None

    def set_frame(self, frame: QImage) -> None:
        self._frame = frame
        self._fit_frame()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_frame()

    def mousePressEvent(self, event) -> None:
        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull() and event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x() - (self.width() - pixmap.width()) / 2
            y = event.position().y() - (self.height() - pixmap.height()) / 2
            if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                self.position_selected.emit(x / pixmap.width(), y / pixmap.height())
        super().mousePressEvent(event)

    def _fit_frame(self) -> None:
        if self._frame is None or self.width() <= 0 or self.height() <= 0:
            return
        pixmap = QPixmap.fromImage(self._frame).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class CaptureWorker(QThread):
    frame_ready = Signal(object, object, object, object, object, object)
    status_changed = Signal(str)
    failed = Signal(str)
    log_message = Signal(str)
    fishing_availability = Signal(bool, str)

    def __init__(
        self,
        mode: ClassifierMode,
        templates: list[CursorTemplate] | None = None,
        auto_move: bool = True,
        cast_key: str = "F12",
        lure_key: str = "F11",
    ) -> None:
        super().__init__()
        self._templates = templates or []
        self._cast_key = cast_key
        self._lure_key = lure_key
        self._rod_position: tuple[float, float] | None = None
        self._frame_origin = (0, 0)
        self._frame_size = (0, 0)
        self._locked_scale: float | None = None
        self.latest_image: Image.Image | None = None
        self.trace: deque = deque(maxlen=900)
        self._scale_misses = 0
        self._device: UInputDevice | None = None
        self._auto_move = threading.Event()
        if auto_move:
            self._auto_move.set()
        self.classifier = PixelClassifier(mode=mode)
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    def set_auto_move(self, enabled: bool) -> None:
        if enabled:
            self._auto_move.set()
        else:
            self._auto_move.clear()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None and self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)

    def run(self) -> None:
        try:
            asyncio.run(self._capture_loop())
        except asyncio.CancelledError:
            pass
        except Exception:
            self.failed.emit(traceback.format_exc())

    async def _capture_loop(self) -> None:
        from .wayland import WaylandScreenCapture

        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        if self._stop_event.is_set():
            return

        capture = WaylandScreenCapture()
        mover: Fisher | None = None
        try:
            self.status_changed.emit("Requesting screen access")
            await capture.start()
            self.status_changed.emit("Live capture")
            if self._templates:
                self.log_message.emit(f"Loaded {len(self._templates)} cursor template(s)")
            else:
                self.log_message.emit("No cursor templates found; add PNGs to the cursors folder")
            cursor_slot = LatestReading()
            mover = self._create_mover(cursor_slot)
            next_scan = 0.0
            next_preview = 0.0
            previous_position = None
            bobber_misses = 0
            bobber_position = None
            cursor_position = None
            cursor_seen: bool | None = None
            next_report = 0.0
            baseline: Image.Image | None = None
            last_known: tuple[int, int] | None = None
            pending: tuple[int, int] | None = None
            cursor_match = None
            cursor_task: asyncio.Task | None = None
            searched = frame_at_launch = None
            while not self._stop_event.is_set():
                frame = await capture.capture()
                self.latest_image = frame.image
                self._frame_origin = frame.origin
                self._frame_size = frame.image.size
                region = _central_region(frame.image.size)
                now = time.monotonic()
                if now >= next_scan:
                    left, top, right, bottom = region
                    scan_image = frame.image.crop((left, top, right, bottom))
                    fishing = mover is not None and self._auto_move.is_set()
                    if fishing and mover.state in (State.IDLE, State.CASTING):
                        if mover.state is State.IDLE:
                            baseline = scan_image
                        previous_position = None
                        bobber_misses = 0
                    elif not fishing:
                        previous_position = None
                        bobber_misses = 0
                    if cursor_task is not None and cursor_task.done():
                        cursor_match = cursor_task.result()
                        searched, cursor_task = frame_at_launch, None
                        if cursor_match is not None and cursor_match.score > _threshold_for(cursor_match):
                            if self._templates and now >= next_report:
                                self.log_message.emit(
                                    f"Closest cursor candidate: {cursor_match.template} "
                                    f"score {cursor_match.score:.0f} (needs <= {_CURSOR_THRESHOLD:.0f})"
                                )
                                next_report = now + 5
                            cursor_match = None
                        if cursor_match is not None:
                            if last_known is not None and math.dist(cursor_match.position, last_known) > _CURSOR_JUMP:
                                earlier, pending = pending, cursor_match.position
                                if earlier is None or math.dist(earlier, cursor_match.position) > 20:
                                    cursor_match = None
                            else:
                                pending = None
                        if cursor_match is not None:
                            last_known = cursor_match.position
                        cursor_position = cursor_match.position if cursor_match is not None else None
                        cursor_slot.set(
                            None
                            if cursor_position is None
                            else (searched.origin[0] + cursor_position[0], searched.origin[1] + cursor_position[1]),
                            searched.captured_at,
                        )
                        if (cursor_position is not None) != cursor_seen and self._templates:
                            cursor_seen = cursor_position is not None
                            self.log_message.emit("Cursor found" if cursor_seen else "Cursor lost")
                    if cursor_task is None:
                        frame_at_launch = frame
                        cursor_task = asyncio.ensure_future(
                            asyncio.to_thread(self._locate_cursor, frame.image, cursor_position)
                        )

                    match = await asyncio.to_thread(
                        _find_bobber_for_scan,
                        scan_image,
                        self.classifier,
                        previous_position,
                        baseline,
                        fishing,
                        mover.state if mover is not None else State.IDLE,
                        cursor_match,
                        (left, top),
                    )
                    bobber_position = match.position if match is not None else None
                    cursor_on_bobber = _is_bobber_cursor(cursor_match)
                    previous_position, bobber_misses = _next_bobber_search(
                        previous_position, bobber_position, bobber_misses
                    )
                    self.trace.append(
                        (
                            round(now, 2),
                            mover.state.value if mover is not None else "-",
                            None if match is None else (*match.position, match.matching_pixels),
                            cursor_position,
                            None if cursor_match is None else (cursor_match.template, round(cursor_match.score, 2)),
                            round(frame.captured_at, 2),
                        )
                    )
                    if mover is not None:
                        if self._auto_move.is_set():
                            mover.resume()
                            bobber_screen = None
                            if bobber_position is not None:
                                bobber_screen = (
                                    frame.origin[0] + region[0] + bobber_position[0],
                                    frame.origin[1] + region[1] + bobber_position[1],
                                )
                            mover.update(
                                bobber_screen,
                                None if match is None else match.matching_pixels,
                                cursor_on_bobber,
                            )
                        else:
                            mover.cancel()
                    next_scan = now + _DETECTION_INTERVAL

                if now >= next_preview:
                    preview = _compose_preview(
                        frame.image, region, bobber_position, cursor_position
                    )
                    self.frame_ready.emit(
                        preview,
                        frame.origin,
                        frame.image.size,
                        bobber_position,
                        region,
                        cursor_position,
                    )
                    next_preview = now + _PREVIEW_INTERVAL
        finally:
            if mover is not None:
                mover.cancel()
            if self._device is not None:
                self._device.close()
            self.status_changed.emit("Stopping capture")
            await capture.close()

    def _create_mover(self, cursor_slot: LatestReading) -> Fisher | None:
        self._device = None
        if not self._templates:
            self.fishing_availability.emit(False, f"Adicione imagens do cursor em {DEFAULT_CURSOR_DIR}")
            return None
        try:
            self._device = UInputDevice([self._cast_key, self._lure_key])
        except MouseError as error:
            self.log_message.emit(f"Auto-fishing unavailable: {error}")
            self.fishing_availability.emit(False, f"Entrada indisponível: {error}")
            return None
        mover = Fisher(
            self._device, cursor_slot, self.log_message.emit, self._cast_key,
            start_delay=0.0, lure_key=self._lure_key, rod_position=self._rod_screen_position,
        )
        self.fishing_availability.emit(True, "Pronto para pescar")
        return mover

    def _rod_screen_position(self) -> tuple[int, int] | None:
        if self._rod_position is None:
            return None
        return (
            self._frame_origin[0] + round(self._rod_position[0] * self._frame_size[0]),
            self._frame_origin[1] + round(self._rod_position[1] * self._frame_size[1]),
        )

    def _locate_cursor(self, image: Image.Image, previous: tuple[int, int] | None):
        templates = self._templates
        if self._locked_scale is not None:
            templates = [t for t in templates if t.scale == self._locked_scale]
        match = self._search_cursor(image, templates, previous)
        if (
            self._locked_scale is not None
            and (
                match is None
                or match.score > _threshold_for(match)
            )
        ):
            alternate_templates = [t for t in self._templates if t.scale != self._locked_scale]
            alternate = self._search_cursor(image, alternate_templates, previous) if alternate_templates else None
            if (
                alternate is not None
                and alternate.score <= _threshold_for(alternate)
                and (match is None or alternate.score < match.score)
            ):
                match = alternate
        if match is not None and match.score <= _threshold_for(match):
            if self._locked_scale != match.scale:
                self._locked_scale = match.scale
                self.log_message.emit(f"Cursor size locked at {match.scale:g}x ({match.template})")
            self._scale_misses = 0
        elif self._locked_scale is not None:
            self._scale_misses += 1
            if self._scale_misses >= _SCALE_UNLOCK_MISSES:
                self._locked_scale, self._scale_misses = None, 0
        return match

    @staticmethod
    def _search_cursor(image, templates, previous):
        if previous is not None:
            radius = _CURSOR_SEARCH_RADIUS
            region = (
                max(0, previous[0] - radius),
                max(0, previous[1] - radius),
                min(image.width, previous[0] + radius),
                min(image.height, previous[1] + radius),
            )
            match = find_cursor(image, templates, region, threshold=None)
            if match is not None and match.score <= _threshold_for(match):
                return match
        region = _central_region(image.size)
        match = find_cursor(image, templates, region, threshold=None)
        if match is not None and match.score <= _threshold_for(match):
            return match
        return find_cursor(image, templates, threshold=None)


class MainWindow(QMainWindow):
    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self._debug = debug
        self._status = "OFFLINE"
        self._worker: CaptureWorker | None = None
        self._detection_found: bool | None = None
        self.setWindowTitle("Pesqueiro")
        self.resize(1040, 720)
        self.setMinimumSize(760, 560)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("PESQUEIRO")
        title.setObjectName("brand")
        header.addWidget(title)
        header.addStretch(1)
        layout.addLayout(header)

        preview_column = QVBoxLayout()
        preview_column.setSpacing(9)
        preview_header = QHBoxLayout()
        self.frame_label = QLabel("Sem imagem")
        self.frame_label.setObjectName("muted")
        preview_header.addStretch(1)
        preview_header.addWidget(self.frame_label)
        preview_column.addLayout(preview_header)
        self.preview = PreviewCanvas()
        self.preview.position_selected.connect(self._select_rod_position)
        preview_column.addWidget(self.preview, 1)
        self.coordinate_label = QLabel("BOIA  --")
        self.coordinate_label.setObjectName("readout")
        preview_column.addWidget(self.coordinate_label)
        self.cursor_label = QLabel("CURSOR  --")
        self.cursor_label.setObjectName("readout")
        preview_column.addWidget(self.cursor_label)
        layout.addLayout(preview_column, 1)

        controls = QFrame()
        controls.setObjectName("panel")
        controls_panel_layout = QVBoxLayout(controls)
        controls_panel_layout.setContentsMargins(14, 10, 14, 10)
        controls_panel_layout.setSpacing(8)
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)
        controls_panel_layout.addLayout(controls_layout)

        self.mode_box = QComboBox()
        self.mode_box.addItem("Vermelha", ClassifierMode.RED)
        self.mode_box.addItem("Azul", ClassifierMode.BLUE)
        self.mode_box.setMinimumWidth(90)
        self.cast_key_box = KeyCaptureEdit()
        self.cast_key_box.setFixedWidth(90)
        self.lure_key_box = KeyCaptureEdit("F11")
        self.lure_key_box.setFixedWidth(90)
        controls_layout.addLayout(self._labelled("PENA", self.mode_box))
        controls_layout.addLayout(self._labelled("TECLA DE LANÇAMENTO", self.cast_key_box))
        controls_layout.addLayout(self._labelled("TECLA DE ISCA", self.lure_key_box))
        self.rod_button = QPushButton("Selecionar vara")
        self.rod_button.setCheckable(True)
        self.rod_button.setEnabled(False)
        self.rod_button.setToolTip("Selecione a vara na barra de ações pela prévia")
        self.rod_button.toggled.connect(self._rod_selection_changed)
        controls_layout.addWidget(self.rod_button)
        controls_layout.addStretch(1)

        actions_layout = QHBoxLayout()
        actions_layout.addStretch(1)
        controls_panel_layout.addLayout(actions_layout)

        self.snapshot_button = QPushButton("Save snapshot")
        self.snapshot_button.setObjectName("ghostButton")
        self.snapshot_button.setToolTip("Save the current raw frame to the debug folder")
        self.snapshot_button.clicked.connect(self._save_snapshot)
        actions_layout.addWidget(self.snapshot_button)
        self.trace_button = QPushButton("Save scan trace")
        self.trace_button.setObjectName("ghostButton")
        self.trace_button.setToolTip("Write the last minutes of bobber and cursor readings to the debug folder")
        self.trace_button.clicked.connect(self._save_trace)
        actions_layout.addWidget(self.trace_button)
        self.snapshot_button.setVisible(self._debug)
        self.trace_button.setVisible(self._debug)

        self.start_button = QPushButton("Iniciar captura de tela")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setMinimumWidth(190)
        self.start_button.clicked.connect(self._toggle_capture)
        actions_layout.addWidget(self.start_button)
        self.fish_button = QPushButton("Iniciar pesca")
        self.fish_button.setObjectName("primaryButton")
        self.fish_button.setMinimumWidth(150)
        self.fish_button.setEnabled(False)
        self.fish_button.clicked.connect(self._toggle_fishing)
        actions_layout.addWidget(self.fish_button)
        self.fishing_status = QLabel("Inicie a captura de tela")
        self.fishing_status.setWordWrap(True)
        self.fishing_status.setObjectName("muted")
        controls_panel_layout.addWidget(self.fishing_status)
        self._fishing = False
        self._countdown = 0
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._countdown_tick)
        layout.addWidget(controls)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(10)
        log_header = QHBoxLayout()
        log_title = QLabel("LOG")
        log_title.setObjectName("sectionTitle")
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        self.copy_log_button = QPushButton("Copy all")
        self.copy_log_button.setToolTip("Copy all log messages to the clipboard")
        self.copy_log_button.clicked.connect(self._copy_logs)
        log_header.addWidget(self.copy_log_button)
        self.clear_log_button = QPushButton("Clear")
        self.clear_log_button.clicked.connect(self.log_output_clear)
        log_header.addWidget(self.clear_log_button)
        log_layout.addLayout(log_header)
        self.log_output = QPlainTextEdit()
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1200)
        self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_output.setPlaceholderText("Capture events and errors will appear here.")
        self.log_output.setFixedHeight(110)
        log_layout.addWidget(self.log_output)
        log_panel.setVisible(self._debug)
        layout.addWidget(log_panel)
        self.setCentralWidget(root)
        self._append_log("Pesqueiro GUI started")

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @classmethod
    def _labelled(cls, text: str, widget: QWidget) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(4)
        column.addWidget(cls._field_label(text))
        column.addWidget(widget)
        return column

    def _apply_style(self) -> None:
        self.setFont(QFont("Noto Sans", 10))
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#282a36;color:#f8f8f2;}"
            "QLabel#brand{font-size:23px;font-weight:700;color:#bd93f9;}"
            "QLabel#fieldLabel{font-size:10px;font-weight:700;color:#6272a4;letter-spacing:1px;}"
            "QLabel#sectionTitle{font-size:12px;font-weight:700;color:#bd93f9;letter-spacing:1px;}"
            "QLabel#muted{color:#6272a4;font-size:11px;}"
            "QLabel#readout{color:#ffb86c;font-family:monospace;font-size:12px;padding:2px;}"
            "QPlainTextEdit#logOutput{background:#21222c;color:#f8f8f2;border:1px solid #44475a;"
            "border-radius:8px;padding:8px;font-family:monospace;font-size:11px;}"
            "QFrame#panel{background:#21222c;border:1px solid #44475a;border-radius:10px;}"
            "QFrame#panel QLabel,QFrame#panel QVBoxLayout{background:transparent;}"
            "QComboBox,QLineEdit{background:#282a36;border:1px solid #44475a;border-radius:8px;"
            "padding:6px 10px;color:#f8f8f2;selection-background-color:#44475a;}"
            "QComboBox:focus,QLineEdit:focus{border-color:#bd93f9;}"
            "QComboBox QAbstractItemView{background:#21222c;color:#f8f8f2;selection-background-color:#44475a;}"
            "QComboBox::drop-down{border:0;width:24px;}"
            "QPushButton{min-height:34px;padding:0 16px;border-radius:8px;font-weight:700;}"
            "QPushButton#primaryButton{background:#50fa7b;color:#282a36;border:0;}"
            "QPushButton#primaryButton:hover{background:#69ff94;}"
            "QPushButton#primaryButton[running=true]{background:#ff5555;color:#f8f8f2;}"
            "QPushButton#primaryButton[running=true]:hover{background:#ff6e6e;}"
            "QPushButton#ghostButton{background:transparent;color:#f8f8f2;border:1px solid #6272a4;}"
            "QPushButton#ghostButton:hover{background:#44475a;}"
            "QPushButton{background:#44475a;color:#f8f8f2;border:0;}"
            "QPushButton:disabled{background:#343746;color:#6272a4;border:0;}"
        )

    def _toggle_capture(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._stop_capture()
        else:
            self._start_capture()

    def _set_running(self, button: QPushButton, running: bool, text: str) -> None:
        button.setText(text)
        button.setProperty("running", running)
        button.style().unpolish(button)
        button.style().polish(button)

    def _start_capture(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = CaptureWorker(
            self.mode_box.currentData(),
            load_available_cursor_templates(scales=DEFAULT_SCALES),
            False,
            self.cast_key_box.value(),
            self.lure_key_box.value(),
        )
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.log_message.connect(self._append_log)
        self._worker.status_changed.connect(self._set_status)
        self._worker.failed.connect(self._capture_failed)
        self._worker.fishing_availability.connect(self._set_fishing_availability)
        self._worker.finished.connect(self._capture_finished)
        self._set_running(self.start_button, True, "Parar captura de tela")
        self.fish_button.setEnabled(False)
        self.fishing_status.setText("Aguardando captura de tela")
        self.rod_button.setEnabled(True)
        self.rod_button.setText("Selecionar vara")
        self.rod_button.setChecked(False)
        self.mode_box.setEnabled(False)
        self.cast_key_box.setEnabled(False)
        self.lure_key_box.setEnabled(False)
        self._set_status("STARTING")
        self._append_log("Screen capture requested; waiting for portal permission")
        self._worker.start()

    def _select_rod_position(self, x: float, y: float) -> None:
        if not self.rod_button.isChecked() or self._worker is None:
            return
        self._worker._rod_position = (x, y)
        self.rod_button.setChecked(False)
        self.rod_button.setText("Vara selecionada")
        self._append_log("Fishing rod action selected")

    def _set_fishing_availability(self, available: bool, message: str) -> None:
        self.fish_button.setEnabled(available)
        self.fishing_status.setText(message)

    def _rod_selection_changed(self, selecting: bool) -> None:
        selected = self._worker is not None and self._worker._rod_position is not None
        self.rod_button.setText(
            "Clique na vara na prévia" if selecting else "Vara selecionada" if selected else "Selecionar vara"
        )

    def _save_snapshot(self) -> None:
        if self._worker is None or self._worker.latest_image is None:
            self._append_log("No frame to save; start capture first")
            return
        self.snapshot_button.setEnabled(False)
        self._append_log("Snapshot in 3 s; switch to the game window and place the cursor")
        QTimer.singleShot(3000, self._write_snapshot)

    def _write_snapshot(self) -> None:
        self.snapshot_button.setEnabled(True)
        image = self._worker.latest_image if self._worker is not None else None
        if image is None:
            self._append_log("Capture stopped before the snapshot was taken")
            return
        folder = DATA_DIR / "debug"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"snapshot-{datetime.now():%H%M%S}.png"
        image.save(path)
        self._append_log(
            f"Saved {path} | {self.coordinate_label.text()} | {self.cursor_label.text()}"
        )

    def _save_trace(self) -> None:
        if self._worker is None or not self._worker.trace:
            self._append_log("No scans recorded yet")
            return
        folder = DATA_DIR / "debug"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"trace-{datetime.now():%H%M%S}.txt"
        lines = ["time state bobber(x,y,matching_pixels) cursor cursor_template(score) frame_time"]
        lines += [
            f"{t} {state} {bobber} {cursor} {cursor_match} {frame_time}"
            for t, state, bobber, cursor, cursor_match, frame_time in list(self._worker.trace)
        ]
        path.write_text("\n".join(lines) + "\n")
        self._append_log(f"Saved {path} ({len(lines) - 1} scans)")

    def _stop_capture(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._set_status("STOPPING")
            self.start_button.setEnabled(False)
            self.fish_button.setEnabled(False)
            self._append_log("Stop requested")
            self._worker.stop()

    def _toggle_fishing(self) -> None:
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
            self._set_running(self.fish_button, False, "Iniciar pesca")
            self._append_log("Auto fishing countdown cancelled")
        elif self._fishing:
            self._set_fishing(False)
        else:
            self._countdown = 3
            self._set_running(self.fish_button, True, f"Iniciando em {self._countdown}...")
            self._countdown_timer.start()

    def _countdown_tick(self) -> None:
        self._countdown -= 1
        if self._countdown > 0:
            self.fish_button.setText(f"Iniciando em {self._countdown}...")
            return
        self._countdown_timer.stop()
        self._set_fishing(True)

    def _set_fishing(self, enabled: bool) -> None:
        self._fishing = enabled
        if self._worker is not None:
            self._worker.set_auto_move(enabled)
        self._set_running(self.fish_button, enabled, "Parar pesca" if enabled else "Iniciar pesca")
        self._append_log("Auto fishing enabled" if enabled else "Auto fishing disabled")

    def _show_frame(self, preview: QImage, origin, size, bobber, region, cursor=None) -> None:
        self.preview.set_frame(preview)
        self.frame_label.setText(f"{size[0]} x {size[1]}  |  {origin[0]},{origin[1]}")
        if cursor is None:
            self.cursor_label.setText("CURSOR  NÃO ENCONTRADO")
            self.cursor_label.setStyleSheet("color:#6272a4;")
        else:
            self.cursor_label.setText(f"CURSOR  {origin[0] + cursor[0]}, {origin[1] + cursor[1]}")
            self.cursor_label.setStyleSheet(f"color:{_CURSOR_COLOUR};")
        if bobber is None:
            self.coordinate_label.setText("BOIA  NÃO ENCONTRADA")
            self.coordinate_label.setStyleSheet("color:#6272a4;")
            if self._detection_found is True:
                self._append_log("Bobber lost in detection region")
            self._detection_found = False
        else:
            screen_x = origin[0] + region[0] + bobber[0]
            screen_y = origin[1] + region[1] + bobber[1]
            self.coordinate_label.setText(f"BOIA  {screen_x}, {screen_y}")
            self.coordinate_label.setStyleSheet("color:#ffb86c;")
            if self._detection_found is not True:
                self._append_log(f"Bobber detected at screen position {screen_x}, {screen_y}")
            self._detection_found = True

    def _set_status(self, status: str) -> None:
        normalized_status = status.upper()
        if self._status != normalized_status:
            self._status = normalized_status
            self._append_log(f"Status: {normalized_status}")

    def _capture_failed(self, message: str) -> None:
        self._set_status("CAPTURE ERROR")
        detail = message.rstrip().splitlines()[-1] if message.strip() else "erro desconhecido"
        self.fishing_status.setText(f"Falha na captura: {detail}")
        self._append_log(f"ERROR\n{message}")

    def _capture_finished(self) -> None:
        self._countdown_timer.stop()
        self._fishing = False
        self.start_button.setEnabled(True)
        self._set_running(self.start_button, False, "Iniciar captura de tela")
        self.fish_button.setEnabled(False)
        if self._status != "CAPTURE ERROR":
            self.fishing_status.setText("Inicie a captura de tela")
        self._set_running(self.fish_button, False, "Iniciar pesca")
        self.mode_box.setEnabled(True)
        self.cast_key_box.setEnabled(True)
        self.lure_key_box.setEnabled(True)
        self.rod_button.setEnabled(False)
        self.rod_button.setChecked(False)
        if self._status not in {"CAPTURE ERROR", "OFFLINE"}:
            self._set_status("OFFLINE")

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message.rstrip()}")

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.log_output.toPlainText())
        self._append_log("Log copied to clipboard")

    def log_output_clear(self) -> None:
        self.log_output.clear()

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(5000)
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(prog="pesqueiro-gui")
    parser.add_argument("--debug", action="store_true", help="show the log and snapshot/trace tools")
    arguments, qt_arguments = parser.parse_known_args()
    app = QApplication([sys.argv[0], *qt_arguments])
    window = MainWindow(debug=arguments.debug)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())