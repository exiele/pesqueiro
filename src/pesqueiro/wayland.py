from __future__ import annotations

import asyncio
import os
import re
import secrets
import select
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image


PORTAL_NAME = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_INTERFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_INTERFACE = "org.freedesktop.portal.Request"
SESSION_INTERFACE = "org.freedesktop.portal.Session"


class WaylandCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapturedFrame:
    image: Image.Image
    origin: tuple[int, int]
    stream_id: int
    captured_at: float = 0.0


def _variant_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _stream_geometry(
    properties: dict[str, Any],
) -> tuple[tuple[int, int] | None, tuple[int, int]]:
    size = _variant_value(properties.get("size"))
    position = _variant_value(properties.get("position", (0, 0)))
    origin = int(position[0]), int(position[1])
    if size is None:
        return None, origin
    if len(size) != 2:
        raise WaylandCaptureError("ScreenCast portal returned invalid stream dimensions")

    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise WaylandCaptureError(f"Invalid PipeWire stream dimensions: {width}x{height}")
    return (width, height), origin


def _caps_dimensions(line: bytes) -> tuple[int, int] | None:
    width = re.search(rb"width=\(int\)\s*(\d+)", line)
    height = re.search(rb"height=\(int\)\s*(\d+)", line)
    if width is None or height is None:
        return None
    dimensions = int(width.group(1)), int(height.group(1))
    if dimensions[0] <= 0 or dimensions[1] <= 0:
        return None
    return dimensions


def _received_unix_fd(body: list[Any], unix_fds: list[int]) -> int:
    if not body:
        raise WaylandCaptureError("ScreenCast portal returned no PipeWire remote handle")
    fd_index = int(body[0])
    if fd_index < 0 or fd_index >= len(unix_fds):
        raise WaylandCaptureError("ScreenCast portal returned an invalid PipeWire remote handle")
    return int(unix_fds[fd_index])


def _decode_rgba_frame(data: bytes, size: tuple[int, int]) -> Image.Image:
    width, height = size
    expected = width * height * 4
    if len(data) != expected:
        raise WaylandCaptureError(
            f"PipeWire frame has {len(data)} bytes; expected {expected} for {width}x{height} RGBA"
        )
    return Image.frombytes("RGBA", size, data).convert("RGB")


class WaylandScreenCapture:
    def __init__(self, frame_timeout: float = 8.0) -> None:
        self.frame_timeout = frame_timeout
        self._bus = None
        self._session_path: str | None = None
        self._pipewire_fd: int | None = None
        self._pipewire_process: subprocess.Popen[bytes] | None = None
        self._raw_frame_fd: int | None = None
        self._responses: dict[str, tuple[int, dict[str, Any]]] = {}
        self._pending_requests: dict[str, asyncio.Future[tuple[int, dict[str, Any]]]] = {}
        self._stream_id: int | None = None
        self._size: tuple[int, int] | None = None
        self._origin = (0, 0)
        self._frame_ready = threading.Condition()
        self._latest: tuple[bytes, float] | None = None
        self._latest_seq = 0
        self._returned_seq = 0
        self._reader_error: str | None = None
        self._reader_stop = threading.Event()
        self._reader: threading.Thread | None = None

    async def start(self) -> None:
        if self._pipewire_process is not None:
            return
        if shutil.which("gst-launch-1.0") is None:
            raise WaylandCaptureError(
                "GStreamer is required for Wayland capture (install gst-launch-1.0 and its PipeWire plugin)"
            )

        try:
            from dbus_next import BusType, Message, MessageType, Variant
            from dbus_next.aio import MessageBus
        except ImportError as error:
            raise WaylandCaptureError(
                "The dbus-next package is required for Wayland portal capture; install Pesqueiro's dependencies"
            ) from error

        self._Message = Message
        self._MessageType = MessageType
        self._Variant = Variant
        try:
            self._bus = await MessageBus(
                bus_type=BusType.SESSION,
                negotiate_unix_fd=True,
            ).connect()
            self._bus.add_message_handler(self._handle_message)
            await self._create_portal_stream()
            self._start_pipewire()
            await asyncio.to_thread(self._wait_for_stream_size)
            await asyncio.to_thread(self._read_exactly, 1)
            self._reader_stop.clear()
            self._reader_error = None
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
        except WaylandCaptureError:
            await self.close()
            raise
        except Exception as error:
            await self.close()
            raise WaylandCaptureError(f"Could not start Wayland ScreenCast capture: {error}") from error

    async def _create_portal_stream(self) -> None:
        token = f"pesqueiro_{secrets.token_hex(8)}"
        handle_token = self._Variant("s", token)

        create_results = await self._portal_request(
            "CreateSession",
            "a{sv}",
            [{"handle_token": handle_token, "session_handle_token": handle_token}],
            token,
        )
        self._session_path = _variant_value(create_results.get("session_handle"))
        if not self._session_path:
            raise WaylandCaptureError("ScreenCast portal did not return a session handle")

        await self._portal_request(
            "SelectSources",
            "oa{sv}",
            [
                self._session_path,
                {
                    "handle_token": self._Variant("s", f"{token}_sources"),
                    "types": self._Variant("u", 3),
                    "multiple": self._Variant("b", False),
                    "cursor_mode": self._Variant("u", 2),
                },
            ],
            f"{token}_sources",
        )

        start_results = await self._portal_request(
            "Start",
            "osa{sv}",
            [
                self._session_path,
                "",
                {"handle_token": self._Variant("s", f"{token}_start")},
            ],
            f"{token}_start",
        )
        streams = _variant_value(start_results.get("streams", []))
        if not streams:
            raise WaylandCaptureError("No screen or window stream was selected")

        self._stream_id = int(streams[0][0])
        stream_properties = _variant_value(streams[0][1])
        self._size, self._origin = _stream_geometry(stream_properties)
        remote_reply = await self._bus.call(
            self._Message(
                destination=PORTAL_NAME,
                path=PORTAL_PATH,
                interface=SCREENCAST_INTERFACE,
                member="OpenPipeWireRemote",
                signature="oa{sv}",
                body=[self._session_path, {}],
            )
        )
        if remote_reply.message_type == self._MessageType.ERROR:
            raise WaylandCaptureError(
                remote_reply.body[0] if remote_reply.body else "Could not open the portal PipeWire connection"
            )
        self._pipewire_fd = _received_unix_fd(remote_reply.body, remote_reply.unix_fds)

    async def _portal_request(
        self,
        method: str,
        signature: str,
        body: list[Any],
        token: str,
    ) -> dict[str, Any]:
        reply = await self._bus.call(
            self._Message(
                destination=PORTAL_NAME,
                path=PORTAL_PATH,
                interface=SCREENCAST_INTERFACE,
                member=method,
                signature=signature,
                body=body,
            )
        )
        if reply.message_type == self._MessageType.ERROR:
            raise WaylandCaptureError(reply.body[0] if reply.body else f"Portal call {method} failed")

        request_path = reply.body[0]
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_path] = future
        if request_path in self._responses:
            future.set_result(self._responses.pop(request_path))

        try:
            response, results = await asyncio.wait_for(future, timeout=120)
        except TimeoutError as error:
            raise WaylandCaptureError(f"Timed out waiting for ScreenCast permission ({method})") from error
        finally:
            self._pending_requests.pop(request_path, None)

        if response == 1:
            raise WaylandCaptureError("ScreenCast was cancelled by the user")
        if response != 0:
            raise WaylandCaptureError(f"ScreenCast portal failed during {method} (response {response})")
        return results

    def _handle_message(self, message: Any) -> bool:
        if (
            message.message_type != self._MessageType.SIGNAL
            or message.interface != REQUEST_INTERFACE
            or message.member != "Response"
        ):
            return False

        request_path = message.path
        response = int(message.body[0])
        results = message.body[1]
        future = self._pending_requests.get(request_path)
        if future is not None and not future.done():
            future.set_result((response, results))
        else:
            self._responses[request_path] = (response, results)
        return False

    def _start_pipewire(self) -> None:
        if self._stream_id is None or self._pipewire_fd is None:
            raise WaylandCaptureError("ScreenCast stream has not been initialized")
        raw_read_fd, raw_write_fd = os.pipe()
        pipeline = _build_pipewire_command(
            self._stream_id,
            self._size,
            self._pipewire_fd,
            raw_write_fd,
        )
        try:
            self._pipewire_process = subprocess.Popen(
                pipeline,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                pass_fds=(self._pipewire_fd, raw_write_fd),
            )
        except OSError as error:
            os.close(raw_read_fd)
            os.close(raw_write_fd)
            raise WaylandCaptureError(f"Could not start GStreamer PipeWire capture: {error}") from error
        os.close(raw_write_fd)
        self._raw_frame_fd = raw_read_fd

    def _wait_for_stream_size(self) -> None:
        if self._pipewire_process is None or self._pipewire_process.stdout is None:
            raise WaylandCaptureError("GStreamer did not expose stream negotiation details")

        descriptor = self._pipewire_process.stdout.fileno()
        deadline = time.monotonic() + self.frame_timeout
        buffered_output = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WaylandCaptureError("Timed out waiting for PipeWire video dimensions")
            ready, _, _ = select.select([descriptor], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(descriptor, 4096)
            if not chunk:
                status = self._pipewire_process.poll()
                raise WaylandCaptureError(
                    f"GStreamer ended before negotiating video dimensions (exit status {status})"
                )
            buffered_output.extend(chunk)
            while b"\n" in buffered_output:
                line, _, remainder = buffered_output.partition(b"\n")
                buffered_output = bytearray(remainder)
                dimensions = _caps_dimensions(line)
                if dimensions is not None:
                    self._size = dimensions
                    threading.Thread(
                        target=self._drain_pipewire_diagnostics,
                        daemon=True,
                    ).start()
                    return

    def _drain_pipewire_diagnostics(self) -> None:
        process = self._pipewire_process
        if process is None or process.stdout is None:
            return
        try:
            while os.read(process.stdout.fileno(), 4096):
                pass
        except OSError:
            pass

    def _read_exactly(self, frame_count: int) -> bytes:
        if self._pipewire_process is None or self._raw_frame_fd is None or self._size is None:
            raise WaylandCaptureError("Wayland capture has not been started")
        byte_count = self._size[0] * self._size[1] * 4 * frame_count
        descriptor = self._raw_frame_fd
        deadline = time.monotonic() + self.frame_timeout
        frame_data = bytearray()
        while len(frame_data) < byte_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WaylandCaptureError("Timed out waiting for a PipeWire screen frame")
            ready, _, _ = select.select([descriptor], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(descriptor, byte_count - len(frame_data))
            if not chunk:
                status = self._pipewire_process.poll()
                raise WaylandCaptureError(f"PipeWire capture stream ended (GStreamer exit status {status})")
            frame_data.extend(chunk)
        return bytes(frame_data)

    def _read_loop(self) -> None:
        descriptor = self._raw_frame_fd
        process = self._pipewire_process
        if descriptor is None or process is None or self._size is None:
            return
        byte_count = self._size[0] * self._size[1] * 4
        buffer = bytearray()
        try:
            while not self._reader_stop.is_set():
                ready, _, _ = select.select([descriptor], [], [], 0.5)
                if not ready:
                    continue
                chunk = os.read(descriptor, byte_count - len(buffer))
                if not chunk:
                    raise WaylandCaptureError(
                        f"PipeWire capture stream ended (GStreamer exit status {process.poll()})"
                    )
                buffer.extend(chunk)
                if len(buffer) == byte_count:
                    with self._frame_ready:
                        self._latest = (bytes(buffer), time.monotonic())
                        self._latest_seq += 1
                        self._frame_ready.notify_all()
                    buffer.clear()
        except (WaylandCaptureError, OSError, ValueError) as error:
            with self._frame_ready:
                if not self._reader_stop.is_set():
                    self._reader_error = str(error)
                self._frame_ready.notify_all()

    def _next_frame(self) -> tuple[bytes, float]:
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._latest_seq != self._returned_seq or self._reader_error is not None,
                timeout=self.frame_timeout,
            )
            if self._latest_seq == self._returned_seq:
                if self._reader_error is not None:
                    raise WaylandCaptureError(self._reader_error)
                raise WaylandCaptureError("Timed out waiting for a PipeWire screen frame")
            self._returned_seq = self._latest_seq
            assert self._latest is not None
            return self._latest

    async def capture(self) -> CapturedFrame:
        if self._pipewire_process is None or self._size is None or self._stream_id is None:
            raise WaylandCaptureError("Call start() before capture()")
        raw_frame, captured_at = await asyncio.to_thread(self._next_frame)
        return CapturedFrame(
            _decode_rgba_frame(raw_frame, self._size), self._origin, self._stream_id, captured_at
        )

    async def close(self) -> None:
        self._reader_stop.set()
        process, self._pipewire_process = self._pipewire_process, None
        if process is not None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 2)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)

        if self._reader is not None:
            await asyncio.to_thread(self._reader.join, 2)
            self._reader = None

        if self._raw_frame_fd is not None:
            try:
                os.close(self._raw_frame_fd)
            except OSError:
                pass
            self._raw_frame_fd = None

        if self._pipewire_fd is not None:
            try:
                os.close(self._pipewire_fd)
            except OSError:
                pass
            self._pipewire_fd = None

        if self._bus is not None:
            if self._session_path:
                try:
                    await self._bus.call(
                        self._Message(
                            destination=PORTAL_NAME,
                            path=self._session_path,
                            interface=SESSION_INTERFACE,
                            member="Close",
                        )
                    )
                except Exception:
                    pass
            self._bus.disconnect()
            self._bus = None
        self._session_path = None

    async def __aenter__(self) -> WaylandScreenCapture:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.close()


def _build_pipewire_command(
    stream_id: int,
    size: tuple[int, int] | None,
    remote_fd: int,
    raw_frame_fd: int,
) -> list[str]:
    pipeline = [
        "gst-launch-1.0",
        "-v",
        "pipewiresrc",
        f"fd={remote_fd}",
        f"path={stream_id}",
        "do-timestamp=true",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=RGBA",
    ]
    if size is not None:
        width, height = size
        pipeline[-1] += f",width={width},height={height}"
    pipeline.extend(["!", "fdsink", f"fd={raw_frame_fd}", "sync=false"])
    return pipeline