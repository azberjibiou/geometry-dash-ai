"""Windows visible-screen capture helpers.

The implementation intentionally uses Win32 GDI through ``ctypes`` so the first
Geometry Dash screenshot spike does not need a third-party screenshot library.
It captures visible pixels from the desktop; the target window must be windowed,
visible, and unobstructed.
"""

from __future__ import annotations

import ctypes
import os
import struct
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


class ScreenCaptureError(RuntimeError):
    """Raised when a visible screen/window capture cannot be completed."""


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """Absolute screen rectangle to capture."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for field_name in ("left", "top", "width", "height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an int")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @classmethod
    def from_string(cls, value: str) -> "CaptureRegion":
        """Parse ``LEFT,TOP,WIDTH,HEIGHT``."""

        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("region must have the form LEFT,TOP,WIDTH,HEIGHT")
        try:
            left, top, width, height = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("region values must be integers") from exc
        return cls(left=left, top=top, width=width, height=height)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CaptureRegion":
        return cls(
            left=_mapping_int(data, "left"),
            top=_mapping_int(data, "top"),
            width=_mapping_int(data, "width"),
            height=_mapping_int(data, "height"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """A 32-bit BGRA frame captured from the visible desktop."""

    width: int
    height: int
    bgra: bytes
    region: CaptureRegion

    def __post_init__(self) -> None:
        if self.width != self.region.width:
            raise ValueError("frame width must match capture region")
        if self.height != self.region.height:
            raise ValueError("frame height must match capture region")
        expected_length = self.width * self.height * 4
        if len(self.bgra) != expected_length:
            raise ValueError(
                f"BGRA payload has {len(self.bgra)} bytes; expected {expected_length}"
            )


@dataclass(frozen=True, slots=True)
class VisibleWindow:
    """One visible top-level Windows window."""

    hwnd: int
    title: str
    region: CaptureRegion

    def to_dict(self) -> dict[str, object]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "region": self.region.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CaptureSource:
    """Capture either an explicit screen region or a matching window title."""

    window_title: str | None = "Geometry Dash"
    region: CaptureRegion | None = None

    def __post_init__(self) -> None:
        if self.region is None and not self.window_title:
            raise ValueError("either window_title or region is required")

    def capture(self) -> CapturedFrame:
        if self.region is not None:
            return capture_visible_region(self.region)
        if not self.window_title:
            raise ScreenCaptureError("window title is required when no region is set")
        return capture_window(self.window_title)

    def describe(self) -> dict[str, object]:
        return {
            "window_title": self.window_title,
            "region": self.region.to_dict() if self.region is not None else None,
            "capture_mode": "region" if self.region is not None else "window",
            "format": "bmp32",
        }


def capture_window(title_substring: str) -> CapturedFrame:
    """Capture the visible pixels of the first matching top-level window."""

    return capture_visible_region(find_window_region(title_substring))


def capture_visible_region(region: CaptureRegion) -> CapturedFrame:
    """Capture a visible screen rectangle using Win32 GDI."""

    _require_windows()
    user32 = _load_user32()
    gdi32 = _load_gdi32()

    screen_dc = user32.GetDC(None)
    if not screen_dc:
        raise _last_windows_error("GetDC")
    memory_dc = gdi32.CreateCompatibleDC(screen_dc)
    if not memory_dc:
        user32.ReleaseDC(None, screen_dc)
        raise _last_windows_error("CreateCompatibleDC")

    bitmap = gdi32.CreateCompatibleBitmap(screen_dc, region.width, region.height)
    if not bitmap:
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)
        raise _last_windows_error("CreateCompatibleBitmap")

    old_object = gdi32.SelectObject(memory_dc, bitmap)
    if not old_object:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)
        raise _last_windows_error("SelectObject")

    try:
        if not gdi32.BitBlt(
            memory_dc,
            0,
            0,
            region.width,
            region.height,
            screen_dc,
            region.left,
            region.top,
            _SRCCOPY,
        ):
            raise _last_windows_error("BitBlt")

        info = _BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        info.bmiHeader.biWidth = region.width
        info.bmiHeader.biHeight = region.height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = _BI_RGB
        info.bmiHeader.biSizeImage = region.width * region.height * 4

        buffer = ctypes.create_string_buffer(info.bmiHeader.biSizeImage)
        scan_lines = gdi32.GetDIBits(
            memory_dc,
            bitmap,
            0,
            region.height,
            buffer,
            ctypes.byref(info),
            _DIB_RGB_COLORS,
        )
        if scan_lines != region.height:
            raise _last_windows_error("GetDIBits")
        return CapturedFrame(
            width=region.width,
            height=region.height,
            bgra=bytes(buffer),
            region=region,
        )
    finally:
        gdi32.SelectObject(memory_dc, old_object)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)


def write_bmp(path: str | Path, frame: CapturedFrame) -> None:
    """Write a captured 32-bit BGRA frame as an uncompressed BMP."""

    bmp_path = Path(path)
    if bmp_path.parent != Path("."):
        bmp_path.parent.mkdir(parents=True, exist_ok=True)

    pixel_data_size = len(frame.bgra)
    pixel_offset = 14 + 40
    file_size = pixel_offset + pixel_data_size
    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, pixel_offset)
    info_header = struct.pack(
        "<IiiHHIIiiII",
        40,
        frame.width,
        frame.height,
        1,
        32,
        _BI_RGB,
        pixel_data_size,
        0,
        0,
        0,
        0,
    )

    with bmp_path.open("wb") as file:
        file.write(file_header)
        file.write(info_header)
        file.write(frame.bgra)


def find_window_region(title_substring: str) -> CaptureRegion:
    """Return the screen region for the first visible window matching a title."""

    matching_windows = [
        window
        for window in list_visible_windows()
        if title_substring.lower() in window.title.lower()
    ]
    if not matching_windows:
        raise ScreenCaptureError(f"no visible window title contains {title_substring!r}")
    return matching_windows[0].region


def list_visible_windows() -> list[VisibleWindow]:
    """List visible top-level windows with non-empty titles."""

    _require_windows()
    user32 = _load_user32()
    windows: list[VisibleWindow] = []

    enum_proc = _WNDENUMPROC(
        lambda hwnd, _lparam: _collect_window(user32, hwnd, windows)
    )
    if not user32.EnumWindows(enum_proc, 0):
        raise _last_windows_error("EnumWindows")
    return windows


def _collect_window(
    user32: ctypes.WinDLL,
    hwnd: int,
    windows: list[VisibleWindow],
) -> bool:
    if not user32.IsWindowVisible(hwnd):
        return True

    title_length = user32.GetWindowTextLengthW(hwnd)
    if title_length <= 0:
        return True
    buffer = ctypes.create_unicode_buffer(title_length + 1)
    copied = user32.GetWindowTextW(hwnd, buffer, title_length + 1)
    if copied <= 0:
        return True
    title = buffer.value.strip()
    if not title:
        return True

    try:
        region = _window_region(hwnd)
    except ScreenCaptureError:
        return True
    windows.append(VisibleWindow(hwnd=int(hwnd), title=title, region=region))
    return True


def _window_region(hwnd: int) -> CaptureRegion:
    rect = _RECT()
    try:
        dwmapi = _load_dwmapi()
        result = dwmapi.DwmGetWindowAttribute(
            hwnd,
            _DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result != 0:
            raise OSError(result)
    except OSError:
        user32 = _load_user32()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise _last_windows_error("GetWindowRect")

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        raise ScreenCaptureError("window rectangle is empty")
    return CaptureRegion(left=rect.left, top=rect.top, width=width, height=height)


def _require_windows() -> None:
    if os.name != "nt":
        raise ScreenCaptureError("Windows screen capture is only available on Windows")


def _last_windows_error(operation: str) -> ScreenCaptureError:
    error_code = ctypes.get_last_error()
    return ScreenCaptureError(f"{operation} failed with Windows error {error_code}")


def _load_user32() -> ctypes.WinDLL:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [
        wintypes.HWND,
        wintypes.LPWSTR,
        ctypes.c_int,
    ]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    return user32


def _load_gdi32() -> ctypes.WinDLL:
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
    ]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.BitBlt.argtypes = [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.POINTER(_BITMAPINFO),
        wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL
    return gdi32


def _load_dwmapi() -> ctypes.WinDLL:
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    dwmapi.DwmGetWindowAttribute.argtypes = [
        wintypes.HWND,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_int
    return dwmapi


def _mapping_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an int")
    return value


_SRCCOPY = 0x00CC0020
_DIB_RGB_COLORS = 0
_BI_RGB = 0
_DWMWA_EXTENDED_FRAME_BOUNDS = 9


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", wintypes.BYTE),
        ("rgbGreen", wintypes.BYTE),
        ("rgbRed", wintypes.BYTE),
        ("rgbReserved", wintypes.BYTE),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", _RGBQUAD * 1),
    ]


_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
