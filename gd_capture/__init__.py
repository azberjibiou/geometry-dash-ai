"""Frame capture helpers for Geometry Dash observation experiments."""

from gd_capture.manifest import (
    FrameCaptureRecord,
    iter_manifest_jsonl,
    load_manifest_jsonl,
    save_manifest_jsonl,
)
from gd_capture.screen_capture import (
    CapturedFrame,
    CaptureRegion,
    CaptureSource,
    ScreenCaptureError,
    activate_window,
    capture_visible_region,
    capture_window,
    foreground_window,
    foreground_window_matches,
    list_visible_windows,
    write_bmp,
)
from gd_capture.validation import (
    FrameImageStats,
    FrameValidationIssue,
    FrameValidationSummary,
    inspect_bmp,
    validate_frame_manifest,
)

__all__ = [
    "CapturedFrame",
    "CaptureRegion",
    "CaptureSource",
    "FrameCaptureRecord",
    "FrameImageStats",
    "FrameValidationIssue",
    "FrameValidationSummary",
    "ScreenCaptureError",
    "activate_window",
    "capture_visible_region",
    "capture_window",
    "foreground_window",
    "foreground_window_matches",
    "inspect_bmp",
    "iter_manifest_jsonl",
    "list_visible_windows",
    "load_manifest_jsonl",
    "save_manifest_jsonl",
    "validate_frame_manifest",
    "write_bmp",
]
