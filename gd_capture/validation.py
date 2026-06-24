"""Validation for captured frame manifests and BMP frame files."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gd_capture.manifest import FrameCaptureRecord, load_manifest_jsonl


@dataclass(frozen=True, slots=True)
class FrameImageStats:
    """Basic image statistics used to sanity-check captured frames."""

    frame_path: str
    width: int
    height: int
    mean_brightness: float
    brightness_variance: float
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameValidationIssue:
    """One validation issue for a captured frame manifest."""

    code: str
    message: str
    frame_path: str | None = None
    tick: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameValidationSummary:
    """Summary of manifest and frame validation results."""

    manifest_path: str
    frame_count: int
    ok: bool
    width_values: list[int]
    height_values: list[int]
    brightness_min: float | None
    brightness_max: float | None
    variance_min: float | None
    variance_max: float | None
    unique_hash_count: int
    issues: list[FrameValidationIssue]
    stats: list[FrameImageStats]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        data["stats"] = [stat.to_dict() for stat in self.stats]
        return data


def validate_frame_manifest(
    manifest_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    min_blank_variance: float = 1.0,
    require_change: bool = True,
) -> FrameValidationSummary:
    """Validate manifest rows and basic BMP frame properties."""

    manifest = Path(manifest_path)
    root = Path(base_dir) if base_dir is not None else manifest.parent
    records = load_manifest_jsonl(manifest)
    issues: list[FrameValidationIssue] = []
    stats: list[FrameImageStats] = []

    previous_tick: int | None = None
    for record in records:
        if previous_tick is not None and record.tick < previous_tick:
            issues.append(
                FrameValidationIssue(
                    code="decreasing_tick",
                    message=f"tick {record.tick} follows tick {previous_tick}",
                    frame_path=record.frame_path,
                    tick=record.tick,
                )
            )
        previous_tick = record.tick

        frame_path = root / record.frame_path
        if not frame_path.exists():
            issues.append(
                FrameValidationIssue(
                    code="missing_frame",
                    message="frame file does not exist",
                    frame_path=record.frame_path,
                    tick=record.tick,
                )
            )
            continue

        try:
            image_stats = inspect_bmp(frame_path)
        except ValueError as exc:
            issues.append(
                FrameValidationIssue(
                    code="invalid_bmp",
                    message=str(exc),
                    frame_path=record.frame_path,
                    tick=record.tick,
                )
            )
            continue

        stats.append(image_stats)
        if image_stats.width != record.capture_width:
            issues.append(
                FrameValidationIssue(
                    code="width_mismatch",
                    message=(
                        f"manifest width {record.capture_width} != "
                        f"image width {image_stats.width}"
                    ),
                    frame_path=record.frame_path,
                    tick=record.tick,
                )
            )
        if image_stats.height != record.capture_height:
            issues.append(
                FrameValidationIssue(
                    code="height_mismatch",
                    message=(
                        f"manifest height {record.capture_height} != "
                        f"image height {image_stats.height}"
                    ),
                    frame_path=record.frame_path,
                    tick=record.tick,
                )
            )
        if image_stats.brightness_variance < min_blank_variance:
            issues.append(
                FrameValidationIssue(
                    code="blank_frame",
                    message=(
                        f"brightness variance {image_stats.brightness_variance:.3f} "
                        f"is below {min_blank_variance:.3f}"
                    ),
                    frame_path=record.frame_path,
                    tick=record.tick,
                )
            )

    unique_hashes = {stat.sha256 for stat in stats}
    if require_change and len(stats) > 1 and len(unique_hashes) <= 1:
        issues.append(
            FrameValidationIssue(
                code="static_hashes",
                message="all captured frame hashes are identical",
            )
        )

    brightness_values = [stat.mean_brightness for stat in stats]
    variance_values = [stat.brightness_variance for stat in stats]

    return FrameValidationSummary(
        manifest_path=str(manifest),
        frame_count=len(records),
        ok=not issues,
        width_values=sorted({stat.width for stat in stats}),
        height_values=sorted({stat.height for stat in stats}),
        brightness_min=min(brightness_values) if brightness_values else None,
        brightness_max=max(brightness_values) if brightness_values else None,
        variance_min=min(variance_values) if variance_values else None,
        variance_max=max(variance_values) if variance_values else None,
        unique_hash_count=len(unique_hashes),
        issues=issues,
        stats=stats,
    )


def inspect_bmp(path: str | Path) -> FrameImageStats:
    """Read a 24-bit or 32-bit uncompressed BMP and compute basic stats."""

    bmp_path = Path(path)
    data = bmp_path.read_bytes()
    if len(data) < 54:
        raise ValueError("BMP is too small")
    if data[:2] != b"BM":
        raise ValueError("file is not a BMP")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_header_size = struct.unpack_from("<I", data, 14)[0]
    if dib_header_size < 40:
        raise ValueError("unsupported BMP DIB header")

    width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bit_count = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    height = abs(signed_height)

    if width <= 0 or height <= 0:
        raise ValueError("BMP dimensions must be positive")
    if planes != 1:
        raise ValueError("BMP planes must be 1")
    if compression != 0:
        raise ValueError("compressed BMP files are not supported")
    if bit_count not in (24, 32):
        raise ValueError("only 24-bit and 32-bit BMP files are supported")

    bytes_per_pixel = bit_count // 8
    row_stride = ((bit_count * width + 31) // 32) * 4
    required_size = pixel_offset + row_stride * height
    if len(data) < required_size:
        raise ValueError("BMP pixel data is truncated")

    pixel_bytes = data[pixel_offset:required_size]
    pixel_count = width * height
    brightness_sum = 0.0
    brightness_square_sum = 0.0

    for row_index in range(height):
        row_start = row_index * row_stride
        row = pixel_bytes[row_start : row_start + row_stride]
        for column_index in range(width):
            pixel_start = column_index * bytes_per_pixel
            blue = row[pixel_start]
            green = row[pixel_start + 1]
            red = row[pixel_start + 2]
            brightness = (red + green + blue) / 3.0
            brightness_sum += brightness
            brightness_square_sum += brightness * brightness

    mean_brightness = brightness_sum / pixel_count
    brightness_variance = (
        brightness_square_sum / pixel_count - mean_brightness * mean_brightness
    )
    return FrameImageStats(
        frame_path=str(bmp_path),
        width=width,
        height=height,
        mean_brightness=mean_brightness,
        brightness_variance=max(0.0, brightness_variance),
        sha256=hashlib.sha256(pixel_bytes).hexdigest(),
    )
