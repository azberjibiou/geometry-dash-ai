"""Load prepared imitation samples into normalized grayscale frame stacks."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gd_imitation.dataset import ImitationSample, load_samples_jsonl


class ImageDatasetError(ValueError):
    """Raised when image-backed imitation samples cannot be loaded."""


@dataclass(frozen=True, slots=True)
class ImageDatasetConfig:
    """Configuration for dependency-free image sample loading."""

    image_width: int = 84
    image_height: int = 84
    required_frame_stack_size: int | None = None
    pad_short_stacks: bool = True
    progress_scale: float = 100.0
    label_mode: str = "events"

    def __post_init__(self) -> None:
        if self.image_width <= 0:
            raise ImageDatasetError("image_width must be positive")
        if self.image_height <= 0:
            raise ImageDatasetError("image_height must be positive")
        if (
            self.required_frame_stack_size is not None
            and self.required_frame_stack_size <= 0
        ):
            raise ImageDatasetError("required_frame_stack_size must be positive")
        if self.progress_scale <= 0.0:
            raise ImageDatasetError("progress_scale must be positive")
        if self.label_mode not in ("events", "target_input_down"):
            raise ImageDatasetError(
                "label_mode must be 'events' or 'target_input_down'"
            )


@dataclass(frozen=True, slots=True)
class GrayscaleFrame:
    """One normalized grayscale image."""

    width: int
    height: int
    pixels: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ImageDatasetError("frame dimensions must be positive")
        if len(self.pixels) != self.height:
            raise ImageDatasetError("pixel row count must match height")
        for row in self.pixels:
            if len(row) != self.width:
                raise ImageDatasetError("pixel column count must match width")
            if any(pixel < 0.0 or pixel > 1.0 for pixel in row):
                raise ImageDatasetError("pixels must be normalized to 0..1")


@dataclass(frozen=True, slots=True)
class ImageInputSample:
    """One prepared sample with loaded frame-stack pixels and labels."""

    sample: ImitationSample
    frame_stack: tuple[GrayscaleFrame, ...]
    scalar_features: tuple[float, ...]
    labels: tuple[float, ...]

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Return ``(stack, height, width)``."""

        if not self.frame_stack:
            return (0, 0, 0)
        first = self.frame_stack[0]
        return (len(self.frame_stack), first.height, first.width)


def load_prepared_image_dataset(
    dataset_dir: str | Path,
    *,
    frame_base_dir: str | Path | None = None,
    config: ImageDatasetConfig | None = None,
) -> list[ImageInputSample]:
    """Load image samples from a directory made by prepare_imitation_dataset.py."""

    root = Path(dataset_dir)
    samples_path = root / "samples.jsonl"
    effective_frame_base_dir = (
        Path(frame_base_dir) if frame_base_dir is not None else _frame_base_from_summary(root)
    )
    return load_image_dataset(
        samples_path,
        frame_base_dir=effective_frame_base_dir,
        config=config,
    )


def load_image_dataset(
    samples_path: str | Path,
    *,
    frame_base_dir: str | Path | None = None,
    config: ImageDatasetConfig | None = None,
) -> list[ImageInputSample]:
    """Load prepared sample rows and their referenced BMP frame stacks."""

    path = Path(samples_path)
    base_dir = Path(frame_base_dir) if frame_base_dir is not None else path.parent
    effective_config = config or ImageDatasetConfig()
    samples = load_samples_jsonl(path)
    frame_cache: dict[Path, GrayscaleFrame] = {}
    return [
        load_image_input_sample(
            sample,
            frame_base_dir=base_dir,
            config=effective_config,
            _frame_cache=frame_cache,
        )
        for sample in samples
    ]


def load_image_input_sample(
    sample: ImitationSample,
    *,
    frame_base_dir: str | Path,
    config: ImageDatasetConfig | None = None,
    _frame_cache: dict[Path, GrayscaleFrame] | None = None,
) -> ImageInputSample:
    """Load one sample's frames, scalar features, and labels."""

    effective_config = config or ImageDatasetConfig()
    base_dir = Path(frame_base_dir)
    frames = tuple(
        _load_resized_frame(
            resolve_frame_path(frame_path, base_dir),
            width=effective_config.image_width,
            height=effective_config.image_height,
            frame_cache=_frame_cache,
        )
        for frame_path in sample.frame_paths
    )
    frames = _normalize_frame_stack(
        frames,
        required_size=effective_config.required_frame_stack_size,
        pad_short=effective_config.pad_short_stacks,
    )
    return ImageInputSample(
        sample=sample,
        frame_stack=frames,
        scalar_features=(
            sample.progress / effective_config.progress_scale,
            1.0 if sample.input_down else 0.0,
        ),
        labels=_labels_for_sample(sample, label_mode=effective_config.label_mode),
    )


def resolve_frame_path(frame_path: str | Path, frame_base_dir: str | Path) -> Path:
    """Resolve a sample frame path against a capture directory."""

    path = Path(frame_path)
    if path.is_absolute():
        return path
    return Path(frame_base_dir) / path


def read_bmp_grayscale(path: str | Path) -> GrayscaleFrame:
    """Read a 24-bit or 32-bit uncompressed BMP as normalized grayscale."""

    bmp_path = Path(path)
    data = bmp_path.read_bytes()
    if len(data) < 54:
        raise ImageDatasetError("BMP is too small")
    if data[:2] != b"BM":
        raise ImageDatasetError("file is not a BMP")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_header_size = struct.unpack_from("<I", data, 14)[0]
    if dib_header_size < 40:
        raise ImageDatasetError("unsupported BMP DIB header")

    width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bit_count = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    height = abs(signed_height)

    if width <= 0 or height <= 0:
        raise ImageDatasetError("BMP dimensions must be positive")
    if planes != 1:
        raise ImageDatasetError("BMP planes must be 1")
    if compression != 0:
        raise ImageDatasetError("compressed BMP files are not supported")
    if bit_count not in (24, 32):
        raise ImageDatasetError("only 24-bit and 32-bit BMP files are supported")

    bytes_per_pixel = bit_count // 8
    row_stride = ((bit_count * width + 31) // 32) * 4
    required_size = pixel_offset + row_stride * height
    if len(data) < required_size:
        raise ImageDatasetError("BMP pixel data is truncated")

    bottom_up = signed_height > 0
    rows: list[tuple[float, ...]] = []
    for output_row in range(height):
        stored_row = height - 1 - output_row if bottom_up else output_row
        row_start = pixel_offset + stored_row * row_stride
        row_values: list[float] = []
        for column in range(width):
            pixel_start = row_start + column * bytes_per_pixel
            blue = data[pixel_start]
            green = data[pixel_start + 1]
            red = data[pixel_start + 2]
            row_values.append(_grayscale(red, green, blue))
        rows.append(tuple(row_values))

    return GrayscaleFrame(width=width, height=height, pixels=tuple(rows))


def read_bmp_grayscale_resized(
    path: str | Path,
    *,
    width: int,
    height: int,
) -> GrayscaleFrame:
    """Read a BMP directly into a nearest-neighbor resized grayscale frame."""

    if width <= 0 or height <= 0:
        raise ImageDatasetError("resize dimensions must be positive")

    bmp_path = Path(path)
    data = bmp_path.read_bytes()
    if len(data) < 54:
        raise ImageDatasetError("BMP is too small")
    if data[:2] != b"BM":
        raise ImageDatasetError("file is not a BMP")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_header_size = struct.unpack_from("<I", data, 14)[0]
    if dib_header_size < 40:
        raise ImageDatasetError("unsupported BMP DIB header")

    source_width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bit_count = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    source_height = abs(signed_height)

    if source_width <= 0 or source_height <= 0:
        raise ImageDatasetError("BMP dimensions must be positive")
    if planes != 1:
        raise ImageDatasetError("BMP planes must be 1")
    if compression != 0:
        raise ImageDatasetError("compressed BMP files are not supported")
    if bit_count not in (24, 32):
        raise ImageDatasetError("only 24-bit and 32-bit BMP files are supported")

    bytes_per_pixel = bit_count // 8
    row_stride = ((bit_count * source_width + 31) // 32) * 4
    required_size = pixel_offset + row_stride * source_height
    if len(data) < required_size:
        raise ImageDatasetError("BMP pixel data is truncated")

    bottom_up = signed_height > 0
    rows: list[tuple[float, ...]] = []
    for output_y in range(height):
        source_y = min(int(output_y * source_height / height), source_height - 1)
        stored_row = source_height - 1 - source_y if bottom_up else source_y
        row_start = pixel_offset + stored_row * row_stride
        row_values: list[float] = []
        for output_x in range(width):
            source_x = min(int(output_x * source_width / width), source_width - 1)
            pixel_start = row_start + source_x * bytes_per_pixel
            blue = data[pixel_start]
            green = data[pixel_start + 1]
            red = data[pixel_start + 2]
            row_values.append(_grayscale(red, green, blue))
        rows.append(tuple(row_values))
    return GrayscaleFrame(width=width, height=height, pixels=tuple(rows))


def resize_grayscale_frame(
    frame: GrayscaleFrame,
    *,
    width: int,
    height: int,
) -> GrayscaleFrame:
    """Nearest-neighbor resize for small training spikes."""

    if width <= 0 or height <= 0:
        raise ImageDatasetError("resize dimensions must be positive")
    if frame.width == width and frame.height == height:
        return frame

    rows: list[tuple[float, ...]] = []
    for output_y in range(height):
        source_y = min(int(output_y * frame.height / height), frame.height - 1)
        row_values: list[float] = []
        for output_x in range(width):
            source_x = min(int(output_x * frame.width / width), frame.width - 1)
            row_values.append(frame.pixels[source_y][source_x])
        rows.append(tuple(row_values))
    return GrayscaleFrame(width=width, height=height, pixels=tuple(rows))


def _load_resized_frame(
    path: Path,
    *,
    width: int,
    height: int,
    frame_cache: dict[Path, GrayscaleFrame] | None,
) -> GrayscaleFrame:
    if frame_cache is None:
        return read_bmp_grayscale_resized(path, width=width, height=height)
    if path not in frame_cache:
        frame_cache[path] = read_bmp_grayscale_resized(
            path,
            width=width,
            height=height,
        )
    return frame_cache[path]


def _normalize_frame_stack(
    frames: tuple[GrayscaleFrame, ...],
    *,
    required_size: int | None,
    pad_short: bool,
) -> tuple[GrayscaleFrame, ...]:
    if required_size is None:
        return frames
    if len(frames) == required_size:
        return frames
    if len(frames) > required_size:
        raise ImageDatasetError(
            f"frame stack has {len(frames)} frames; expected {required_size}"
        )
    if not pad_short:
        raise ImageDatasetError(
            f"frame stack has {len(frames)} frames; expected {required_size}"
        )
    if not frames:
        raise ImageDatasetError("frame stack is empty")
    padding = (frames[0],) * (required_size - len(frames))
    return padding + frames


def _frame_base_from_summary(dataset_dir: Path) -> Path:
    summary_path = dataset_dir / "summary.json"
    try:
        with summary_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ImageDatasetError(
            "frame_base_dir is required when summary.json is missing"
        ) from exc
    if not isinstance(data, Mapping):
        raise ImageDatasetError("summary.json must contain an object")
    inputs = data.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ImageDatasetError("summary.json must contain inputs")
    manifest_path = inputs.get("manifest_jsonl")
    if not isinstance(manifest_path, str):
        raise ImageDatasetError("summary inputs must contain manifest_jsonl")
    return Path(manifest_path).parent


def _labels_for_sample(
    sample: ImitationSample,
    *,
    label_mode: str,
) -> tuple[float, ...]:
    if label_mode == "events":
        return (
            1.0 if sample.press_event else 0.0,
            1.0 if sample.release_event else 0.0,
        )
    if label_mode == "target_input_down":
        return (1.0 if sample.target_input_down else 0.0,)
    raise ImageDatasetError("unsupported label_mode")


def _grayscale(red: int, green: int, blue: int) -> float:
    return (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
