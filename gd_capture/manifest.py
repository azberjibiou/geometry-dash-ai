"""Manifest schema for frames aligned to Geode bridge observations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping

from gd_capture.screen_capture import CaptureRegion
from gd_env import BridgeObservation


class ManifestError(ValueError):
    """Raised when a frame manifest row is malformed."""


@dataclass(frozen=True, slots=True)
class FrameCaptureRecord:
    """One captured frame aligned with one bridge observation."""

    frame_path: str
    tick: int
    time_ms: float
    percent: float
    x: float
    y: float
    x_vel: float
    y_vel: float
    rotation: float
    mode: str
    gravity: str
    input_down: bool
    dead: bool
    completed: bool
    death_reason: str | None
    capture_width: int
    capture_height: int
    capture_region: dict[str, int]
    window_title: str | None = None

    def __post_init__(self) -> None:
        if not self.frame_path:
            raise ManifestError("frame_path must be non-empty")
        if self.tick < 0:
            raise ManifestError("tick must be non-negative")
        if self.time_ms < 0.0:
            raise ManifestError("time_ms must be non-negative")
        if not 0.0 <= self.percent <= 100.0:
            raise ManifestError("percent must be between 0 and 100")
        if not self.mode:
            raise ManifestError("mode must be non-empty")
        if not self.gravity:
            raise ManifestError("gravity must be non-empty")
        if self.capture_width <= 0:
            raise ManifestError("capture_width must be positive")
        if self.capture_height <= 0:
            raise ManifestError("capture_height must be positive")
        region = CaptureRegion.from_mapping(self.capture_region)
        if region.width != self.capture_width:
            raise ManifestError("capture_width must match capture_region.width")
        if region.height != self.capture_height:
            raise ManifestError("capture_height must match capture_region.height")

    @classmethod
    def from_observation(
        cls,
        observation: BridgeObservation,
        *,
        frame_path: str,
        capture_width: int,
        capture_height: int,
        capture_region: CaptureRegion,
        fps: int,
        window_title: str | None = None,
    ) -> "FrameCaptureRecord":
        """Build a manifest record from one bridge observation."""

        return cls(
            frame_path=frame_path,
            tick=observation.tick,
            time_ms=observation.tick * 1000.0 / fps,
            percent=observation.percent,
            x=observation.x,
            y=observation.y,
            x_vel=observation.x_vel,
            y_vel=observation.y_vel,
            rotation=observation.rotation,
            mode=observation.mode,
            gravity=observation.gravity,
            input_down=observation.input_down,
            dead=observation.dead,
            completed=observation.completed,
            death_reason=observation.death_reason,
            capture_width=capture_width,
            capture_height=capture_height,
            capture_region=capture_region.to_dict(),
            window_title=window_title,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "FrameCaptureRecord":
        return cls(
            frame_path=_as_str(data, "frame_path"),
            tick=_as_int(data, "tick"),
            time_ms=_as_float(data, "time_ms"),
            percent=_as_float(data, "percent"),
            x=_as_float(data, "x"),
            y=_as_float(data, "y"),
            x_vel=_as_float(data, "x_vel", default=0.0),
            y_vel=_as_float(data, "y_vel"),
            rotation=_as_float(data, "rotation", default=0.0),
            mode=_as_str(data, "mode"),
            gravity=_as_str(data, "gravity"),
            input_down=_as_bool(data, "input_down"),
            dead=_as_bool(data, "dead"),
            completed=_as_bool(data, "completed", default=False),
            death_reason=_as_optional_str(data, "death_reason", default=None),
            capture_width=_as_int(data, "capture_width"),
            capture_height=_as_int(data, "capture_height"),
            capture_region=_as_region_dict(data),
            window_title=_as_optional_str(data, "window_title", default=None),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_manifest_jsonl(
    records: list[FrameCaptureRecord],
    path: str | Path,
) -> None:
    """Save frame manifest rows as JSONL."""

    manifest_path = Path(path)
    if manifest_path.parent != Path("."):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)
            )
            file.write("\n")


def iter_manifest_jsonl(path: str | Path) -> Iterator[FrameCaptureRecord]:
    """Yield validated frame manifest records from JSONL."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise ManifestError(f"{manifest_path}:{line_number}: row must be an object")
            try:
                yield FrameCaptureRecord.from_mapping(data)
            except ManifestError as exc:
                raise ManifestError(f"{manifest_path}:{line_number}: {exc}") from exc


def load_manifest_jsonl(path: str | Path) -> list[FrameCaptureRecord]:
    """Load and validate a complete frame manifest."""

    records = list(iter_manifest_jsonl(path))
    _validate_ticks_non_decreasing(records)
    return records


def _validate_ticks_non_decreasing(records: list[FrameCaptureRecord]) -> None:
    previous_tick: int | None = None
    for index, record in enumerate(records):
        if previous_tick is not None and record.tick < previous_tick:
            raise ManifestError(
                f"manifest ticks must be non-decreasing: row {index + 1} "
                f"has tick {record.tick} after {previous_tick}"
            )
        previous_tick = record.tick


def _as_int(
    data: Mapping[str, object],
    key: str,
    default: int | None = None,
) -> int:
    value = data.get(key, default)
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{key} must be an int")
    return value


def _as_float(
    data: Mapping[str, object],
    key: str,
    default: float | None = None,
) -> float:
    value = data.get(key, default)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{key} must be a number")
    return float(value)


def _as_bool(
    data: Mapping[str, object],
    key: str,
    default: bool | None = None,
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ManifestError(f"{key} must be a bool")
    return value


def _as_str(
    data: Mapping[str, object],
    key: str,
    default: str | None = None,
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ManifestError(f"{key} must be a string")
    return value


def _as_optional_str(
    data: Mapping[str, object],
    key: str,
    default: str | None = None,
) -> str | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(f"{key} must be a string or null")
    return value


def _as_region_dict(data: Mapping[str, object]) -> dict[str, int]:
    value = data.get("capture_region")
    if not isinstance(value, Mapping):
        raise ManifestError("capture_region must be an object")
    region = CaptureRegion.from_mapping(value)
    return region.to_dict()
