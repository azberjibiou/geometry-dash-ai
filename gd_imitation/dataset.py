"""Prepare aligned frame/trace/event samples for imitation learning."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from gd_capture.manifest import FrameCaptureRecord, load_manifest_jsonl
from gd_human_model.events import Event, EventKind, Player, sort_events
from gd_trace.load_trace import load_macro_json, load_trace_jsonl
from gd_trace.macro_schema import event_from_mapping, event_to_dict
from gd_trace.trace_schema import TraceRow, validate_trace_sequence


class DatasetPrepError(ValueError):
    """Raised when captured artifacts cannot be aligned into samples."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for turning captured artifacts into supervised samples."""

    frame_stack_size: int = 4
    label_shift_frames: int = 0
    event_label_radius_frames: int = 0
    player: Player = "p1"
    include_partial_windows: bool = False
    drop_incomplete_label_horizon: bool = True
    drop_unmatched_frame_ticks: bool = False

    def __post_init__(self) -> None:
        if self.frame_stack_size <= 0:
            raise DatasetPrepError("frame_stack_size must be positive")
        if self.label_shift_frames < 0:
            raise DatasetPrepError("label_shift_frames must be non-negative")
        if self.event_label_radius_frames < 0:
            raise DatasetPrepError("event_label_radius_frames must be non-negative")
        if self.player not in ("p1", "p2"):
            raise DatasetPrepError("player must be 'p1' or 'p2'")


@dataclass(frozen=True, slots=True)
class ImitationSample:
    """One frame-window sample with current state and shifted event labels."""

    index: int
    tick: int
    label_tick: int
    frame_ticks: tuple[int, ...]
    frame_paths: tuple[str, ...]
    progress: float
    input_down: bool
    x: float
    y: float
    x_vel: float
    y_vel: float
    rotation: float
    mode: str
    gravity: str
    dead: bool
    death_reason: str | None
    fps: int
    cbf: bool
    physics_bypass: bool
    press_event: bool
    release_event: bool
    events: tuple[Event, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ImitationSample":
        """Build one prepared sample from parsed JSON data."""

        return cls(
            index=_as_int(data, "index"),
            tick=_as_int(data, "tick"),
            label_tick=_as_int(data, "label_tick"),
            frame_ticks=tuple(_as_int_sequence(data, "frame_ticks")),
            frame_paths=tuple(_as_str_sequence(data, "frame_paths")),
            progress=_as_float(data, "progress"),
            input_down=_as_bool(data, "input_down"),
            x=_as_float(data, "x"),
            y=_as_float(data, "y"),
            x_vel=_as_float(data, "x_vel"),
            y_vel=_as_float(data, "y_vel"),
            rotation=_as_float(data, "rotation"),
            mode=_as_str(data, "mode"),
            gravity=_as_str(data, "gravity"),
            dead=_as_bool(data, "dead"),
            death_reason=_as_optional_str(data, "death_reason"),
            fps=_as_int(data, "fps"),
            cbf=_as_bool(data, "cbf"),
            physics_bypass=_as_bool(data, "physics_bypass"),
            press_event=_as_bool(data, "press_event"),
            release_event=_as_bool(data, "release_event"),
            events=tuple(_events_from_data(data.get("events", []))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable sample metadata."""

        data = asdict(self)
        data["frame_ticks"] = list(self.frame_ticks)
        data["frame_paths"] = list(self.frame_paths)
        data["events"] = [event_to_dict(event) for event in self.events]
        return data


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A deterministic train/validation split over prepared samples."""

    train: tuple[ImitationSample, ...]
    validation: tuple[ImitationSample, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_fraction: float
    shuffle: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly split metadata without embedding samples."""

        return {
            "train_count": len(self.train),
            "validation_count": len(self.validation),
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "validation_fraction": self.validation_fraction,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }


def load_imitation_samples(
    manifest_path: str | Path,
    trace_path: str | Path,
    macro_path: str | Path,
    *,
    config: DatasetConfig | None = None,
) -> list[ImitationSample]:
    """Load artifacts from disk and prepare aligned imitation samples."""

    manifest_records = load_manifest_jsonl(manifest_path)
    trace_rows = load_trace_jsonl(trace_path)
    macro = load_macro_json(macro_path)
    return build_imitation_samples(
        manifest_records,
        trace_rows,
        macro.events,
        config=config,
    )


def iter_samples_jsonl(path: str | Path) -> Iterator[ImitationSample]:
    """Yield prepared imitation samples from JSONL."""

    samples_path = Path(path)
    with samples_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetPrepError(
                    f"{samples_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(data, Mapping):
                raise DatasetPrepError(f"{samples_path}:{line_number}: row must be an object")
            try:
                yield ImitationSample.from_mapping(data)
            except (DatasetPrepError, ValueError, TypeError) as exc:
                raise DatasetPrepError(f"{samples_path}:{line_number}: {exc}") from exc


def load_samples_jsonl(path: str | Path) -> list[ImitationSample]:
    """Load prepared imitation samples from JSONL."""

    samples = list(iter_samples_jsonl(path))
    _validate_sample_indices(samples)
    return samples


def build_imitation_samples(
    frame_records: Sequence[FrameCaptureRecord],
    trace_rows: Sequence[TraceRow],
    events: Sequence[Event],
    *,
    config: DatasetConfig | None = None,
) -> list[ImitationSample]:
    """Align frames, trace state, and shifted macro events into samples."""

    effective_config = config or DatasetConfig()
    _validate_frame_ticks_non_decreasing(frame_records)
    trace_list = list(trace_rows)
    try:
        validate_trace_sequence(trace_list)
    except ValueError as exc:
        raise DatasetPrepError(str(exc)) from exc

    trace_by_tick = _index_trace_by_tick(trace_list)
    events_by_tick = _events_by_tick(events, player=effective_config.player)
    aligned_frame_records = _aligned_frame_records(
        frame_records,
        trace_by_tick,
        drop_unmatched=effective_config.drop_unmatched_frame_ticks,
    )
    samples: list[ImitationSample] = []

    for frame_index, record in enumerate(aligned_frame_records):
        window_records = _frame_window(
            aligned_frame_records,
            end_index=frame_index,
            size=effective_config.frame_stack_size,
            include_partial=effective_config.include_partial_windows,
        )
        if window_records is None:
            continue

        label_tick = record.tick + effective_config.label_shift_frames
        if label_tick not in trace_by_tick:
            if effective_config.drop_incomplete_label_horizon:
                continue
            raise DatasetPrepError(f"label tick {label_tick} is missing from trace")

        row = trace_by_tick[record.tick]
        label_events = _events_for_label_tick(
            events_by_tick,
            label_tick=label_tick,
            radius=effective_config.event_label_radius_frames,
        )
        samples.append(
            ImitationSample(
                index=len(samples),
                tick=record.tick,
                label_tick=label_tick,
                frame_ticks=tuple(item.tick for item in window_records),
                frame_paths=tuple(item.frame_path for item in window_records),
                progress=row.percent,
                input_down=row.input_down,
                x=row.x,
                y=row.y,
                x_vel=row.x_vel,
                y_vel=row.y_vel,
                rotation=row.rotation,
                mode=row.mode,
                gravity=row.gravity,
                dead=row.dead,
                death_reason=row.death_reason,
                fps=row.fps,
                cbf=row.cbf,
                physics_bypass=row.physics_bypass,
                press_event=_has_event(label_events, "press"),
                release_event=_has_event(label_events, "release"),
                events=label_events,
            )
        )

    return samples


def split_imitation_samples(
    samples: Sequence[ImitationSample],
    *,
    validation_fraction: float = 0.2,
    shuffle: bool = False,
    seed: int = 0,
) -> DatasetSplit:
    """Return a deterministic train/validation split."""

    if not 0.0 <= validation_fraction <= 1.0:
        raise DatasetPrepError("validation_fraction must be between 0 and 1")

    sample_list = list(samples)
    indices = list(range(len(sample_list)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    validation_count = int(len(indices) * validation_fraction)
    if validation_fraction > 0.0 and indices and validation_count == 0:
        validation_count = 1
    if validation_fraction < 1.0 and len(indices) > 1:
        validation_count = min(validation_count, len(indices) - 1)

    if validation_count == 0:
        validation_indices: tuple[int, ...] = ()
    elif shuffle:
        validation_indices = tuple(sorted(indices[:validation_count]))
    else:
        validation_indices = tuple(indices[-validation_count:])

    validation_index_set = set(validation_indices)
    train_indices = tuple(
        index for index in range(len(sample_list)) if index not in validation_index_set
    )

    return DatasetSplit(
        train=tuple(sample_list[index] for index in train_indices),
        validation=tuple(sample_list[index] for index in validation_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
        validation_fraction=validation_fraction,
        shuffle=shuffle,
        seed=seed,
    )


def _validate_frame_ticks_non_decreasing(
    records: Sequence[FrameCaptureRecord],
) -> None:
    previous_tick: int | None = None
    for index, record in enumerate(records):
        if previous_tick is not None and record.tick < previous_tick:
            raise DatasetPrepError(
                f"frame ticks must be non-decreasing: record {index} "
                f"has tick {record.tick} after {previous_tick}"
            )
        previous_tick = record.tick


def _index_trace_by_tick(rows: Sequence[TraceRow]) -> dict[int, TraceRow]:
    index: dict[int, TraceRow] = {}
    for row in rows:
        if row.tick in index:
            raise DatasetPrepError(f"duplicate trace tick {row.tick}")
        index[row.tick] = row
    return index


def _aligned_frame_records(
    records: Sequence[FrameCaptureRecord],
    trace_by_tick: Mapping[int, TraceRow],
    *,
    drop_unmatched: bool,
) -> list[FrameCaptureRecord]:
    aligned_records: list[FrameCaptureRecord] = []
    for record in records:
        if record.tick in trace_by_tick:
            aligned_records.append(record)
            continue
        if not drop_unmatched:
            raise DatasetPrepError(f"frame tick {record.tick} is missing from trace")
    return aligned_records


def _events_by_tick(
    events: Sequence[Event],
    *,
    player: Player,
) -> dict[int, tuple[Event, ...]]:
    grouped: dict[int, list[Event]] = {}
    for event in sort_events(events):
        if event.player != player:
            continue
        grouped.setdefault(event.tick, []).append(event)
    return {tick: tuple(values) for tick, values in grouped.items()}


def _events_for_label_tick(
    events_by_tick: Mapping[int, tuple[Event, ...]],
    *,
    label_tick: int,
    radius: int,
) -> tuple[Event, ...]:
    if radius == 0:
        return tuple(events_by_tick.get(label_tick, ()))

    events: list[Event] = []
    for tick in range(label_tick - radius, label_tick + radius + 1):
        events.extend(events_by_tick.get(tick, ()))
    return tuple(sort_events(events))


def _frame_window(
    records: Sequence[FrameCaptureRecord],
    *,
    end_index: int,
    size: int,
    include_partial: bool,
) -> tuple[FrameCaptureRecord, ...] | None:
    start_index = end_index - size + 1
    if start_index < 0:
        if not include_partial:
            return None
        start_index = 0
    return tuple(records[start_index : end_index + 1])


def _has_event(events: Sequence[Event], kind: EventKind) -> bool:
    return any(event.kind == kind for event in events)


def _validate_sample_indices(samples: Sequence[ImitationSample]) -> None:
    for expected_index, sample in enumerate(samples):
        if sample.index != expected_index:
            raise DatasetPrepError(
                f"sample indices must be contiguous: expected {expected_index}, "
                f"got {sample.index}"
            )


def _events_from_data(data: Any) -> list[Event]:
    if not isinstance(data, list):
        raise DatasetPrepError("events must be a list")
    try:
        return [event_from_mapping(item) for item in data]
    except ValueError as exc:
        raise DatasetPrepError(str(exc)) from exc


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise DatasetPrepError(f"missing required field: {key}")
    return data[key]


def _as_int(data: Mapping[str, Any], key: str) -> int:
    value = _required(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetPrepError(f"{key} must be an int")
    return value


def _as_float(data: Mapping[str, Any], key: str) -> float:
    value = _required(data, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetPrepError(f"{key} must be a number")
    return float(value)


def _as_bool(data: Mapping[str, Any], key: str) -> bool:
    value = _required(data, key)
    if not isinstance(value, bool):
        raise DatasetPrepError(f"{key} must be a bool")
    return value


def _as_str(data: Mapping[str, Any], key: str) -> str:
    value = _required(data, key)
    if not isinstance(value, str):
        raise DatasetPrepError(f"{key} must be a string")
    return value


def _as_optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = _required(data, key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetPrepError(f"{key} must be a string or null")
    return value


def _as_int_sequence(data: Mapping[str, Any], key: str) -> list[int]:
    value = _required(data, key)
    if not isinstance(value, list):
        raise DatasetPrepError(f"{key} must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise DatasetPrepError(f"{key} must contain only ints")
    return value


def _as_str_sequence(data: Mapping[str, Any], key: str) -> list[str]:
    value = _required(data, key)
    if not isinstance(value, list):
        raise DatasetPrepError(f"{key} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise DatasetPrepError(f"{key} must contain only strings")
    return value
