"""Decode future input state predictions into replayable macro events."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from gd_human_model.events import Event, Player, sort_events


class StateDecoderError(ValueError):
    """Raised when state prediction rows or settings are invalid."""


@dataclass(frozen=True, slots=True)
class StateDecoderConfig:
    """Configuration for converting target-input probabilities to events."""

    threshold: float = 0.5
    initial_input_down: bool = False
    min_state_run_frames: int = 1
    event_tick_field: str = "label_tick"
    player: Player = "p1"

    def __post_init__(self) -> None:
        _validate_probability(self.threshold, "threshold")
        if not isinstance(self.initial_input_down, bool):
            raise StateDecoderError("initial_input_down must be a bool")
        if (
            isinstance(self.min_state_run_frames, bool)
            or not isinstance(self.min_state_run_frames, int)
        ):
            raise StateDecoderError("min_state_run_frames must be an int")
        if self.min_state_run_frames <= 0:
            raise StateDecoderError("min_state_run_frames must be positive")
        if self.event_tick_field not in ("tick", "label_tick"):
            raise StateDecoderError("event_tick_field must be 'tick' or 'label_tick'")
        if self.player not in ("p1", "p2"):
            raise StateDecoderError("player must be 'p1' or 'p2'")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly config data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class StatePredictionPoint:
    """One predicted future input state aligned to a sample tick."""

    tick: int
    event_tick: int
    target_input_down_probability: float
    target_input_down_label: bool = False
    split: str | None = None
    position: int | None = None
    index: int | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        event_tick_field: str = "label_tick",
    ) -> "StatePredictionPoint":
        """Parse one state prediction row."""

        tick = _required_int(data, "tick")
        if event_tick_field == "label_tick":
            event_tick = _optional_int(data, "label_tick", default=tick)
        elif event_tick_field == "tick":
            event_tick = tick
        else:
            raise StateDecoderError("event_tick_field must be 'tick' or 'label_tick'")

        probabilities = data.get("probabilities")
        if isinstance(probabilities, Mapping):
            probability = _required_probability(
                probabilities,
                "target_input_down",
            )
        else:
            probability = _required_probability(
                data,
                "target_input_down_probability",
            )

        labels = data.get("labels")
        if isinstance(labels, Mapping):
            label = _optional_bool(labels, "target_input_down", default=False)
        else:
            label = _optional_bool(
                data,
                "target_input_down_label",
                default=False,
            )

        return cls(
            tick=tick,
            event_tick=event_tick,
            target_input_down_probability=probability,
            target_input_down_label=label,
            split=_optional_str(data, "split"),
            position=_optional_int(data, "position"),
            index=_optional_int(data, "index"),
        )

    def predicted_input_down(self, *, threshold: float) -> bool:
        """Return thresholded predicted state."""

        _validate_probability(threshold, "threshold")
        return self.target_input_down_probability >= threshold


def load_state_prediction_jsonl(
    path: str | Path,
    *,
    event_tick_field: str = "label_tick",
) -> list[dict[str, Any]]:
    """Load state prediction rows from JSONL."""

    return list(iter_state_prediction_jsonl(path, event_tick_field=event_tick_field))


def iter_state_prediction_jsonl(
    path: str | Path,
    *,
    event_tick_field: str = "label_tick",
) -> Iterator[dict[str, Any]]:
    """Yield raw state prediction rows with validation."""

    predictions_path = Path(path)
    with predictions_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise StateDecoderError(
                    f"{predictions_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise StateDecoderError(
                    f"{predictions_path}:{line_number}: row must be an object"
                )
            try:
                StatePredictionPoint.from_mapping(
                    data,
                    event_tick_field=event_tick_field,
                )
            except StateDecoderError as exc:
                raise StateDecoderError(
                    f"{predictions_path}:{line_number}: {exc}"
                ) from exc
            yield data


def decode_state_predictions(
    predictions: Sequence[Mapping[str, Any] | StatePredictionPoint],
    *,
    config: StateDecoderConfig | None = None,
) -> list[Event]:
    """Decode thresholded input state changes into press/release events."""

    effective_config = config or StateDecoderConfig()
    points = _coerce_state_prediction_points(
        predictions,
        event_tick_field=effective_config.event_tick_field,
    )
    input_down = effective_config.initial_input_down
    candidate_state: bool | None = None
    candidate_start: StatePredictionPoint | None = None
    candidate_length = 0
    events: list[Event] = []

    for point in points:
        predicted_down = point.predicted_input_down(
            threshold=effective_config.threshold,
        )
        if predicted_down == input_down:
            candidate_state = None
            candidate_start = None
            candidate_length = 0
            continue

        if predicted_down == candidate_state:
            candidate_length += 1
        else:
            candidate_state = predicted_down
            candidate_start = point
            candidate_length = 1

        if candidate_length < effective_config.min_state_run_frames:
            continue

        if candidate_start is None:
            raise StateDecoderError("internal state decoder error")
        events.append(
            Event(
                candidate_start.event_tick,
                "press" if predicted_down else "release",
                effective_config.player,
            )
        )
        input_down = predicted_down
        candidate_state = None
        candidate_start = None
        candidate_length = 0

    return sort_events(events)


def state_level_metrics(
    predictions: Sequence[Mapping[str, Any] | StatePredictionPoint],
    decoded_events: Sequence[Event],
    *,
    config: StateDecoderConfig | None = None,
) -> dict[str, Any]:
    """Return lightweight state and transition metrics for a decoded macro."""

    effective_config = config or StateDecoderConfig()
    points = _coerce_state_prediction_points(
        predictions,
        event_tick_field=effective_config.event_tick_field,
    )
    actual = [point.target_input_down_label for point in points]
    predicted = [
        point.predicted_input_down(threshold=effective_config.threshold)
        for point in points
    ]
    return {
        "state": _binary_metrics(actual=actual, predicted=predicted),
        "labeled_transition_ticks": _transition_ticks(
            points,
            states=actual,
            initial_input_down=effective_config.initial_input_down,
        ),
        "predicted_transition_ticks": [
            event.tick for event in sort_events(decoded_events)
        ],
    }


def _coerce_state_prediction_points(
    predictions: Sequence[Mapping[str, Any] | StatePredictionPoint],
    *,
    event_tick_field: str,
) -> list[StatePredictionPoint]:
    points = [
        item
        if isinstance(item, StatePredictionPoint)
        else StatePredictionPoint.from_mapping(
            item,
            event_tick_field=event_tick_field,
        )
        for item in predictions
    ]
    sorted_points = sorted(points, key=lambda point: point.tick)
    previous_tick: int | None = None
    for point in sorted_points:
        if previous_tick == point.tick:
            raise StateDecoderError(f"duplicate prediction tick {point.tick}")
        previous_tick = point.tick
    return sorted_points


def _transition_ticks(
    points: Sequence[StatePredictionPoint],
    *,
    states: Sequence[bool],
    initial_input_down: bool,
) -> list[int]:
    ticks: list[int] = []
    input_down = initial_input_down
    for point, state in zip(points, states):
        if state == input_down:
            continue
        ticks.append(point.event_tick)
        input_down = state
    return ticks


def _binary_metrics(
    *,
    actual: Sequence[bool],
    predicted: Sequence[bool],
) -> dict[str, Any]:
    if len(actual) != len(predicted):
        raise StateDecoderError("actual and predicted lengths differ")
    total = len(actual)
    true_positive = sum(1 for a, p in zip(actual, predicted) if a and p)
    false_positive = sum(1 for a, p in zip(actual, predicted) if not a and p)
    true_negative = sum(1 for a, p in zip(actual, predicted) if not a and not p)
    false_negative = sum(1 for a, p in zip(actual, predicted) if a and not p)
    return {
        "count": total,
        "actual_positive_count": true_positive + false_negative,
        "predicted_positive_count": true_positive + false_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / total if total else None,
    }


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateDecoderError(f"{key} must be an int")
    if value < 0:
        raise StateDecoderError(f"{key} must be non-negative")
    return value


def _optional_int(
    data: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateDecoderError(f"{key} must be an int")
    if value < 0:
        raise StateDecoderError(f"{key} must be non-negative")
    return value


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StateDecoderError(f"{key} must be a string")
    return value


def _optional_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise StateDecoderError(f"{key} must be a bool")
    return value


def _required_probability(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateDecoderError(f"{key} must be a number")
    probability = float(value)
    _validate_probability(probability, key)
    return probability


def _validate_probability(value: object, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateDecoderError(f"{key} must be a number")
    if value < 0.0 or value > 1.0:
        raise StateDecoderError(f"{key} must be between 0 and 1")
