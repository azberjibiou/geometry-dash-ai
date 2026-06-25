"""Decode imitation-policy probabilities into valid macro events."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from gd_human_model.events import Event, EventKind, Player, sort_events


class EventDecoderError(ValueError):
    """Raised when prediction rows or decoder settings are invalid."""


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Configuration for converting per-tick probabilities into events."""

    press_threshold: float = 0.5
    release_threshold: float = 0.5
    non_max_radius_frames: int = 4
    min_event_spacing_frames: int = 2
    initial_input_down: bool = False
    player: Player = "p1"

    def __post_init__(self) -> None:
        _validate_probability(self.press_threshold, "press_threshold")
        _validate_probability(self.release_threshold, "release_threshold")
        if (
            isinstance(self.non_max_radius_frames, bool)
            or not isinstance(self.non_max_radius_frames, int)
        ):
            raise EventDecoderError("non_max_radius_frames must be an int")
        if self.non_max_radius_frames < 0:
            raise EventDecoderError("non_max_radius_frames must be non-negative")
        if (
            isinstance(self.min_event_spacing_frames, bool)
            or not isinstance(self.min_event_spacing_frames, int)
        ):
            raise EventDecoderError("min_event_spacing_frames must be an int")
        if self.min_event_spacing_frames < 0:
            raise EventDecoderError("min_event_spacing_frames must be non-negative")
        if not isinstance(self.initial_input_down, bool):
            raise EventDecoderError("initial_input_down must be a bool")
        if self.player not in ("p1", "p2"):
            raise EventDecoderError("player must be 'p1' or 'p2'")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly config data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionPoint:
    """One policy output row aligned to a game tick."""

    tick: int
    press_probability: float
    release_probability: float
    press_label: bool = False
    release_label: bool = False
    split: str | None = None
    position: int | None = None
    index: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PredictionPoint":
        """Parse a baseline prediction JSON row."""

        tick = _required_int(data, "tick")
        probabilities = data.get("probabilities")
        if isinstance(probabilities, Mapping):
            press_probability = _required_probability(probabilities, "press_event")
            release_probability = _required_probability(
                probabilities,
                "release_event",
            )
        else:
            press_probability = _required_probability(data, "press_probability")
            release_probability = _required_probability(
                data,
                "release_probability",
            )

        labels = data.get("labels")
        if isinstance(labels, Mapping):
            press_label = _optional_bool(labels, "press_event", default=False)
            release_label = _optional_bool(labels, "release_event", default=False)
        else:
            press_label = _optional_bool(data, "press_label", default=False)
            release_label = _optional_bool(data, "release_label", default=False)

        return cls(
            tick=tick,
            press_probability=press_probability,
            release_probability=release_probability,
            press_label=press_label,
            release_label=release_label,
            split=_optional_str(data, "split"),
            position=_optional_int(data, "position"),
            index=_optional_int(data, "index"),
        )


@dataclass(frozen=True, slots=True)
class EventCandidate:
    """A thresholded probability peak before legality filtering."""

    tick: int
    kind: EventKind
    probability: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly candidate data."""

        return {
            "tick": self.tick,
            "kind": self.kind,
            "probability": self.probability,
        }


def load_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load baseline prediction rows from JSONL."""

    return list(iter_prediction_jsonl(path))


def iter_prediction_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield raw prediction rows from JSONL with helpful validation errors."""

    predictions_path = Path(path)
    with predictions_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EventDecoderError(
                    f"{predictions_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise EventDecoderError(
                    f"{predictions_path}:{line_number}: row must be an object"
                )
            try:
                PredictionPoint.from_mapping(data)
            except EventDecoderError as exc:
                raise EventDecoderError(
                    f"{predictions_path}:{line_number}: {exc}"
                ) from exc
            yield data


def decode_predictions(
    predictions: Sequence[Mapping[str, Any] | PredictionPoint],
    *,
    config: DecoderConfig | None = None,
) -> list[Event]:
    """Decode prediction rows into state-valid press/release events."""

    effective_config = config or DecoderConfig()
    candidates = select_event_candidates(predictions, config=effective_config)
    input_down = effective_config.initial_input_down
    last_event_tick: int | None = None
    events: list[Event] = []

    for candidate in sorted(candidates, key=_candidate_sort_key):
        if (
            last_event_tick is not None
            and candidate.tick - last_event_tick
            < effective_config.min_event_spacing_frames
        ):
            continue
        if candidate.kind == "press":
            if input_down:
                continue
            events.append(Event(candidate.tick, "press", effective_config.player))
            input_down = True
            last_event_tick = candidate.tick
        else:
            if not input_down:
                continue
            events.append(Event(candidate.tick, "release", effective_config.player))
            input_down = False
            last_event_tick = candidate.tick

    return sort_events(events)


def select_event_candidates(
    predictions: Sequence[Mapping[str, Any] | PredictionPoint],
    *,
    config: DecoderConfig | None = None,
) -> list[EventCandidate]:
    """Return non-max-suppressed press/release probability peaks."""

    effective_config = config or DecoderConfig()
    points = _coerce_prediction_points(predictions)
    return sorted(
        [
            *_select_kind_candidates(points, "press", config=effective_config),
            *_select_kind_candidates(points, "release", config=effective_config),
        ],
        key=_candidate_sort_key,
    )


def event_level_metrics(
    predictions: Sequence[Mapping[str, Any] | PredictionPoint],
    decoded_events: Sequence[Event],
    *,
    match_tolerance_frames: int = 2,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute event-level timing metrics for decoded macro events."""

    if match_tolerance_frames < 0:
        raise EventDecoderError("match_tolerance_frames must be non-negative")
    if top_k < 0:
        raise EventDecoderError("top_k must be non-negative")

    points = _coerce_prediction_points(predictions)
    events = sort_events(decoded_events)
    return {
        "match_tolerance_frames": match_tolerance_frames,
        "press": _kind_metrics(
            points,
            events,
            kind="press",
            match_tolerance_frames=match_tolerance_frames,
            top_k=top_k,
        ),
        "release": _kind_metrics(
            points,
            events,
            kind="release",
            match_tolerance_frames=match_tolerance_frames,
            top_k=top_k,
        ),
    }


def _select_kind_candidates(
    points: Sequence[PredictionPoint],
    kind: EventKind,
    *,
    config: DecoderConfig,
) -> list[EventCandidate]:
    threshold = (
        config.press_threshold if kind == "press" else config.release_threshold
    )
    candidates = [
        EventCandidate(
            tick=point.tick,
            kind=kind,
            probability=_probability_for_kind(point, kind),
        )
        for point in points
        if _probability_for_kind(point, kind) >= threshold
    ]
    if config.non_max_radius_frames == 0:
        return sorted(candidates, key=_candidate_sort_key)

    kept: list[EventCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.probability, item.tick)):
        if any(
            abs(candidate.tick - selected.tick) <= config.non_max_radius_frames
            for selected in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=_candidate_sort_key)


def _kind_metrics(
    points: Sequence[PredictionPoint],
    events: Sequence[Event],
    *,
    kind: EventKind,
    match_tolerance_frames: int,
    top_k: int,
) -> dict[str, Any]:
    labeled_ticks = [point.tick for point in points if _label_for_kind(point, kind)]
    predicted_ticks = [event.tick for event in events if event.kind == kind]
    default_match = _match_ticks(
        labeled_ticks,
        predicted_ticks,
        tolerance=match_tolerance_frames,
    )
    return {
        "labeled_event_ticks": labeled_ticks,
        "predicted_event_ticks": predicted_ticks,
        "top_probability_ticks": _top_probability_ticks(points, kind, top_k=top_k),
        "nearest_label_distance_frames": _nearest_label_distances(
            labeled_ticks,
            predicted_ticks,
        ),
        "within_1_frame": _match_ticks(labeled_ticks, predicted_ticks, tolerance=1),
        "within_2_frames": _match_ticks(labeled_ticks, predicted_ticks, tolerance=2),
        "within_5_frames": _match_ticks(labeled_ticks, predicted_ticks, tolerance=5),
        "missed_labeled_events": default_match["missed_labeled_events"],
        "extra_predicted_events": default_match["extra_predicted_events"],
        "precision": default_match["precision"],
        "recall": default_match["recall"],
        "f1": default_match["f1"],
    }


def _match_ticks(
    labeled_ticks: Sequence[int],
    predicted_ticks: Sequence[int],
    *,
    tolerance: int,
) -> dict[str, Any]:
    candidate_pairs: list[tuple[int, int, int]] = []
    for label_index, label_tick in enumerate(labeled_ticks):
        for predicted_index, predicted_tick in enumerate(predicted_ticks):
            distance = abs(label_tick - predicted_tick)
            if distance <= tolerance:
                candidate_pairs.append((distance, label_index, predicted_index))

    used_labels: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[dict[str, int]] = []
    for distance, label_index, predicted_index in sorted(candidate_pairs):
        if label_index in used_labels or predicted_index in used_predictions:
            continue
        used_labels.add(label_index)
        used_predictions.add(predicted_index)
        matches.append(
            {
                "label_tick": labeled_ticks[label_index],
                "predicted_tick": predicted_ticks[predicted_index],
                "distance_frames": distance,
            }
        )

    missed = [
        tick for index, tick in enumerate(labeled_ticks) if index not in used_labels
    ]
    extra = [
        tick
        for index, tick in enumerate(predicted_ticks)
        if index not in used_predictions
    ]
    precision = len(matches) / len(predicted_ticks) if predicted_ticks else None
    recall = len(matches) / len(labeled_ticks) if labeled_ticks else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return {
        "tolerance_frames": tolerance,
        "matched_count": len(matches),
        "label_count": len(labeled_ticks),
        "predicted_count": len(predicted_ticks),
        "matches": matches,
        "missed_labeled_events": missed,
        "extra_predicted_events": extra,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _nearest_label_distances(
    labeled_ticks: Sequence[int],
    predicted_ticks: Sequence[int],
) -> list[dict[str, int | None]]:
    distances: list[dict[str, int | None]] = []
    for predicted_tick in predicted_ticks:
        if not labeled_ticks:
            distances.append(
                {
                    "predicted_tick": predicted_tick,
                    "nearest_label_tick": None,
                    "distance_frames": None,
                }
            )
            continue
        nearest_label_tick = min(
            labeled_ticks,
            key=lambda label_tick: (abs(label_tick - predicted_tick), label_tick),
        )
        distances.append(
            {
                "predicted_tick": predicted_tick,
                "nearest_label_tick": nearest_label_tick,
                "distance_frames": abs(nearest_label_tick - predicted_tick),
            }
        )
    return distances


def _top_probability_ticks(
    points: Sequence[PredictionPoint],
    kind: EventKind,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        points,
        key=lambda point: (-_probability_for_kind(point, kind), point.tick),
    )
    return [
        {
            "tick": point.tick,
            "probability": _probability_for_kind(point, kind),
            "label": point.press_label if kind == "press" else point.release_label,
        }
        for point in ranked[:top_k]
    ]


def _coerce_prediction_points(
    predictions: Sequence[Mapping[str, Any] | PredictionPoint],
) -> list[PredictionPoint]:
    points = [
        item if isinstance(item, PredictionPoint) else PredictionPoint.from_mapping(item)
        for item in predictions
    ]
    sorted_points = sorted(points, key=lambda point: point.tick)
    previous_tick: int | None = None
    for point in sorted_points:
        if previous_tick == point.tick:
            raise EventDecoderError(f"duplicate prediction tick {point.tick}")
        previous_tick = point.tick
    return sorted_points


def _probability_for_kind(point: PredictionPoint, kind: EventKind) -> float:
    if kind == "press":
        return point.press_probability
    return point.release_probability


def _label_for_kind(point: PredictionPoint, kind: EventKind) -> bool:
    if kind == "press":
        return point.press_label
    return point.release_label


def _candidate_sort_key(candidate: EventCandidate) -> tuple[int, int]:
    kind_order = 0 if candidate.kind == "press" else 1
    return (candidate.tick, kind_order)


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventDecoderError(f"{key} must be an int")
    if value < 0:
        raise EventDecoderError(f"{key} must be non-negative")
    return value


def _optional_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventDecoderError(f"{key} must be an int")
    return value


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise EventDecoderError(f"{key} must be a string")
    return value


def _optional_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise EventDecoderError(f"{key} must be a bool")
    return value


def _required_probability(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventDecoderError(f"{key} must be a number")
    probability = float(value)
    _validate_probability(probability, key)
    return probability


def _validate_probability(value: object, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventDecoderError(f"{key} must be a number")
    if value < 0.0 or value > 1.0:
        raise EventDecoderError(f"{key} must be between 0 and 1")
