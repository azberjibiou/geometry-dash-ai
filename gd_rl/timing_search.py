"""Small timing-window search utilities for Phase C practice learning."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Mapping, Sequence

from gd_human_model import Event
from gd_human_model.events import EventKind, Player

FAILED_SCORE = -1e18


@dataclass(frozen=True, slots=True)
class TimingEventWindow:
    """Search distribution and hard bounds for one intended input event."""

    name: str
    kind: EventKind
    mean_tick: float
    std_tick: float
    min_tick: int
    max_tick: int
    player: Player = "p1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("event window name must be non-empty")
        if self.kind not in ("press", "release"):
            raise ValueError("event window kind must be 'press' or 'release'")
        if self.player not in ("p1", "p2"):
            raise ValueError("event window player must be 'p1' or 'p2'")
        if self.min_tick < 0:
            raise ValueError("event window min_tick must be non-negative")
        if self.max_tick < self.min_tick:
            raise ValueError("event window max_tick must be >= min_tick")
        if not self.min_tick <= self.mean_tick <= self.max_tick:
            raise ValueError("event window mean_tick must be within bounds")
        if self.std_tick <= 0.0:
            raise ValueError("event window std_tick must be positive")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TimingEventWindow":
        return cls(
            name=str(_required(data, "name")),
            kind=_event_kind(_required(data, "kind")),
            mean_tick=float(_required(data, "mean_tick")),
            std_tick=float(_required(data, "std_tick")),
            min_tick=int(_required(data, "min_tick")),
            max_tick=int(_required(data, "max_tick")),
            player=_player(data.get("player", "p1")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_distribution(self, *, mean_tick: float, std_tick: float) -> "TimingEventWindow":
        return TimingEventWindow(
            name=self.name,
            kind=self.kind,
            mean_tick=_clamp_float(mean_tick, self.min_tick, self.max_tick),
            std_tick=std_tick,
            min_tick=self.min_tick,
            max_tick=self.max_tick,
            player=self.player,
        )


@dataclass(frozen=True, slots=True)
class TimingSearchConfig:
    """Configuration for a compact CEM/random timing search."""

    level_id: str
    event_windows: list[TimingEventWindow]
    generations: int = 3
    population_size: int = 8
    elite_fraction: float = 0.25
    attempts_per_candidate: int = 1
    seed: int = 0
    min_event_spacing: int = 1
    min_std_tick: float = 2.0
    max_std_tick: float = 80.0
    update_smoothing: float = 0.25
    continue_on_candidate_error: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.level_id:
            raise ValueError("level_id must be non-empty")
        if not self.event_windows:
            raise ValueError("at least one event window is required")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if self.attempts_per_candidate <= 0:
            raise ValueError("attempts_per_candidate must be positive")
        if self.min_event_spacing < 0:
            raise ValueError("min_event_spacing must be non-negative")
        if self.min_std_tick <= 0.0:
            raise ValueError("min_std_tick must be positive")
        if self.max_std_tick < self.min_std_tick:
            raise ValueError("max_std_tick must be >= min_std_tick")
        if not 0.0 <= self.update_smoothing <= 1.0:
            raise ValueError("update_smoothing must be between 0 and 1")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")

    @property
    def elite_count(self) -> int:
        return max(1, math.ceil(self.population_size * self.elite_fraction))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_windows"] = [window.to_dict() for window in self.event_windows]
        return data


@dataclass(frozen=True, slots=True)
class TimingCandidate:
    """One sampled intended event schedule."""

    candidate_id: str
    generation_index: int
    population_index: int
    events: list[Event]
    event_names: list[str]

    @property
    def ticks(self) -> list[int]:
        return [event.tick for event in self.events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation_index": self.generation_index,
            "population_index": self.population_index,
            "event_names": list(self.event_names),
            "events": [
                {"tick": event.tick, "kind": event.kind, "player": event.player}
                for event in self.events
            ],
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Evaluation result for one sampled timing candidate."""

    candidate: TimingCandidate
    score: float
    summary: dict[str, Any] | None = None
    output_dir: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score,
            "summary": self.summary,
            "output_dir": self.output_dir,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """All candidates and updated windows for one search generation."""

    generation_index: int
    starting_windows: list[TimingEventWindow]
    evaluations: list[CandidateEvaluation]
    elite_candidate_ids: list[str]
    updated_windows: list[TimingEventWindow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_index": self.generation_index,
            "starting_windows": [
                window.to_dict() for window in self.starting_windows
            ],
            "evaluations": [
                evaluation.to_dict() for evaluation in self.evaluations
            ],
            "elite_candidate_ids": list(self.elite_candidate_ids),
            "updated_windows": [
                window.to_dict() for window in self.updated_windows
            ],
        }


@dataclass(frozen=True, slots=True)
class TimingSearchResult:
    """Complete Phase C timing search result."""

    config: TimingSearchConfig
    generations: list[GenerationResult]
    best_evaluation: CandidateEvaluation | None
    final_windows: list[TimingEventWindow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "generations": [generation.to_dict() for generation in self.generations],
            "best_evaluation": (
                self.best_evaluation.to_dict()
                if self.best_evaluation is not None
                else None
            ),
            "final_windows": [window.to_dict() for window in self.final_windows],
        }


CandidateEvaluator = Callable[[TimingCandidate], CandidateEvaluation]


def load_timing_windows_json(path: str | Path) -> tuple[list[TimingEventWindow], dict[str, Any]]:
    """Load timing event windows from a JSON file."""

    window_path = Path(path)
    with window_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{window_path} must contain a JSON object")
    events_data = data.get("events")
    if not isinstance(events_data, list):
        raise ValueError(f"{window_path} must contain an events list")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{window_path} metadata must be an object")
    return (
        [
            TimingEventWindow.from_mapping(_mapping(item, window_path))
            for item in events_data
        ],
        dict(metadata),
    )


def sample_candidate(
    *,
    windows: Sequence[TimingEventWindow],
    rng: random.Random,
    generation_index: int,
    population_index: int,
    min_event_spacing: int = 1,
) -> TimingCandidate:
    """Sample a legal candidate from the current event windows."""

    if min_event_spacing < 0:
        raise ValueError("min_event_spacing must be non-negative")

    for _ in range(100):
        ticks = [
            _clamp_int(
                round(rng.gauss(window.mean_tick, window.std_tick)),
                window.min_tick,
                window.max_tick,
            )
            for window in windows
        ]
        if _is_valid_spacing(ticks, min_event_spacing):
            return _candidate_from_ticks(
                windows=windows,
                ticks=ticks,
                generation_index=generation_index,
                population_index=population_index,
            )

    ticks = _repair_ticks(windows, min_event_spacing)
    return _candidate_from_ticks(
        windows=windows,
        ticks=ticks,
        generation_index=generation_index,
        population_index=population_index,
    )


def update_windows_from_elites(
    *,
    windows: Sequence[TimingEventWindow],
    elites: Sequence[CandidateEvaluation],
    min_std_tick: float,
    max_std_tick: float,
    update_smoothing: float,
) -> list[TimingEventWindow]:
    """Move timing distributions toward the elite candidates."""

    if not elites:
        return list(windows)
    if min_std_tick <= 0.0:
        raise ValueError("min_std_tick must be positive")
    if max_std_tick < min_std_tick:
        raise ValueError("max_std_tick must be >= min_std_tick")
    if not 0.0 <= update_smoothing <= 1.0:
        raise ValueError("update_smoothing must be between 0 and 1")

    updated: list[TimingEventWindow] = []
    for event_index, window in enumerate(windows):
        elite_ticks = [
            evaluation.candidate.events[event_index].tick
            for evaluation in elites
            if evaluation.succeeded
        ]
        if not elite_ticks:
            updated.append(window)
            continue

        elite_mean = mean(elite_ticks)
        elite_std = pstdev(elite_ticks) if len(elite_ticks) > 1 else window.std_tick
        elite_std = _clamp_float(elite_std, min_std_tick, max_std_tick)
        smoothed_mean = (
            update_smoothing * window.mean_tick
            + (1.0 - update_smoothing) * elite_mean
        )
        smoothed_std = (
            update_smoothing * window.std_tick
            + (1.0 - update_smoothing) * elite_std
        )
        updated.append(
            window.with_distribution(
                mean_tick=smoothed_mean,
                std_tick=_clamp_float(smoothed_std, min_std_tick, max_std_tick),
            )
        )
    return updated


def run_timing_search(
    *,
    config: TimingSearchConfig,
    evaluator: CandidateEvaluator,
) -> TimingSearchResult:
    """Run a compact CEM-style timing search."""

    rng = random.Random(config.seed)
    windows = list(config.event_windows)
    generation_results: list[GenerationResult] = []
    best_evaluation: CandidateEvaluation | None = None

    for generation_index in range(config.generations):
        starting_windows = list(windows)
        evaluations: list[CandidateEvaluation] = []
        for population_index in range(config.population_size):
            candidate = sample_candidate(
                windows=windows,
                rng=rng,
                generation_index=generation_index,
                population_index=population_index,
                min_event_spacing=config.min_event_spacing,
            )
            try:
                evaluation = evaluator(candidate)
            except Exception as exc:
                if not config.continue_on_candidate_error:
                    raise
                evaluation = CandidateEvaluation(
                    candidate=candidate,
                    score=FAILED_SCORE,
                    error=f"{type(exc).__name__}: {exc}",
                )
            evaluations.append(evaluation)
            if evaluation.succeeded and (
                best_evaluation is None or evaluation.score > best_evaluation.score
            ):
                best_evaluation = evaluation

        ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
        elites = [evaluation for evaluation in ranked if evaluation.succeeded][
            : config.elite_count
        ]
        windows = update_windows_from_elites(
            windows=windows,
            elites=elites,
            min_std_tick=config.min_std_tick,
            max_std_tick=config.max_std_tick,
            update_smoothing=config.update_smoothing,
        )
        generation_results.append(
            GenerationResult(
                generation_index=generation_index,
                starting_windows=starting_windows,
                evaluations=evaluations,
                elite_candidate_ids=[
                    evaluation.candidate.candidate_id for evaluation in elites
                ],
                updated_windows=windows,
            )
        )

    return TimingSearchResult(
        config=config,
        generations=generation_results,
        best_evaluation=best_evaluation,
        final_windows=windows,
    )


def _candidate_from_ticks(
    *,
    windows: Sequence[TimingEventWindow],
    ticks: Sequence[int],
    generation_index: int,
    population_index: int,
) -> TimingCandidate:
    return TimingCandidate(
        candidate_id=f"g{generation_index:03d}_c{population_index:03d}",
        generation_index=generation_index,
        population_index=population_index,
        events=[
            Event(tick=tick, kind=window.kind, player=window.player)
            for tick, window in zip(ticks, windows, strict=True)
        ],
        event_names=[window.name for window in windows],
    )


def _repair_ticks(
    windows: Sequence[TimingEventWindow],
    min_event_spacing: int,
) -> list[int]:
    ticks: list[int] = []
    previous_tick: int | None = None
    for window in windows:
        tick = _clamp_int(round(window.mean_tick), window.min_tick, window.max_tick)
        if previous_tick is not None:
            tick = max(tick, previous_tick + min_event_spacing)
        if tick > window.max_tick:
            raise ValueError(
                f"cannot satisfy event spacing before window {window.name!r}"
            )
        ticks.append(tick)
        previous_tick = tick
    return ticks


def _is_valid_spacing(ticks: Sequence[int], min_event_spacing: int) -> bool:
    return all(
        tick >= previous + min_event_spacing
        for previous, tick in zip(ticks, ticks[1:])
    )


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"missing required field {key!r}")
    return data[key]


def _mapping(value: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} event window entries must be objects")
    return value


def _event_kind(value: Any) -> EventKind:
    if value not in ("press", "release"):
        raise ValueError("event kind must be 'press' or 'release'")
    return value


def _player(value: Any) -> Player:
    if value not in ("p1", "p2"):
        raise ValueError("event player must be 'p1' or 'p2'")
    return value


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return min(max(value, lower), upper)


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
