"""Result dataclasses for level-specific practice runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """Persistable summary for one practice attempt."""

    level_id: str
    attempt_index: int
    human_profile: dict[str, Any]
    seed: int
    trace_path: str
    intended_events_path: str
    executed_events_path: str
    humanization_path: str
    row_count: int
    playtime_seconds: float
    final_percent: float
    best_percent: float
    death_tick: int | None
    death_percent: float | None
    cleared: bool
    total_reward: float
    reward_terms: dict[str, float]
    intended_event_count: int
    executed_event_count: int
    dropped_event_count: int
    trace_input_events_path: str | None = None
    trace_input_event_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PracticeRunSummary:
    """Aggregate metrics across repeated practice attempts."""

    level_id: str
    attempt_count: int
    clears: int
    deaths: int
    clear_rate: float
    attempts_to_first_clear: int | None
    playtime_to_first_clear_seconds: float | None
    average_final_percent: float
    best_percent: float
    total_reward: float
    reward_curve: list[float]
    final_percent_by_attempt: list[float]
    best_percent_by_attempt: list[float]
    death_tick_histogram: dict[str, int]
    death_percent_histogram: dict[str, int]
    attempts: list[AttemptResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_attempts(
        cls,
        *,
        level_id: str,
        attempts: Sequence[AttemptResult],
        metadata: dict[str, Any] | None = None,
    ) -> "PracticeRunSummary":
        """Build an aggregate summary from completed attempts."""

        attempt_list = list(attempts)
        clears = sum(1 for attempt in attempt_list if attempt.cleared)
        deaths = sum(1 for attempt in attempt_list if attempt.death_tick is not None)
        first_clear_index = next(
            (
                index
                for index, attempt in enumerate(attempt_list)
                if attempt.cleared
            ),
            None,
        )
        final_percents = [attempt.final_percent for attempt in attempt_list]
        best_by_attempt = [attempt.best_percent for attempt in attempt_list]
        reward_curve = [attempt.total_reward for attempt in attempt_list]

        return cls(
            level_id=level_id,
            attempt_count=len(attempt_list),
            clears=clears,
            deaths=deaths,
            clear_rate=(clears / len(attempt_list) if attempt_list else 0.0),
            attempts_to_first_clear=(
                first_clear_index + 1 if first_clear_index is not None else None
            ),
            playtime_to_first_clear_seconds=(
                sum(
                    attempt.playtime_seconds
                    for attempt in attempt_list[: first_clear_index + 1]
                )
                if first_clear_index is not None
                else None
            ),
            average_final_percent=(
                float(mean(final_percents)) if final_percents else 0.0
            ),
            best_percent=max(best_by_attempt) if best_by_attempt else 0.0,
            total_reward=sum(reward_curve),
            reward_curve=reward_curve,
            final_percent_by_attempt=final_percents,
            best_percent_by_attempt=best_by_attempt,
            death_tick_histogram=_histogram(
                attempt.death_tick
                for attempt in attempt_list
                if attempt.death_tick is not None
            ),
            death_percent_histogram=_histogram(
                int(attempt.death_percent)
                for attempt in attempt_list
                if attempt.death_percent is not None
            ),
            attempts=attempt_list,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return data


def _histogram(values: Sequence[int]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram
