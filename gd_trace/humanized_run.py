"""Outcome summaries for humanized macro replay experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

from gd_human_model.macro_humanizer import HumanizedMacroEvent
from gd_trace.compare_trace import first_death_tick
from gd_trace.trace_schema import TraceRow


@dataclass(frozen=True, slots=True)
class NumericDistribution:
    """Small JSON-friendly summary for a numeric sample."""

    count: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return JSON-serializable distribution data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class HumanizedAttemptSummary:
    """Per-attempt outcome and humanization summary."""

    attempt: int
    rows: int
    first_tick: int | None
    last_tick: int | None
    final_percent: float
    death_tick: int | None
    death_percent: float | None
    cleared: bool
    intended_event_count: int
    actual_event_count: int
    missed_event_count: int
    timing_delta_frames: list[int]
    delay_adjusted_delta_frames: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable attempt data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class HumanizedRunSummary:
    """Aggregate metrics for repeated humanized macro attempts."""

    attempt_count: int
    clears: int
    deaths: int
    clear_rate: float
    survival_rate: float
    attempts_to_first_clear: int | None
    playtime_to_first_clear_seconds: float | None
    average_progress: float
    best_percent: float
    final_percent_by_attempt: list[float]
    death_ticks: list[int | None]
    death_tick_histogram: dict[str, int]
    death_percent_histogram: dict[str, int]
    intended_event_count: int
    actual_event_count_by_attempt: list[int]
    missed_event_count_by_attempt: list[int]
    total_missed_events: int
    timing_delta_frames: NumericDistribution
    delay_adjusted_delta_frames: NumericDistribution
    attempts: list[HumanizedAttemptSummary]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable run data."""

        data = asdict(self)
        data["timing_delta_frames"] = self.timing_delta_frames.to_dict()
        data["delay_adjusted_delta_frames"] = (
            self.delay_adjusted_delta_frames.to_dict()
        )
        data["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return data


def summarize_humanized_attempts(
    traces: Sequence[Sequence[TraceRow]],
    event_results_by_attempt: Sequence[Sequence[HumanizedMacroEvent]],
    *,
    success_percent: float = 100.0,
) -> HumanizedRunSummary:
    """Summarize repeated attempts with separately humanized macros."""

    if len(traces) != len(event_results_by_attempt):
        raise ValueError("traces and event_results_by_attempt must have the same length")

    attempts = [
        _summarize_attempt(
            attempt_index=index,
            trace=trace,
            event_results=event_results,
            success_percent=success_percent,
        )
        for index, (trace, event_results) in enumerate(
            zip(traces, event_results_by_attempt),
            start=1,
        )
    ]
    final_percents = [attempt.final_percent for attempt in attempts]
    death_ticks = [attempt.death_tick for attempt in attempts]
    clear_flags = [attempt.cleared for attempt in attempts]
    all_timing_deltas = [
        delta for attempt in attempts for delta in attempt.timing_delta_frames
    ]
    all_delay_adjusted_deltas = [
        delta
        for attempt in attempts
        for delta in attempt.delay_adjusted_delta_frames
    ]
    first_clear_index = next(
        (index for index, cleared in enumerate(clear_flags) if cleared),
        None,
    )

    return HumanizedRunSummary(
        attempt_count=len(attempts),
        clears=sum(1 for cleared in clear_flags if cleared),
        deaths=sum(1 for tick in death_ticks if tick is not None),
        clear_rate=_fraction(clear_flags),
        survival_rate=_fraction(tick is None for tick in death_ticks),
        attempts_to_first_clear=(
            first_clear_index + 1 if first_clear_index is not None else None
        ),
        playtime_to_first_clear_seconds=(
            _playtime_seconds(traces[: first_clear_index + 1])
            if first_clear_index is not None
            else None
        ),
        average_progress=float(mean(final_percents)) if final_percents else 0.0,
        best_percent=max(final_percents) if final_percents else 0.0,
        final_percent_by_attempt=final_percents,
        death_ticks=death_ticks,
        death_tick_histogram=_histogram_int(
            tick for tick in death_ticks if tick is not None
        ),
        death_percent_histogram=_death_percent_histogram(
            attempt.death_percent for attempt in attempts if attempt.death_percent is not None
        ),
        intended_event_count=(
            len(event_results_by_attempt[0]) if event_results_by_attempt else 0
        ),
        actual_event_count_by_attempt=[
            attempt.actual_event_count for attempt in attempts
        ],
        missed_event_count_by_attempt=[
            attempt.missed_event_count for attempt in attempts
        ],
        total_missed_events=sum(attempt.missed_event_count for attempt in attempts),
        timing_delta_frames=_distribution(all_timing_deltas),
        delay_adjusted_delta_frames=_distribution(all_delay_adjusted_deltas),
        attempts=attempts,
    )


def _summarize_attempt(
    *,
    attempt_index: int,
    trace: Sequence[TraceRow],
    event_results: Sequence[HumanizedMacroEvent],
    success_percent: float,
) -> HumanizedAttemptSummary:
    death_row = _first_death_row(trace)
    actual_results = [
        result for result in event_results if result.actual_event is not None
    ]
    timing_deltas = [
        result.actual_delta_frames
        for result in actual_results
        if result.actual_delta_frames is not None
    ]
    delay_adjusted_deltas = [
        result.delay_adjusted_delta_frames
        for result in actual_results
        if result.delay_adjusted_delta_frames is not None
    ]
    final_percent = trace[-1].percent if trace else 0.0

    return HumanizedAttemptSummary(
        attempt=attempt_index,
        rows=len(trace),
        first_tick=trace[0].tick if trace else None,
        last_tick=trace[-1].tick if trace else None,
        final_percent=final_percent,
        death_tick=first_death_tick(trace),
        death_percent=death_row.percent if death_row is not None else None,
        cleared=final_percent >= success_percent,
        intended_event_count=len(event_results),
        actual_event_count=len(actual_results),
        missed_event_count=len(event_results) - len(actual_results),
        timing_delta_frames=timing_deltas,
        delay_adjusted_delta_frames=delay_adjusted_deltas,
    )


def _first_death_row(trace: Sequence[TraceRow]) -> TraceRow | None:
    for row in trace:
        if row.dead:
            return row
    return None


def _playtime_seconds(traces: Sequence[Sequence[TraceRow]]) -> float:
    return sum((trace[-1].time_ms / 1000.0) for trace in traces if trace)


def _distribution(values: Sequence[float | int]) -> NumericDistribution:
    if not values:
        return NumericDistribution(count=0, mean=None, std=None, min=None, max=None)
    return NumericDistribution(
        count=len(values),
        mean=float(mean(values)),
        std=float(pstdev(values)) if len(values) > 1 else 0.0,
        min=float(min(values)),
        max=float(max(values)),
    )


def _histogram_int(values: Iterable[int]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _death_percent_histogram(values: Iterable[float]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _fraction(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for value in items if value) / len(items)
