"""Deterministic replay metrics for repeated identical macros."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence

from gd_human_model.events import Event, EventKind, Player, sort_events
from gd_trace.compare_trace import first_death_tick
from gd_trace.trace_schema import TraceRow


@dataclass(frozen=True, slots=True)
class ObservedInputTransition:
    """One observed p1 input state transition in a trace."""

    tick: int
    kind: EventKind


@dataclass(frozen=True, slots=True)
class InputLatencySummary:
    """Observed latency for one intended macro event across all trials."""

    event_index: int
    intended_tick: int
    kind: EventKind
    player: Player
    matched_count: int
    missing_count: int
    mean_frames: float | None
    std_frames: float | None
    min_frames: int | None
    max_frames: int | None
    observed_ticks_by_trial: list[int | None]
    latency_frames_by_trial: list[int | None]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable latency data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class MacroApplicationSummary:
    """Mod-side application timing for one queued macro event."""

    event_index: int
    intended_tick: int
    kind: EventKind
    player: Player
    matched_count: int
    missing_count: int
    mean_frames: float | None
    std_frames: float | None
    min_frames: int | None
    max_frames: int | None
    applied_ticks_by_trial: list[int | None]
    latency_frames_by_trial: list[int | None]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable application timing data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplayCheckSummary:
    """Summary metrics for deterministic replay trials."""

    trace_count: int
    row_counts: list[int]
    compared_tick_count: int
    final_percents: list[float]
    final_percent_std: float
    death_ticks: list[int | None]
    death_tick_std: float | None
    success_percent: float
    success_rate: float
    survival_rate: float
    x_position_max_diff: float
    y_position_max_diff: float
    input_state_mismatch_ticks: int
    first_movement_ticks: list[int | None]
    zero_movement_step_counts: list[int]
    double_movement_step_counts: list[int]
    input_latency_mean_frames: float | None
    input_latency_std_frames: float | None
    input_latency_by_event: list[InputLatencySummary]
    macro_application_by_event: list[MacroApplicationSummary]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable replay summary data."""

        return asdict(self)


def summarize_replay_check(
    traces: Sequence[Sequence[TraceRow]],
    macro_events: Sequence[Event],
    *,
    success_percent: float = 100.0,
    diagnostics_by_trial: Sequence[Sequence[Mapping[str, Any] | Any]] | None = None,
) -> ReplayCheckSummary:
    """Summarize repeated trace outcomes for one identical macro."""

    trace_lists = [list(trace) for trace in traces]
    trace_count = len(trace_lists)
    row_counts = [len(trace) for trace in trace_lists]
    final_percents = [trace[-1].percent if trace else 0.0 for trace in trace_lists]
    death_ticks = [first_death_tick(trace) for trace in trace_lists]
    death_tick_values = [tick for tick in death_ticks if tick is not None]
    x_max_diff, y_max_diff, input_mismatch_ticks, compared_tick_count = (
        _aligned_tick_metrics(trace_lists)
    )
    input_latency = summarize_input_latency(trace_lists, macro_events)
    all_latencies = [
        latency
        for event_summary in input_latency
        for latency in event_summary.latency_frames_by_trial
        if latency is not None
    ]
    movement_diagnostics = [_movement_diagnostics(trace) for trace in trace_lists]
    macro_application = (
        summarize_macro_applications(
            macro_events,
            diagnostics_by_trial,
        )
        if diagnostics_by_trial is not None
        else []
    )

    return ReplayCheckSummary(
        trace_count=trace_count,
        row_counts=row_counts,
        compared_tick_count=compared_tick_count,
        final_percents=final_percents,
        final_percent_std=_population_std(final_percents),
        death_ticks=death_ticks,
        death_tick_std=(
            _population_std(death_tick_values) if death_tick_values else None
        ),
        success_percent=success_percent,
        success_rate=_fraction(
            final_percent >= success_percent for final_percent in final_percents
        ),
        survival_rate=_fraction(tick is None for tick in death_ticks),
        x_position_max_diff=x_max_diff,
        y_position_max_diff=y_max_diff,
        input_state_mismatch_ticks=input_mismatch_ticks,
        first_movement_ticks=[
            diagnostic.first_movement_tick for diagnostic in movement_diagnostics
        ],
        zero_movement_step_counts=[
            diagnostic.zero_movement_steps for diagnostic in movement_diagnostics
        ],
        double_movement_step_counts=[
            diagnostic.double_movement_steps for diagnostic in movement_diagnostics
        ],
        input_latency_mean_frames=_mean_or_none(all_latencies),
        input_latency_std_frames=_std_or_none(all_latencies),
        input_latency_by_event=input_latency,
        macro_application_by_event=macro_application,
    )


def summarize_input_latency(
    traces: Sequence[Sequence[TraceRow]],
    macro_events: Sequence[Event],
) -> list[InputLatencySummary]:
    """Compare intended p1 macro events with observed input_down transitions."""

    sorted_events = sort_events(macro_events)
    p1_events = [
        (event_index, event)
        for event_index, event in enumerate(sorted_events)
        if event.player == "p1"
    ]
    per_trace_matches = [
        _match_input_transitions(trace, sorted_events) for trace in traces
    ]
    summaries: list[InputLatencySummary] = []

    for event_index, event in p1_events:
        observed_ticks = [
            matches.get(event_index).tick if matches.get(event_index) is not None else None
            for matches in per_trace_matches
        ]
        latencies = [
            observed_tick - event.tick if observed_tick is not None else None
            for observed_tick in observed_ticks
        ]
        matched_latencies = [latency for latency in latencies if latency is not None]
        summaries.append(
            InputLatencySummary(
                event_index=event_index,
                intended_tick=event.tick,
                kind=event.kind,
                player=event.player,
                matched_count=len(matched_latencies),
                missing_count=len(latencies) - len(matched_latencies),
                mean_frames=_mean_or_none(matched_latencies),
                std_frames=_std_or_none(matched_latencies),
                min_frames=min(matched_latencies) if matched_latencies else None,
                max_frames=max(matched_latencies) if matched_latencies else None,
                observed_ticks_by_trial=observed_ticks,
                latency_frames_by_trial=latencies,
            )
        )

    return summaries


def detect_input_transitions(trace: Sequence[TraceRow]) -> list[ObservedInputTransition]:
    """Return p1 press/release transitions observed in trace input_down state."""

    if not trace:
        return []

    transitions: list[ObservedInputTransition] = []
    previous_down = trace[0].input_down
    if previous_down:
        transitions.append(ObservedInputTransition(trace[0].tick, "press"))

    for row in trace[1:]:
        if row.input_down == previous_down:
            continue
        transitions.append(
            ObservedInputTransition(
                tick=row.tick,
                kind="press" if row.input_down else "release",
            )
        )
        previous_down = row.input_down

    return transitions


def summarize_macro_applications(
    macro_events: Sequence[Event],
    diagnostics_by_trial: Sequence[Sequence[Mapping[str, Any] | Any]],
) -> list[MacroApplicationSummary]:
    """Summarize mod-side application diagnostics for queued macro events."""

    sorted_events = sort_events(macro_events)
    matches_by_trial = [
        _macro_application_matches(diagnostics) for diagnostics in diagnostics_by_trial
    ]
    summaries: list[MacroApplicationSummary] = []

    for event_index, event in enumerate(sorted_events):
        applied_ticks = [
            matches.get(event_index) for matches in matches_by_trial
        ]
        latencies = [
            applied_tick - event.tick if applied_tick is not None else None
            for applied_tick in applied_ticks
        ]
        matched_latencies = [latency for latency in latencies if latency is not None]
        summaries.append(
            MacroApplicationSummary(
                event_index=event_index,
                intended_tick=event.tick,
                kind=event.kind,
                player=event.player,
                matched_count=len(matched_latencies),
                missing_count=len(latencies) - len(matched_latencies),
                mean_frames=_mean_or_none(matched_latencies),
                std_frames=_std_or_none(matched_latencies),
                min_frames=min(matched_latencies) if matched_latencies else None,
                max_frames=max(matched_latencies) if matched_latencies else None,
                applied_ticks_by_trial=applied_ticks,
                latency_frames_by_trial=latencies,
            )
        )

    return summaries


def _match_input_transitions(
    trace: Sequence[TraceRow],
    sorted_macro_events: Sequence[Event],
) -> dict[int, ObservedInputTransition]:
    transitions = detect_input_transitions(trace)
    matches: dict[int, ObservedInputTransition] = {}
    next_transition_index = 0

    for event_index, event in enumerate(sorted_macro_events):
        if event.player != "p1":
            continue

        for transition_index in range(next_transition_index, len(transitions)):
            transition = transitions[transition_index]
            if transition.kind == event.kind and transition.tick >= event.tick:
                matches[event_index] = transition
                next_transition_index = transition_index + 1
                break

    return matches


@dataclass(frozen=True, slots=True)
class _MovementDiagnostics:
    first_movement_tick: int | None
    zero_movement_steps: int
    double_movement_steps: int


def _movement_diagnostics(trace: Sequence[TraceRow]) -> _MovementDiagnostics:
    epsilon = 1e-6
    if len(trace) < 2:
        return _MovementDiagnostics(
            first_movement_tick=None,
            zero_movement_steps=0,
            double_movement_steps=0,
        )

    first = trace[0]
    first_movement_tick: int | None = None
    zero_movement_steps = 0
    positive_x_steps: list[float] = []

    for previous, current in zip(trace, trace[1:]):
        x_delta = abs(current.x - previous.x)
        y_delta = abs(current.y - previous.y)
        if first_movement_tick is None and (
            abs(current.x - first.x) > epsilon or abs(current.y - first.y) > epsilon
        ):
            first_movement_tick = current.tick
        if x_delta <= epsilon and y_delta <= epsilon:
            zero_movement_steps += 1
        if x_delta > epsilon:
            positive_x_steps.append(x_delta)

    if not positive_x_steps:
        return _MovementDiagnostics(
            first_movement_tick=first_movement_tick,
            zero_movement_steps=zero_movement_steps,
            double_movement_steps=0,
        )

    baseline_step = float(median(positive_x_steps))
    double_threshold = baseline_step * 1.5
    double_movement_steps = sum(
        1
        for previous, current in zip(trace, trace[1:])
        if abs(current.x - previous.x) > double_threshold
    )
    return _MovementDiagnostics(
        first_movement_tick=first_movement_tick,
        zero_movement_steps=zero_movement_steps,
        double_movement_steps=double_movement_steps,
    )


def _macro_application_matches(
    diagnostics: Sequence[Mapping[str, Any] | Any],
) -> dict[int, int]:
    matches: dict[int, int] = {}
    for diagnostic in diagnostics:
        diagnostic_data = _diagnostic_to_mapping(diagnostic)
        if diagnostic_data.get("kind") != "macro_event_applied":
            continue

        data = diagnostic_data.get("data", {})
        if not isinstance(data, Mapping):
            continue
        event_index = data.get("event_index")
        applied_tick = data.get("applied_tick", diagnostic_data.get("tick"))
        if (
            isinstance(event_index, bool)
            or not isinstance(event_index, int)
            or isinstance(applied_tick, bool)
            or not isinstance(applied_tick, int)
        ):
            continue
        matches.setdefault(event_index, applied_tick)
    return matches


def _diagnostic_to_mapping(diagnostic: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(diagnostic, Mapping):
        return diagnostic
    to_dict = getattr(diagnostic, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return data
    return {}


def _aligned_tick_metrics(
    traces: Sequence[Sequence[TraceRow]],
) -> tuple[float, float, int, int]:
    if not traces:
        return 0.0, 0.0, 0, 0

    rows_by_tick = [{row.tick: row for row in trace} for trace in traces]
    shared_ticks = set(rows_by_tick[0])
    for rows in rows_by_tick[1:]:
        shared_ticks.intersection_update(rows)

    x_max_diff = 0.0
    y_max_diff = 0.0
    input_mismatch_ticks = 0

    for tick in shared_ticks:
        rows = [mapping[tick] for mapping in rows_by_tick]
        xs = [row.x for row in rows]
        ys = [row.y for row in rows]
        x_max_diff = max(x_max_diff, max(xs) - min(xs))
        y_max_diff = max(y_max_diff, max(ys) - min(ys))
        if len({row.input_down for row in rows}) > 1:
            input_mismatch_ticks += 1

    return x_max_diff, y_max_diff, input_mismatch_ticks, len(shared_ticks)


def _population_std(values: Sequence[float | int]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _std_or_none(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return _population_std(values)


def _mean_or_none(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _fraction(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)
