"""Utilities for comparing two recorded traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from gd_trace.trace_schema import TraceRow


@dataclass(frozen=True, slots=True)
class TraceComparison:
    """Summary metrics for two traces."""

    row_count_a: int
    row_count_b: int
    compared_rows: int
    death_tick_a: int | None
    death_tick_b: int | None
    final_percent_a: float
    final_percent_b: float
    final_percent_diff: float
    x_position_max_diff: float
    y_position_max_diff: float
    input_mismatch_count: int
    tick_mismatch_count: int

    def to_dict(self) -> dict[str, int | float | None]:
        """Return JSON-serializable comparison data."""

        return asdict(self)


def compare_traces(trace_a: Sequence[TraceRow], trace_b: Sequence[TraceRow]) -> TraceComparison:
    """Compare two traces row-by-row over their shared prefix."""

    compared_rows = min(len(trace_a), len(trace_b))
    x_position_max_diff = 0.0
    y_position_max_diff = 0.0
    input_mismatch_count = 0
    tick_mismatch_count = 0

    for row_a, row_b in zip(trace_a, trace_b):
        x_position_max_diff = max(x_position_max_diff, abs(row_a.x - row_b.x))
        y_position_max_diff = max(y_position_max_diff, abs(row_a.y - row_b.y))
        if row_a.input_down != row_b.input_down:
            input_mismatch_count += 1
        if row_a.tick != row_b.tick:
            tick_mismatch_count += 1

    final_percent_a = trace_a[-1].percent if trace_a else 0.0
    final_percent_b = trace_b[-1].percent if trace_b else 0.0

    return TraceComparison(
        row_count_a=len(trace_a),
        row_count_b=len(trace_b),
        compared_rows=compared_rows,
        death_tick_a=first_death_tick(trace_a),
        death_tick_b=first_death_tick(trace_b),
        final_percent_a=final_percent_a,
        final_percent_b=final_percent_b,
        final_percent_diff=abs(final_percent_a - final_percent_b),
        x_position_max_diff=x_position_max_diff,
        y_position_max_diff=y_position_max_diff,
        input_mismatch_count=input_mismatch_count,
        tick_mismatch_count=tick_mismatch_count,
    )


def first_death_tick(trace: Sequence[TraceRow]) -> int | None:
    """Return the first tick where a trace row is dead."""

    for row in trace:
        if row.dead:
            return row.tick
    return None
