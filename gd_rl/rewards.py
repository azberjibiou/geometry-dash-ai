"""Reward computation for repeated Geometry Dash practice attempts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from typing import Sequence

from gd_trace.trace_schema import TraceRow


@dataclass(frozen=True, slots=True)
class TraceOutcome:
    """Terminal and progress facts derived deterministically from a trace."""

    row_count: int
    first_tick: int | None
    last_tick: int | None
    playtime_seconds: float
    start_percent: float
    final_percent: float
    best_percent: float
    death_tick: int | None
    death_percent: float | None
    cleared: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for the first practice-loop reward function."""

    success_percent: float = 100.0
    progress_scale: float = 1.0
    best_progress_bonus_scale: float = 0.5
    section_size_percent: float = 10.0
    section_survival_bonus: float = 0.25
    clear_bonus: float = 100.0
    death_penalty: float = 10.0
    excessive_input_free_events: int = 0
    excessive_input_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_percent <= 100.0:
            raise ValueError("success_percent must be between 0 and 100")
        if self.section_size_percent <= 0.0:
            raise ValueError("section_size_percent must be positive")
        if self.excessive_input_free_events < 0:
            raise ValueError("excessive_input_free_events must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Named reward terms plus their total."""

    terms: dict[str, float]
    total: float

    def to_dict(self) -> dict[str, float | dict[str, float]]:
        return {"terms": dict(self.terms), "total": self.total}


def summarize_trace_outcome(
    rows: Sequence[TraceRow],
    *,
    success_percent: float = 100.0,
) -> TraceOutcome:
    """Derive final progress, death, clear, and playtime from trace rows."""

    if not 0.0 <= success_percent <= 100.0:
        raise ValueError("success_percent must be between 0 and 100")

    first_row = rows[0] if rows else None
    last_row = rows[-1] if rows else None
    death_row = next((row for row in rows if row.dead), None)
    best_percent = max((row.percent for row in rows), default=0.0)
    final_percent = last_row.percent if last_row is not None else 0.0

    return TraceOutcome(
        row_count=len(rows),
        first_tick=first_row.tick if first_row is not None else None,
        last_tick=last_row.tick if last_row is not None else None,
        playtime_seconds=(last_row.time_ms / 1000.0 if last_row is not None else 0.0),
        start_percent=first_row.percent if first_row is not None else 0.0,
        final_percent=final_percent,
        best_percent=best_percent,
        death_tick=death_row.tick if death_row is not None else None,
        death_percent=death_row.percent if death_row is not None else None,
        cleared=best_percent >= success_percent,
    )


def compute_reward(
    rows: Sequence[TraceRow],
    *,
    config: RewardConfig | None = None,
    previous_best_percent: float = 0.0,
    intended_event_count: int = 0,
    executed_event_count: int = 0,
) -> RewardBreakdown:
    """Compute deterministic reward terms from a trace and attempt metadata."""

    reward_config = config or RewardConfig()
    outcome = summarize_trace_outcome(
        rows,
        success_percent=reward_config.success_percent,
    )
    if intended_event_count < 0 or executed_event_count < 0:
        raise ValueError("event counts must be non-negative")

    progress_delta = max(0.0, outcome.best_percent - outcome.start_percent)
    best_progress_delta = max(0.0, outcome.best_percent - previous_best_percent)
    sections_reached = floor(outcome.best_percent / reward_config.section_size_percent)
    event_count_for_penalty = max(intended_event_count, executed_event_count)
    excessive_events = max(
        0,
        event_count_for_penalty - reward_config.excessive_input_free_events,
    )

    death_penalty = 0.0
    if outcome.death_tick is not None and not outcome.cleared:
        death_percent = outcome.death_percent if outcome.death_percent is not None else 0.0
        death_penalty = -reward_config.death_penalty * max(
            0.0,
            1.0 - death_percent / max(reward_config.success_percent, 1e-9),
        )

    terms = {
        "progress_delta": progress_delta * reward_config.progress_scale,
        "best_progress_bonus": (
            best_progress_delta * reward_config.best_progress_bonus_scale
        ),
        "section_survival_bonus": (
            sections_reached * reward_config.section_survival_bonus
        ),
        "clear_bonus": reward_config.clear_bonus if outcome.cleared else 0.0,
        "death_penalty": death_penalty,
        "illegal_or_excessive_input_penalty": (
            -excessive_events * reward_config.excessive_input_penalty
        ),
    }
    return RewardBreakdown(terms=terms, total=sum(terms.values()))
