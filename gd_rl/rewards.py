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

    reward_style: str = "progress"
    success_percent: float = 100.0
    progress_scale: float = 1.0
    best_progress_bonus_scale: float = 0.5
    section_size_percent: float = 10.0
    section_survival_bonus: float = 0.25
    clear_bonus: float = 100.0
    death_penalty: float = 10.0
    excessive_input_free_events: int = 0
    excessive_input_penalty: float = 0.0
    default_reward: float = 0.01
    jump_punishment: float = 0.0
    checkpoint_reward: float = 0.0
    checkpoint_size_percent: float = 3.0

    def __post_init__(self) -> None:
        if self.reward_style not in {"progress", "picklegawd"}:
            raise ValueError("reward_style must be 'progress' or 'picklegawd'")
        if not 0.0 <= self.success_percent <= 100.0:
            raise ValueError("success_percent must be between 0 and 100")
        if self.section_size_percent <= 0.0:
            raise ValueError("section_size_percent must be positive")
        if self.checkpoint_size_percent <= 0.0:
            raise ValueError("checkpoint_size_percent must be positive")
        if self.excessive_input_free_events < 0:
            raise ValueError("excessive_input_free_events must be non-negative")

    def to_dict(self) -> dict[str, float | int | str]:
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

    if reward_config.reward_style == "picklegawd":
        return _compute_picklegawd_reward(
            rows,
            config=reward_config,
            intended_event_count=intended_event_count,
            executed_event_count=executed_event_count,
        )

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


def compute_picklegawd_step_reward_terms(
    *,
    current_percent: float,
    next_percent: float,
    previous_best_percent: float | None = None,
    input_down: bool,
    dead: bool,
    cleared: bool,
    config: RewardConfig | None = None,
    newly_charged_excessive_events: int = 0,
) -> dict[str, float]:
    """Compute the PickleGawd-style single-step reward terms.

    The source environment uses a small survival reward for every non-terminal
    tick, optional additive input cost, optional checkpoint rewards, death
    punishment on death, and then adds a level-completion bonus.
    """

    reward_config = config or RewardConfig(reward_style="picklegawd")
    if reward_config.reward_style != "picklegawd":
        raise ValueError("picklegawd step terms require reward_style='picklegawd'")

    terms = _empty_picklegawd_terms()
    best_percent_before_step = (
        current_percent if previous_best_percent is None else previous_best_percent
    )
    progress_delta = max(0.0, next_percent - current_percent)
    best_progress_delta = max(0.0, next_percent - best_percent_before_step)
    terms["progress_delta"] = progress_delta * reward_config.progress_scale
    terms["best_progress_bonus"] = (
        best_progress_delta * reward_config.best_progress_bonus_scale
    )
    terms["default_reward"] = reward_config.default_reward
    if input_down:
        terms["jump_punishment"] = reward_config.jump_punishment

    if _crossed_percent_checkpoint(
        current_percent,
        next_percent,
        checkpoint_size=reward_config.checkpoint_size_percent,
    ):
        terms["checkpoint_reward"] = reward_config.checkpoint_reward

    if dead:
        terms["default_reward"] = 0.0
        terms["jump_punishment"] = 0.0
        terms["checkpoint_reward"] = 0.0
        terms["death_penalty"] = -reward_config.death_penalty

    if cleared:
        terms["clear_bonus"] = reward_config.clear_bonus

    terms["illegal_or_excessive_input_penalty"] = (
        -newly_charged_excessive_events * reward_config.excessive_input_penalty
    )
    return terms


def _compute_picklegawd_reward(
    rows: Sequence[TraceRow],
    *,
    config: RewardConfig,
    intended_event_count: int,
    executed_event_count: int,
) -> RewardBreakdown:
    terms = _empty_picklegawd_terms()
    best_percent = rows[0].percent if rows else 0.0
    if len(rows) >= 2:
        for current, next_row in zip(rows, rows[1:]):
            step_terms = compute_picklegawd_step_reward_terms(
                current_percent=current.percent,
                next_percent=next_row.percent,
                previous_best_percent=best_percent,
                input_down=next_row.input_down,
                dead=next_row.dead,
                cleared=next_row.percent >= config.success_percent,
                config=config,
            )
            for key, value in step_terms.items():
                terms[key] += value
            best_percent = max(best_percent, next_row.percent)

    event_count_for_penalty = max(intended_event_count, executed_event_count)
    excessive_events = max(
        0,
        event_count_for_penalty - config.excessive_input_free_events,
    )
    terms["illegal_or_excessive_input_penalty"] += (
        -excessive_events * config.excessive_input_penalty
    )
    return RewardBreakdown(terms=terms, total=sum(terms.values()))


def _empty_picklegawd_terms() -> dict[str, float]:
    return {
        "progress_delta": 0.0,
        "best_progress_bonus": 0.0,
        "default_reward": 0.0,
        "jump_punishment": 0.0,
        "checkpoint_reward": 0.0,
        "clear_bonus": 0.0,
        "death_penalty": 0.0,
        "illegal_or_excessive_input_penalty": 0.0,
    }


def _crossed_percent_checkpoint(
    previous_percent: float,
    next_percent: float,
    *,
    checkpoint_size: float,
) -> bool:
    return (
        next_percent > previous_percent
        and (next_percent % checkpoint_size)
        < (previous_percent % checkpoint_size)
    )
