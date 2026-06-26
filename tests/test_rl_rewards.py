import pytest

from gd_rl import (
    RewardConfig,
    compute_picklegawd_step_reward_terms,
    compute_reward,
    summarize_trace_outcome,
)
from gd_trace.trace_schema import TraceRow
from tests.test_trace_io import make_row


def row(tick: int, *, percent: float = 0.0, dead: bool = False) -> TraceRow:
    data = make_row(tick, percent=percent).to_dict()
    data["dead"] = dead
    return TraceRow.from_mapping(data)


def input_row(
    tick: int,
    *,
    percent: float = 0.0,
    dead: bool = False,
    input_down: bool = False,
) -> TraceRow:
    data = make_row(tick, percent=percent).to_dict()
    data["dead"] = dead
    data["input_down"] = input_down
    return TraceRow.from_mapping(data)


def test_trace_outcome_reports_progress_death_and_clear() -> None:
    rows = [
        row(0, percent=1.0),
        row(20, percent=40.0),
        row(30, percent=35.0, dead=True),
    ]

    outcome = summarize_trace_outcome(rows, success_percent=100.0)

    assert outcome.row_count == 3
    assert outcome.first_tick == 0
    assert outcome.last_tick == 30
    assert outcome.start_percent == 1.0
    assert outcome.final_percent == 35.0
    assert outcome.best_percent == 40.0
    assert outcome.death_tick == 30
    assert outcome.death_percent == 35.0
    assert outcome.cleared is False


def test_reward_uses_progress_best_bonus_sections_and_death_penalty() -> None:
    rows = [
        row(0, percent=0.0),
        row(40, percent=60.0, dead=True),
    ]
    config = RewardConfig(
        progress_scale=1.0,
        best_progress_bonus_scale=0.5,
        section_size_percent=10.0,
        section_survival_bonus=1.0,
        clear_bonus=100.0,
        death_penalty=10.0,
    )

    reward = compute_reward(
        rows,
        config=config,
        previous_best_percent=40.0,
    )

    assert reward.terms["progress_delta"] == 60.0
    assert reward.terms["best_progress_bonus"] == 10.0
    assert reward.terms["section_survival_bonus"] == 6.0
    assert reward.terms["clear_bonus"] == 0.0
    assert reward.terms["death_penalty"] == -4.0
    assert reward.total == 72.0


def test_reward_gives_clear_bonus_without_death_penalty() -> None:
    rows = [row(0), row(100, percent=100.0)]
    config = RewardConfig(clear_bonus=50.0, death_penalty=10.0)

    reward = compute_reward(rows, config=config)

    assert reward.terms["clear_bonus"] == 50.0
    assert reward.terms["death_penalty"] == 0.0
    assert reward.total > 100.0


def test_reward_can_penalize_excessive_input() -> None:
    rows = [row(0), row(10, percent=5.0)]
    config = RewardConfig(
        excessive_input_free_events=2,
        excessive_input_penalty=0.25,
    )

    reward = compute_reward(
        rows,
        config=config,
        intended_event_count=5,
        executed_event_count=4,
    )

    assert reward.terms["illegal_or_excessive_input_penalty"] == -0.75


def test_picklegawd_reward_gives_survival_tick_reward_for_idle_and_hold() -> None:
    rows = [
        input_row(0, percent=0.0, input_down=False),
        input_row(1, percent=1.0, input_down=False),
        input_row(2, percent=3.1, input_down=True),
        input_row(3, percent=4.0, input_down=True, dead=True),
    ]
    config = RewardConfig(reward_style="picklegawd")

    reward = compute_reward(rows, config=config)

    assert reward.terms["progress_delta"] == 4.0
    assert reward.terms["best_progress_bonus"] == 2.0
    assert reward.terms["default_reward"] == 0.02
    assert reward.terms["jump_punishment"] == 0.0
    assert reward.terms["checkpoint_reward"] == 0.0
    assert reward.terms["death_penalty"] == -10.0
    assert reward.total == pytest.approx(-3.98)


def test_picklegawd_reward_does_not_penalize_hold_ticks_by_default() -> None:
    rows = [
        input_row(0, percent=0.0, input_down=False),
        *[
            input_row(tick, percent=tick * 26.658 / 205.0, input_down=True)
            for tick in range(1, 205)
        ],
        input_row(205, percent=26.658, input_down=True, dead=True),
    ]
    config = RewardConfig(reward_style="picklegawd")

    reward = compute_reward(rows, config=config)

    assert reward.total == pytest.approx(32.027)
    assert reward.terms["death_penalty"] == -10.0
    assert reward.terms["jump_punishment"] == 0.0
    assert reward.terms["default_reward"] > 0.0


def test_picklegawd_step_reward_adds_clear_bonus_after_action_reward() -> None:
    config = RewardConfig(reward_style="picklegawd", success_percent=99.0)

    reward = compute_picklegawd_step_reward_terms(
        current_percent=98.8,
        next_percent=99.5,
        previous_best_percent=98.8,
        input_down=True,
        dead=False,
        cleared=True,
        config=config,
    )

    assert reward["progress_delta"] == pytest.approx(0.7)
    assert reward["best_progress_bonus"] == pytest.approx(0.35)
    assert reward["default_reward"] == 0.01
    assert reward["jump_punishment"] == 0.0
    assert reward["clear_bonus"] == 100.0
    assert sum(reward.values()) == pytest.approx(101.06)
