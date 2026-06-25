import pytest

from gd_env import BridgeObservation
from scripts.capture_geode_frames import (
    _observations_to_trace_rows,
    _pre_capture_terminal_reason,
    _summarize_observations,
    _terminal_reason,
    _validate_observation_progress,
    _validate_start_observation,
)


def test_capture_start_guard_rejects_wrong_level_start() -> None:
    observation = _observation(tick=0, x=100.0, percent=12.0)

    with pytest.raises(ValueError, match="fresh start check failed"):
        _validate_start_observation(
            observation,
            require_percent_max=2.0,
            require_x_max=50.0,
        )


def test_capture_progress_guard_accepts_expected_progress() -> None:
    observations = [
        _observation(tick=0, percent=0.0),
        _observation(tick=120, percent=17.0),
    ]

    _validate_observation_progress(
        observations,
        require_tick=120,
        require_percent_min=10.0,
    )


def test_capture_progress_guard_rejects_wrong_level() -> None:
    observations = [
        _observation(tick=0, percent=0.0),
        _observation(tick=120, percent=0.5),
    ]

    with pytest.raises(ValueError, match="progress check failed"):
        _validate_observation_progress(
            observations,
            require_tick=120,
            require_percent_min=10.0,
        )


def test_capture_progress_guard_rejects_short_capture() -> None:
    observations = [_observation(tick=0, percent=0.0)]

    with pytest.raises(ValueError, match="ended before progress guard tick"):
        _validate_observation_progress(
            observations,
            require_tick=120,
            require_percent_min=10.0,
        )


def test_terminal_reason_prefers_death_before_success() -> None:
    observation = _observation(tick=10, percent=100.0, dead=True)

    assert (
        _terminal_reason(
            observation,
            stop_on_death=True,
            stop_on_success=True,
            success_percent=100.0,
        )
        == "death"
    )


def test_terminal_reason_detects_success() -> None:
    observation = _observation(tick=10, percent=100.0)

    assert (
        _terminal_reason(
            observation,
            stop_on_death=True,
            stop_on_success=True,
            success_percent=100.0,
        )
        == "success"
    )


def test_pre_capture_terminal_reason_detects_completion() -> None:
    observation = _observation(tick=10, percent=98.0, completed=True)

    assert (
        _pre_capture_terminal_reason(
            observation,
            stop_before_completion=True,
        )
        == "completed"
    )


def test_pre_capture_terminal_reason_ignores_completion_when_disabled() -> None:
    observation = _observation(tick=10, percent=98.0, completed=True)

    assert (
        _pre_capture_terminal_reason(
            observation,
            stop_before_completion=False,
        )
        is None
    )


def test_observation_summary_reports_terminal_state() -> None:
    summary = _summarize_observations(
        [
            _observation(tick=0, percent=0.0),
            _observation(tick=42, percent=100.0, completed=True),
        ],
        stop_reason="success",
    )

    assert summary == {
        "stop_reason": "success",
        "first_tick": 0,
        "last_tick": 42,
        "start_percent": 0.0,
        "final_percent": 100.0,
        "dead": False,
        "completed": True,
    }


def test_observations_to_trace_rows_preserves_capture_metadata() -> None:
    rows = _observations_to_trace_rows(
        [_observation(tick=7, percent=12.5)],
        fps=240,
        cbf=False,
        physics_bypass=True,
    )

    assert len(rows) == 1
    assert rows[0].tick == 7
    assert rows[0].time_ms == 7 * 1000 / 240
    assert rows[0].percent == 12.5
    assert rows[0].fps == 240
    assert rows[0].cbf is False
    assert rows[0].physics_bypass is True


def _observation(
    *,
    tick: int,
    x: float = 0.0,
    percent: float = 0.0,
    dead: bool = False,
    completed: bool = False,
) -> BridgeObservation:
    return BridgeObservation(
        tick=tick,
        x=x,
        y=0.0,
        y_vel=0.0,
        mode="cube",
        gravity="normal",
        percent=percent,
        dead=dead,
        input_down=False,
        completed=completed,
    )
