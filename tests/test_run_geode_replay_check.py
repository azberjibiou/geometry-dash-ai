import pytest

from gd_env import BridgeObservation
from scripts.run_geode_replay_check import (
    _validate_start_observation,
    _validate_trace_progress,
)
from tests.test_trace_io import make_row


def test_start_guard_rejects_wrong_late_attempt() -> None:
    observation = BridgeObservation(
        tick=0,
        x=100.0,
        y=105.0,
        y_vel=0.0,
        mode="cube",
        gravity="normal",
        percent=20.0,
        dead=False,
        input_down=False,
    )

    with pytest.raises(ValueError, match="fresh start check failed"):
        _validate_start_observation(
            observation,
            trial_number=1,
            require_percent_max=2.0,
            require_x_max=50.0,
        )


def test_progress_guard_rejects_wrong_long_level() -> None:
    rows = [make_row(0, percent=0.0), make_row(120, percent=0.6)]

    with pytest.raises(ValueError, match="progress check failed"):
        _validate_trace_progress(
            rows,
            trial_number=1,
            require_tick=120,
            require_percent_min=10.0,
        )


def test_progress_guard_accepts_expected_short_level() -> None:
    rows = [make_row(0, percent=0.0), make_row(120, percent=17.0)]

    _validate_trace_progress(
        rows,
        trial_number=1,
        require_tick=120,
        require_percent_min=10.0,
    )
