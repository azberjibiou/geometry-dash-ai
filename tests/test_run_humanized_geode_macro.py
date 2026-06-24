import json

import pytest

from gd_env import BridgeObservation
from gd_human_model import HumanProfile
from gd_trace.trace_schema import TraceRow
from scripts.run_humanized_geode_macro import (
    _is_terminal_trace,
    _load_profile,
    _validate_start_observation,
    _validate_trace_progress,
)
from tests.test_trace_io import make_row


def test_load_profile_accepts_builtin_alias() -> None:
    profile = _load_profile("top_player", None)

    assert profile.name == "TopPlayer"


def test_load_profile_accepts_json_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "profile.json"
    data = {
        "name": "NoisyTest",
        "visual_delay_frames": 1,
        "motor_delay_frames": 2,
        "base_press_std_frames": 3.0,
        "base_release_std_frames": 4.0,
        "close_amp": 0.0,
        "close_tau": 10.0,
        "long_amp": 0.0,
        "long_tau": 120.0,
        "error_rho": 0.0,
        "miss_prob_base": 0.0,
        "miss_prob_close_amp": 0.0,
        "miss_prob_close_tau": 8.0,
        "random_seed": 99,
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    profile = _load_profile("Advanced", path)

    assert isinstance(profile, HumanProfile)
    assert profile.name == "NoisyTest"
    assert profile.base_release_std_frames == 4.0


def test_start_guard_rejects_wrong_attempt() -> None:
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
            attempt_number=1,
            require_percent_max=2.0,
            require_x_max=50.0,
        )


def test_progress_guard_accepts_expected_short_level() -> None:
    rows = [make_row(0, percent=0.0), make_row(120, percent=17.0)]

    _validate_trace_progress(
        rows,
        attempt_number=1,
        require_tick=120,
        require_percent_min=10.0,
    )


def test_progress_guard_rejects_wrong_level() -> None:
    rows = [make_row(0, percent=0.0), make_row(120, percent=0.6)]

    with pytest.raises(ValueError, match="progress check failed"):
        _validate_trace_progress(
            rows,
            attempt_number=1,
            require_tick=120,
            require_percent_min=10.0,
        )


def test_terminal_trace_detects_death_or_success() -> None:
    dead_row = make_row(10, percent=50.0).to_dict()
    dead_row["dead"] = True

    assert _is_terminal_trace(
        [make_row(0), make_row(10, percent=100.0)],
        success_percent=100.0,
    )
    assert _is_terminal_trace(
        [make_row(0), TraceRow.from_mapping(dead_row)],
        success_percent=100.0,
    )
    assert not _is_terminal_trace(
        [make_row(0), make_row(10, percent=50.0)],
        success_percent=100.0,
    )
