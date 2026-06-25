import argparse

from gd_human_model import Event, HumanProfile
from scripts.run_rl_practice_geode import (
    _build_geode_config,
    _build_policy,
    _build_reward_config,
    _load_profile,
)
from gd_rl import NoInputPolicy, RandomEventPolicy, ScriptedEventPolicy
from gd_trace import Macro, save_macro_json


def args(**overrides: object) -> argparse.Namespace:
    data = {
        "policy": "no-input",
        "macro_json": None,
        "random_max_events": 12,
        "random_min_tick": 0,
        "random_max_tick": 1200,
        "random_min_spacing": 4,
        "random_seed": 7,
        "success_percent": 100.0,
        "stop_after_first_clear": False,
        "progress_scale": 1.0,
        "best_progress_bonus_scale": 0.5,
        "section_size_percent": 10.0,
        "section_survival_bonus": 0.25,
        "clear_bonus": 100.0,
        "death_penalty": 10.0,
        "excessive_input_free_events": 0,
        "excessive_input_penalty": 0.0,
        "host": "127.0.0.1",
        "port": 29430,
        "timeout_seconds": 5.0,
        "max_observations": 1200,
        "reset_wait_observations": 600,
        "fps": 240,
        "cbf": False,
        "physics_bypass": False,
        "stop_on_success": True,
        "post_terminal_delay_seconds": 0.0,
        "start_guard_reset_retries": 2,
        "start_guard_retry_delay_seconds": 0.2,
        "require_start_percent_max": 2.0,
        "require_start_x_max": 50.0,
        "require_progress_tick": 120,
        "require_progress_percent_min": 10.0,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_load_profile_accepts_builtin_alias() -> None:
    profile = _load_profile("advanced", None)

    assert profile.name == "Advanced"


def test_load_profile_accepts_json_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "profile.json"
    path.write_text(
        """{
  "name": "ScriptProfile",
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
  "random_seed": 99
}
""",
        encoding="utf-8",
    )

    profile = _load_profile("Advanced", path)

    assert isinstance(profile, HumanProfile)
    assert profile.name == "ScriptProfile"


def test_build_policy_supports_no_input_random_and_scripted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(_build_policy(args(policy="no-input")), NoInputPolicy)
    assert isinstance(_build_policy(args(policy="random")), RandomEventPolicy)

    macro_path = tmp_path / "macro.json"
    save_macro_json(
        Macro(events=[Event(10, "press"), Event(20, "release")]),
        macro_path,
    )
    policy = _build_policy(args(policy="scripted", macro_json=macro_path))

    assert isinstance(policy, ScriptedEventPolicy)
    assert policy.events == [Event(10, "press"), Event(20, "release")]


def test_build_reward_and_geode_configs_from_args() -> None:
    namespace = args()

    reward_config = _build_reward_config(namespace)
    geode_config = _build_geode_config(namespace)

    assert reward_config.success_percent == 100.0
    assert geode_config.stop_on_success is True
    assert geode_config.require_progress_tick == 120
