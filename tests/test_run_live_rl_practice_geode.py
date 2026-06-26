import argparse
import json
from pathlib import Path

import pytest

from gd_env import BridgeDiagnostic, BridgeObservation
from gd_human_model import Event
from scripts.run_live_rl_practice_geode import (
    _build_actor_critic_config,
    _build_env_config,
    _build_policy_config,
    _build_reinforce_config,
    main,
)
from scripts.run_rl_practice_geode import _build_reward_config


def args(**overrides: object) -> argparse.Namespace:
    data = {
        "level_id": "local_live_smoke",
        "attempts": 1,
        "base_seed": None,
        "output_dir": Path("artifacts") / "test",
        "host": "127.0.0.1",
        "port": 29430,
        "timeout_seconds": 5.0,
        "max_steps": 600,
        "reset_wait_observations": 600,
        "fps": 240,
        "cbf": False,
        "physics_bypass": False,
        "success_percent": 100.0,
        "action_horizon_ticks": 1,
        "observation_buffer_size": None,
        "post_terminal_delay_seconds": 5.0,
        "start_guard_reset_retries": 3,
        "start_guard_retry_delay_seconds": 1.0,
        "require_start_percent_max": 2.0,
        "require_start_x_max": 50.0,
        "hidden_size": 32,
        "policy_seed": 0,
        "device": "cpu",
        "encoder_max_tick": None,
        "encoder_x_scale": 1000.0,
        "encoder_y_scale": 500.0,
        "encoder_velocity_scale": 20.0,
        "encoder_rotation_scale": 360.0,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "entropy_bonus": 0.0,
        "max_grad_norm": 1.0,
        "no_grad_clip": False,
        "normalize_returns": False,
        "deterministic_actions": False,
        "learner_seed": 0,
        "progress_scale": 1.0,
        "best_progress_bonus_scale": 0.5,
        "section_size_percent": 10.0,
        "section_survival_bonus": 0.25,
        "clear_bonus": 100.0,
        "death_penalty": 10.0,
        "excessive_input_free_events": 0,
        "excessive_input_penalty": 0.0,
        "algorithm": "a2c",
        "value_loss_weight": 0.5,
        "normalize_advantages": False,
        "history_length": 4,
        "death_local_window": 24,
        "death_local_penalty": 1.0,
        "input_rate_penalty": 0.0,
        "min_dwell_ticks": 4,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_build_live_rl_configs_keep_smoke_defaults() -> None:
    namespace = args()
    reward_config = _build_reward_config(namespace)

    env_config = _build_env_config(
        namespace,
        output_dir=namespace.output_dir,
        reward_config=reward_config,
        run_id="test_run",
    )
    policy_config = _build_policy_config(namespace)
    reinforce_config = _build_reinforce_config(namespace)
    actor_critic_config = _build_actor_critic_config(namespace)

    assert env_config.max_steps == 600
    assert env_config.post_terminal_delay_seconds == 5.0
    assert env_config.start_guard_reset_retries == 3
    assert env_config.require_start_percent_max == 2.0
    assert env_config.require_start_x_max == 50.0
    assert policy_config.device == "cpu"
    assert policy_config.encoder.max_tick == namespace.max_steps
    assert reinforce_config.attempts == 1
    assert reinforce_config.min_dwell_ticks == 4
    assert reinforce_config.max_grad_norm == 1.0
    assert _build_reinforce_config(args(no_grad_clip=True)).max_grad_norm is None
    assert actor_critic_config.attempts == 1
    assert actor_critic_config.history_length == 4
    assert actor_critic_config.death_local_window == 24
    assert actor_critic_config.min_dwell_ticks == 4
    assert actor_critic_config.max_grad_norm == 1.0
    assert _build_actor_critic_config(args(no_grad_clip=True)).max_grad_norm is None


def test_live_rl_cli_runs_with_fake_client_and_writes_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    fake_client = OneStepTerminalClient()
    output_dir = tmp_path / "live_smoke"

    exit_code = main(
        [
            "--level-id",
            "fake_local_level",
            "--output-dir",
            str(output_dir),
            "--attempts",
            "1",
            "--max-steps",
            "1",
            "--post-terminal-delay-seconds",
            "0",
            "--require-start-percent-max",
            "2",
            "--require-start-x-max",
            "50",
        ],
        client_factory=lambda: fake_client,
    )

    stdout = json.loads(capsys.readouterr().out)
    training_summary = json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    attempt_summary = json.loads(
        (output_dir / "attempt_001" / "summary.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert fake_client.connected is True
    assert fake_client.closed is True
    assert fake_client.reset_reasons == ["live_practice_attempt_1"]
    assert stdout["training_summary_json"] == str(output_dir / "training_summary.json")
    assert training_summary["attempt_count"] == 1
    assert training_summary["config"]["algorithm"] == "tiny_actor_critic"
    assert training_summary["config"]["policy"]["device"] == "cpu"
    assert training_summary["config"]["policy"]["history_length"] == 4
    assert attempt_summary["metadata"]["executor"] == "geode_live_step"
    assert attempt_summary["cleared"] is True


class OneStepTerminalClient:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.reset_reasons: list[str] = []
        self.sent_events: list[Event] = []

    def connect(self) -> "OneStepTerminalClient":
        self.connected = True
        return self

    def close(self) -> None:
        self.closed = True

    def reset_attempt(
        self,
        reason: str = "requested",
        *,
        max_observations: int = 600,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        self.reset_reasons.append(reason)
        self.sent_events.clear()
        if diagnostics is not None:
            diagnostics.append(
                BridgeDiagnostic(
                    kind="fake_reset",
                    tick=0,
                    data={"reason": reason, "max_observations": max_observations},
                )
            )
        return _observation(0, percent=0.0)

    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        return _observation(
            1,
            percent=100.0,
            completed=True,
            input_down=any(event.kind == "press" for event in self.sent_events),
        )

    def send_event(self, event: Event) -> None:
        self.sent_events.append(event)


def _observation(
    tick: int,
    *,
    percent: float,
    completed: bool = False,
    input_down: bool = False,
) -> BridgeObservation:
    return BridgeObservation(
        tick=tick,
        x=float(tick),
        y=0.0,
        y_vel=0.0,
        mode="cube",
        gravity="normal",
        percent=percent,
        dead=False,
        input_down=input_down,
        completed=completed,
        x_vel=1.0,
    )
