import json
from pathlib import Path

import pytest

from gd_env import BridgeDiagnostic, BridgeObservation
from gd_human_model import Event, HumanProfile
from gd_rl import (
    ButtonStateIntentAdapter,
    IntendedAction,
    LiveObservationEncoderConfig,
    LivePracticeEnv,
    LivePracticeEnvConfig,
    LivePracticeObservation,
    NeuralPolicyConfig,
    ReinforceConfig,
    TinyLivePolicyNetwork,
    encode_live_observation,
    live_observation_feature_dim,
    run_reinforce_training,
)


def test_live_observation_encoder_uses_policy_visible_observation() -> None:
    latest = _observation(20, percent=80.0, x=200.0)
    delayed = _observation(10, percent=25.0, x=50.0)

    features = encode_live_observation(
        LivePracticeObservation(latest=latest, policy_observation=delayed),
        config=LiveObservationEncoderConfig(max_tick=100, x_scale=100.0),
    )

    assert len(features) == live_observation_feature_dim()
    assert features[0] == 1.0
    assert features[1] == pytest.approx(0.1)
    assert features[2] == pytest.approx(0.25)
    assert features[3] == pytest.approx(0.5)


def test_live_observation_encoder_masks_missing_policy_observation() -> None:
    latest = _observation(20, percent=80.0, x=200.0)

    features = encode_live_observation(
        LivePracticeObservation(latest=latest, policy_observation=None)
    )

    assert len(features) == live_observation_feature_dim()
    assert features == [0.0] * live_observation_feature_dim()


def test_button_state_adapter_uses_intended_state_not_executed_state() -> None:
    adapter = ButtonStateIntentAdapter()
    adapter.reset(
        LivePracticeObservation(
            latest=_observation(0, percent=0.0, input_down=False),
            policy_observation=_observation(0, percent=0.0, input_down=False),
        )
    )

    assert adapter.intent_for_desired_state("down", tick=100) == IntendedAction.press(
        100
    )
    assert adapter.intent_for_desired_state("down", tick=101).kind == "no_op"
    assert adapter.intent_for_desired_state("up", tick=102) == IntendedAction.release(
        102
    )
    assert adapter.intent_for_desired_state("up", tick=103).kind == "no_op"


def test_reinforce_training_updates_down_probability_and_writes_summary(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    fake_client = OneStepPressRewardClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="one_step_press_reward",
            output_dir=tmp_path / "attempts",
            max_steps=1,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLivePolicyNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=7)
    )
    _bias_policy_toward_down(torch, policy)
    observation = LivePracticeObservation(
        latest=_observation(0, percent=0.0),
        policy_observation=_observation(0, percent=0.0),
    )
    before_down_probability = policy.action_probabilities(observation)[1]

    with env:
        summary = run_reinforce_training(
            env,
            policy,
            config=ReinforceConfig(
                attempts=1,
                learning_rate=0.05,
                deterministic_actions=True,
                normalize_returns=False,
            ),
            summary_path=tmp_path / "learner_summary.json",
        )

    after_down_probability = policy.action_probabilities(observation)[1]
    written_summary = json.loads(
        (tmp_path / "learner_summary.json").read_text(encoding="utf-8")
    )

    assert fake_client.sent_events == [Event(0, "press")]
    assert after_down_probability > before_down_probability
    assert summary.attempt_count == 1
    assert summary.attempts[0].action_counts["down"] == 1
    assert summary.attempts[0].intent_counts["press"] == 1
    assert summary.attempts[0].attempt_result["cleared"] is True
    assert written_summary["attempt_count"] == 1


class OneStepPressRewardClient:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.sent_events: list[Event] = []

    def connect(self) -> "OneStepPressRewardClient":
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

    def receive_observation(self) -> BridgeObservation:
        pressed = any(event.kind == "press" for event in self.sent_events)
        return _observation(
            1,
            percent=100.0 if pressed else 0.0,
            completed=pressed,
            dead=not pressed,
            input_down=pressed,
        )

    def send_event(self, event: Event) -> None:
        self.sent_events.append(event)


def _bias_policy_toward_down(torch, policy: TinyLivePolicyNetwork) -> None:  # type: ignore[no-untyped-def]
    with torch.no_grad():
        for parameter in policy.model.parameters():
            parameter.zero_()
        policy.model[2].bias.copy_(
            torch.tensor([0.0, 1.0], dtype=torch.float32)
        )


def _profile() -> HumanProfile:
    return HumanProfile(
        name="DeterministicTest",
        visual_delay_frames=0,
        motor_delay_frames=0,
        base_press_std_frames=0.0,
        base_release_std_frames=0.0,
        close_amp=0.0,
        close_tau=1.0,
        long_amp=0.0,
        long_tau=1.0,
        error_rho=0.0,
        miss_prob_base=0.0,
        miss_prob_close_amp=0.0,
        miss_prob_close_tau=1.0,
        random_seed=0,
    )


def _observation(
    tick: int,
    *,
    percent: float,
    x: float | None = None,
    dead: bool = False,
    completed: bool = False,
    input_down: bool = False,
) -> BridgeObservation:
    return BridgeObservation(
        tick=tick,
        x=float(tick if x is None else x),
        y=0.0,
        y_vel=0.0,
        mode="cube",
        gravity="normal",
        percent=percent,
        dead=dead,
        input_down=input_down,
        completed=completed,
        x_vel=1.0,
    )
