import json
from pathlib import Path

import pytest

from gd_env import BridgeDiagnostic, BridgeObservation
from gd_human_model import Event, HumanProfile
from gd_rl import (
    ActorCriticConfig,
    ButtonStateIntentAdapter,
    DQNConfig,
    IntendedAction,
    LiveActionHistory,
    LiveObservationEncoderConfig,
    LivePracticeEnv,
    LivePracticeEnvConfig,
    LivePracticeObservation,
    NeuralPolicyConfig,
    ReinforceConfig,
    TinyLiveActorCriticNetwork,
    TinyLiveDQNNetwork,
    TinyLivePolicyNetwork,
    actor_critic_feature_dim,
    dqn_feature_dim,
    encode_actor_critic_observation,
    encode_dqn_observation,
    encode_live_observation,
    live_observation_feature_dim,
    run_actor_critic_training,
    run_dqn_training,
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

    assert adapter.intent_for_desired_state("hold", tick=100) == IntendedAction.press(
        100
    )
    assert adapter.intent_for_desired_state("hold", tick=101).kind == "no_op"
    assert adapter.intent_for_desired_state("idle", tick=102) == IntendedAction.release(
        102
    )
    assert adapter.intent_for_desired_state("idle", tick=103).kind == "no_op"


def test_button_state_adapter_enforces_minimum_dwell_ticks() -> None:
    adapter = ButtonStateIntentAdapter(min_dwell_ticks=4)
    adapter.reset(
        LivePracticeObservation(
            latest=_observation(0, percent=0.0, input_down=False),
            policy_observation=_observation(0, percent=0.0, input_down=False),
        )
    )

    assert adapter.intent_for_desired_state("hold", tick=10) == IntendedAction.press(
        10
    )
    assert adapter.effective_input_state == "hold"
    assert adapter.intent_for_desired_state("idle", tick=11).kind == "no_op"
    assert adapter.effective_input_state == "hold"
    assert adapter.intent_for_desired_state("idle", tick=13).kind == "no_op"
    assert adapter.intent_for_desired_state("idle", tick=14) == IntendedAction.release(
        14
    )
    assert adapter.effective_input_state == "idle"
    assert adapter.intent_for_desired_state("hold", tick=17).kind == "no_op"
    assert adapter.intent_for_desired_state("hold", tick=18) == IntendedAction.press(
        18
    )


def test_actor_critic_encoder_appends_recent_intended_history() -> None:
    observation = LivePracticeObservation(
        latest=_observation(20, percent=80.0, x=200.0),
        policy_observation=_observation(10, percent=25.0, x=50.0),
    )
    history = LiveActionHistory(length=2)
    history.append(desired_input_state="hold", intent_kind="press")

    features = encode_actor_critic_observation(
        observation,
        history=history,
        history_length=2,
        config=LiveObservationEncoderConfig(max_tick=100, x_scale=100.0),
    )

    assert len(features) == actor_critic_feature_dim(2)
    assert features[: live_observation_feature_dim()] == encode_live_observation(
        observation,
        config=LiveObservationEncoderConfig(max_tick=100, x_scale=100.0),
    )
    assert features[-8:] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_dqn_encoder_uses_same_delay_aware_features() -> None:
    observation = LivePracticeObservation(
        latest=_observation(20, percent=80.0, x=200.0),
        policy_observation=_observation(10, percent=25.0, x=50.0),
    )
    history = LiveActionHistory(length=1)
    history.append(desired_input_state="hold", intent_kind="press")

    features = encode_dqn_observation(
        observation,
        history=history,
        history_length=1,
        config=LiveObservationEncoderConfig(max_tick=100, x_scale=100.0),
    )

    assert len(features) == dqn_feature_dim(1)
    assert features == encode_actor_critic_observation(
        observation,
        history=history,
        history_length=1,
        config=LiveObservationEncoderConfig(max_tick=100, x_scale=100.0),
    )


def test_reinforce_training_updates_hold_probability_and_writes_summary(
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
    _bias_policy_toward_hold(torch, policy)
    observation = LivePracticeObservation(
        latest=_observation(0, percent=0.0),
        policy_observation=_observation(0, percent=0.0),
    )
    before_hold_probability = policy.action_probabilities(observation)[1]

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

    after_hold_probability = policy.action_probabilities(observation)[1]
    written_summary = json.loads(
        (tmp_path / "learner_summary.json").read_text(encoding="utf-8")
    )

    assert fake_client.sent_events == [Event(0, "press")]
    assert after_hold_probability > before_hold_probability
    assert summary.attempt_count == 1
    assert summary.attempts[0].action_counts["hold"] == 1
    assert summary.attempts[0].intent_counts["press"] == 1
    assert summary.attempts[0].attempt_result["cleared"] is True
    assert written_summary["attempt_count"] == 1


def test_actor_critic_training_updates_hold_probability_and_writes_summary(
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
    policy = TinyLiveActorCriticNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=9),
        history_length=2,
    )
    _bias_actor_critic_toward_hold(torch, policy)
    observation = LivePracticeObservation(
        latest=_observation(0, percent=0.0),
        policy_observation=_observation(0, percent=0.0),
    )
    before_hold_probability = policy.action_probabilities(observation)[1]

    with env:
        summary = run_actor_critic_training(
            env,
            policy,
            config=ActorCriticConfig(
                attempts=1,
                learning_rate=0.05,
                deterministic_actions=True,
                history_length=2,
                death_local_penalty=0.0,
            ),
            summary_path=tmp_path / "a2c_summary.json",
        )

    after_hold_probability = policy.action_probabilities(observation)[1]
    written_summary = json.loads(
        (tmp_path / "a2c_summary.json").read_text(encoding="utf-8")
    )

    assert fake_client.sent_events == [Event(0, "press")]
    assert after_hold_probability > before_hold_probability
    assert summary.attempt_count == 1
    assert summary.attempts[0].action_counts["hold"] == 1
    assert summary.attempts[0].intent_counts["press"] == 1
    assert summary.attempts[0].value_loss >= 0.0
    assert summary.attempts[0].attempt_result["cleared"] is True
    assert written_summary["config"]["algorithm"] == "tiny_actor_critic"
    assert written_summary["config"]["policy"]["history_length"] == 2


def test_actor_critic_death_local_feedback_weights_recent_window(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    fake_client = OneStepDeathClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="one_step_death",
            output_dir=tmp_path / "attempts",
            max_steps=1,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLiveActorCriticNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=11),
        history_length=1,
    )
    _bias_actor_critic_toward_idle(torch, policy)

    with env:
        summary = run_actor_critic_training(
            env,
            policy,
            config=ActorCriticConfig(
                attempts=1,
                learning_rate=0.01,
                deterministic_actions=True,
                history_length=1,
                death_local_window=4,
                death_local_penalty=2.0,
            ),
        )

    attempt = summary.attempts[0]
    stats = attempt.death_local_stats

    assert stats["applied"] is True
    assert stats["affected_step_count"] == 1
    assert stats["penalty_total"] == pytest.approx(-2.0)
    assert stats["intent_counts"]["no_op"] == 1
    assert stats["recent_steps"][0]["death_local_penalty"] == pytest.approx(-2.0)
    assert attempt.total_training_reward == pytest.approx(
        attempt.total_step_reward - 2.0
    )


def test_dqn_training_updates_hold_q_value_and_writes_summary(
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
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=13),
        history_length=1,
    )
    _bias_dqn_toward_hold(torch, policy)
    observation = LivePracticeObservation(
        latest=_observation(0, percent=0.0),
        policy_observation=_observation(0, percent=0.0),
    )
    before_hold_q = policy.q_values(observation)[1]

    with env:
        summary = run_dqn_training(
            env,
            policy,
            config=DQNConfig(
                attempts=1,
                learning_rate=0.05,
                deterministic_actions=True,
                history_length=1,
                batch_size=1,
                replay_capacity=4,
                warmup_steps=1,
            ),
            summary_path=tmp_path / "dqn_summary.json",
        )

    after_hold_q = policy.q_values(observation)[1]
    written_summary = json.loads(
        (tmp_path / "dqn_summary.json").read_text(encoding="utf-8")
    )

    assert fake_client.sent_events == [Event(0, "press")]
    assert after_hold_q > before_hold_q
    assert summary.attempt_count == 1
    assert summary.attempts[0].action_counts["hold"] == 1
    assert summary.attempts[0].intent_counts["press"] == 1
    assert summary.attempts[0].update_count == 1
    assert summary.attempts[0].replay_appended_count == 1
    assert summary.attempts[0].replay_skipped is False
    assert summary.attempts[0].replay_skip_reason is None
    assert summary.attempts[0].attempt_result["cleared"] is True
    assert written_summary["config"]["algorithm"] == "tiny_dqn"
    assert written_summary["config"]["policy"]["history_length"] == 1


def test_dqn_training_skips_tick_rewind_reset_attempts(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    fake_client = OneStepTickRewindClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="tick_rewind_artifact",
            output_dir=tmp_path / "attempts",
            max_steps=4,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=17),
        history_length=1,
    )
    _bias_dqn_toward_hold(torch, policy)
    observation = LivePracticeObservation(
        latest=_observation(0, percent=0.0),
        policy_observation=_observation(0, percent=0.0),
    )
    before_hold_q = policy.q_values(observation)[1]

    with env:
        summary = run_dqn_training(
            env,
            policy,
            config=DQNConfig(
                attempts=1,
                learning_rate=0.05,
                deterministic_actions=True,
                history_length=1,
                batch_size=1,
                replay_capacity=4,
                warmup_steps=1,
            ),
            summary_path=tmp_path / "dqn_rewind_summary.json",
        )

    after_hold_q = policy.q_values(observation)[1]
    written_summary = json.loads(
        (tmp_path / "dqn_rewind_summary.json").read_text(encoding="utf-8")
    )

    attempt = summary.attempts[0]
    assert attempt.step_count == 1
    assert attempt.update_count == 0
    assert attempt.replay_size == 0
    assert attempt.replay_appended_count == 0
    assert attempt.replay_skipped is True
    assert attempt.replay_skip_reason == "death_reason:tick_rewind_reset"
    assert attempt.attempt_result["death_tick"] == 0
    assert after_hold_q == pytest.approx(before_hold_q)
    assert written_summary["attempts"][0]["replay_skipped"] is True


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

    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
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


class OneStepDeathClient(OneStepPressRewardClient):
    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        return _observation(
            1,
            percent=0.0,
            dead=True,
            input_down=any(event.kind == "press" for event in self.sent_events),
        )


class OneStepTickRewindClient(OneStepPressRewardClient):
    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        return _observation(
            0,
            percent=0.0,
            dead=False,
            input_down=any(event.kind == "press" for event in self.sent_events),
        )


def _bias_policy_toward_hold(torch, policy: TinyLivePolicyNetwork) -> None:  # type: ignore[no-untyped-def]
    with torch.no_grad():
        for parameter in policy.model.parameters():
            parameter.zero_()
        policy.model[2].bias.copy_(
            torch.tensor([0.0, 1.0], dtype=torch.float32)
        )


def _bias_actor_critic_toward_hold(torch, policy: TinyLiveActorCriticNetwork) -> None:  # type: ignore[no-untyped-def]
    _zero_actor_critic(torch, policy)
    with torch.no_grad():
        policy.actor_head.bias.copy_(
            torch.tensor([0.0, 1.0], dtype=torch.float32)
        )


def _bias_actor_critic_toward_idle(torch, policy: TinyLiveActorCriticNetwork) -> None:  # type: ignore[no-untyped-def]
    _zero_actor_critic(torch, policy)
    with torch.no_grad():
        policy.actor_head.bias.copy_(
            torch.tensor([1.0, 0.0], dtype=torch.float32)
        )


def _zero_actor_critic(torch, policy: TinyLiveActorCriticNetwork) -> None:  # type: ignore[no-untyped-def]
    with torch.no_grad():
        for module in (policy.shared, policy.actor_head, policy.value_head):
            for parameter in module.parameters():
                parameter.zero_()


def _bias_dqn_toward_hold(torch, policy: TinyLiveDQNNetwork) -> None:  # type: ignore[no-untyped-def]
    with torch.no_grad():
        for parameter in policy.q_network.parameters():
            parameter.zero_()
        policy.q_network[2].bias.copy_(
            torch.tensor([0.0, 1.0], dtype=torch.float32)
        )
    policy.sync_target()


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
