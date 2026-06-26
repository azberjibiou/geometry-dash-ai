import json
from pathlib import Path

import pytest

import gd_rl.live_learner as live_learner_module
from gd_env import BridgeDiagnostic, BridgeObservation
from gd_human_model import Event, HumanProfile
from gd_rl import (
    ActorCriticConfig,
    ButtonStateIntentAdapter,
    DQNConfig,
    DQNReplayBuffer,
    DQNTransition,
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
    latest = _observation(20, percent=80.0, x=200.0, input_down=True)
    delayed = _observation(10, percent=25.0, x=50.0, input_down=False)

    features = encode_live_observation(
        LivePracticeObservation(
            latest=latest,
            policy_observation=delayed,
            pending_event_count=2,
        ),
        config=LiveObservationEncoderConfig(
            max_tick=100,
            x_scale=100.0,
            pending_event_scale=4.0,
        ),
    )

    assert len(features) == live_observation_feature_dim()
    assert features[0] == 1.0
    assert features[1] == pytest.approx(0.1)
    assert features[2] == pytest.approx(0.25)
    assert features[3] == pytest.approx(0.5)
    assert features[-4:] == pytest.approx([1.0, 1.0, 0.5, 0.1])


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
    history.append(
        selected_desired_state="hold",
        commanded_input_state="hold",
        intent_kind="press",
        dwell_blocked=False,
    )

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
    assert features[-12:] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]


def test_action_history_preserves_dwell_blocked_desired_and_commanded_state() -> None:
    history = LiveActionHistory(length=1)
    history.append(
        selected_desired_state="idle",
        commanded_input_state="hold",
        intent_kind="no_op",
        dwell_blocked=True,
    )

    assert history.entries[0].selected_desired_state == "idle"
    assert history.entries[0].commanded_input_state == "hold"
    assert history.entries[0].dwell_blocked is True
    assert history.features() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]


def test_dqn_dwell_blocked_decision_history_keeps_selected_idle() -> None:
    torch = pytest.importorskip("torch")
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=23),
        history_length=1,
    )
    adapter = ButtonStateIntentAdapter(min_dwell_ticks=4)
    adapter.reset(
        LivePracticeObservation(
            latest=_observation(10, percent=0.0, input_down=False),
            policy_observation=_observation(10, percent=0.0, input_down=False),
        )
    )
    history = LiveActionHistory(length=1)

    _bias_dqn_toward_hold(torch, policy)
    first = policy.act(
        LivePracticeObservation(
            latest=_observation(10, percent=0.0, input_down=False),
            policy_observation=_observation(10, percent=0.0, input_down=False),
        ),
        intent_adapter=adapter,
        history=history,
        deterministic=True,
    )
    assert first.intent.kind == "press"

    _bias_dqn_toward_idle(torch, policy)
    blocked = policy.act(
        LivePracticeObservation(
            latest=_observation(11, percent=1.0, input_down=False),
            policy_observation=_observation(11, percent=1.0, input_down=False),
        ),
        intent_adapter=adapter,
        history=history,
        deterministic=True,
    )
    history.append(
        selected_desired_state=blocked.desired_input_state,
        commanded_input_state=blocked.effective_input_state,
        intent_kind=blocked.intent.kind,
        dwell_blocked=blocked.dwell_blocked,
    )

    assert blocked.desired_input_state == "idle"
    assert blocked.effective_input_state == "hold"
    assert blocked.dwell_blocked is True
    assert history.features() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]


def test_dqn_encoder_uses_same_delay_aware_features() -> None:
    observation = LivePracticeObservation(
        latest=_observation(20, percent=80.0, x=200.0),
        policy_observation=_observation(10, percent=25.0, x=50.0),
    )
    history = LiveActionHistory(length=1)
    history.append(
        selected_desired_state="hold",
        commanded_input_state="hold",
        intent_kind="press",
        dwell_blocked=False,
    )

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


def test_dqn_decision_stride_repeats_noop_steps_for_one_transition(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    fake_client = MultiStepProgressClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="decision_stride_progress",
            output_dir=tmp_path / "attempts",
            max_steps=4,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=29),
        history_length=1,
    )
    _bias_dqn_toward_hold(torch, policy)

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
                decision_stride=4,
            ),
        )

    attempt = summary.attempts[0]
    assert fake_client.sent_events == [Event(0, "press")]
    assert attempt.step_count == 4
    assert attempt.decision_count == 1
    assert attempt.replay_appended_count == 1
    assert attempt.action_counts["hold"] == 1
    assert attempt.intent_counts["press"] == 1
    assert attempt.attempt_result["cleared"] is False


def test_dqn_replay_buffer_overwrites_oldest_transition() -> None:
    features = [0.0] * dqn_feature_dim(history_length=0)
    replay = DQNReplayBuffer(capacity=2)

    for reward in (1.0, 2.0, 3.0):
        replay.append(
            DQNTransition(
                features=features,
                action_index=1,
                reward=reward,
                next_features=features,
                done=False,
                terminated=False,
            )
        )

    assert len(replay) == 2
    assert {transition.reward for transition in replay.sample(2)} == {2.0, 3.0}


def test_dqn_replay_buffer_can_reserve_success_and_terminal_samples() -> None:
    features = [0.0] * dqn_feature_dim(history_length=0)
    replay = DQNReplayBuffer(capacity=5)

    replay.append(
        DQNTransition(
            features=features,
            action_index=0,
            reward=0.0,
            next_features=features,
            done=False,
            terminated=False,
        )
    )
    replay.append(
        DQNTransition(
            features=features,
            action_index=1,
            reward=90.0,
            next_features=features,
            done=True,
            terminated=True,
            contains_clear_bonus=True,
            terminal_kind="clear",
        )
    )
    replay.append(
        DQNTransition(
            features=features,
            action_index=1,
            reward=60.0,
            next_features=features,
            done=False,
            terminated=False,
        )
    )
    replay.append(
        DQNTransition(
            features=features,
            action_index=0,
            reward=-10.0,
            next_features=features,
            done=True,
            terminated=True,
            terminal_kind="death",
        )
    )
    replay.append(
        DQNTransition(
            features=features,
            action_index=0,
            reward=1.0,
            next_features=features,
            done=False,
            terminated=False,
        )
    )

    batch = replay.sample(
        4,
        success_fraction=0.5,
        terminal_fraction=0.25,
        success_reward_threshold=50.0,
    )

    assert len(batch) == 4
    assert sum(
        transition.contains_clear_bonus or transition.reward >= 50.0
        for transition in batch
    ) >= 2
    assert any(transition.terminal_kind == "death" for transition in batch)


def test_dqn_training_runs_periodic_greedy_evaluation(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    fake_client = OneStepPressRewardClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="greedy_eval",
            output_dir=tmp_path / "attempts",
            max_steps=1,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=31),
        history_length=1,
    )
    _bias_dqn_toward_hold(torch, policy)
    callback_runs = []
    summary_path = tmp_path / "dqn_eval_summary.json"
    callback_snapshots = []

    def capture_evaluation(evaluation):  # type: ignore[no-untyped-def]
        callback_runs.append(evaluation)
        callback_snapshots.append(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )

    with env:
        summary = run_dqn_training(
            env,
            policy,
            config=DQNConfig(
                attempts=2,
                learning_rate=0.05,
                deterministic_actions=True,
                history_length=1,
                batch_size=1,
                replay_capacity=8,
                warmup_steps=1,
                eval_attempts=2,
                eval_interval_attempts=1,
            ),
            summary_path=summary_path,
            evaluation_callback=capture_evaluation,
        )

    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    first_eval = summary.evaluation_runs[0]
    second_eval = summary.evaluation_runs[1]

    assert len(summary.evaluation_runs) == 2
    assert len(callback_runs) == 2
    assert [snapshot["attempt_count"] for snapshot in callback_snapshots] == [1, 2]
    assert [len(snapshot["evaluation_runs"]) for snapshot in callback_snapshots] == [
        1,
        2,
    ]
    assert first_eval.after_attempt_index == 1
    assert first_eval.clear_count == 2
    assert first_eval.clear_rate == pytest.approx(1.0)
    assert first_eval.attempts[0].attempt_index == 3
    assert first_eval.attempts[1].attempt_index == 4
    assert second_eval.after_attempt_index == 2
    assert second_eval.attempts[0].attempt_index == 5
    assert written_summary["evaluation_runs"][0]["clear_count"] == 2
    assert written_summary["evaluation_runs"][0]["attempts"][0]["eval_index"] == 1


def test_dqn_repeat_action_penalty_reduces_training_reward(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    fake_client = MultiStepProgressClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="repeat_penalty",
            output_dir=tmp_path / "attempts",
            max_steps=4,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=37),
        history_length=1,
    )
    _bias_dqn_toward_hold(torch, policy)

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
                replay_capacity=8,
                warmup_steps=1,
                repeat_action_penalty=0.25,
                repeat_action_penalty_free_decisions=1,
            ),
        )

    attempt = summary.attempts[0]
    assert attempt.decision_count == 4
    assert attempt.total_training_reward == pytest.approx(
        attempt.total_step_reward - 0.75
    )
    assert attempt.action_diagnostics["repeat_action_penalty_total"] == pytest.approx(
        -0.75
    )
    assert attempt.action_diagnostics["max_selected_state_run"] == 4
    assert attempt.action_diagnostics["collapse_flags"]["selected_action_collapse"]
    assert attempt.greedy_action_counts["hold"] == 4
    assert attempt.q_margin_stats["mean"] > 0.0


def test_dqn_target_sync_waits_for_successful_update(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    fake_client = OneStepPressRewardClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="target_sync_warmup",
            output_dir=tmp_path / "attempts",
            max_steps=1,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=41),
        history_length=0,
    )
    _set_dqn_online_target_biases(
        torch,
        policy,
        online_biases=[0.0, 1.0],
        target_biases=[0.0, 0.0],
    )
    features = [0.0] * dqn_feature_dim(history_length=0)
    before_target_hold_q = _dqn_target_action_q(
        torch,
        policy,
        features,
        action_index=1,
    )

    with env:
        summary = run_dqn_training(
            env,
            policy,
            config=DQNConfig(
                attempts=1,
                deterministic_actions=True,
                history_length=0,
                batch_size=1,
                replay_capacity=4,
                warmup_steps=2,
                target_update_interval=1,
            ),
        )

    after_target_hold_q = _dqn_target_action_q(
        torch,
        policy,
        features,
        action_index=1,
    )

    assert summary.attempts[0].replay_appended_count == 1
    assert summary.attempts[0].update_count == 0
    assert after_target_hold_q == pytest.approx(before_target_hold_q)


def test_dqn_target_sync_uses_global_update_count(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    fake_client = OneStepPressRewardClient()
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="target_sync_global_updates",
            output_dir=tmp_path / "attempts",
            max_steps=1,
            action_horizon_ticks=0,
            success_percent=100.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=43),
        history_length=0,
    )
    _set_dqn_online_target_biases(
        torch,
        policy,
        online_biases=[0.0, 1.0],
        target_biases=[0.0, 0.0],
    )

    with env:
        summary = run_dqn_training(
            env,
            policy,
            config=DQNConfig(
                attempts=2,
                deterministic_actions=True,
                history_length=0,
                batch_size=1,
                replay_capacity=4,
                warmup_steps=1,
                target_update_interval=2,
            ),
        )

    assert [attempt.update_count for attempt in summary.attempts] == [1, 1]
    assert _dqn_networks_match(torch, policy)


def test_dqn_picklegawd_epsilon_schedule_decays_by_attempt() -> None:
    config = DQNConfig(
        epsilon_schedule="picklegawd",
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_rate=0.995,
    )

    assert live_learner_module._dqn_epsilon(
        config,
        0,
        attempt_index=1,
    ) == pytest.approx(1.0)
    assert live_learner_module._dqn_epsilon(
        config,
        9999,
        attempt_index=2,
    ) == pytest.approx(0.995)
    assert live_learner_module._dqn_epsilon(
        config,
        0,
        attempt_index=5000,
    ) == pytest.approx(0.01)


def test_dqn_linear_epsilon_schedule_uses_global_step() -> None:
    config = DQNConfig(
        epsilon_schedule="linear",
        epsilon_start=0.2,
        epsilon_end=0.05,
        epsilon_decay_steps=100,
    )

    assert live_learner_module._dqn_epsilon(
        config,
        0,
        attempt_index=99,
    ) == pytest.approx(0.2)
    assert live_learner_module._dqn_epsilon(
        config,
        50,
        attempt_index=99,
    ) == pytest.approx(0.125)
    assert live_learner_module._dqn_epsilon(
        config,
        1000,
        attempt_index=99,
    ) == pytest.approx(0.05)


def test_dqn_target_bootstraps_truncated_but_not_terminated_transition() -> None:
    torch = pytest.importorskip("torch")
    features = [0.0] * dqn_feature_dim(history_length=0)

    terminal_policy = _zero_dqn_with_target_next_q(torch)
    terminal_optimizer = terminal_policy.make_optimizer(
        DQNConfig(
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
        )
    )
    terminal_replay = DQNReplayBuffer(capacity=2)
    terminal_replay.append(
        DQNTransition(
            features=features,
            action_index=1,
            reward=0.0,
            next_features=features,
            done=True,
            terminated=True,
            truncated=False,
        )
    )

    terminal_before = _dqn_action_q(torch, terminal_policy, features, action_index=1)
    terminal_loss = live_learner_module._optimize_dqn(
        terminal_policy,
        terminal_optimizer,
        terminal_replay,
        config=DQNConfig(
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
        ),
    )
    terminal_after = _dqn_action_q(torch, terminal_policy, features, action_index=1)

    truncated_policy = _zero_dqn_with_target_next_q(torch)
    truncated_optimizer = truncated_policy.make_optimizer(
        DQNConfig(
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
        )
    )
    truncated_replay = DQNReplayBuffer(capacity=2)
    truncated_replay.append(
        DQNTransition(
            features=features,
            action_index=1,
            reward=0.0,
            next_features=features,
            done=True,
            terminated=False,
            truncated=True,
        )
    )

    truncated_before = _dqn_action_q(torch, truncated_policy, features, action_index=1)
    truncated_loss = live_learner_module._optimize_dqn(
        truncated_policy,
        truncated_optimizer,
        truncated_replay,
        config=DQNConfig(
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
        ),
    )
    truncated_after = _dqn_action_q(torch, truncated_policy, features, action_index=1)

    assert terminal_loss == pytest.approx(0.0)
    assert terminal_after == pytest.approx(terminal_before)
    assert truncated_loss is not None
    assert truncated_loss > 0.0
    assert truncated_after > truncated_before


def test_dqn_n_step_transitions_fold_discounted_rewards() -> None:
    features = [0.0] * dqn_feature_dim(history_length=0)
    transitions = [
        DQNTransition(
            features=features,
            action_index=1,
            reward=1.0,
            next_features=[1.0] * len(features),
            done=False,
            terminated=False,
        ),
        DQNTransition(
            features=[1.0] * len(features),
            action_index=0,
            reward=2.0,
            next_features=[2.0] * len(features),
            done=False,
            terminated=False,
        ),
        DQNTransition(
            features=[2.0] * len(features),
            action_index=1,
            reward=4.0,
            next_features=[3.0] * len(features),
            done=True,
            terminated=True,
        ),
    ]

    expanded = live_learner_module._expand_dqn_n_step_transitions(
        transitions,
        config=DQNConfig(
            gamma=0.5,
            n_step_return=3,
            batch_size=1,
            replay_capacity=4,
        ),
    )

    assert len(expanded) == 3
    assert expanded[0].reward == pytest.approx(3.0)
    assert expanded[0].discount == pytest.approx(0.125)
    assert expanded[0].next_features == [3.0] * len(features)
    assert expanded[0].terminated is True
    assert expanded[1].reward == pytest.approx(4.0)
    assert expanded[1].discount == pytest.approx(0.25)
    assert expanded[2].reward == pytest.approx(4.0)
    assert expanded[2].discount == pytest.approx(0.5)


def test_dqn_optimizer_uses_double_dqn_target_action_selection() -> None:
    torch = pytest.importorskip("torch")
    features = [0.0] * dqn_feature_dim(history_length=0)

    double_policy = _dqn_with_split_online_target_values(torch)
    double_optimizer = double_policy.make_optimizer(
        DQNConfig(
            gamma=1.0,
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
            double_dqn=True,
        )
    )
    double_replay = DQNReplayBuffer(capacity=2)
    double_replay.append(
        DQNTransition(
            features=features,
            action_index=1,
            reward=0.0,
            next_features=features,
            done=False,
            terminated=False,
            discount=1.0,
        )
    )

    standard_policy = _dqn_with_split_online_target_values(torch)
    standard_optimizer = standard_policy.make_optimizer(
        DQNConfig(
            gamma=1.0,
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
            double_dqn=False,
        )
    )
    standard_replay = DQNReplayBuffer(capacity=2)
    standard_replay.append(
        DQNTransition(
            features=features,
            action_index=1,
            reward=0.0,
            next_features=features,
            done=False,
            terminated=False,
            discount=1.0,
        )
    )

    double_loss = live_learner_module._optimize_dqn(
        double_policy,
        double_optimizer,
        double_replay,
        config=DQNConfig(
            gamma=1.0,
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
            double_dqn=True,
        ),
    )
    standard_loss = live_learner_module._optimize_dqn(
        standard_policy,
        standard_optimizer,
        standard_replay,
        config=DQNConfig(
            gamma=1.0,
            batch_size=1,
            replay_capacity=2,
            warmup_steps=1,
            max_grad_norm=None,
            double_dqn=False,
        ),
    )

    assert double_loss == pytest.approx(0.5)
    assert standard_loss == pytest.approx(8.5)


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
    assert attempt.replay_skip_reason == "aborted:tick_rewind_reset"
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


class MultiStepProgressClient(OneStepPressRewardClient):
    def __init__(self) -> None:
        super().__init__()
        self.tick = 0

    def reset_attempt(
        self,
        reason: str = "requested",
        *,
        max_observations: int = 600,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        self.tick = 0
        return super().reset_attempt(
            reason,
            max_observations=max_observations,
            diagnostics=diagnostics,
        )

    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        self.tick += 1
        input_down = any(event.kind == "press" for event in self.sent_events)
        return _observation(
            self.tick,
            percent=float(self.tick * 10),
            dead=False,
            completed=False,
            input_down=input_down,
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


def _bias_dqn_toward_idle(torch, policy: TinyLiveDQNNetwork) -> None:  # type: ignore[no-untyped-def]
    with torch.no_grad():
        for parameter in policy.q_network.parameters():
            parameter.zero_()
        policy.q_network[2].bias.copy_(
            torch.tensor([1.0, 0.0], dtype=torch.float32)
        )
    policy.sync_target()


def _zero_dqn_with_target_next_q(torch) -> TinyLiveDQNNetwork:  # type: ignore[no-untyped-def]
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=19),
        history_length=0,
    )
    with torch.no_grad():
        for parameter in policy.q_network.parameters():
            parameter.zero_()
        for parameter in policy.target_network.parameters():
            parameter.zero_()
        policy.target_network[2].bias.copy_(
            torch.tensor([1.0, 1.0], dtype=torch.float32)
        )
    return policy


def _dqn_with_split_online_target_values(torch) -> TinyLiveDQNNetwork:  # type: ignore[no-untyped-def]
    policy = TinyLiveDQNNetwork.from_encoder_config(
        NeuralPolicyConfig(hidden_size=4, seed=23),
        history_length=0,
    )
    with torch.no_grad():
        for parameter in policy.q_network.parameters():
            parameter.zero_()
        for parameter in policy.target_network.parameters():
            parameter.zero_()
        policy.q_network[2].bias.copy_(
            torch.tensor([0.0, 1.0], dtype=torch.float32)
        )
        policy.target_network[2].bias.copy_(
            torch.tensor([10.0, 2.0], dtype=torch.float32)
        )
    return policy


def _set_dqn_online_target_biases(
    torch,  # type: ignore[no-untyped-def]
    policy: TinyLiveDQNNetwork,
    *,
    online_biases: list[float],
    target_biases: list[float],
) -> None:
    with torch.no_grad():
        for parameter in policy.q_network.parameters():
            parameter.zero_()
        for parameter in policy.target_network.parameters():
            parameter.zero_()
        policy.q_network[2].bias.copy_(
            torch.tensor(online_biases, dtype=torch.float32)
        )
        policy.target_network[2].bias.copy_(
            torch.tensor(target_biases, dtype=torch.float32)
        )


def _dqn_action_q(
    torch,  # type: ignore[no-untyped-def]
    policy: TinyLiveDQNNetwork,
    features: list[float],
    *,
    action_index: int,
) -> float:
    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(0)
    with torch.no_grad():
        values = policy.q_network(feature_tensor)
    return float(values[0, action_index].detach().cpu().item())


def _dqn_target_action_q(
    torch,  # type: ignore[no-untyped-def]
    policy: TinyLiveDQNNetwork,
    features: list[float],
    *,
    action_index: int,
) -> float:
    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(0)
    with torch.no_grad():
        values = policy.target_network(feature_tensor)
    return float(values[0, action_index].detach().cpu().item())


def _dqn_networks_match(torch, policy: TinyLiveDQNNetwork) -> bool:  # type: ignore[no-untyped-def]
    return all(
        bool(torch.equal(online, target))
        for online, target in zip(
            policy.q_network.parameters(),
            policy.target_network.parameters(),
        )
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
