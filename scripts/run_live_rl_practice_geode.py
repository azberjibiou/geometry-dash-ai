"""Run a tiny live neural RL-practice smoke against the Geode bridge."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_env import GeometryDashClient
from gd_rl import (
    ActorCriticConfig,
    DQNConfig,
    LiveLearnerError,
    LiveObservationEncoderConfig,
    LivePracticeEnv,
    LivePracticeEnvConfig,
    NeuralPolicyConfig,
    ReinforceConfig,
    RewardConfig,
    TinyLiveActorCriticNetwork,
    TinyLiveDQNNetwork,
    TinyLivePolicyNetwork,
    run_actor_critic_training,
    run_dqn_training,
    run_reinforce_training,
)
from gd_rl.live_env import LiveGeodeClientLike
from scripts.run_rl_practice_geode import _build_reward_config, _load_profile


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], LiveGeodeClientLike] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a guarded, tiny live neural practice smoke through the "
            "human model and Geode bridge."
        )
    )
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--profile", default="Advanced")
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="load a complete HumanProfile JSON object instead of a built-in profile",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="print only run-level learning metrics; full summary remains on disk",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        help="first humanization seed; defaults to the selected profile seed",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="write the final neural policy checkpoint after training",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="load a compatible neural policy checkpoint before training",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29430)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--reset-wait-observations", type=int, default=600)
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--cbf", action="store_true")
    parser.add_argument("--physics-bypass", action="store_true")
    parser.add_argument("--success-percent", type=float, default=100.0)
    parser.add_argument("--action-horizon-ticks", type=int, default=1)
    parser.add_argument("--observation-buffer-size", type=int)
    parser.add_argument("--post-terminal-delay-seconds", type=float, default=5.0)
    parser.add_argument("--start-guard-reset-retries", type=int, default=3)
    parser.add_argument("--start-guard-retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--require-start-percent-max", type=float, default=2.0)
    parser.add_argument("--require-start-x-max", type=float, default=50.0)

    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoder-max-tick",
        type=int,
        help="feature scale for tick; defaults to --max-steps",
    )
    parser.add_argument("--encoder-x-scale", type=float, default=1000.0)
    parser.add_argument("--encoder-y-scale", type=float, default=500.0)
    parser.add_argument("--encoder-velocity-scale", type=float, default=20.0)
    parser.add_argument("--encoder-rotation-scale", type=float, default=360.0)

    parser.add_argument(
        "--algorithm",
        choices=("a2c", "reinforce", "dqn"),
        default="a2c",
        help="tiny live learner to run; defaults to the Phase D actor-critic path",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--no-grad-clip",
        action="store_true",
        help="disable gradient clipping for this tiny smoke run",
    )
    parser.add_argument("--normalize-returns", action="store_true")
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument("--learner-seed", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--death-local-window", type=int, default=24)
    parser.add_argument("--death-local-penalty", type=float, default=1.0)
    parser.add_argument("--input-rate-penalty", type=float, default=0.0)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1000)
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-replay-capacity", type=int, default=2048)
    parser.add_argument("--dqn-warmup-steps", type=int, default=32)
    parser.add_argument("--dqn-target-update-interval", type=int, default=100)
    parser.add_argument(
        "--min-dwell-ticks",
        type=int,
        default=4,
        help=(
            "minimum ticks to keep an idle/hold state before allowing the "
            "opposite edge; 4 matches the PickleGawd-style step repeat"
        ),
    )

    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--best-progress-bonus-scale", type=float, default=0.5)
    parser.add_argument("--section-size-percent", type=float, default=10.0)
    parser.add_argument("--section-survival-bonus", type=float, default=0.25)
    parser.add_argument("--clear-bonus", type=float, default=100.0)
    parser.add_argument("--death-penalty", type=float, default=10.0)
    parser.add_argument("--excessive-input-free-events", type=int, default=0)
    parser.add_argument("--excessive-input-penalty", type=float, default=0.0)

    args = parser.parse_args(argv)

    try:
        _validate_args(args, parser)
        profile = _load_profile(args.profile, args.profile_json)
        reward_config = _build_reward_config(args)
        reinforce_config = _build_reinforce_config(args)
        actor_critic_config = _build_actor_critic_config(args)
        dqn_config = _build_dqn_config(args)
        policy_config = _build_policy_config(args)
    except (ValueError, LiveLearnerError) as exc:
        parser.error(str(exc))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("artifacts") / f"live_rl_practice_{run_id}"
    try:
        env_config = _build_env_config(
            args,
            output_dir=output_dir,
            reward_config=reward_config,
            run_id=run_id,
        )
    except ValueError as exc:
        parser.error(str(exc))
    bridge_client_factory = client_factory or _build_client_factory(args)

    summary_path = output_dir / "training_summary.json"
    try:
        with LivePracticeEnv(
            config=env_config,
            human_profile=profile,
            client_factory=bridge_client_factory,
        ) as env:
            if args.algorithm == "reinforce":
                policy = TinyLivePolicyNetwork.from_encoder_config(policy_config)
                _load_policy_checkpoint(policy, args.load_checkpoint)
                summary = run_reinforce_training(
                    env,
                    policy,
                    config=reinforce_config,
                    summary_path=summary_path,
                )
            elif args.algorithm == "dqn":
                policy = TinyLiveDQNNetwork.from_encoder_config(
                    policy_config,
                    history_length=dqn_config.history_length,
                )
                _load_policy_checkpoint(policy, args.load_checkpoint)
                summary = run_dqn_training(
                    env,
                    policy,
                    config=dqn_config,
                    summary_path=summary_path,
                )
            else:
                policy = TinyLiveActorCriticNetwork.from_encoder_config(
                    policy_config,
                    history_length=actor_critic_config.history_length,
                )
                _load_policy_checkpoint(policy, args.load_checkpoint)
                summary = run_actor_critic_training(
                    env,
                    policy,
                    config=actor_critic_config,
                    summary_path=summary_path,
                )
    except LiveLearnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, TimeoutError, EOFError) as exc:
        print(f"error: bridge communication failed: {exc}", file=sys.stderr)
        return 1

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None:
        _save_policy_checkpoint(
            policy,
            checkpoint_path,
            algorithm=args.algorithm,
            policy_config=policy_config,
            summary_path=summary_path,
        )

    payload = {
        "output_dir": str(output_dir),
        "training_summary_json": str(summary_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "summary": summary.to_dict(),
    }
    if args.compact_output:
        payload = _compact_training_payload(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if args.port <= 0:
        parser.error("--port must be positive")
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.reset_wait_observations <= 0:
        parser.error("--reset-wait-observations must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.action_horizon_ticks < 0:
        parser.error("--action-horizon-ticks must be non-negative")
    if args.observation_buffer_size is not None and args.observation_buffer_size <= 0:
        parser.error("--observation-buffer-size must be positive")
    if args.history_length < 0:
        parser.error("--history-length must be non-negative")
    if args.death_local_window < 0:
        parser.error("--death-local-window must be non-negative")
    if args.min_dwell_ticks < 0:
        parser.error("--min-dwell-ticks must be non-negative")


def _build_env_config(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    reward_config: RewardConfig,
    run_id: str,
) -> LivePracticeEnvConfig:
    return LivePracticeEnvConfig(
        level_id=args.level_id,
        output_dir=output_dir,
        max_steps=args.max_steps,
        reset_wait_observations=args.reset_wait_observations,
        fps=args.fps,
        cbf=args.cbf,
        physics_bypass=args.physics_bypass,
        success_percent=args.success_percent,
        base_seed=args.base_seed,
        action_horizon_ticks=args.action_horizon_ticks,
        observation_buffer_size=args.observation_buffer_size,
        post_terminal_delay_seconds=args.post_terminal_delay_seconds,
        start_guard_reset_retries=args.start_guard_reset_retries,
        start_guard_retry_delay_seconds=args.start_guard_retry_delay_seconds,
        require_start_percent_max=args.require_start_percent_max,
        require_start_x_max=args.require_start_x_max,
        reward_config=reward_config,
        metadata={
            "run_id": run_id,
            "executor": "geode_live_step",
            "learner": _learner_name(args.algorithm),
            "algorithm": args.algorithm,
            "script": "run_live_rl_practice_geode.py",
        },
    )


def _build_policy_config(args: argparse.Namespace) -> NeuralPolicyConfig:
    encoder = LiveObservationEncoderConfig(
        max_tick=args.encoder_max_tick or args.max_steps,
        x_scale=args.encoder_x_scale,
        y_scale=args.encoder_y_scale,
        velocity_scale=args.encoder_velocity_scale,
        rotation_scale=args.encoder_rotation_scale,
    )
    return NeuralPolicyConfig(
        hidden_size=args.hidden_size,
        seed=args.policy_seed,
        device=args.device,
        encoder=encoder,
    )


def _build_reinforce_config(args: argparse.Namespace) -> ReinforceConfig:
    return ReinforceConfig(
        attempts=args.attempts,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        entropy_bonus=args.entropy_bonus,
        max_grad_norm=None if args.no_grad_clip else args.max_grad_norm,
        normalize_returns=args.normalize_returns,
        deterministic_actions=args.deterministic_actions,
        seed=args.learner_seed,
        min_dwell_ticks=args.min_dwell_ticks,
    )


def _build_actor_critic_config(args: argparse.Namespace) -> ActorCriticConfig:
    return ActorCriticConfig(
        attempts=args.attempts,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        entropy_bonus=args.entropy_bonus,
        value_loss_weight=args.value_loss_weight,
        max_grad_norm=None if args.no_grad_clip else args.max_grad_norm,
        normalize_advantages=args.normalize_advantages,
        deterministic_actions=args.deterministic_actions,
        seed=args.learner_seed,
        history_length=args.history_length,
        death_local_window=args.death_local_window,
        death_local_penalty=args.death_local_penalty,
        input_rate_penalty=args.input_rate_penalty,
        min_dwell_ticks=args.min_dwell_ticks,
    )


def _build_dqn_config(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        attempts=args.attempts,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        batch_size=args.dqn_batch_size,
        replay_capacity=args.dqn_replay_capacity,
        warmup_steps=args.dqn_warmup_steps,
        target_update_interval=args.dqn_target_update_interval,
        max_grad_norm=None if args.no_grad_clip else args.max_grad_norm,
        deterministic_actions=args.deterministic_actions,
        seed=args.learner_seed,
        history_length=args.history_length,
        input_rate_penalty=args.input_rate_penalty,
        min_dwell_ticks=args.min_dwell_ticks,
    )


def _learner_name(algorithm: str) -> str:
    if algorithm == "a2c":
        return "tiny_actor_critic"
    if algorithm == "dqn":
        return "tiny_dqn"
    return "tiny_reinforce"


def _compact_training_payload(payload: dict[str, object]) -> dict[str, object]:
    summary = payload["summary"]
    if not isinstance(summary, dict):
        return payload
    attempts = summary.get("attempts", [])
    compact_attempts: list[dict[str, object]] = []
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            result = attempt.get("attempt_result", {})
            if not isinstance(result, dict):
                result = {}
            compact_attempts.append(
                {
                    "attempt": attempt.get("attempt_index"),
                    "cleared": result.get("cleared"),
                    "death_tick": result.get("death_tick"),
                    "final_percent": result.get("final_percent"),
                    "best_percent": result.get("best_percent"),
                    "total_reward": result.get("total_reward"),
                    "step_count": attempt.get("step_count"),
                    "action_counts": attempt.get("action_counts"),
                    "intent_counts": attempt.get("intent_counts"),
                    "executed_event_count": result.get("executed_event_count"),
                    "dropped_event_count": result.get("dropped_event_count"),
                    "update_count": attempt.get("update_count"),
                    "mean_loss": attempt.get("mean_loss"),
                    "epsilon_start": attempt.get("epsilon_start"),
                    "epsilon_end": attempt.get("epsilon_end"),
                }
            )

    best_percent_values = [
        float(attempt["best_percent"])
        for attempt in compact_attempts
        if attempt.get("best_percent") is not None
    ]
    clear_count = sum(
        1 for attempt in compact_attempts if bool(attempt.get("cleared"))
    )
    return {
        "output_dir": payload["output_dir"],
        "training_summary_json": payload["training_summary_json"],
        "checkpoint_path": payload.get("checkpoint_path"),
        "attempt_count": summary.get("attempt_count"),
        "clear_count": clear_count,
        "best_percent_overall": (
            max(best_percent_values) if best_percent_values else None
        ),
        "attempts": compact_attempts,
    }


def _save_policy_checkpoint(
    policy: object,
    path: Path,
    *,
    algorithm: str,
    policy_config: NeuralPolicyConfig,
    summary_path: Path,
) -> None:
    torch = getattr(policy, "torch")
    checkpoint = {
        "algorithm": algorithm,
        "policy_config": {
            "hidden_size": policy_config.hidden_size,
            "seed": policy_config.seed,
            "device": policy_config.device,
            "encoder": {
                "max_tick": policy_config.encoder.max_tick,
                "x_scale": policy_config.encoder.x_scale,
                "y_scale": policy_config.encoder.y_scale,
                "velocity_scale": policy_config.encoder.velocity_scale,
                "rotation_scale": policy_config.encoder.rotation_scale,
            },
        },
        "summary_path": str(summary_path),
    }
    if hasattr(policy, "q_network"):
        checkpoint["q_network"] = policy.q_network.state_dict()
        checkpoint["target_network"] = policy.target_network.state_dict()
        checkpoint["history_length"] = policy.history_length
    elif hasattr(policy, "shared"):
        checkpoint["shared"] = policy.shared.state_dict()
        checkpoint["actor_head"] = policy.actor_head.state_dict()
        checkpoint["value_head"] = policy.value_head.state_dict()
        checkpoint["history_length"] = policy.history_length
    elif hasattr(policy, "model"):
        checkpoint["model"] = policy.model.state_dict()
    else:
        raise LiveLearnerError("unsupported policy type for checkpoint saving")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load_policy_checkpoint(policy: object, path: Path | None) -> None:
    if path is None:
        return
    torch = getattr(policy, "torch")
    checkpoint = torch.load(path, map_location=getattr(policy, "device"))
    if hasattr(policy, "q_network"):
        policy.q_network.load_state_dict(checkpoint["q_network"])
        if "target_network" in checkpoint:
            policy.target_network.load_state_dict(checkpoint["target_network"])
        else:
            policy.sync_target()
    elif hasattr(policy, "shared"):
        policy.shared.load_state_dict(checkpoint["shared"])
        policy.actor_head.load_state_dict(checkpoint["actor_head"])
        policy.value_head.load_state_dict(checkpoint["value_head"])
    elif hasattr(policy, "model"):
        policy.model.load_state_dict(checkpoint["model"])
    else:
        raise LiveLearnerError("unsupported policy type for checkpoint loading")


def _build_client_factory(
    args: argparse.Namespace,
) -> Callable[[], GeometryDashClient]:
    return lambda: GeometryDashClient(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
