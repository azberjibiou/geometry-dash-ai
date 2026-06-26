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
    LiveLearnerError,
    LiveObservationEncoderConfig,
    LivePracticeEnv,
    LivePracticeEnvConfig,
    NeuralPolicyConfig,
    ReinforceConfig,
    RewardConfig,
    TinyLiveActorCriticNetwork,
    TinyLivePolicyNetwork,
    run_actor_critic_training,
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
        "--base-seed",
        type=int,
        help="first humanization seed; defaults to the selected profile seed",
    )
    parser.add_argument("--output-dir", type=Path)

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
        choices=("a2c", "reinforce"),
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
                summary = run_reinforce_training(
                    env,
                    policy,
                    config=reinforce_config,
                    summary_path=summary_path,
                )
            else:
                policy = TinyLiveActorCriticNetwork.from_encoder_config(
                    policy_config,
                    history_length=actor_critic_config.history_length,
                )
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

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "training_summary_json": str(summary_path),
                "summary": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
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
            "learner": (
                "tiny_actor_critic"
                if args.algorithm == "a2c"
                else "tiny_reinforce"
            ),
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
