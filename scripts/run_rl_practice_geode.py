"""Run Phase A RL-practice attempts against the live Geode bridge."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_human_model import HumanProfile, profile_by_name
from gd_rl import (
    GeodeExecutorConfig,
    GeodePracticeExecutor,
    NoInputPolicy,
    PracticeRunConfig,
    PracticeRunner,
    RandomEventPolicy,
    RewardConfig,
    ScriptedEventPolicy,
)
from gd_rl.policy import PracticePolicy
from gd_trace import load_macro_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Phase A practice policy through the human model and live "
            "queued Geode replay."
        )
    )
    parser.add_argument("--level-id", required=True)
    parser.add_argument(
        "--policy",
        choices=("no-input", "scripted", "random"),
        default="scripted",
    )
    parser.add_argument(
        "--macro-json",
        type=Path,
        help="required for --policy scripted; intended macro before humanization",
    )
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--profile", default="Advanced")
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="load a complete HumanProfile JSON object instead of a built-in profile",
    )
    parser.add_argument(
        "--timing-reference",
        choices=("target", "decision"),
        default="target",
        help=(
            "target centers actual clicks on intended macro ticks; decision "
            "treats ticks as policy decision times"
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        help="first humanization seed; defaults to the selected profile seed",
    )
    parser.add_argument("--max-tick", type=int)
    parser.add_argument("--output-dir", type=Path)

    parser.add_argument("--random-max-events", type=int, default=12)
    parser.add_argument("--random-min-tick", type=int, default=0)
    parser.add_argument("--random-max-tick", type=int, default=1200)
    parser.add_argument("--random-min-spacing", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=0)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29430)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-observations", type=int, default=1200)
    parser.add_argument("--reset-wait-observations", type=int, default=600)
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--cbf", action="store_true")
    parser.add_argument("--physics-bypass", action="store_true")
    parser.add_argument("--success-percent", type=float, default=100.0)
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="stop each trace as soon as observation.percent reaches --success-percent",
    )
    parser.add_argument(
        "--stop-after-first-clear",
        action="store_true",
        help=(
            "end the whole practice run after the first clear; useful while "
            "Geometry Dash/Geode reset-after-clear is unstable"
        ),
    )
    parser.add_argument(
        "--post-terminal-delay-seconds",
        type=float,
        default=0.0,
        help="wait after death/success before the next reset",
    )
    parser.add_argument(
        "--start-guard-reset-retries",
        type=int,
        default=2,
        help="retry reset this many times when the fresh-start guard fails",
    )
    parser.add_argument(
        "--start-guard-retry-delay-seconds",
        type=float,
        default=0.2,
        help="delay between fresh-start reset retries",
    )
    parser.add_argument(
        "--require-start-percent-max",
        type=float,
        help="fail if a fresh tick-0 observation starts above this percent",
    )
    parser.add_argument(
        "--require-start-x-max",
        type=float,
        help="fail if a fresh tick-0 observation starts beyond this x position",
    )
    parser.add_argument(
        "--require-progress-tick",
        type=int,
        help="tick used with --require-progress-percent-min to verify the expected level",
    )
    parser.add_argument(
        "--require-progress-percent-min",
        type=float,
        help="minimum percent required at --require-progress-tick",
    )

    parser.add_argument(
        "--reward-style",
        choices=("progress", "picklegawd"),
        default="progress",
        help="reward formula to use for attempt and live-step rewards",
    )
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--best-progress-bonus-scale", type=float, default=0.5)
    parser.add_argument("--section-size-percent", type=float, default=10.0)
    parser.add_argument("--section-survival-bonus", type=float, default=0.25)
    parser.add_argument("--clear-bonus", type=float, default=100.0)
    parser.add_argument("--death-penalty", type=float, default=10.0)
    parser.add_argument("--excessive-input-free-events", type=int, default=0)
    parser.add_argument("--excessive-input-penalty", type=float, default=0.0)
    parser.add_argument("--default-reward", type=float, default=0.01)
    parser.add_argument("--jump-punishment", type=float, default=0.0)
    parser.add_argument("--checkpoint-reward", type=float, default=0.0)
    parser.add_argument("--checkpoint-size-percent", type=float, default=3.0)

    args = parser.parse_args(argv)

    try:
        _validate_args(args, parser)
        profile = _load_profile(args.profile, args.profile_json)
        policy = _build_policy(args)
        reward_config = _build_reward_config(args)
        geode_config = _build_geode_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("artifacts") / f"rl_practice_geode_{run_id}"
    practice_config = PracticeRunConfig(
        level_id=args.level_id,
        attempts=args.attempts,
        output_dir=output_dir,
        max_tick=args.max_tick,
        success_percent=args.success_percent,
        stop_after_first_clear=args.stop_after_first_clear,
        base_seed=args.base_seed,
        timing_reference=args.timing_reference,
        reward_config=reward_config,
        metadata={
            "run_id": run_id,
            "policy": args.policy,
            "macro_json": str(args.macro_json) if args.macro_json else None,
            "executor": "geode_queued_macro",
        },
    )

    try:
        with GeodePracticeExecutor(geode_config) as executor:
            summary = PracticeRunner(
                policy=policy,
                executor=executor,
                human_profile=profile,
                config=practice_config,
            ).run()
    except (OSError, TimeoutError, EOFError) as exc:
        print(f"error: bridge communication failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(output_dir / "summary.json"),
                "summary": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if args.max_tick is not None and args.max_tick < 0:
        parser.error("--max-tick must be non-negative")
    if args.policy == "scripted" and args.macro_json is None:
        parser.error("--macro-json is required when --policy scripted")
    if args.policy != "scripted" and args.macro_json is not None:
        parser.error("--macro-json can only be used with --policy scripted")
    if args.random_max_events < 0:
        parser.error("--random-max-events must be non-negative")
    if args.random_min_tick < 0:
        parser.error("--random-min-tick must be non-negative")
    if args.random_max_tick < args.random_min_tick:
        parser.error("--random-max-tick must be >= --random-min-tick")
    if args.random_min_spacing < 0:
        parser.error("--random-min-spacing must be non-negative")


def _load_profile(profile_name: str, profile_json: Path | None) -> HumanProfile:
    if profile_json is None:
        return profile_by_name(profile_name)

    with profile_json.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("--profile-json must contain a JSON object")
    try:
        return HumanProfile(**data)
    except TypeError as exc:
        raise ValueError(f"invalid HumanProfile JSON: {exc}") from exc


def _build_policy(args: argparse.Namespace) -> PracticePolicy:
    if args.policy == "no-input":
        return NoInputPolicy()
    if args.policy == "random":
        return RandomEventPolicy(
            max_events=args.random_max_events,
            min_tick=args.random_min_tick,
            max_tick=args.random_max_tick,
            min_spacing=args.random_min_spacing,
            seed=args.random_seed,
        )

    if args.macro_json is None:
        raise ValueError("--macro-json is required when --policy scripted")
    macro = load_macro_json(args.macro_json)
    return ScriptedEventPolicy(
        events=macro.events,
        name=f"scripted_events:{args.macro_json.name}",
    )


def _build_reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        reward_style=args.reward_style,
        success_percent=args.success_percent,
        progress_scale=args.progress_scale,
        best_progress_bonus_scale=args.best_progress_bonus_scale,
        section_size_percent=args.section_size_percent,
        section_survival_bonus=args.section_survival_bonus,
        clear_bonus=args.clear_bonus,
        death_penalty=args.death_penalty,
        excessive_input_free_events=args.excessive_input_free_events,
        excessive_input_penalty=args.excessive_input_penalty,
        default_reward=args.default_reward,
        jump_punishment=args.jump_punishment,
        checkpoint_reward=args.checkpoint_reward,
        checkpoint_size_percent=args.checkpoint_size_percent,
    )


def _build_geode_config(args: argparse.Namespace) -> GeodeExecutorConfig:
    return GeodeExecutorConfig(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        max_observations=args.max_observations,
        reset_wait_observations=args.reset_wait_observations,
        fps=args.fps,
        cbf=args.cbf,
        physics_bypass=args.physics_bypass,
        success_percent=args.success_percent,
        stop_on_success=args.stop_on_success,
        post_terminal_delay_seconds=args.post_terminal_delay_seconds,
        start_guard_reset_retries=args.start_guard_reset_retries,
        start_guard_retry_delay_seconds=args.start_guard_retry_delay_seconds,
        require_start_percent_max=args.require_start_percent_max,
        require_start_x_max=args.require_start_x_max,
        require_progress_tick=args.require_progress_tick,
        require_progress_percent_min=args.require_progress_percent_min,
    )


if __name__ == "__main__":
    raise SystemExit(main())
