"""Run Phase C timing-window search against the live Geode bridge."""

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

from gd_rl import (
    CandidateEvaluation,
    GeodePracticeExecutor,
    PracticeRunConfig,
    PracticeRunner,
    ScriptedEventPolicy,
    TimingCandidate,
    TimingSearchConfig,
    load_timing_windows_json,
    run_timing_search,
)
from gd_trace import Macro, save_macro_json
from scripts.run_rl_practice_geode import (
    _build_geode_config,
    _build_reward_config,
    _load_profile,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact Phase C CEM/random timing search through the human "
            "model and live queued Geode replay."
        )
    )
    parser.add_argument("--level-id", required=True)
    parser.add_argument(
        "--window-json",
        type=Path,
        required=True,
        help="JSON event timing windows for intended policy events",
    )
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--attempts-per-candidate", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--min-event-spacing", type=int, default=1)
    parser.add_argument("--min-std-tick", type=float, default=2.0)
    parser.add_argument("--max-std-tick", type=float, default=80.0)
    parser.add_argument("--update-smoothing", type=float, default=0.25)
    parser.add_argument(
        "--score-metric",
        choices=(
            "average_reward",
            "total_reward",
            "average_final_percent",
            "best_percent",
            "clear_rate",
        ),
        default="average_reward",
    )
    parser.add_argument(
        "--fail-on-candidate-error",
        action="store_true",
        help="abort the whole search if one candidate run fails",
    )
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
        help="target centers actual clicks on intended ticks",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        help="first humanization seed for each candidate evaluation",
    )
    parser.add_argument("--max-tick", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stop-after-first-clear",
        action="store_true",
        help="stop a candidate evaluation after its first clear",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29430)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-observations", type=int, default=1800)
    parser.add_argument("--reset-wait-observations", type=int, default=600)
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--cbf", action="store_true")
    parser.add_argument("--physics-bypass", action="store_true")
    parser.add_argument("--success-percent", type=float, default=100.0)
    parser.add_argument(
        "--stop-on-success",
        dest="stop_on_success",
        action="store_true",
        default=True,
        help="stop each trace as soon as success percent is reached",
    )
    parser.add_argument(
        "--no-stop-on-success",
        dest="stop_on_success",
        action="store_false",
    )
    parser.add_argument("--post-terminal-delay-seconds", type=float, default=5.0)
    parser.add_argument("--start-guard-reset-retries", type=int, default=3)
    parser.add_argument("--start-guard-retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--require-start-percent-max", type=float)
    parser.add_argument("--require-start-x-max", type=float)
    parser.add_argument("--require-progress-tick", type=int)
    parser.add_argument("--require-progress-percent-min", type=float)

    parser.add_argument(
        "--reward-style",
        choices=("progress", "picklegawd"),
        default="progress",
        help="reward formula to use for candidate scoring",
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
        event_windows, window_metadata = load_timing_windows_json(args.window_json)
        profile = _load_profile(args.profile, args.profile_json)
        reward_config = _build_reward_config(args)
        geode_config = _build_geode_config(args)
        search_config = TimingSearchConfig(
            level_id=args.level_id,
            event_windows=event_windows,
            generations=args.generations,
            population_size=args.population_size,
            elite_fraction=args.elite_fraction,
            attempts_per_candidate=args.attempts_per_candidate,
            seed=args.search_seed,
            min_event_spacing=args.min_event_spacing,
            min_std_tick=args.min_std_tick,
            max_std_tick=args.max_std_tick,
            update_smoothing=args.update_smoothing,
            continue_on_candidate_error=not args.fail_on_candidate_error,
            metadata={
                "window_json": str(args.window_json),
                "window_metadata": window_metadata,
                "score_metric": args.score_metric,
            },
        )
    except ValueError as exc:
        parser.error(str(exc))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("artifacts") / f"rl_timing_search_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with GeodePracticeExecutor(geode_config) as executor:
        result = run_timing_search(
            config=search_config,
            evaluator=_build_live_evaluator(
                args=args,
                executor=executor,
                output_dir=output_dir,
                profile=profile,
                reward_config=reward_config,
            ),
        )

    summary_path = output_dir / "search_summary.json"
    _write_json(result.to_dict(), summary_path)
    final_windows_path = output_dir / "final_windows.json"
    _write_json(
        {
            "metadata": {
                "kind": "phase_c_final_timing_windows",
                "level_id": args.level_id,
                "source_window_json": str(args.window_json),
                "search_summary_json": str(summary_path),
            },
            "events": [window.to_dict() for window in result.final_windows],
        },
        final_windows_path,
    )
    best_macro_path = None
    if result.best_evaluation is not None:
        best_macro_path = output_dir / "best_intended_macro.json"
        save_macro_json(
            Macro(
                events=result.best_evaluation.candidate.events,
                metadata={
                    "kind": "phase_c_best_intended_candidate",
                    "level_id": args.level_id,
                    "candidate_id": result.best_evaluation.candidate.candidate_id,
                    "score": result.best_evaluation.score,
                    "score_metric": args.score_metric,
                },
            ),
            best_macro_path,
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "search_summary_json": str(summary_path),
                "final_windows_json": str(final_windows_path),
                "best_macro_json": str(best_macro_path) if best_macro_path else None,
                "best_score": (
                    result.best_evaluation.score
                    if result.best_evaluation is not None
                    else None
                ),
                "best_candidate": (
                    result.best_evaluation.candidate.to_dict()
                    if result.best_evaluation is not None
                    else None
                ),
                "final_windows": [
                    window.to_dict() for window in result.final_windows
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.best_evaluation is not None else 1


def _build_live_evaluator(
    *,
    args: argparse.Namespace,
    executor: GeodePracticeExecutor,
    output_dir: Path,
    profile: object,
    reward_config: object,
):
    def evaluate(candidate: TimingCandidate) -> CandidateEvaluation:
        candidate_dir = (
            output_dir
            / f"generation_{candidate.generation_index:03d}"
            / f"candidate_{candidate.population_index:03d}"
        )
        policy = ScriptedEventPolicy(
            events=candidate.events,
            name=f"timing_search:{candidate.candidate_id}",
        )
        practice_config = PracticeRunConfig(
            level_id=args.level_id,
            attempts=args.attempts_per_candidate,
            output_dir=candidate_dir,
            max_tick=args.max_tick,
            success_percent=args.success_percent,
            stop_after_first_clear=args.stop_after_first_clear,
            base_seed=args.base_seed,
            timing_reference=args.timing_reference,
            reward_config=reward_config,
            metadata={
                "executor": "geode_queued_macro",
                "policy": "timing_search",
                "candidate": candidate.to_dict(),
                "score_metric": args.score_metric,
            },
        )
        summary = PracticeRunner(
            policy=policy,
            executor=executor,
            human_profile=profile,
            config=practice_config,
        ).run()
        score = _score_summary(summary, args.score_metric)
        print(
            (
                f"{candidate.candidate_id}: score={score:.6g} "
                f"clears={summary.clears}/{summary.attempt_count} "
                f"best={summary.best_percent:.3f} "
                f"avg_final={summary.average_final_percent:.3f} "
                f"ticks={candidate.ticks}"
            ),
            file=sys.stderr,
            flush=True,
        )
        return CandidateEvaluation(
            candidate=candidate,
            score=score,
            summary=summary.to_dict(),
            output_dir=str(candidate_dir),
        )

    return evaluate


def _score_summary(summary: object, metric: str) -> float:
    attempt_count = getattr(summary, "attempt_count")
    if metric == "average_reward":
        return float(getattr(summary, "total_reward")) / max(attempt_count, 1)
    if metric == "total_reward":
        return float(getattr(summary, "total_reward"))
    if metric == "average_final_percent":
        return float(getattr(summary, "average_final_percent"))
    if metric == "best_percent":
        return float(getattr(summary, "best_percent"))
    if metric == "clear_rate":
        return float(getattr(summary, "clear_rate"))
    raise ValueError(f"unknown score metric {metric!r}")


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
