"""Run humanized macro attempts against the live Geode bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_env import BridgeObservation, GeometryDashClient
from gd_human_model import HumanProfile, humanize_macro, profile_by_name
from gd_trace import (
    TraceRow,
    load_macro_json,
    save_macro_json,
    summarize_humanized_attempts,
)
from gd_trace.compare_trace import first_death_tick


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Humanize an intended macro with a HumanProfile and replay each "
            "attempt through the live queued Geode bridge."
        )
    )
    parser.add_argument("macro_json")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--profile", default="Advanced")
    parser.add_argument(
        "--timing-reference",
        choices=("target", "decision"),
        default="target",
        help=(
            "target centers actual clicks on macro ticks; decision treats macro "
            "ticks as policy-decision times and applies visual+motor delay"
        ),
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="load a complete HumanProfile JSON object instead of a built-in profile",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        help="first attempt seed; defaults to the selected profile's random_seed",
    )
    parser.add_argument("--max-observations", type=int, default=1200)
    parser.add_argument("--reset-wait-observations", type=int, default=600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29430)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
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
        "--post-terminal-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "wait after death/success before the next reset; useful after clears "
            "so Geometry Dash can settle its completion transition"
        ),
    )
    parser.add_argument(
        "--require-start-percent-max",
        type=float,
        help="fail if a trial's fresh tick-0 observation starts above this percent",
    )
    parser.add_argument(
        "--require-start-x-max",
        type=float,
        help="fail if a trial's fresh tick-0 observation starts beyond this x position",
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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    try:
        _validate_args(args, parser)
        profile = _load_profile(args.profile, args.profile_json)
    except ValueError as exc:
        parser.error(str(exc))

    macro_path = Path(args.macro_json)
    intended_macro = load_macro_json(macro_path)
    base_seed = profile.random_seed if args.base_seed is None else args.base_seed
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("artifacts") / f"humanized_macro_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    traces: list[list[TraceRow]] = []
    event_results_by_attempt = []
    trial_results = []

    with GeometryDashClient(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        for attempt_index in range(args.attempts):
            attempt_number = attempt_index + 1
            attempt_seed = base_seed + attempt_index
            humanized = humanize_macro(
                intended_macro,
                profile,
                seed=attempt_seed,
                attempt_index=attempt_number,
                timing_reference=args.timing_reference,
            )
            actual_macro = humanized.to_macro(
                metadata={
                    "source_macro_path": str(macro_path),
                    "run_id": run_id,
                }
            )
            actual_macro_path = output_dir / f"actual_macro_{attempt_number:03d}.json"
            humanization_path = output_dir / f"humanization_{attempt_number:03d}.json"
            trace_path = output_dir / f"attempt_{attempt_number:03d}.jsonl"
            save_macro_json(actual_macro, actual_macro_path)
            _write_json(humanized.to_dict(), humanization_path)

            diagnostics = []
            client.load_macro(actual_macro.events, metadata=actual_macro.metadata)
            initial_observation = client.reset_attempt(
                f"humanized_macro_attempt_{attempt_number}",
                max_observations=args.reset_wait_observations,
                diagnostics=diagnostics,
            )
            try:
                _validate_start_observation(
                    initial_observation,
                    attempt_number=attempt_number,
                    require_percent_max=args.require_start_percent_max,
                    require_x_max=args.require_start_x_max,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1

            rows = client.run_loaded_macro(
                max_observations=args.max_observations,
                fps=args.fps,
                cbf=args.cbf,
                physics_bypass=args.physics_bypass,
                trace_path=trace_path,
                initial_observation=initial_observation,
                diagnostics=diagnostics,
                stop_percent=args.success_percent if args.stop_on_success else None,
            )
            if (
                args.post_terminal_delay_seconds > 0.0
                and _is_terminal_trace(rows, success_percent=args.success_percent)
            ):
                time.sleep(args.post_terminal_delay_seconds)
            try:
                _validate_trace_progress(
                    rows,
                    attempt_number=attempt_number,
                    require_tick=args.require_progress_tick,
                    require_percent_min=args.require_progress_percent_min,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1

            traces.append(rows)
            event_results_by_attempt.append(humanized.event_results)
            diagnostics_document = [
                diagnostic.to_dict() for diagnostic in diagnostics
            ]
            trial_results.append(
                {
                    "attempt": attempt_number,
                    "seed": attempt_seed,
                    "actual_macro_path": str(actual_macro_path),
                    "humanization_path": str(humanization_path),
                    "trace_path": str(trace_path),
                    "rows": len(rows),
                    "first_tick": rows[0].tick if rows else None,
                    "start_percent": rows[0].percent if rows else None,
                    "start_x": rows[0].x if rows else None,
                    "last_tick": rows[-1].tick if rows else None,
                    "final_percent": rows[-1].percent if rows else 0.0,
                    "death_tick": first_death_tick(rows),
                    "intended_event_count": len(humanized.event_results),
                    "actual_event_count": len(humanized.actual_events),
                    "missed_event_count": humanized.missed_event_count,
                    "diagnostics": diagnostics_document,
                }
            )

    run_summary = summarize_humanized_attempts(
        traces,
        event_results_by_attempt,
        success_percent=args.success_percent,
    )
    summary_path = output_dir / "summary.json"
    summary_document = {
        "run_id": run_id,
        "macro_path": str(macro_path),
        "macro_metadata": intended_macro.metadata,
        "profile": asdict(profile),
        "settings": {
            "host": args.host,
            "port": args.port,
            "fps": args.fps,
            "cbf": args.cbf,
            "physics_bypass": args.physics_bypass,
            "attempts": args.attempts,
            "base_seed": base_seed,
            "timing_reference": args.timing_reference,
            "max_observations": args.max_observations,
            "success_percent": args.success_percent,
            "stop_on_success": args.stop_on_success,
            "post_terminal_delay_seconds": args.post_terminal_delay_seconds,
            "require_start_percent_max": args.require_start_percent_max,
            "require_start_x_max": args.require_start_x_max,
            "require_progress_tick": args.require_progress_tick,
            "require_progress_percent_min": args.require_progress_percent_min,
        },
        "attempts": trial_results,
        "summary": run_summary.to_dict(),
    }
    _write_json(summary_document, summary_path)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(summary_path),
                "summary": run_summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if args.max_observations <= 0:
        parser.error("--max-observations must be positive")
    if args.reset_wait_observations <= 0:
        parser.error("--reset-wait-observations must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.post_terminal_delay_seconds < 0.0:
        parser.error("--post-terminal-delay-seconds must be non-negative")
    if not 0.0 <= args.success_percent <= 100.0:
        parser.error("--success-percent must be between 0 and 100")
    if (
        args.require_start_percent_max is not None
        and not 0.0 <= args.require_start_percent_max <= 100.0
    ):
        parser.error("--require-start-percent-max must be between 0 and 100")
    if args.require_start_x_max is not None and args.require_start_x_max < 0.0:
        parser.error("--require-start-x-max must be non-negative")
    if args.require_progress_tick is not None and args.require_progress_tick < 0:
        parser.error("--require-progress-tick must be non-negative")
    if (
        args.require_progress_percent_min is not None
        and not 0.0 <= args.require_progress_percent_min <= 100.0
    ):
        parser.error("--require-progress-percent-min must be between 0 and 100")
    if (args.require_progress_tick is None) != (
        args.require_progress_percent_min is None
    ):
        parser.error(
            "--require-progress-tick and --require-progress-percent-min must be used together"
        )


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


def _validate_start_observation(
    observation: BridgeObservation,
    *,
    attempt_number: int,
    require_percent_max: float | None,
    require_x_max: float | None,
) -> None:
    failures = []
    if (
        require_percent_max is not None
        and observation.percent > require_percent_max
    ):
        failures.append(
            f"percent {observation.percent:.3f} > {require_percent_max:.3f}"
        )
    if require_x_max is not None and observation.x > require_x_max:
        failures.append(f"x {observation.x:.3f} > {require_x_max:.3f}")
    if failures:
        raise ValueError(
            f"attempt {attempt_number} fresh start check failed: "
            + "; ".join(failures)
        )


def _validate_trace_progress(
    rows: list[TraceRow],
    *,
    attempt_number: int,
    require_tick: int | None,
    require_percent_min: float | None,
) -> None:
    if require_tick is None or require_percent_min is None:
        return

    row = next((candidate for candidate in rows if candidate.tick >= require_tick), None)
    if row is None:
        raise ValueError(
            f"attempt {attempt_number} ended before progress guard tick {require_tick}"
        )
    if row.percent < require_percent_min:
        raise ValueError(
            f"attempt {attempt_number} progress check failed at tick {row.tick}: "
            f"percent {row.percent:.3f} < {require_percent_min:.3f}"
        )


def _is_terminal_trace(rows: list[TraceRow], *, success_percent: float) -> bool:
    if not rows:
        return False
    last = rows[-1]
    return last.dead or last.percent >= success_percent


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TimeoutError, EOFError) as exc:
        print(f"error: bridge communication failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
