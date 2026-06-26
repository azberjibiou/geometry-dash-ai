"""Run repeated identical macros against the live Geode bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_env import BridgeDiagnostic, BridgeObservation, GeometryDashClient
from gd_trace import TraceRow, load_macro_json, summarize_replay_check
from gd_trace.compare_trace import first_death_tick


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic replay check against the live Geode bridge."
    )
    parser.add_argument("macro_json")
    parser.add_argument("--trials", type=int, default=5)
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
        "--post-success-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "sleep after a successful trial before the next reset, useful when "
            "Geometry Dash shows delayed clear UI"
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
        "--start-guard-reset-retries",
        type=int,
        default=0,
        help="retry reset when fresh-start percent/x guards fail",
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
    parser.add_argument(
        "--live-send",
        action="store_true",
        help="send actions from Python as observations arrive instead of using mod-side queued replay",
    )
    args = parser.parse_args(argv)

    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.max_observations <= 0:
        parser.error("--max-observations must be positive")
    if args.post_success_delay_seconds < 0.0:
        parser.error("--post-success-delay-seconds must be non-negative")
    if (
        args.require_start_percent_max is not None
        and not 0.0 <= args.require_start_percent_max <= 100.0
    ):
        parser.error("--require-start-percent-max must be between 0 and 100")
    if args.require_start_x_max is not None and args.require_start_x_max < 0.0:
        parser.error("--require-start-x-max must be non-negative")
    if args.start_guard_reset_retries < 0:
        parser.error("--start-guard-reset-retries must be non-negative")
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

    macro_path = Path(args.macro_json)
    macro = load_macro_json(macro_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("artifacts") / f"replay_check_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    trial_results = []
    diagnostics_by_trial = []
    replay_mode = "live_send" if args.live_send else "queued_macro"

    with GeometryDashClient(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        if args.live_send:
            # Clear any queued macro left in the mod from a prior replay check.
            client.load_macro([], metadata={"replay_mode": "live_send_clear"})

        for trial_index in range(args.trials):
            trial_number = trial_index + 1
            diagnostics = []
            if not args.live_send:
                client.load_macro(
                    macro.events,
                    metadata={
                        **macro.metadata,
                        "replay_check_trial": trial_number,
                    },
                )
            try:
                initial_observation, reset_attempts = _reset_until_valid_start(
                    client,
                    trial_number=trial_number,
                    max_observations=args.reset_wait_observations,
                    max_retries=args.start_guard_reset_retries,
                    diagnostics=diagnostics,
                    require_percent_max=args.require_start_percent_max,
                    require_x_max=args.require_start_x_max,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            trace_path = output_dir / f"trial_{trial_number:03d}.jsonl"
            if args.live_send:
                rows = client.run_scripted_events(
                    macro.events,
                    max_observations=args.max_observations,
                    fps=args.fps,
                    cbf=args.cbf,
                    physics_bypass=args.physics_bypass,
                    trace_path=trace_path,
                    initial_observation=initial_observation,
                    stop_percent=args.success_percent if args.stop_on_success else None,
                )
            else:
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
            try:
                _validate_trace_progress(
                    rows,
                    trial_number=trial_number,
                    require_tick=args.require_progress_tick,
                    require_percent_min=args.require_progress_percent_min,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            traces.append(rows)
            diagnostics_document = [
                diagnostic.to_dict() for diagnostic in diagnostics
            ]
            diagnostics_by_trial.append(diagnostics_document)
            trial_results.append(
                {
                    "trial": trial_number,
                    "trace_path": str(trace_path),
                    "rows": len(rows),
                    "first_tick": rows[0].tick if rows else None,
                    "start_percent": rows[0].percent if rows else None,
                    "start_x": rows[0].x if rows else None,
                    "reset_attempts": reset_attempts,
                    "last_tick": rows[-1].tick if rows else None,
                    "final_percent": rows[-1].percent if rows else 0.0,
                    "death_tick": first_death_tick(rows),
                    "diagnostics": diagnostics_document,
                }
            )
            if (
                rows
                and rows[-1].percent >= args.success_percent
                and args.post_success_delay_seconds > 0.0
            ):
                time.sleep(args.post_success_delay_seconds)

    summary = summarize_replay_check(
        traces,
        macro.events,
        success_percent=args.success_percent,
        diagnostics_by_trial=None if args.live_send else diagnostics_by_trial,
    )
    summary_path = output_dir / "summary.json"
    summary_document = {
        "run_id": run_id,
        "macro_path": str(macro_path),
        "macro_metadata": macro.metadata,
        "replay_mode": replay_mode,
        "settings": {
            "host": args.host,
            "port": args.port,
            "fps": args.fps,
            "cbf": args.cbf,
            "physics_bypass": args.physics_bypass,
            "max_observations": args.max_observations,
            "success_percent": args.success_percent,
            "stop_on_success": args.stop_on_success,
            "post_success_delay_seconds": args.post_success_delay_seconds,
            "require_start_percent_max": args.require_start_percent_max,
            "require_start_x_max": args.require_start_x_max,
            "start_guard_reset_retries": args.start_guard_reset_retries,
            "require_progress_tick": args.require_progress_tick,
            "require_progress_percent_min": args.require_progress_percent_min,
            "live_send": args.live_send,
            "reload_macro_each_trial": not args.live_send,
        },
        "trials": trial_results,
        "summary": summary.to_dict(),
    }

    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary_document, file, indent=2, sort_keys=True)
        file.write("\n")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(summary_path),
                "summary": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _reset_until_valid_start(
    client: GeometryDashClient,
    *,
    trial_number: int,
    max_observations: int,
    max_retries: int,
    diagnostics: list,
    require_percent_max: float | None,
    require_x_max: float | None,
) -> tuple[BridgeObservation, int]:
    last_error: ValueError | None = None
    for reset_index in range(max_retries + 1):
        observation = client.reset_attempt(
            f"replay_check_trial_{trial_number}_reset_{reset_index + 1}",
            max_observations=max_observations,
            diagnostics=diagnostics,
        )
        try:
            _validate_start_observation(
                observation,
                trial_number=trial_number,
                require_percent_max=require_percent_max,
                require_x_max=require_x_max,
            )
            return observation, reset_index + 1
        except ValueError as exc:
            last_error = exc
            diagnostics.append(
                BridgeDiagnostic(
                    kind="fresh_start_guard_retry",
                    tick=observation.tick,
                    data={
                        "trial": trial_number,
                        "reset_attempt": reset_index + 1,
                        "percent": observation.percent,
                        "x": observation.x,
                        "error": str(exc),
                    },
                )
            )
    if last_error is None:
        raise RuntimeError("unreachable start guard state")
    raise last_error


def _validate_start_observation(
    observation: BridgeObservation,
    *,
    trial_number: int,
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
            f"trial {trial_number} fresh start check failed: " + "; ".join(failures)
        )


def _validate_trace_progress(
    rows: list[TraceRow],
    *,
    trial_number: int,
    require_tick: int | None,
    require_percent_min: float | None,
) -> None:
    if require_tick is None or require_percent_min is None:
        return

    row = next((candidate for candidate in rows if candidate.tick >= require_tick), None)
    if row is None:
        raise ValueError(
            f"trial {trial_number} ended before progress guard tick {require_tick}"
        )
    if row.percent < require_percent_min:
        raise ValueError(
            f"trial {trial_number} progress check failed at tick {row.tick}: "
            f"percent {row.percent:.3f} < {require_percent_min:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
