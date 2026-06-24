"""Run repeated identical macros against the live Geode bridge."""

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

from gd_env import GeometryDashClient
from gd_trace import load_macro_json, summarize_replay_check
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
        else:
            client.load_macro(macro.events, metadata=macro.metadata)

        for trial_index in range(args.trials):
            trial_number = trial_index + 1
            diagnostics = []
            initial_observation = client.reset_attempt(
                f"replay_check_trial_{trial_number}",
                max_observations=args.reset_wait_observations,
                diagnostics=diagnostics,
            )
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
                )
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
                    "last_tick": rows[-1].tick if rows else None,
                    "final_percent": rows[-1].percent if rows else 0.0,
                    "death_tick": first_death_tick(rows),
                    "diagnostics": diagnostics_document,
                }
            )

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
            "live_send": args.live_send,
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


if __name__ == "__main__":
    raise SystemExit(main())
