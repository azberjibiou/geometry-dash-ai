"""Run a scripted macro against the live Geode bridge and save a trace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_env import GeometryDashClient
from gd_trace import load_macro_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("macro_json")
    parser.add_argument("trace_jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29430)
    parser.add_argument("--max-observations", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--cbf", action="store_true")
    parser.add_argument("--physics-bypass", action="store_true")
    args = parser.parse_args()

    macro = load_macro_json(args.macro_json)
    trace_path = Path(args.trace_jsonl)

    with GeometryDashClient(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        rows = client.run_scripted_events(
            macro.events,
            max_observations=args.max_observations,
            fps=args.fps,
            cbf=args.cbf,
            physics_bypass=args.physics_bypass,
            trace_path=trace_path,
        )

    print(f"saved {len(rows)} trace rows to {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
