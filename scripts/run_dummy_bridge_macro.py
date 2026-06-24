"""Run a scripted macro against the dummy bridge server and save a trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from gd_env import DummyGeometryDashServer, GeometryDashClient
from gd_trace import load_macro_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("macro_json")
    parser.add_argument("trace_jsonl")
    parser.add_argument("--max-observations", type=int, default=120)
    args = parser.parse_args()

    macro = load_macro_json(args.macro_json)
    trace_path = Path(args.trace_jsonl)

    with DummyGeometryDashServer(max_ticks=args.max_observations + 5) as server:
        assert server.port is not None
        with GeometryDashClient(port=server.port) as client:
            rows = client.run_scripted_events(
                macro.events,
                max_observations=args.max_observations,
                trace_path=trace_path,
            )

    print(f"saved {len(rows)} trace rows to {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
