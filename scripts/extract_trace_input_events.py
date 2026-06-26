"""Extract a replayable observed-input macro from a trace JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_trace import load_trace_jsonl, save_macro_json, trace_input_macro  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build trace_input_events.json from observed input_down transitions "
            "in a captured trace.jsonl."
        )
    )
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-macro-json",
        type=Path,
        help="defaults to TRACE_PARENT/trace_input_events.json",
    )
    parser.add_argument("--level-id")
    parser.add_argument("--attempt-index", type=int)
    parser.add_argument("--player", choices=("p1", "p2"), default="p1")
    args = parser.parse_args(argv)

    if args.attempt_index is not None and args.attempt_index <= 0:
        parser.error("--attempt-index must be positive")

    output_path = args.output_macro_json or (
        args.trace_jsonl.parent / "trace_input_events.json"
    )

    try:
        rows = load_trace_jsonl(args.trace_jsonl)
        metadata = {
            "generated_by": "scripts/extract_trace_input_events.py",
            "source_trace_jsonl": str(args.trace_jsonl),
        }
        if args.level_id is not None:
            metadata["level_id"] = args.level_id
        if args.attempt_index is not None:
            metadata["attempt_index"] = args.attempt_index
        macro = trace_input_macro(rows, player=args.player, metadata=metadata)
        save_macro_json(macro, output_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "trace_jsonl": str(args.trace_jsonl),
                "output_macro_json": str(output_path),
                "event_count": len(macro.events),
                "first_trace_tick": macro.metadata["first_tick"],
                "last_trace_tick": macro.metadata["last_tick"],
                "first_event_tick": macro.metadata["first_event_tick"],
                "last_event_tick": macro.metadata["last_event_tick"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
