"""Small command-line tools for validating and comparing Phase 2 files."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from gd_trace.click_window import analyze_click_windows
from gd_trace.compare_trace import compare_traces
from gd_trace.load_trace import load_macro_json, load_trace_jsonl


def trace_validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gd-trace-validate")
    parser.add_argument("trace_jsonl")
    args = parser.parse_args(argv)

    try:
        rows = load_trace_jsonl(args.trace_jsonl)
    except Exception as exc:  # pragma: no cover - exercised by CLI users.
        print(f"invalid trace: {exc}", file=sys.stderr)
        return 1

    summary = {
        "rows": len(rows),
        "first_tick": rows[0].tick if rows else None,
        "last_tick": rows[-1].tick if rows else None,
        "fps": rows[0].fps if rows else None,
        "final_percent": rows[-1].percent if rows else None,
        "death_tick": next((row.tick for row in rows if row.dead), None),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def macro_validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gd-macro-validate")
    parser.add_argument("macro_json")
    args = parser.parse_args(argv)

    try:
        macro = load_macro_json(args.macro_json)
    except Exception as exc:  # pragma: no cover - exercised by CLI users.
        print(f"invalid macro: {exc}", file=sys.stderr)
        return 1

    summary = {
        "events": len(macro.events),
        "first_tick": macro.events[0].tick if macro.events else None,
        "last_tick": macro.events[-1].tick if macro.events else None,
        "presses": sum(1 for event in macro.events if event.kind == "press"),
        "releases": sum(1 for event in macro.events if event.kind == "release"),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def trace_compare_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gd-trace-compare")
    parser.add_argument("trace_a_jsonl")
    parser.add_argument("trace_b_jsonl")
    args = parser.parse_args(argv)

    try:
        trace_a = load_trace_jsonl(args.trace_a_jsonl)
        trace_b = load_trace_jsonl(args.trace_b_jsonl)
    except Exception as exc:  # pragma: no cover - exercised by CLI users.
        print(f"invalid trace: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(compare_traces(trace_a, trace_b).to_dict(), sort_keys=True))
    return 0


def click_window_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gd-click-window")
    parser.add_argument("macro_json")
    parser.add_argument("--radius-frames", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        macro = load_macro_json(args.macro_json)
        windows = analyze_click_windows(macro, radius_frames=args.radius_frames)
    except Exception as exc:  # pragma: no cover - exercised by CLI users.
        print(f"invalid macro: {exc}", file=sys.stderr)
        return 1

    print(json.dumps([window.to_dict() for window in windows], sort_keys=True))
    return 0
