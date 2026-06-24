"""Save trace JSONL files and macro JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from gd_trace.macro_schema import Macro
from gd_trace.trace_schema import TraceRow, validate_trace_sequence


def save_trace_jsonl(rows: Iterable[TraceRow], path: str | Path) -> None:
    """Save trace rows as canonical JSONL."""

    trace_rows = list(rows)
    validate_trace_sequence(trace_rows)
    trace_path = Path(path)
    if trace_path.parent != Path("."):
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    with trace_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in trace_rows:
            file.write(json.dumps(row.to_dict(), separators=(",", ":"), sort_keys=True))
            file.write("\n")


def save_macro_json(macro: Macro, path: str | Path) -> None:
    """Save a macro as canonical JSON."""

    macro_path = Path(path)
    if macro_path.parent != Path("."):
        macro_path.parent.mkdir(parents=True, exist_ok=True)

    with macro_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(macro.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
