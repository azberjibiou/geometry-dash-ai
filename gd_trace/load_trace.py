"""Load trace JSONL files and macro JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from gd_trace.macro_schema import Macro, MacroSchemaError
from gd_trace.trace_schema import TraceRow, TraceSchemaError, validate_trace_sequence


def iter_trace_jsonl(path: str | Path) -> Iterator[TraceRow]:
    """Yield validated trace rows from a JSONL file."""

    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TraceSchemaError(
                    f"{trace_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise TraceSchemaError(f"{trace_path}:{line_number}: row must be an object")
            try:
                yield TraceRow.from_mapping(data)
            except TraceSchemaError as exc:
                raise TraceSchemaError(f"{trace_path}:{line_number}: {exc}") from exc


def load_trace_jsonl(path: str | Path) -> list[TraceRow]:
    """Load and validate a whole trace JSONL file."""

    rows = list(iter_trace_jsonl(path))
    validate_trace_sequence(rows)
    return rows


def load_macro_json(path: str | Path) -> Macro:
    """Load and validate a macro JSON file."""

    macro_path = Path(path)
    try:
        with macro_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise MacroSchemaError(f"{macro_path}: invalid JSON: {exc.msg}") from exc

    try:
        return Macro.from_data(data)
    except MacroSchemaError as exc:
        raise MacroSchemaError(f"{macro_path}: {exc}") from exc
