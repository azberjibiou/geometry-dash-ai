"""Compare Phase A/B practice run summary JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True, slots=True)
class PracticeRunComparisonRow:
    """Compact comparison row for one persisted PracticeRunSummary."""

    run_path: str
    level_id: str
    policy_name: str | None
    human_profile_name: str | None
    attempts: int
    clears: int
    clear_rate: float
    average_final_percent: float
    best_percent: float
    deaths: int
    total_reward: float
    reward_curve: list[float]
    death_histogram: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_path": self.run_path,
            "level_id": self.level_id,
            "policy_name": self.policy_name,
            "human_profile_name": self.human_profile_name,
            "attempts": self.attempts,
            "clears": self.clears,
            "clear_rate": self.clear_rate,
            "average_final_percent": self.average_final_percent,
            "best_percent": self.best_percent,
            "deaths": self.deaths,
            "total_reward": self.total_reward,
            "reward_curve": self.reward_curve,
            "death_histogram": self.death_histogram,
        }


def load_comparison_rows(paths: Sequence[Path]) -> list[PracticeRunComparisonRow]:
    """Load one comparison row per run summary path or run directory."""

    if not paths:
        raise ValueError("at least one summary path is required")
    return [load_comparison_row(path) for path in paths]


def load_comparison_row(path: Path) -> PracticeRunComparisonRow:
    """Load a compact comparison row from a PracticeRunSummary JSON file."""

    summary_path = _resolve_summary_path(path)
    with summary_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{summary_path} must contain a JSON object")

    metadata = _mapping(data.get("metadata"))
    attempts = _list(data.get("attempts"))
    run_path = str(summary_path.parent if summary_path.name == "summary.json" else summary_path)

    return PracticeRunComparisonRow(
        run_path=run_path,
        level_id=str(_required(data, "level_id", summary_path)),
        policy_name=_optional_str(
            data.get("policy_name")
            or metadata.get("policy_name")
            or _nested(metadata, "config", "metadata", "policy")
            or _first_attempt_metadata(attempts, "policy_name")
        ),
        human_profile_name=_optional_str(
            data.get("human_profile_name")
            or metadata.get("human_profile_name")
            or _first_attempt_human_profile_name(attempts)
        ),
        attempts=_attempt_count(data, attempts),
        clears=_int(data.get("clears", 0)),
        clear_rate=_float(data.get("clear_rate", 0.0)),
        average_final_percent=_float(data.get("average_final_percent", 0.0)),
        best_percent=_float(data.get("best_percent", 0.0)),
        deaths=_int(data.get("deaths", 0)),
        total_reward=_float(data.get("total_reward", 0.0)),
        reward_curve=[_float(value) for value in _list(data.get("reward_curve"))],
        death_histogram={
            "tick": _int_histogram(data.get("death_tick_histogram")),
            "percent": _int_histogram(data.get("death_percent_histogram")),
        },
    )


def format_table(rows: Sequence[PracticeRunComparisonRow]) -> str:
    """Render rows as a compact plain-text table."""

    headers = [
        "run_path",
        "level_id",
        "policy_name",
        "human_profile_name",
        "attempts",
        "clears",
        "clear_rate",
        "average_final_percent",
        "best_percent",
        "deaths",
        "total_reward",
        "reward_curve",
        "death_histogram",
    ]
    table_rows = [[_format_cell(row, header) for header in headers] for row in rows]
    widths = [
        max(len(header), *(len(table_row[index]) for table_row in table_rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(table_row))
        for table_row in table_rows
    )
    return "\n".join(lines)


def format_json(rows: Sequence[PracticeRunComparisonRow]) -> str:
    """Render rows as stable pretty JSON."""

    return json.dumps(
        [row.to_dict() for row in rows],
        indent=2,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare one or more Phase A/B PracticeRunSummary JSON files."
    )
    parser.add_argument(
        "summary_paths",
        nargs="+",
        type=Path,
        help="summary.json files, or run directories containing summary.json",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the comparison to this path instead of stdout",
    )
    args = parser.parse_args(argv)

    try:
        rows = load_comparison_rows(args.summary_paths)
        output = format_json(rows) if args.format == "json" else format_table(rows)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8", newline="\n")
    return 0


def _resolve_summary_path(path: Path) -> Path:
    if path.is_dir():
        return path / "summary.json"
    return path


def _required(data: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise ValueError(f"{path} is missing required field {key!r}")
    return data[key]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _attempt_count(data: Mapping[str, Any], attempts: Sequence[Any]) -> int:
    if "attempt_count" in data:
        return _int(data["attempt_count"])
    raw_attempts = data.get("attempts")
    if isinstance(raw_attempts, list):
        return len(raw_attempts)
    if raw_attempts is not None:
        return _int(raw_attempts)
    return len(attempts)


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _nested(data: Mapping[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_attempt_metadata(attempts: Sequence[Any], key: str) -> Any:
    for attempt in attempts:
        if isinstance(attempt, dict):
            metadata = attempt.get("metadata")
            if isinstance(metadata, dict) and key in metadata:
                return metadata[key]
    return None


def _first_attempt_human_profile_name(attempts: Sequence[Any]) -> Any:
    for attempt in attempts:
        if isinstance(attempt, dict):
            human_profile = attempt.get("human_profile")
            if isinstance(human_profile, dict) and "name" in human_profile:
                return human_profile["name"]
    return None


def _int_histogram(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(count) for key, count in value.items()}


def _format_cell(row: PracticeRunComparisonRow, header: str) -> str:
    value = getattr(row, header)
    if isinstance(value, float):
        return f"{value:.6g}"
    if header == "death_histogram":
        return _compact_json(value)
    if isinstance(value, list):
        return _compact_json(value)
    if value is None:
        return ""
    return str(value)


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
