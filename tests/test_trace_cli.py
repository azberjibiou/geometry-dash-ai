import json

from gd_human_model import Event
from gd_trace.cli import macro_validate_main, trace_compare_main, trace_validate_main
from gd_trace.macro_schema import Macro
from gd_trace.save_trace import save_macro_json, save_trace_jsonl
from tests.test_trace_io import make_row


def test_trace_validate_cli_prints_summary(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "trace.jsonl"
    save_trace_jsonl([make_row(0), make_row(1, percent=3.0)], path)

    assert trace_validate_main([str(path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 2
    assert summary["final_percent"] == 3.0


def test_macro_validate_cli_prints_summary(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "macro.json"
    save_macro_json(Macro(events=[Event(10, "press"), Event(20, "release")]), path)

    assert macro_validate_main([str(path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["events"] == 2
    assert summary["presses"] == 1


def test_trace_compare_cli_prints_summary(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    save_trace_jsonl([make_row(0, x=1.0)], path_a)
    save_trace_jsonl([make_row(0, x=3.0)], path_b)

    assert trace_compare_main([str(path_a), str(path_b)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["x_position_max_diff"] == 2.0
