import json

from gd_human_model import Event
from gd_trace import TraceRow, load_macro_json, save_trace_jsonl
from scripts.extract_trace_input_events import main


def test_extract_trace_input_events_writes_observed_input_macro(
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    trace_path = tmp_path / "attempt_001" / "trace.jsonl"
    output_path = tmp_path / "teacher.json"
    save_trace_jsonl(
        [
            _row(0, input_down=False),
            _row(1, input_down=True),
            _row(2, input_down=True),
            _row(3, input_down=False),
        ],
        trace_path,
    )

    exit_code = main(
        [
            "--trace-jsonl",
            str(trace_path),
            "--output-macro-json",
            str(output_path),
            "--level-id",
            "fixture",
            "--attempt-index",
            "1",
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    macro = load_macro_json(output_path)

    assert printed["event_count"] == 2
    assert macro.metadata["kind"] == "trace_observed_input"
    assert macro.metadata["level_id"] == "fixture"
    assert macro.metadata["attempt_index"] == 1
    assert macro.events == [Event(1, "press"), Event(3, "release")]


def _row(tick: int, *, input_down: bool) -> TraceRow:
    return TraceRow(
        tick=tick,
        time_ms=tick * 1000.0 / 240,
        input_down=input_down,
        x=float(tick),
        y=2.0,
        x_vel=1.0,
        y_vel=0.0,
        rotation=0.0,
        mode="cube",
        gravity="normal",
        percent=float(tick),
        dead=False,
        death_reason=None,
        fps=240,
        cbf=False,
        physics_bypass=False,
    )
