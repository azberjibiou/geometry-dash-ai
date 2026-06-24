from gd_human_model import Event
from gd_trace.load_trace import load_macro_json, load_trace_jsonl
from gd_trace.macro_schema import Macro
from gd_trace.save_trace import save_macro_json, save_trace_jsonl
from gd_trace.trace_schema import TraceRow


def make_row(tick: int, *, x: float = 0.0, percent: float = 0.0) -> TraceRow:
    return TraceRow(
        tick=tick,
        time_ms=tick * 1000 / 240,
        input_down=False,
        x=x,
        y=0.0,
        x_vel=0.0,
        y_vel=0.0,
        rotation=0.0,
        mode="cube",
        gravity="normal",
        percent=percent,
        dead=False,
        death_reason=None,
        fps=240,
        cbf=False,
        physics_bypass=False,
    )


def test_trace_jsonl_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "trace.jsonl"
    rows = [make_row(0), make_row(1, x=2.0, percent=1.0)]

    save_trace_jsonl(rows, path)

    assert load_trace_jsonl(path) == rows


def test_macro_json_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "macro.json"
    macro = Macro(
        metadata={"level_name": "roundtrip"},
        events=[Event(10, "press"), Event(20, "release")],
    )

    save_macro_json(macro, path)

    assert load_macro_json(path) == macro
