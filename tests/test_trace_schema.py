import pytest

from gd_trace.trace_schema import TraceRow, TraceSchemaError, validate_trace_sequence


def row_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "tick": 0,
        "time_ms": 0.0,
        "input_down": False,
        "x": 1.0,
        "y": 2.0,
        "x_vel": 3.0,
        "y_vel": 4.0,
        "rotation": 0.0,
        "mode": "cube",
        "gravity": "normal",
        "percent": 0.0,
        "dead": False,
        "death_reason": None,
        "fps": 240,
        "cbf": False,
        "physics_bypass": False,
    }
    data.update(overrides)
    return data


def test_trace_row_from_mapping_validates_and_serializes() -> None:
    row = TraceRow.from_mapping(row_data(tick=5, percent=12.5))

    assert row.tick == 5
    assert row.percent == 12.5
    assert row.to_dict()["mode"] == "cube"


def test_trace_row_rejects_missing_or_bad_fields() -> None:
    missing = row_data()
    missing.pop("x")

    with pytest.raises(TraceSchemaError, match="missing required field"):
        TraceRow.from_mapping(missing)

    with pytest.raises(TraceSchemaError, match="percent"):
        TraceRow.from_mapping(row_data(percent=101.0))


def test_trace_sequence_requires_strict_ticks_and_fixed_settings() -> None:
    rows = [
        TraceRow.from_mapping(row_data(tick=0)),
        TraceRow.from_mapping(row_data(tick=1)),
    ]
    validate_trace_sequence(rows)

    with pytest.raises(TraceSchemaError, match="strictly increasing"):
        validate_trace_sequence([rows[0], rows[0]])

    with pytest.raises(TraceSchemaError, match="fps changed"):
        validate_trace_sequence(
            [
                TraceRow.from_mapping(row_data(tick=0, fps=240)),
                TraceRow.from_mapping(row_data(tick=1, fps=120)),
            ]
        )
