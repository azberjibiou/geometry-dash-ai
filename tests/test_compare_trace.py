from gd_trace.compare_trace import compare_traces
from tests.test_trace_io import make_row


def test_compare_traces_reports_position_and_death_differences() -> None:
    trace_a = [make_row(0, x=0.0), make_row(1, x=1.0, percent=5.0)]
    trace_b = [make_row(0, x=0.0), make_row(1, x=3.5, percent=7.0)]
    trace_b[1] = trace_b[1].__class__(**{**trace_b[1].to_dict(), "dead": True})

    comparison = compare_traces(trace_a, trace_b)

    assert comparison.compared_rows == 2
    assert comparison.death_tick_a is None
    assert comparison.death_tick_b == 1
    assert comparison.final_percent_diff == 2.0
    assert comparison.x_position_max_diff == 2.5
