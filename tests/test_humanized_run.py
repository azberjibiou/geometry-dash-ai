import pytest

from gd_human_model import Event, humanize_macro_events
from gd_trace.humanized_run import summarize_humanized_attempts
from gd_trace.trace_schema import TraceRow
from tests.test_macro_humanizer import quiet_profile
from tests.test_trace_io import make_row


def row(tick: int, *, percent: float = 0.0, dead: bool = False) -> TraceRow:
    data = make_row(tick, percent=percent).to_dict()
    data["dead"] = dead
    return TraceRow.from_mapping(data)


def test_humanized_run_summary_reports_clears_deaths_and_progress() -> None:
    profile = quiet_profile(visual_delay_frames=2, motor_delay_frames=3)
    humanized = humanize_macro_events(
        [Event(10, "press"), Event(20, "release")],
        profile,
    )
    traces = [
        [row(0), row(50, percent=100.0)],
        [row(0), row(30, percent=42.4, dead=True)],
    ]

    summary = summarize_humanized_attempts(
        traces,
        [humanized.event_results, humanized.event_results],
        success_percent=100.0,
    )

    assert summary.attempt_count == 2
    assert summary.clears == 1
    assert summary.deaths == 1
    assert summary.clear_rate == 0.5
    assert summary.survival_rate == 0.5
    assert summary.average_progress == 71.2
    assert summary.best_percent == 100.0
    assert summary.final_percent_by_attempt == [100.0, 42.4]
    assert summary.death_ticks == [None, 30]
    assert summary.death_tick_histogram == {"30": 1}
    assert summary.death_percent_histogram == {"42": 1}
    assert summary.attempts_to_first_clear == 1
    assert summary.playtime_to_first_clear_seconds == traces[0][-1].time_ms / 1000.0


def test_humanized_run_summary_reports_event_deltas_and_misses() -> None:
    hit_profile = quiet_profile(visual_delay_frames=2, motor_delay_frames=3)
    miss_profile = quiet_profile(miss_prob_base=1.0)
    intended = [Event(10, "press"), Event(20, "release")]
    hit = humanize_macro_events(intended, hit_profile)
    miss = humanize_macro_events(intended, miss_profile)

    summary = summarize_humanized_attempts(
        [[row(0), row(20, percent=10.0)], [row(0), row(20, percent=5.0)]],
        [hit.event_results, miss.event_results],
        success_percent=100.0,
    )

    assert summary.actual_event_count_by_attempt == [2, 0]
    assert summary.missed_event_count_by_attempt == [0, 2]
    assert summary.total_missed_events == 2
    assert summary.timing_delta_frames.count == 2
    assert summary.timing_delta_frames.mean == 0.0
    assert summary.delay_adjusted_delta_frames.mean == 0.0
    assert summary.attempts[0].timing_delta_frames == [0, 0]


def test_humanized_run_summary_requires_matching_attempt_counts() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_humanized_attempts([[row(0)]], [])
