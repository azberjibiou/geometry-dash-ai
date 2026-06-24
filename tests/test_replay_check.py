from gd_human_model import Event
from gd_trace.replay_check import (
    detect_input_transitions,
    summarize_replay_check,
    summarize_macro_applications,
)
from gd_trace.trace_schema import TraceRow
from tests.test_trace_io import make_row


def row(
    tick: int,
    *,
    x: float = 0.0,
    y: float = 0.0,
    percent: float = 0.0,
    dead: bool = False,
    input_down: bool = False,
) -> TraceRow:
    data = make_row(tick, x=x, percent=percent).to_dict()
    data.update({"y": y, "dead": dead, "input_down": input_down})
    return TraceRow.from_mapping(data)


def test_replay_summary_reports_identical_traces_as_stable() -> None:
    trace = [
        row(0, x=0.0, y=0.0),
        row(1, x=1.0, y=2.0, percent=5.0),
        row(2, x=2.0, y=3.0, percent=10.0),
    ]

    summary = summarize_replay_check([trace, list(trace)], [], success_percent=10.0)

    assert summary.trace_count == 2
    assert summary.row_counts == [3, 3]
    assert summary.compared_tick_count == 3
    assert summary.final_percent_std == 0.0
    assert summary.death_tick_std is None
    assert summary.x_position_max_diff == 0.0
    assert summary.y_position_max_diff == 0.0
    assert summary.success_rate == 1.0
    assert summary.survival_rate == 1.0


def test_replay_summary_reports_outcome_and_position_spread() -> None:
    trace_a = [
        row(0, x=0.0, y=0.0),
        row(1, x=1.0, y=1.0),
        row(2, x=2.0, y=2.0, percent=20.0, dead=True),
    ]
    trace_b = [
        row(0, x=0.0, y=0.0),
        row(1, x=3.0, y=5.0),
        row(2, x=2.5, y=3.0),
        row(3, x=3.0, y=4.0, percent=24.0, dead=True),
    ]

    summary = summarize_replay_check([trace_a, trace_b], [], success_percent=20.0)

    assert summary.final_percents == [20.0, 24.0]
    assert summary.final_percent_std == 2.0
    assert summary.death_ticks == [2, 3]
    assert summary.death_tick_std == 0.5
    assert summary.compared_tick_count == 3
    assert summary.x_position_max_diff == 2.0
    assert summary.y_position_max_diff == 4.0
    assert summary.survival_rate == 0.0


def test_replay_summary_reports_observed_input_latency() -> None:
    macro = [Event(2, "press"), Event(5, "release")]
    trace_a = [
        row(0),
        row(1),
        row(2),
        row(3, input_down=True),
        row(4, input_down=True),
        row(5, input_down=True),
        row(6),
    ]
    trace_b = [
        row(0),
        row(1),
        row(2),
        row(3),
        row(4, input_down=True),
        row(5, input_down=True),
        row(6, input_down=True),
        row(7, input_down=True),
        row(8),
    ]

    summary = summarize_replay_check([trace_a, trace_b], macro)

    press_latency, release_latency = summary.input_latency_by_event
    assert press_latency.latency_frames_by_trial == [1, 2]
    assert press_latency.observed_ticks_by_trial == [3, 4]
    assert press_latency.mean_frames == 1.5
    assert press_latency.std_frames == 0.5
    assert release_latency.latency_frames_by_trial == [1, 3]
    assert release_latency.max_frames == 3
    assert summary.input_latency_mean_frames == 1.75


def test_replay_summary_counts_missing_input_transitions() -> None:
    macro = [Event(2, "press"), Event(5, "release")]
    trace = [row(0), row(1), row(2), row(3), row(4), row(5)]

    summary = summarize_replay_check([trace], macro)

    assert summary.input_latency_by_event[0].missing_count == 1
    assert summary.input_latency_by_event[0].matched_count == 0
    assert summary.input_latency_by_event[0].mean_frames is None
    assert summary.input_latency_mean_frames is None


def test_detect_input_transitions_includes_initial_pressed_state() -> None:
    trace = [row(0, input_down=True), row(1, input_down=True), row(2)]

    transitions = detect_input_transitions(trace)

    assert [(transition.tick, transition.kind) for transition in transitions] == [
        (0, "press"),
        (2, "release"),
    ]


def test_replay_summary_reports_movement_step_diagnostics() -> None:
    trace = [
        row(0, x=0.0),
        row(1, x=0.0),
        row(2, x=1.0),
        row(3, x=3.0),
        row(4, x=4.0),
    ]

    summary = summarize_replay_check([trace], [])

    assert summary.first_movement_ticks == [2]
    assert summary.zero_movement_step_counts == [1]
    assert summary.double_movement_step_counts == [1]


def test_replay_summary_reports_macro_application_diagnostics() -> None:
    macro = [Event(2, "press"), Event(5, "release")]
    diagnostics_by_trial = [
        [
            {
                "kind": "macro_event_applied",
                "tick": 2,
                "data": {
                    "event_index": 0,
                    "intended_tick": 2,
                    "applied_tick": 2,
                },
            },
            {
                "kind": "macro_event_applied",
                "tick": 5,
                "data": {
                    "event_index": 1,
                    "intended_tick": 5,
                    "applied_tick": 5,
                },
            },
        ],
        [
            {
                "kind": "macro_event_applied",
                "tick": 3,
                "data": {
                    "event_index": 0,
                    "intended_tick": 2,
                    "applied_tick": 3,
                },
            },
            {
                "kind": "macro_event_applied",
                "tick": 6,
                "data": {
                    "event_index": 1,
                    "intended_tick": 5,
                    "applied_tick": 6,
                },
            },
        ],
    ]

    summaries = summarize_macro_applications(macro, diagnostics_by_trial)

    assert summaries[0].applied_ticks_by_trial == [2, 3]
    assert summaries[0].latency_frames_by_trial == [0, 1]
    assert summaries[0].mean_frames == 0.5
    assert summaries[1].applied_ticks_by_trial == [5, 6]

    summary = summarize_replay_check(
        [[row(0), row(1)], [row(0), row(1)]],
        macro,
        diagnostics_by_trial=diagnostics_by_trial,
    )
    assert summary.macro_application_by_event[0].max_frames == 1
