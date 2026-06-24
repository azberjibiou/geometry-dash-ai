import statistics
from dataclasses import replace

from gd_human_model import ADVANCED, Event, HumanProfile, humanize_macro_events


def quiet_profile(**overrides: object) -> HumanProfile:
    profile = replace(
        ADVANCED,
        visual_delay_frames=0,
        motor_delay_frames=0,
        base_press_std_frames=0.0,
        base_release_std_frames=0.0,
        close_amp=0.0,
        long_amp=0.0,
        error_rho=0.0,
        miss_prob_base=0.0,
        miss_prob_close_amp=0.0,
        random_seed=123,
    )
    return replace(profile, **overrides)


def alternating_events(count: int, *, spacing: int = 30) -> list[Event]:
    events = []
    for index in range(count):
        tick = index * spacing
        events.append(Event(tick, "press"))
        events.append(Event(tick + spacing // 2, "release"))
    return events


def test_humanized_macro_is_deterministic_for_fixed_seed() -> None:
    profile = replace(
        ADVANCED,
        random_seed=99,
        miss_prob_base=0.0,
        miss_prob_close_amp=0.0,
    )
    intended = alternating_events(3)

    first = humanize_macro_events(intended, profile, seed=77)
    second = humanize_macro_events(intended, profile, seed=77)

    assert first.actual_events == second.actual_events
    assert [result.to_dict() for result in first.event_results] == [
        result.to_dict() for result in second.event_results
    ]


def test_target_reference_centers_actual_macro_without_noise() -> None:
    profile = quiet_profile(visual_delay_frames=5, motor_delay_frames=8)

    humanized = humanize_macro_events(
        [Event(10, "press"), Event(20, "release")],
        profile,
    )

    assert humanized.actual_events == [Event(10, "press"), Event(20, "release")]
    assert humanized.event_results[0].decision_event == Event(2, "press")
    assert humanized.event_results[0].actual_delta_frames == 0
    assert humanized.event_results[0].delay_adjusted_delta_frames == 0
    assert humanized.to_macro().metadata["timing_reference"] == "target"


def test_decision_reference_applies_visual_and_motor_delay() -> None:
    profile = quiet_profile(visual_delay_frames=5, motor_delay_frames=8)

    humanized = humanize_macro_events(
        [Event(10, "press"), Event(20, "release")],
        profile,
        timing_reference="decision",
    )

    assert humanized.actual_events == [Event(23, "press"), Event(33, "release")]
    assert humanized.event_results[0].decision_event == Event(15, "press")
    assert humanized.event_results[0].actual_delta_frames == 13
    assert humanized.event_results[0].delay_adjusted_delta_frames == 0
    assert humanized.to_macro().metadata["timing_reference"] == "decision"


def test_higher_base_std_increases_humanized_timing_spread() -> None:
    intended = alternating_events(100, spacing=50)
    low = quiet_profile(base_press_std_frames=0.0, base_release_std_frames=0.0)
    high = quiet_profile(base_press_std_frames=4.0, base_release_std_frames=4.0)

    low_result = humanize_macro_events(intended, low, seed=5)
    high_result = humanize_macro_events(intended, high, seed=5)
    low_deltas = [
        result.delay_adjusted_delta_frames
        for result in low_result.event_results
        if result.delay_adjusted_delta_frames is not None
    ]
    high_deltas = [
        result.delay_adjusted_delta_frames
        for result in high_result.event_results
        if result.delay_adjusted_delta_frames is not None
    ]

    assert statistics.pstdev(low_deltas) == 0.0
    assert statistics.pstdev(high_deltas) > 2.0


def test_miss_probability_can_drop_macro_events() -> None:
    profile = quiet_profile(miss_prob_base=1.0)

    humanized = humanize_macro_events(
        [Event(10, "press"), Event(20, "release")],
        profile,
    )

    assert humanized.actual_events == []
    assert humanized.missed_event_count == 2
    assert [result.drop_reason for result in humanized.event_results] == [
        "miss",
        "miss",
    ]


def test_generated_macro_is_sorted_and_valid() -> None:
    profile = quiet_profile(
        motor_delay_frames=4,
        base_press_std_frames=6.0,
        base_release_std_frames=6.0,
        random_seed=8,
    )

    humanized = humanize_macro_events(alternating_events(10), profile)
    actual_macro = humanized.to_macro()

    assert actual_macro.events == sorted(actual_macro.events, key=lambda event: event.tick)
    assert all(event.tick >= 0 for event in actual_macro.events)
    assert actual_macro.metadata["humanized"] is True
    assert actual_macro.metadata["missed_event_count"] == humanized.missed_event_count
