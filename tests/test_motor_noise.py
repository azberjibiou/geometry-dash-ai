import statistics
from dataclasses import replace

from gd_human_model import ADVANCED, Event, HumanProfile, MotorNoiseModel


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


def lag_correlation(values: list[float]) -> float:
    left = values[:-1]
    right = values[1:]
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    return numerator / (left_var * right_var) ** 0.5


def test_same_random_seed_gives_identical_actual_events() -> None:
    profile = replace(
        ADVANCED,
        random_seed=99,
        miss_prob_base=0.0,
        miss_prob_close_amp=0.0,
    )
    intended = [
        Event(10, "press"),
        Event(20, "release"),
        Event(35, "press"),
        Event(45, "release"),
    ]

    first = MotorNoiseModel(profile).humanize_events(intended)
    second = MotorNoiseModel(profile).humanize_events(intended)

    assert first == second


def test_larger_close_amp_increases_std_for_close_clicks() -> None:
    low = quiet_profile(base_press_std_frames=1.0, close_amp=0.0)
    high = quiet_profile(base_press_std_frames=1.0, close_amp=5.0)

    assert MotorNoiseModel(high).timing_std("press", delta_frames=2) > MotorNoiseModel(
        low
    ).timing_std("press", delta_frames=2)


def test_miss_probability_can_drop_events() -> None:
    profile = quiet_profile(miss_prob_base=1.0)
    actual = MotorNoiseModel(profile).humanize_events(
        [Event(10, "press"), Event(20, "release")]
    )

    assert actual == []


def test_correlated_errors_have_positive_lag_correlation() -> None:
    uncorrelated = quiet_profile(
        base_press_std_frames=3.0,
        error_rho=0.0,
        random_seed=7,
    )
    correlated = quiet_profile(
        base_press_std_frames=3.0,
        error_rho=0.85,
        random_seed=7,
    )

    uncorrelated_model = MotorNoiseModel(uncorrelated)
    correlated_model = MotorNoiseModel(correlated)

    uncorrelated_errors = [
        uncorrelated_model.sample_timing_error("press", 20) for _ in range(800)
    ]
    correlated_errors = [
        correlated_model.sample_timing_error("press", 20) for _ in range(800)
    ]

    assert lag_correlation(correlated_errors) > lag_correlation(uncorrelated_errors) + 0.5


def test_event_jitter_distribution_matches_configured_std() -> None:
    profile = quiet_profile(base_press_std_frames=2.0, random_seed=42)
    model = MotorNoiseModel(profile)

    errors = [model.sample_timing_error("press", 50) for _ in range(5000)]

    assert abs(statistics.mean(errors)) < 0.1
    assert 1.9 < statistics.pstdev(errors) < 2.1
