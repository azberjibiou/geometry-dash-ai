from dataclasses import replace

from gd_human_model import ADVANCED, Event, HumanizedAgent, MotorNoiseModel


def no_noise_profile(**overrides: object):
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
        random_seed=0,
    )
    return replace(profile, **overrides)


def test_motor_delay_schedules_events_later() -> None:
    profile = no_noise_profile(motor_delay_frames=7)
    agent = HumanizedAgent[object](profile)

    actual = agent.submit_intended_event(Event(10, "press"))

    assert actual == Event(17, "press")
    assert agent.pop_due_events(16) == []
    assert agent.pop_due_events(17) == [Event(17, "press")]


def test_press_release_order_is_preserved_after_reordering_jitter() -> None:
    class ReorderingNoise(MotorNoiseModel):
        def __init__(self):
            super().__init__(no_noise_profile())
            self.errors = iter([10.0, -10.0])

        def sample_timing_error(self, kind, delta_frames, player="p1"):  # type: ignore[no-untyped-def]
            return next(self.errors)

    agent = HumanizedAgent[object](no_noise_profile(), motor_noise=ReorderingNoise())

    actual = agent.submit_intended_events([Event(10, "press"), Event(12, "release")])

    assert actual == [Event(20, "press"), Event(21, "release")]


def test_impossible_release_is_dropped() -> None:
    agent = HumanizedAgent[object](no_noise_profile())

    assert agent.submit_intended_event(Event(10, "release")) is None
    assert agent.pending_count == 0
