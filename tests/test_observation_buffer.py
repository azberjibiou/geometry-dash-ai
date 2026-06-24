from gd_human_model import ADVANCED, HumanizedAgent, ObservationBuffer


def test_visual_delay_returns_older_observation() -> None:
    agent = HumanizedAgent[str](ADVANCED)

    for tick in range(20):
        agent.observe(tick, f"obs-{tick}")

    assert agent.delayed_observation(19) == "obs-11"


def test_larger_visual_delay_returns_older_observation() -> None:
    buffer = ObservationBuffer[str]()
    for tick in range(10):
        buffer.add(tick, f"obs-{tick}")

    assert buffer.get_delayed(current_tick=9, delay_frames=2) == "obs-7"
    assert buffer.get_delayed(current_tick=9, delay_frames=5) == "obs-4"


def test_buffer_trims_old_observations() -> None:
    buffer = ObservationBuffer[int](maxlen=3)
    for tick in range(5):
        buffer.add(tick, tick)

    assert len(buffer) == 3
    assert buffer.get(1) is None
    assert buffer.get(2) == 2
