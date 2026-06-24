import pytest

from gd_human_model import Event


def test_event_validates_kind_and_tick() -> None:
    assert Event(10, "press").shifted(5) == Event(15, "press")

    with pytest.raises(ValueError):
        Event(-1, "press")

    with pytest.raises(ValueError):
        Event(1, "tap")  # type: ignore[arg-type]
