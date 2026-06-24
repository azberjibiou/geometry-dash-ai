import pytest

from gd_human_model import Event
from gd_trace.macro_schema import Macro, MacroSchemaError, event_from_mapping


def test_macro_accepts_object_and_sorts_events() -> None:
    macro = Macro.from_data(
        {
            "metadata": {"level_name": "test"},
            "events": [
                {"tick": 20, "kind": "release", "player": "p1"},
                {"tick": 10, "kind": "press", "player": "p1"},
            ],
        }
    )

    assert macro.metadata == {"level_name": "test"}
    assert macro.events == [Event(10, "press"), Event(20, "release")]
    assert macro.to_dict()["events"][0]["kind"] == "press"


def test_macro_accepts_bare_event_list() -> None:
    macro = Macro.from_data([{"tick": 4, "kind": "press"}])

    assert macro.events == [Event(4, "press")]


def test_macro_rejects_invalid_events() -> None:
    with pytest.raises(MacroSchemaError, match="event.kind"):
        event_from_mapping({"tick": 1, "kind": "tap"})

    with pytest.raises(MacroSchemaError, match="events list"):
        Macro.from_data({"events": "bad"})
