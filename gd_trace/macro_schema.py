"""Canonical macro schema for discrete input events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gd_human_model.events import Event, sort_events


class MacroSchemaError(ValueError):
    """Raised when macro data does not match the canonical schema."""


@dataclass(frozen=True, slots=True)
class Macro:
    """A sorted input macro plus optional metadata."""

    events: list[Event]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.events, list):
            raise MacroSchemaError("events must be a list")
        sorted_events = sort_events(self.events)
        object.__setattr__(self, "events", sorted_events)
        if not isinstance(self.metadata, dict):
            raise MacroSchemaError("metadata must be a dict")

    @classmethod
    def from_data(cls, data: Any) -> "Macro":
        """Build a macro from parsed JSON data.

        Accepted forms:
        - [{"tick": 10, "kind": "press", "player": "p1"}, ...]
        - {"metadata": {...}, "events": [...]}
        """

        if isinstance(data, list):
            return cls(events=[event_from_mapping(item) for item in data])
        if isinstance(data, Mapping):
            events_data = data.get("events")
            if not isinstance(events_data, list):
                raise MacroSchemaError("macro object must contain an events list")
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise MacroSchemaError("metadata must be a dict")
            return cls(
                events=[event_from_mapping(item) for item in events_data],
                metadata=dict(metadata),
            )
        raise MacroSchemaError("macro must be a list or object")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable macro data."""

        return {
            "metadata": self.metadata,
            "events": [event_to_dict(event) for event in self.events],
        }


def event_from_mapping(data: Any) -> Event:
    """Parse and validate one macro event."""

    if not isinstance(data, Mapping):
        raise MacroSchemaError("macro event must be an object")

    tick = data.get("tick")
    kind = data.get("kind")
    player = data.get("player", "p1")

    if isinstance(tick, bool) or not isinstance(tick, int):
        raise MacroSchemaError("event.tick must be an int")
    if kind not in ("press", "release"):
        raise MacroSchemaError("event.kind must be 'press' or 'release'")
    if player not in ("p1", "p2"):
        raise MacroSchemaError("event.player must be 'p1' or 'p2'")

    try:
        return Event(tick=tick, kind=kind, player=player)
    except ValueError as exc:
        raise MacroSchemaError(str(exc)) from exc


def event_to_dict(event: Event) -> dict[str, Any]:
    """Return JSON-serializable event data."""

    return {
        "tick": event.tick,
        "kind": event.kind,
        "player": event.player,
    }
