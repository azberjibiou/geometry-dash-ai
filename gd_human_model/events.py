"""Discrete input events used by policies, macros, and human wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

EventKind = Literal["press", "release"]
Player = Literal["p1", "p2"]

_KIND_ORDER: dict[str, int] = {"press": 0, "release": 1}


@dataclass(frozen=True, slots=True)
class Event:
    """A discrete input event at a game tick."""

    tick: int
    kind: EventKind
    player: Player = "p1"

    def __post_init__(self) -> None:
        if not isinstance(self.tick, int):
            raise TypeError("tick must be an int")
        if self.tick < 0:
            raise ValueError("tick must be non-negative")
        if self.kind not in ("press", "release"):
            raise ValueError("kind must be 'press' or 'release'")
        if self.player not in ("p1", "p2"):
            raise ValueError("player must be 'p1' or 'p2'")

    def shifted(self, delta_frames: int) -> "Event":
        """Return a copy shifted by a frame delta."""

        return Event(self.tick + delta_frames, self.kind, self.player)


def event_sort_key(event: Event) -> tuple[int, str, int]:
    """Stable-ish ordering for events that share a tick."""

    return (event.tick, event.player, _KIND_ORDER[event.kind])


def sort_events(events: Iterable[Event]) -> list[Event]:
    """Return events sorted by tick, player, then press-before-release."""

    return sorted(events, key=event_sort_key)
