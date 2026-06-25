"""Policy intent representations for practice attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from gd_human_model.events import Event, Player, sort_events

ActionKind = Literal["no_op", "press", "release"]


@dataclass(frozen=True, slots=True)
class IntendedAction:
    """A policy's intended action before humanization.

    The action is not a game input. Press/release actions become intended
    events, then the human model decides whether and when they execute.
    """

    tick: int
    kind: ActionKind
    player: Player = "p1"

    def __post_init__(self) -> None:
        if not isinstance(self.tick, int):
            raise TypeError("tick must be an int")
        if self.tick < 0:
            raise ValueError("tick must be non-negative")
        if self.kind not in ("no_op", "press", "release"):
            raise ValueError("kind must be 'no_op', 'press', or 'release'")
        if self.player not in ("p1", "p2"):
            raise ValueError("player must be 'p1' or 'p2'")

    @classmethod
    def no_op(cls, tick: int) -> "IntendedAction":
        """Return a no-op intent."""

        return cls(tick=tick, kind="no_op")

    @classmethod
    def press(cls, tick: int, player: Player = "p1") -> "IntendedAction":
        """Return a press intent."""

        return cls(tick=tick, kind="press", player=player)

    @classmethod
    def release(cls, tick: int, player: Player = "p1") -> "IntendedAction":
        """Return a release intent."""

        return cls(tick=tick, kind="release", player=player)

    def to_event(self) -> Event | None:
        """Return the intended event represented by this action, if any."""

        if self.kind == "no_op":
            return None
        return Event(tick=self.tick, kind=self.kind, player=self.player)


def actions_to_events(actions: Iterable[IntendedAction]) -> list[Event]:
    """Convert press/release actions to canonical intended events."""

    return sort_events(
        event for action in actions if (event := action.to_event()) is not None
    )
