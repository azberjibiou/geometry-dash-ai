"""Simple click-window summaries for macro events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from gd_human_model.events import Event, EventKind, Player, sort_events
from gd_trace.macro_schema import Macro


@dataclass(frozen=True, slots=True)
class ClickWindow:
    """A timing window centered on one macro event."""

    center_tick: int
    start_tick: int
    end_tick: int
    kind: EventKind
    player: Player = "p1"

    @property
    def width_frames(self) -> int:
        return self.end_tick - self.start_tick + 1

    def to_dict(self) -> dict[str, int | str]:
        data = asdict(self)
        data["width_frames"] = self.width_frames
        return data


def analyze_click_windows(
    events_or_macro: Iterable[Event] | Macro,
    *,
    radius_frames: int = 2,
) -> list[ClickWindow]:
    """Create coarse event windows around each macro event."""

    if radius_frames < 0:
        raise ValueError("radius_frames must be non-negative")

    if isinstance(events_or_macro, Macro):
        events = events_or_macro.events
    else:
        events = list(events_or_macro)

    windows: list[ClickWindow] = []
    for event in sort_events(events):
        windows.append(
            ClickWindow(
                center_tick=event.tick,
                start_tick=max(0, event.tick - radius_frames),
                end_tick=event.tick + radius_frames,
                kind=event.kind,
                player=event.player,
            )
        )
    return windows
