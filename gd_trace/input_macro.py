"""Build replayable input macros from observed trace state."""

from __future__ import annotations

from typing import Any, Sequence

from gd_human_model.events import Event, Player
from gd_trace.macro_schema import Macro
from gd_trace.replay_check import detect_input_transitions
from gd_trace.trace_schema import TraceRow


TRACE_INPUT_MACRO_KIND = "trace_observed_input"


def trace_input_events(
    rows: Sequence[TraceRow],
    *,
    player: Player = "p1",
) -> list[Event]:
    """Return press/release events from observed ``input_down`` transitions."""

    _validate_player(player)
    return [
        Event(transition.tick, transition.kind, player)
        for transition in detect_input_transitions(rows)
    ]


def trace_input_macro(
    rows: Sequence[TraceRow],
    *,
    player: Player = "p1",
    metadata: dict[str, Any] | None = None,
) -> Macro:
    """Return the canonical replayable macro for a trace's observed input."""

    row_list = list(rows)
    events = trace_input_events(row_list, player=player)
    macro_metadata: dict[str, Any] = {
        "kind": TRACE_INPUT_MACRO_KIND,
        "player": player,
        "trace_row_count": len(row_list),
        "first_tick": row_list[0].tick if row_list else None,
        "last_tick": row_list[-1].tick if row_list else None,
        "event_count": len(events),
        "first_event_tick": events[0].tick if events else None,
        "last_event_tick": events[-1].tick if events else None,
    }
    if metadata:
        macro_metadata.update(metadata)
    macro_metadata["kind"] = TRACE_INPUT_MACRO_KIND

    return Macro(
        events=events,
        metadata=macro_metadata,
    )


def _validate_player(player: Player) -> None:
    if player not in ("p1", "p2"):
        raise ValueError("player must be 'p1' or 'p2'")
