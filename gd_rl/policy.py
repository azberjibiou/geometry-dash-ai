"""Small policy interfaces and baseline policies for Phase A practice loops."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from gd_human_model.events import Event, Player, sort_events

from gd_rl.actions import IntendedAction

if TYPE_CHECKING:
    from gd_rl.results import AttemptResult


@dataclass(frozen=True, slots=True)
class PracticeContext:
    """Attempt-level context passed to simple event-schedule policies."""

    level_id: str
    attempt_index: int
    max_tick: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PracticePolicy:
    """Base contract for policies that output intent, not direct game input.

    Phase A uses ``plan_attempt`` because queued replay consumes a whole event
    schedule. Later live-RL policies can override ``act`` for per-observation
    decisions while preserving the same human-model boundary.
    """

    name = "practice_policy"

    def reset(self, level_id: str, attempt_index: int) -> None:
        """Reset per-attempt policy state."""

    def act(self, observation: Any) -> IntendedAction:
        """Return a per-observation intent for future live environments."""

        tick = getattr(observation, "tick", 0)
        return IntendedAction.no_op(tick)

    def plan_attempt(self, context: PracticeContext) -> Sequence[Event]:
        """Return intended events for a queued-replay practice attempt."""

        return []

    def update(self, attempt_result: "AttemptResult") -> None:
        """Receive the completed attempt result."""


class NoInputPolicy(PracticePolicy):
    """A baseline policy that never clicks."""

    name = "no_input"


@dataclass(slots=True)
class ScriptedEventPolicy(PracticePolicy):
    """A policy that replays a fixed intended event schedule."""

    events: Sequence[Event]
    name: str = "scripted_events"

    def plan_attempt(self, context: PracticeContext) -> Sequence[Event]:
        return sort_events(self.events)


@dataclass(slots=True)
class RandomEventPolicy(PracticePolicy):
    """Generate a legal random press/release schedule for smoke tests."""

    max_events: int
    min_tick: int = 0
    max_tick: int = 1200
    min_spacing: int = 4
    seed: int = 0
    player: Player = "p1"
    name: str = "random_events"

    def plan_attempt(self, context: PracticeContext) -> Sequence[Event]:
        if self.max_events <= 0:
            return []
        if self.min_tick < 0 or self.max_tick < self.min_tick:
            raise ValueError("random policy tick bounds are invalid")
        if self.min_spacing < 0:
            raise ValueError("min_spacing must be non-negative")

        attempt_seed = self.seed + context.attempt_index - 1
        rng = random.Random(attempt_seed)
        tick_upper = context.max_tick if context.max_tick is not None else self.max_tick
        tick_upper = min(tick_upper, self.max_tick)
        if tick_upper < self.min_tick:
            return []

        candidate_ticks = sorted(
            rng.sample(
                range(self.min_tick, tick_upper + 1),
                k=min(self.max_events, tick_upper - self.min_tick + 1),
            )
        )
        filtered_ticks: list[int] = []
        last_tick: int | None = None
        for tick in candidate_ticks:
            if last_tick is not None and tick - last_tick < self.min_spacing:
                continue
            filtered_ticks.append(tick)
            last_tick = tick

        input_down = False
        events: list[Event] = []
        for tick in filtered_ticks:
            kind = "release" if input_down else "press"
            events.append(Event(tick=tick, kind=kind, player=self.player))
            input_down = not input_down
        return events
