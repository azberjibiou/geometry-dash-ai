"""Motor delay, timing jitter, correlation, misses, and event repair."""

from __future__ import annotations

import math
import random
from typing import Iterable

from gd_human_model.events import Event, EventKind, Player, sort_events
from gd_human_model.profile import HumanProfile


class MotorNoiseModel:
    """Converts intended events into delayed, noisy actual events."""

    def __init__(self, profile: HumanProfile, rng: random.Random | None = None) -> None:
        self.profile = profile
        self.rng = rng if rng is not None else random.Random(profile.random_seed)
        self.reset()

    def reset(self) -> None:
        """Reset all temporal, correlation, and button-state memory."""

        self._previous_intended_tick: dict[Player, int] = {}
        self._previous_error: dict[Player, float] = {"p1": 0.0, "p2": 0.0}
        self._button_down: dict[Player, bool] = {"p1": False, "p2": False}
        self._next_allowed_tick: dict[Player, int] = {"p1": 0, "p2": 0}

    def timing_std(self, kind: EventKind, delta_frames: int) -> float:
        """Return interval-dependent timing standard deviation in frames."""

        if delta_frames < 0:
            raise ValueError("delta_frames must be non-negative")
        base_std = self.profile.base_std_for(kind)
        close_term = self.profile.close_amp * math.exp(-delta_frames / self.profile.close_tau)
        long_term = self.profile.long_amp * math.log1p(delta_frames / self.profile.long_tau)
        return base_std + close_term + long_term

    def miss_probability(self, delta_frames: int) -> float:
        """Return interval-dependent probability that an event is dropped."""

        if delta_frames < 0:
            raise ValueError("delta_frames must be non-negative")
        close_term = self.profile.miss_prob_close_amp * math.exp(
            -delta_frames / self.profile.miss_prob_close_tau
        )
        return min(1.0, max(0.0, self.profile.miss_prob_base + close_term))

    def should_miss(self, delta_frames: int) -> bool:
        """Sample whether an intended event is dropped."""

        return self.rng.random() < self.miss_probability(delta_frames)

    def sample_timing_error(
        self,
        kind: EventKind,
        delta_frames: int,
        player: Player = "p1",
    ) -> float:
        """Sample an AR(1)-style timing error in frames."""

        std = self.timing_std(kind, delta_frames)
        rho = self.profile.error_rho
        innovation_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        error = (
            rho * self._previous_error[player]
            + innovation_scale * std * self.rng.gauss(0.0, 1.0)
        )
        self._previous_error[player] = error
        return error

    def humanize_event(self, event: Event) -> Event | None:
        """Convert one intended event into one repaired actual event, or drop it."""

        delta = self._delta_since_previous_intended(event)
        self._previous_intended_tick[event.player] = event.tick

        if self.should_miss(delta):
            return None

        error = self.sample_timing_error(event.kind, delta, event.player)
        raw_tick = round(event.tick + self.profile.motor_delay_frames + error)
        raw_tick = max(0, int(raw_tick))
        return self._repair_button_order(event, raw_tick)

    def humanize_events(self, events: Iterable[Event]) -> list[Event]:
        """Humanize a batch of intended events in intended-time order."""

        actual_events: list[Event] = []
        for event in sort_events(events):
            actual_event = self.humanize_event(event)
            if actual_event is not None:
                actual_events.append(actual_event)
        return sort_events(actual_events)

    def _delta_since_previous_intended(self, event: Event) -> int:
        previous_tick = self._previous_intended_tick.get(event.player)
        if previous_tick is None:
            return int(round(self.profile.long_tau))
        return max(0, event.tick - previous_tick)

    def _repair_button_order(self, intended_event: Event, raw_tick: int) -> Event | None:
        player = intended_event.player
        is_down = self._button_down[player]

        if intended_event.kind == "press":
            if is_down:
                return None
            next_down = True
        else:
            if not is_down:
                return None
            next_down = False

        repaired_tick = max(raw_tick, self._next_allowed_tick[player])
        actual_event = Event(repaired_tick, intended_event.kind, player)
        self._button_down[player] = next_down
        self._next_allowed_tick[player] = repaired_tick + 1
        return actual_event
