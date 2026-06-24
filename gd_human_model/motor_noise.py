"""Motor delay, timing jitter, correlation, misses, and event repair."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable
from typing import Literal

from gd_human_model.events import Event, EventKind, Player, sort_events
from gd_human_model.profile import HumanProfile

DropReason = Literal["miss", "button_order"]


@dataclass(frozen=True, slots=True)
class HumanizedEventResult:
    """Detailed result for one intended event after human-like execution."""

    intended_event: Event
    actual_event: Event | None
    delta_since_previous_intended: int
    motor_delay_frames: int
    timing_std_frames: float
    miss_probability: float
    sampled_error_frames: float | None
    raw_tick: int | None
    drop_reason: DropReason | None

    @property
    def dropped(self) -> bool:
        """Return whether the intended event produced no actual event."""

        return self.actual_event is None

    @property
    def actual_delta_frames(self) -> int | None:
        """Return actual_tick - intended_tick when an event was produced."""

        if self.actual_event is None:
            return None
        return self.actual_event.tick - self.intended_event.tick

    @property
    def delay_adjusted_delta_frames(self) -> int | None:
        """Return timing delta after subtracting the configured motor delay."""

        if self.actual_event is None:
            return None
        return (
            self.actual_event.tick
            - self.intended_event.tick
            - self.motor_delay_frames
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable result data."""

        data = asdict(self)
        data["dropped"] = self.dropped
        data["actual_delta_frames"] = self.actual_delta_frames
        data["delay_adjusted_delta_frames"] = self.delay_adjusted_delta_frames
        return data


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

        return self.humanize_event_result(event).actual_event

    def humanize_event_result(self, event: Event) -> HumanizedEventResult:
        """Convert one event and return detailed timing/drop provenance."""

        delta = self._delta_since_previous_intended(event)
        self._previous_intended_tick[event.player] = event.tick
        timing_std = self.timing_std(event.kind, delta)
        miss_probability = self.miss_probability(delta)

        if self.should_miss(delta):
            return HumanizedEventResult(
                intended_event=event,
                actual_event=None,
                delta_since_previous_intended=delta,
                motor_delay_frames=self.profile.motor_delay_frames,
                timing_std_frames=timing_std,
                miss_probability=miss_probability,
                sampled_error_frames=None,
                raw_tick=None,
                drop_reason="miss",
            )

        error = self.sample_timing_error(event.kind, delta, event.player)
        raw_tick = round(event.tick + self.profile.motor_delay_frames + error)
        raw_tick = max(0, int(raw_tick))
        actual_event = self._repair_button_order(event, raw_tick)
        return HumanizedEventResult(
            intended_event=event,
            actual_event=actual_event,
            delta_since_previous_intended=delta,
            motor_delay_frames=self.profile.motor_delay_frames,
            timing_std_frames=timing_std,
            miss_probability=miss_probability,
            sampled_error_frames=error,
            raw_tick=raw_tick,
            drop_reason=None if actual_event is not None else "button_order",
        )

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
