"""Wrapper that combines delayed observations with humanized event scheduling."""

from __future__ import annotations

import heapq
from typing import Generic, Iterable, TypeVar

from gd_human_model.events import Event, event_sort_key
from gd_human_model.motor_noise import HumanizedEventResult, MotorNoiseModel
from gd_human_model.observation_buffer import ObservationBuffer
from gd_human_model.profile import HumanProfile

ObservationT = TypeVar("ObservationT")


class HumanizedAgent(Generic[ObservationT]):
    """Queues intended events after applying human-like execution noise."""

    def __init__(
        self,
        profile: HumanProfile,
        *,
        observation_buffer_size: int | None = None,
        motor_noise: MotorNoiseModel | None = None,
    ) -> None:
        self.profile = profile
        self.observations: ObservationBuffer[ObservationT] = ObservationBuffer(
            observation_buffer_size
        )
        self.motor_noise = motor_noise if motor_noise is not None else MotorNoiseModel(profile)
        self._pending: list[tuple[int, int, Event]] = []
        self._sequence = 0

    def reset(self, *, keep_observations: bool = False) -> None:
        """Reset pending input and motor state between attempts."""

        self.motor_noise.reset()
        self._pending.clear()
        self._sequence = 0
        if not keep_observations:
            self.observations.clear()

    def observe(self, tick: int, observation: ObservationT) -> None:
        """Store an observation from the environment."""

        self.observations.add(tick, observation)

    def delayed_observation(self, current_tick: int) -> ObservationT | None:
        """Return what the policy may observe after visual delay."""

        return self.observations.get_delayed(current_tick, self.profile.visual_delay_frames)

    def submit_intended_event(self, event: Event) -> Event | None:
        """Humanize and queue a single intended event."""

        result = self.submit_intended_event_result(event)
        return result.actual_event

    def submit_intended_event_result(self, event: Event) -> HumanizedEventResult:
        """Humanize and queue one event, returning detailed provenance."""

        result = self.motor_noise.humanize_event_result(event)
        actual_event = result.actual_event
        if actual_event is not None:
            self._queue(actual_event)
        return result

    def submit_intended_events(self, events: Iterable[Event]) -> list[Event]:
        """Humanize and queue intended events."""

        actual_events = self.motor_noise.humanize_events(events)
        for actual_event in actual_events:
            self._queue(actual_event)
        return actual_events

    def pop_due_events(self, current_tick: int) -> list[Event]:
        """Return all actual events scheduled at or before the current tick."""

        due_events: list[Event] = []
        while self._pending and self._pending[0][0] <= current_tick:
            _, _, event = heapq.heappop(self._pending)
            due_events.append(event)
        return sorted(due_events, key=event_sort_key)

    def flush_pending_events(self) -> list[Event]:
        """Return all queued events, sorted by actual tick."""

        events = [item[2] for item in self._pending]
        self._pending.clear()
        return sorted(events, key=event_sort_key)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _queue(self, event: Event) -> None:
        heapq.heappush(self._pending, (event.tick, self._sequence, event))
        self._sequence += 1
