"""Tick-indexed observation storage with delayed lookup."""

from __future__ import annotations

from collections import deque
from typing import Deque, Generic, TypeVar

ObservationT = TypeVar("ObservationT")


class ObservationBuffer(Generic[ObservationT]):
    """Stores recent observations and returns older observations by delay."""

    def __init__(self, maxlen: int | None = None) -> None:
        if maxlen is not None and maxlen <= 0:
            raise ValueError("maxlen must be positive or None")
        self.maxlen = maxlen
        self._ticks: Deque[int] = deque()
        self._observations: dict[int, ObservationT] = {}
        self._last_tick: int | None = None

    def __len__(self) -> int:
        return len(self._ticks)

    def clear(self) -> None:
        self._ticks.clear()
        self._observations.clear()
        self._last_tick = None

    def add(self, tick: int, observation: ObservationT) -> None:
        """Store an observation for a non-decreasing game tick."""

        self._validate_tick(tick)
        if self._last_tick is not None and tick < self._last_tick:
            raise ValueError("observations must be added in non-decreasing tick order")

        if tick not in self._observations:
            self._ticks.append(tick)
        self._observations[tick] = observation
        self._last_tick = tick
        self._trim()

    def get(self, tick: int) -> ObservationT | None:
        """Return the exact observation for a tick, if present."""

        self._validate_tick(tick)
        return self._observations.get(tick)

    def get_at_or_before(self, tick: int) -> ObservationT | None:
        """Return the newest observation at or before a target tick."""

        self._validate_tick(tick)
        for stored_tick in reversed(self._ticks):
            if stored_tick <= tick:
                return self._observations[stored_tick]
        return None

    def get_delayed(self, current_tick: int, delay_frames: int) -> ObservationT | None:
        """Return the observation visible after a visual frame delay."""

        self._validate_tick(current_tick)
        if delay_frames < 0:
            raise ValueError("delay_frames must be non-negative")
        target_tick = current_tick - delay_frames
        if target_tick < 0:
            return None
        return self.get_at_or_before(target_tick)

    def latest(self) -> ObservationT | None:
        """Return the most recently stored observation."""

        if self._last_tick is None:
            return None
        return self._observations[self._last_tick]

    def _trim(self) -> None:
        if self.maxlen is None:
            return
        while len(self._ticks) > self.maxlen:
            old_tick = self._ticks.popleft()
            self._observations.pop(old_tick, None)

    @staticmethod
    def _validate_tick(tick: int) -> None:
        if not isinstance(tick, int):
            raise TypeError("tick must be an int")
        if tick < 0:
            raise ValueError("tick must be non-negative")
