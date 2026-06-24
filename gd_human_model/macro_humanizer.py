"""Convert ideal macros into human-executed macros with provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Sequence

from gd_human_model.events import Event, sort_events
from gd_human_model.motor_noise import HumanizedEventResult, MotorNoiseModel
from gd_human_model.profile import HumanProfile

if TYPE_CHECKING:
    from gd_trace.macro_schema import Macro

TimingReference = Literal["target", "decision"]


@dataclass(frozen=True, slots=True)
class HumanizedMacroEvent:
    """Mapping from one ideal intended event to its humanized outcome."""

    event_index: int
    intended_event: Event
    decision_event: Event
    motor_result: HumanizedEventResult
    visual_delay_frames: int
    timing_reference: TimingReference
    expected_center_tick: int

    @property
    def actual_event(self) -> Event | None:
        """Return the executed event, or None when dropped."""

        return self.motor_result.actual_event

    @property
    def drop_reason(self) -> str | None:
        """Return why the event was dropped, when applicable."""

        return self.motor_result.drop_reason

    @property
    def actual_delta_frames(self) -> int | None:
        """Return actual_tick - original_intended_tick."""

        if self.actual_event is None:
            return None
        return self.actual_event.tick - self.intended_event.tick

    @property
    def delay_adjusted_delta_frames(self) -> int | None:
        """Return timing error relative to this mode's expected center."""

        if self.actual_event is None:
            return None
        return self.actual_event.tick - self.expected_center_tick

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable event provenance."""

        return {
            "event_index": self.event_index,
            "intended_event": asdict(self.intended_event),
            "decision_event": asdict(self.decision_event),
            "actual_event": (
                asdict(self.actual_event) if self.actual_event is not None else None
            ),
            "timing_reference": self.timing_reference,
            "visual_delay_frames": self.visual_delay_frames,
            "motor_delay_frames": self.motor_result.motor_delay_frames,
            "expected_center_tick": self.expected_center_tick,
            "expected_center_delta_frames": (
                self.expected_center_tick - self.intended_event.tick
            ),
            "delta_since_previous_intended": (
                self.motor_result.delta_since_previous_intended
            ),
            "timing_std_frames": self.motor_result.timing_std_frames,
            "miss_probability": self.motor_result.miss_probability,
            "sampled_error_frames": self.motor_result.sampled_error_frames,
            "raw_tick": self.motor_result.raw_tick,
            "drop_reason": self.drop_reason,
            "dropped": self.actual_event is None,
            "actual_delta_frames": self.actual_delta_frames,
            "delay_adjusted_delta_frames": self.delay_adjusted_delta_frames,
        }


@dataclass(frozen=True, slots=True)
class HumanizedMacro:
    """A humanized macro plus the per-event mapping used to produce it."""

    profile: HumanProfile
    seed: int
    attempt_index: int | None
    timing_reference: TimingReference
    source_metadata: dict[str, Any]
    event_results: list[HumanizedMacroEvent]

    @property
    def intended_events(self) -> list[Event]:
        """Return original intended events in canonical order."""

        return [result.intended_event for result in self.event_results]

    @property
    def actual_events(self) -> list[Event]:
        """Return produced actual events in canonical order."""

        return sort_events(
            result.actual_event
            for result in self.event_results
            if result.actual_event is not None
        )

    @property
    def missed_event_count(self) -> int:
        """Return the number of intended events that produced no event."""

        return sum(1 for result in self.event_results if result.actual_event is None)

    def to_macro(self, *, metadata: dict[str, Any] | None = None) -> Macro:
        """Return the actual events as a canonical macro."""

        from gd_trace.macro_schema import Macro

        macro_metadata = self.default_metadata()
        if metadata is not None:
            macro_metadata.update(metadata)
        return Macro(events=self.actual_events, metadata=macro_metadata)

    def default_metadata(self) -> dict[str, Any]:
        """Return canonical metadata for the generated macro."""

        return {
            "humanized": True,
            "profile": _profile_to_dict(self.profile),
            "profile_name": self.profile.name,
            "seed": self.seed,
            "attempt_index": self.attempt_index,
            "timing_reference": self.timing_reference,
            "source_metadata": self.source_metadata,
            "intended_event_count": len(self.event_results),
            "actual_event_count": len(self.actual_events),
            "missed_event_count": self.missed_event_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable macro provenance."""

        return {
            "metadata": self.default_metadata(),
            "intended_events": [asdict(event) for event in self.intended_events],
            "actual_events": [asdict(event) for event in self.actual_events],
            "event_results": [result.to_dict() for result in self.event_results],
        }


def humanize_macro(
    macro: Macro,
    profile: HumanProfile,
    *,
    seed: int | None = None,
    attempt_index: int | None = None,
    timing_reference: TimingReference = "target",
) -> HumanizedMacro:
    """Humanize a canonical macro with a profile and deterministic seed."""

    return humanize_macro_events(
        macro.events,
        profile,
        seed=seed,
        attempt_index=attempt_index,
        timing_reference=timing_reference,
        source_metadata=macro.metadata,
    )


def humanize_macro_events(
    events: Sequence[Event],
    profile: HumanProfile,
    *,
    seed: int | None = None,
    attempt_index: int | None = None,
    timing_reference: TimingReference = "target",
    source_metadata: dict[str, Any] | None = None,
) -> HumanizedMacro:
    """Humanize ideal intended events into actual replay events.

    In target mode, macro ticks are desired click timings, so actual events are
    centered on the macro ticks and jitter/misses happen around that target.
    In decision mode, macro ticks are policy-decision timings, so visual and
    motor delay shift actual events later.
    """

    if timing_reference not in ("target", "decision"):
        raise ValueError("timing_reference must be 'target' or 'decision'")

    effective_seed = profile.random_seed if seed is None else seed
    run_profile = replace(profile, random_seed=effective_seed)
    motor_noise = MotorNoiseModel(run_profile)
    results: list[HumanizedMacroEvent] = []

    for event_index, intended_event in enumerate(sort_events(events)):
        decision_event = _decision_event_for_reference(
            intended_event,
            run_profile,
            timing_reference=timing_reference,
        )
        motor_result = motor_noise.humanize_event_result(decision_event)
        expected_center_tick = decision_event.tick + run_profile.motor_delay_frames
        results.append(
            HumanizedMacroEvent(
                event_index=event_index,
                intended_event=intended_event,
                decision_event=decision_event,
                motor_result=motor_result,
                visual_delay_frames=run_profile.visual_delay_frames,
                timing_reference=timing_reference,
                expected_center_tick=expected_center_tick,
            )
        )

    return HumanizedMacro(
        profile=run_profile,
        seed=effective_seed,
        attempt_index=attempt_index,
        timing_reference=timing_reference,
        source_metadata=dict(source_metadata or {}),
        event_results=results,
    )


def _decision_event_for_reference(
    event: Event,
    profile: HumanProfile,
    *,
    timing_reference: TimingReference,
) -> Event:
    if timing_reference == "decision":
        return event.shifted(profile.visual_delay_frames)

    decision_tick = max(0, event.tick - profile.motor_delay_frames)
    return Event(decision_tick, event.kind, event.player)


def _profile_to_dict(profile: HumanProfile) -> dict[str, Any]:
    return asdict(profile)
