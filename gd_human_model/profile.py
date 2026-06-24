"""Configurable human-like execution profiles."""

from __future__ import annotations

from dataclasses import dataclass

from gd_human_model.events import EventKind


@dataclass(frozen=True, slots=True)
class HumanProfile:
    """Parameters for delayed, noisy, human-like input execution."""

    name: str

    visual_delay_frames: int
    motor_delay_frames: int

    base_press_std_frames: float
    base_release_std_frames: float

    close_amp: float
    close_tau: float
    long_amp: float
    long_tau: float

    error_rho: float

    miss_prob_base: float
    miss_prob_close_amp: float
    miss_prob_close_tau: float

    random_seed: int

    def __post_init__(self) -> None:
        if self.visual_delay_frames < 0:
            raise ValueError("visual_delay_frames must be non-negative")
        if self.motor_delay_frames < 0:
            raise ValueError("motor_delay_frames must be non-negative")
        for field_name in (
            "base_press_std_frames",
            "base_release_std_frames",
            "close_amp",
            "long_amp",
            "miss_prob_base",
            "miss_prob_close_amp",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in ("close_tau", "long_tau", "miss_prob_close_tau"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not -1.0 <= self.error_rho <= 1.0:
            raise ValueError("error_rho must be between -1 and 1")
        if self.miss_prob_base > 1.0:
            raise ValueError("miss_prob_base must be at most 1")

    def base_std_for(self, kind: EventKind) -> float:
        """Return the base timing standard deviation for an event kind."""

        if kind == "press":
            return self.base_press_std_frames
        if kind == "release":
            return self.base_release_std_frames
        raise ValueError("kind must be 'press' or 'release'")


BEGINNER = HumanProfile(
    name="Beginner",
    visual_delay_frames=18,
    motor_delay_frames=18,
    base_press_std_frames=5.0,
    base_release_std_frames=5.0,
    close_amp=5.0,
    close_tau=12.0,
    long_amp=0.8,
    long_tau=120.0,
    error_rho=0.4,
    miss_prob_base=0.03,
    miss_prob_close_amp=0.07,
    miss_prob_close_tau=10.0,
    random_seed=0,
)

INTERMEDIATE = HumanProfile(
    name="Intermediate",
    visual_delay_frames=12,
    motor_delay_frames=12,
    base_press_std_frames=3.0,
    base_release_std_frames=3.0,
    close_amp=3.0,
    close_tau=10.0,
    long_amp=0.5,
    long_tau=120.0,
    error_rho=0.3,
    miss_prob_base=0.01,
    miss_prob_close_amp=0.03,
    miss_prob_close_tau=8.0,
    random_seed=0,
)

ADVANCED = HumanProfile(
    name="Advanced",
    visual_delay_frames=8,
    motor_delay_frames=8,
    base_press_std_frames=1.5,
    base_release_std_frames=1.5,
    close_amp=1.5,
    close_tau=8.0,
    long_amp=0.3,
    long_tau=120.0,
    error_rho=0.25,
    miss_prob_base=0.003,
    miss_prob_close_amp=0.01,
    miss_prob_close_tau=8.0,
    random_seed=0,
)

TOP_PLAYER = HumanProfile(
    name="TopPlayer",
    visual_delay_frames=5,
    motor_delay_frames=5,
    base_press_std_frames=0.7,
    base_release_std_frames=0.7,
    close_amp=0.8,
    close_tau=6.0,
    long_amp=0.15,
    long_tau=120.0,
    error_rho=0.2,
    miss_prob_base=0.0005,
    miss_prob_close_amp=0.003,
    miss_prob_close_tau=6.0,
    random_seed=0,
)
