"""Live step-based practice environment for observation-conditioned RL."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from math import floor
from pathlib import Path
from typing import Any, Callable, Protocol

from gd_env import BridgeDiagnostic, BridgeObservation, GeometryDashClient
from gd_human_model import Event, HumanProfile, HumanizedAgent
from gd_rl.actions import IntendedAction
from gd_rl.results import AttemptResult
from gd_rl.rewards import RewardConfig, compute_reward, summarize_trace_outcome
from gd_trace import Macro, TraceRow, save_macro_json, save_trace_jsonl


class LiveGeodeClientLike(Protocol):
    """Subset of GeometryDashClient needed by the live step environment."""

    def connect(self) -> "LiveGeodeClientLike":
        ...

    def close(self) -> None:
        ...

    def reset_attempt(
        self,
        reason: str = "requested",
        *,
        max_observations: int = 600,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        ...

    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        ...

    def send_event(self, event: Event) -> None:
        ...


@dataclass(frozen=True, slots=True)
class LivePracticeObservation:
    """Observation returned to a live policy.

    ``policy_observation`` is the delayed observation the policy should use for
    decisions. ``latest`` is included for bookkeeping, logging, and learners
    that need the current live tick for recurrent state alignment.
    """

    latest: BridgeObservation
    policy_observation: BridgeObservation | None

    @property
    def tick(self) -> int:
        return self.latest.tick

    @property
    def percent(self) -> float:
        return self.latest.percent

    @property
    def input_down(self) -> bool:
        return self.latest.input_down

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest": self.latest.to_dict(),
            "policy_observation": (
                self.policy_observation.to_dict()
                if self.policy_observation is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class LiveStepResult:
    """Result of one live environment step."""

    observation: LivePracticeObservation
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LivePracticeEnvConfig:
    """Configuration for one local/offline live practice environment."""

    level_id: str
    output_dir: Path
    max_steps: int = 1200
    reset_wait_observations: int = 600
    fps: int = 240
    cbf: bool = False
    physics_bypass: bool = False
    success_percent: float = 100.0
    base_seed: int | None = None
    action_horizon_ticks: int = 1
    observation_buffer_size: int | None = None
    post_terminal_delay_seconds: float = 0.0
    start_guard_reset_retries: int = 0
    start_guard_retry_delay_seconds: float = 0.0
    require_start_percent_max: float | None = None
    require_start_x_max: float | None = None
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.level_id:
            raise ValueError("level_id must be non-empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reset_wait_observations <= 0:
            raise ValueError("reset_wait_observations must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= self.success_percent <= 100.0:
            raise ValueError("success_percent must be between 0 and 100")
        if self.action_horizon_ticks < 0:
            raise ValueError("action_horizon_ticks must be non-negative")
        if self.post_terminal_delay_seconds < 0.0:
            raise ValueError("post_terminal_delay_seconds must be non-negative")
        if self.start_guard_reset_retries < 0:
            raise ValueError("start_guard_reset_retries must be non-negative")
        if self.start_guard_retry_delay_seconds < 0.0:
            raise ValueError("start_guard_retry_delay_seconds must be non-negative")
        if (
            self.require_start_percent_max is not None
            and not 0.0 <= self.require_start_percent_max <= 100.0
        ):
            raise ValueError("require_start_percent_max must be between 0 and 100")
        if self.require_start_x_max is not None and self.require_start_x_max < 0.0:
            raise ValueError("require_start_x_max must be non-negative")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["reward_config"] = self.reward_config.to_dict()
        return data


class LivePracticeEnv:
    """Reset/step environment that keeps policy intent behind a human model."""

    def __init__(
        self,
        *,
        config: LivePracticeEnvConfig,
        human_profile: HumanProfile,
        client_factory: Callable[[], LiveGeodeClientLike] | None = None,
    ) -> None:
        self.config = config
        self.human_profile = human_profile
        self._client_factory = client_factory or self._default_client_factory
        self._client: LiveGeodeClientLike | None = None

        self._human: HumanizedAgent[BridgeObservation] | None = None
        self._run_profile: HumanProfile | None = None
        self._current_observation: BridgeObservation | None = None
        self._attempt_index = 0
        self._attempt_seed: int | None = None
        self._attempt_dir: Path | None = None
        self._saved_result: AttemptResult | None = None

        self._rows: list[TraceRow] = []
        self._intended_events: list[Event] = []
        self._executed_events: list[Event] = []
        self._humanization_details: list[dict[str, Any]] = []
        self._diagnostics: list[BridgeDiagnostic] = []
        self._step_rewards: list[dict[str, Any]] = []

        self._step_count = 0
        self._best_percent = 0.0
        self._completed_seen = False
        self._done = False
        self._total_step_reward = 0.0
        self._charged_excessive_events = 0

    def __enter__(self) -> "LivePracticeEnv":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    def connect(self) -> "LivePracticeEnv":
        if self._client is None:
            self._client = self._client_factory().connect()
        return self

    def close(self) -> None:
        if self._rows and self._saved_result is None:
            self.save_attempt()
        if self._client is not None:
            self._client.close()
            self._client = None

    def reset(self, *, attempt_index: int | None = None) -> LivePracticeObservation:
        """Reset the live level and start a new step-based attempt."""

        if self._rows and self._saved_result is None:
            self.save_attempt()

        client = self._ensure_client()
        self._attempt_index = (
            self._attempt_index + 1 if attempt_index is None else attempt_index
        )
        if self._attempt_index <= 0:
            raise ValueError("attempt_index must be positive")

        base_seed = (
            self.human_profile.random_seed
            if self.config.base_seed is None
            else self.config.base_seed
        )
        self._attempt_seed = base_seed + self._attempt_index - 1
        self._run_profile = replace(self.human_profile, random_seed=self._attempt_seed)
        self._human = HumanizedAgent(
            self._run_profile,
            observation_buffer_size=self.config.observation_buffer_size,
        )
        self._human.reset()

        self._attempt_dir = self.config.output_dir / f"attempt_{self._attempt_index:03d}"
        self._attempt_dir.mkdir(parents=True, exist_ok=True)
        self._saved_result = None
        self._rows = []
        self._intended_events = []
        self._executed_events = []
        self._humanization_details = []
        self._diagnostics = []
        self._step_rewards = []
        self._step_count = 0
        self._completed_seen = False
        self._done = False
        self._total_step_reward = 0.0
        self._charged_excessive_events = 0

        initial_observation, reset_attempts = self._reset_until_fresh_start(client)
        self._diagnostics.append(
            BridgeDiagnostic(
                kind="live_reset_complete",
                tick=initial_observation.tick,
                data={"reset_attempts": reset_attempts},
            )
        )
        self._current_observation = initial_observation
        self._best_percent = initial_observation.percent
        self._record_observation(initial_observation)
        self._done = _is_terminal_observation(
            initial_observation,
            success_percent=self.config.success_percent,
        )
        return self._make_live_observation(initial_observation)

    def step(self, intent: IntendedAction) -> LiveStepResult:
        """Advance one observation after humanizing and dispatching policy intent."""

        if self._current_observation is None:
            raise RuntimeError("reset must be called before step")
        if self._done:
            raise RuntimeError("cannot step after the attempt is done")
        if not isinstance(intent, IntendedAction):
            raise TypeError("intent must be an IntendedAction")

        client = self._ensure_client()
        human = self._ensure_human()
        current = self._current_observation
        intended_event = self._event_from_intent(intent, current_tick=current.tick)
        event_result_info: dict[str, Any] | None = None
        if intended_event is not None:
            self._intended_events.append(intended_event)
            event_index = len(self._intended_events) - 1
            event_result = human.submit_intended_event_result(intended_event)
            event_result_info = event_result.to_dict()
            event_result_info.update(
                {
                    "event_index": event_index,
                    "requested_action": asdict(intent),
                    "live_decision_tick": current.tick,
                }
            )
            self._humanization_details.append(event_result_info)

        dispatch_cutoff_tick = current.tick + self.config.action_horizon_ticks
        dispatched_events = human.pop_due_events(dispatch_cutoff_tick)
        for event in dispatched_events:
            client.send_event(event)
            self._executed_events.append(event)

        next_observation = client.receive_observation(diagnostics=self._diagnostics)
        tick_rewound = next_observation.tick <= current.tick
        if tick_rewound:
            self._diagnostics.append(
                BridgeDiagnostic(
                    kind="live_tick_rewind_terminal",
                    tick=current.tick,
                    data={
                        "current_tick": current.tick,
                        "rewound_tick": next_observation.tick,
                        "current_percent": current.percent,
                        "rewound_percent": next_observation.percent,
                    },
                )
            )
            next_observation = replace(
                current,
                dead=True,
                completed=False,
                death_reason=current.death_reason or "tick_rewind_reset",
            )

        self._step_count += 1
        self._current_observation = next_observation
        if tick_rewound:
            self._replace_last_observation_record(next_observation)
        else:
            self._record_observation(next_observation)

        reward_terms = self._compute_step_reward(
            current=current,
            next_observation=next_observation,
        )
        reward = sum(reward_terms.values())
        self._total_step_reward += reward
        self._best_percent = max(self._best_percent, next_observation.percent)
        self._done = (
            _is_terminal_observation(
                next_observation,
                success_percent=self.config.success_percent,
            )
            or self._step_count >= self.config.max_steps
        )

        info = {
            "attempt_index": self._attempt_index,
            "step_index": self._step_count,
            "level_id": self.config.level_id,
            "intended_event": (
                asdict(intended_event) if intended_event is not None else None
            ),
            "humanization": event_result_info,
            "dispatched_events": [asdict(event) for event in dispatched_events],
            "reward_terms": reward_terms,
            "raw_observation": next_observation.to_dict(),
            "pending_executed_event_count": human.pending_count,
            "max_steps_reached": self._step_count >= self.config.max_steps,
        }
        self._step_rewards.append(
            {
                "step_index": self._step_count,
                "tick": next_observation.tick,
                "reward": reward,
                "terms": reward_terms,
            }
        )

        if self._done:
            terminal_cleared = _is_clear_observation(
                next_observation,
                success_percent=self.config.success_percent,
            )
            if terminal_cleared and self.config.post_terminal_delay_seconds > 0.0:
                time.sleep(self.config.post_terminal_delay_seconds)
            result = self.save_attempt()
            info["attempt_result"] = result.to_dict()

        return LiveStepResult(
            observation=self._make_live_observation(next_observation),
            reward=reward,
            done=self._done,
            info=info,
        )

    def save_attempt(self) -> AttemptResult:
        """Persist the current live attempt and return its summary."""

        if self._saved_result is not None:
            return self._saved_result
        if self._attempt_dir is None:
            raise RuntimeError("reset must be called before save_attempt")
        if self._run_profile is None or self._attempt_seed is None:
            raise RuntimeError("attempt profile was not initialized")

        attempt_dir = self._attempt_dir
        intended_path = attempt_dir / "policy_intended_events.json"
        executed_path = attempt_dir / "human_executed_events.json"
        humanization_path = attempt_dir / "humanization_details.json"
        trace_path = attempt_dir / "trace.jsonl"
        diagnostics_path = attempt_dir / "geode_diagnostics.json"

        intended_macro = Macro(
            events=list(self._intended_events),
            metadata={
                "level_id": self.config.level_id,
                "attempt_index": self._attempt_index,
                "kind": "policy_intent",
                "source": "live_step_env",
            },
        )
        executed_macro = Macro(
            events=list(self._executed_events),
            metadata={
                "level_id": self.config.level_id,
                "attempt_index": self._attempt_index,
                "kind": "human_executed_input",
                "source": "live_step_env",
            },
        )
        save_macro_json(intended_macro, intended_path)
        save_macro_json(executed_macro, executed_path)
        save_trace_jsonl(self._rows, trace_path)

        pending_events = self._ensure_human().flush_pending_events()
        dropped_event_count = sum(
            1 for detail in self._humanization_details if detail.get("dropped")
        )
        _write_json(
            {
                "metadata": {
                    "online": True,
                    "profile": asdict(self._run_profile),
                    "profile_name": self._run_profile.name,
                    "seed": self._attempt_seed,
                    "attempt_index": self._attempt_index,
                    "source_metadata": dict(self.config.metadata),
                    "intended_event_count": len(self._intended_events),
                    "executed_event_count": len(self._executed_events),
                    "dropped_event_count": dropped_event_count,
                    "pending_not_dispatched_count": len(pending_events),
                },
                "intended_events": [asdict(event) for event in self._intended_events],
                "actual_events": [asdict(event) for event in self._executed_events],
                "pending_not_dispatched_events": [
                    asdict(event) for event in pending_events
                ],
                "event_results": self._humanization_details,
            },
            humanization_path,
        )
        _write_json(
            {
                "executor": "geode_live_step",
                "config": self.config.to_dict(),
                "diagnostics": [diagnostic.to_dict() for diagnostic in self._diagnostics],
            },
            diagnostics_path,
        )

        outcome = summarize_trace_outcome(
            self._rows,
            success_percent=self.config.success_percent,
        )
        cleared = outcome.cleared or self._completed_seen
        attempt_reward = compute_reward(
            self._rows,
            config=self.config.reward_config,
            previous_best_percent=0.0,
            intended_event_count=len(self._intended_events),
            executed_event_count=len(self._executed_events),
        )
        result = AttemptResult(
            level_id=self.config.level_id,
            attempt_index=self._attempt_index,
            human_profile=asdict(self._run_profile),
            seed=self._attempt_seed,
            trace_path=str(trace_path),
            intended_events_path=str(intended_path),
            executed_events_path=str(executed_path),
            humanization_path=str(humanization_path),
            row_count=outcome.row_count,
            playtime_seconds=outcome.playtime_seconds,
            final_percent=outcome.final_percent,
            best_percent=outcome.best_percent,
            death_tick=outcome.death_tick,
            death_percent=outcome.death_percent,
            cleared=cleared,
            total_reward=attempt_reward.total,
            reward_terms=attempt_reward.terms,
            intended_event_count=len(self._intended_events),
            executed_event_count=len(self._executed_events),
            dropped_event_count=dropped_event_count,
            metadata={
                "executor": "geode_live_step",
                "outcome": {
                    **outcome.to_dict(),
                    "completed_seen": self._completed_seen,
                    "cleared_from_completed_flag": cleared and not outcome.cleared,
                },
                "step_count": self._step_count,
                "step_reward_total": self._total_step_reward,
                "step_rewards": self._step_rewards,
                "pending_not_dispatched_count": len(pending_events),
                "config": self.config.to_dict(),
            },
        )
        _write_json(result.to_dict(), attempt_dir / "summary.json")
        self._saved_result = result
        return result

    def _reset_until_fresh_start(
        self,
        client: LiveGeodeClientLike,
    ) -> tuple[BridgeObservation, int]:
        last_error: TimeoutError | ValueError | None = None
        max_resets = self.config.start_guard_reset_retries + 1

        for reset_index in range(max_resets):
            try:
                initial_observation = client.reset_attempt(
                    f"live_practice_attempt_{self._attempt_index}",
                    max_observations=self.config.reset_wait_observations,
                    diagnostics=self._diagnostics,
                )
                _validate_start_observation(
                    initial_observation,
                    attempt_index=self._attempt_index,
                    require_percent_max=self.config.require_start_percent_max,
                    require_x_max=self.config.require_start_x_max,
                )
                return initial_observation, reset_index + 1
            except (TimeoutError, ValueError) as exc:
                last_error = exc
                if reset_index + 1 >= max_resets:
                    raise
                if self.config.start_guard_retry_delay_seconds > 0.0:
                    time.sleep(self.config.start_guard_retry_delay_seconds)

        raise RuntimeError("unreachable reset retry state") from last_error

    def _record_observation(self, observation: BridgeObservation) -> None:
        human = self._ensure_human()
        self._rows.append(
            observation.to_trace_row(
                fps=self.config.fps,
                cbf=self.config.cbf,
                physics_bypass=self.config.physics_bypass,
            )
        )
        self._completed_seen = self._completed_seen or observation.completed
        human.observe(observation.tick, observation)

    def _replace_last_observation_record(self, observation: BridgeObservation) -> None:
        if not self._rows:
            raise RuntimeError("cannot replace missing trace row")
        self._rows[-1] = observation.to_trace_row(
            fps=self.config.fps,
            cbf=self.config.cbf,
            physics_bypass=self.config.physics_bypass,
        )
        self._completed_seen = self._completed_seen or observation.completed

    def _make_live_observation(
        self,
        observation: BridgeObservation,
    ) -> LivePracticeObservation:
        human = self._ensure_human()
        return LivePracticeObservation(
            latest=observation,
            policy_observation=human.delayed_observation(observation.tick),
        )

    def _event_from_intent(
        self,
        intent: IntendedAction,
        *,
        current_tick: int,
    ) -> Event | None:
        if intent.kind == "no_op":
            return None
        return Event(tick=current_tick, kind=intent.kind, player=intent.player)

    def _compute_step_reward(
        self,
        *,
        current: BridgeObservation,
        next_observation: BridgeObservation,
    ) -> dict[str, float]:
        reward_config = self.config.reward_config
        progress_delta = max(0.0, next_observation.percent - current.percent)
        best_progress_delta = max(0.0, next_observation.percent - self._best_percent)
        previous_sections = floor(self._best_percent / reward_config.section_size_percent)
        next_sections = floor(
            max(self._best_percent, next_observation.percent)
            / reward_config.section_size_percent
        )
        crossed_sections = max(0, next_sections - previous_sections)

        cleared = _is_clear_observation(
            next_observation,
            success_percent=self.config.success_percent,
        )
        death_penalty = 0.0
        if next_observation.dead and not cleared:
            death_penalty = -reward_config.death_penalty * max(
                0.0,
                1.0
                - next_observation.percent
                / max(reward_config.success_percent, 1e-9),
            )

        event_count_for_penalty = max(
            len(self._intended_events),
            len(self._executed_events),
        )
        excessive_events = max(
            0,
            event_count_for_penalty - reward_config.excessive_input_free_events,
        )
        newly_charged_excessive_events = max(
            0,
            excessive_events - self._charged_excessive_events,
        )
        self._charged_excessive_events += newly_charged_excessive_events

        return {
            "progress_delta": progress_delta * reward_config.progress_scale,
            "best_progress_bonus": (
                best_progress_delta * reward_config.best_progress_bonus_scale
            ),
            "section_survival_bonus": (
                crossed_sections * reward_config.section_survival_bonus
            ),
            "clear_bonus": reward_config.clear_bonus if cleared else 0.0,
            "death_penalty": death_penalty,
            "illegal_or_excessive_input_penalty": (
                -newly_charged_excessive_events
                * reward_config.excessive_input_penalty
            ),
        }

    def _ensure_client(self) -> LiveGeodeClientLike:
        if self._client is None:
            self.connect()
        if self._client is None:
            raise RuntimeError("failed to initialize Geode client")
        return self._client

    def _ensure_human(self) -> HumanizedAgent[BridgeObservation]:
        if self._human is None:
            raise RuntimeError("reset must be called before using the human model")
        return self._human

    def _default_client_factory(self) -> LiveGeodeClientLike:
        return GeometryDashClient()


def _is_clear_observation(
    observation: BridgeObservation,
    *,
    success_percent: float,
) -> bool:
    return observation.completed or observation.percent >= success_percent


def _is_terminal_observation(
    observation: BridgeObservation,
    *,
    success_percent: float,
) -> bool:
    return observation.dead or _is_clear_observation(
        observation,
        success_percent=success_percent,
    )


def _validate_start_observation(
    observation: BridgeObservation,
    *,
    attempt_index: int,
    require_percent_max: float | None,
    require_x_max: float | None,
) -> None:
    failures = []
    if (
        require_percent_max is not None
        and observation.percent > require_percent_max
    ):
        failures.append(
            f"percent {observation.percent:.3f} > {require_percent_max:.3f}"
        )
    if require_x_max is not None and observation.x > require_x_max:
        failures.append(f"x {observation.x:.3f} > {require_x_max:.3f}")
    if failures:
        raise ValueError(
            f"attempt {attempt_index} fresh start check failed: "
            + "; ".join(failures)
        )


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
