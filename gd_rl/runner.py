"""Repeated-attempt practice runner with a mandatory human-model boundary."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from gd_human_model import HumanProfile, humanize_macro
from gd_trace import Macro, TraceRow, save_macro_json, save_trace_jsonl

from gd_rl.policy import PracticeContext, PracticePolicy
from gd_rl.results import AttemptResult, PracticeRunSummary
from gd_rl.rewards import RewardConfig, compute_reward, summarize_trace_outcome


class PracticeAttemptExecutor(Protocol):
    """Runs one attempt from a humanized executed macro and returns a trace."""

    def run_attempt(
        self,
        *,
        attempt_index: int,
        executed_macro: Macro,
        attempt_dir: Path,
        metadata: dict[str, Any],
    ) -> Sequence[TraceRow]:
        """Execute one attempt and return recorded trace rows."""


@dataclass(frozen=True, slots=True)
class PracticeRunConfig:
    """Configuration for a Phase A repeated-attempt practice run."""

    level_id: str
    attempts: int
    output_dir: Path
    max_tick: int | None = None
    success_percent: float = 100.0
    stop_after_first_clear: bool = False
    base_seed: int | None = None
    timing_reference: str = "target"
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")
        if not 0.0 <= self.success_percent <= 100.0:
            raise ValueError("success_percent must be between 0 and 100")
        if self.timing_reference not in ("target", "decision"):
            raise ValueError("timing_reference must be 'target' or 'decision'")


class PracticeRunner:
    """Run a policy through the human model over repeated attempts."""

    def __init__(
        self,
        *,
        policy: PracticePolicy,
        executor: PracticeAttemptExecutor,
        human_profile: HumanProfile,
        config: PracticeRunConfig,
    ) -> None:
        self.policy = policy
        self.executor = executor
        self.human_profile = human_profile
        self.config = config

    def run(self) -> PracticeRunSummary:
        """Run all attempts, persist artifacts, and return an aggregate summary."""

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        base_seed = (
            self.human_profile.random_seed
            if self.config.base_seed is None
            else self.config.base_seed
        )
        attempt_results: list[AttemptResult] = []
        previous_best_percent = 0.0

        for attempt_index in range(1, self.config.attempts + 1):
            attempt_dir = output_dir / f"attempt_{attempt_index:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_seed = base_seed + attempt_index - 1
            context = PracticeContext(
                level_id=self.config.level_id,
                attempt_index=attempt_index,
                max_tick=self.config.max_tick,
                metadata=dict(self.config.metadata),
            )

            self.policy.reset(self.config.level_id, attempt_index)
            intended_events = list(self.policy.plan_attempt(context))
            intended_macro = Macro(
                events=intended_events,
                metadata={
                    "level_id": self.config.level_id,
                    "attempt_index": attempt_index,
                    "policy_name": self.policy.name,
                    "kind": "policy_intent",
                },
            )
            intended_path = attempt_dir / "policy_intended_events.json"
            save_macro_json(intended_macro, intended_path)

            humanized = humanize_macro(
                intended_macro,
                self.human_profile,
                seed=attempt_seed,
                attempt_index=attempt_index,
                timing_reference=self.config.timing_reference,  # type: ignore[arg-type]
            )
            executed_macro = humanized.to_macro(
                metadata={
                    "level_id": self.config.level_id,
                    "attempt_index": attempt_index,
                    "policy_name": self.policy.name,
                    "kind": "human_executed_input",
                }
            )
            executed_path = attempt_dir / "human_executed_events.json"
            humanization_path = attempt_dir / "humanization_details.json"
            save_macro_json(executed_macro, executed_path)
            _write_json(humanized.to_dict(), humanization_path)

            rows = list(
                self.executor.run_attempt(
                    attempt_index=attempt_index,
                    executed_macro=executed_macro,
                    attempt_dir=attempt_dir,
                    metadata={
                        "level_id": self.config.level_id,
                        "policy_name": self.policy.name,
                        "seed": attempt_seed,
                    },
                )
            )
            trace_path = attempt_dir / "trace.jsonl"
            save_trace_jsonl(rows, trace_path)
            outcome = summarize_trace_outcome(
                rows,
                success_percent=self.config.success_percent,
            )
            reward = compute_reward(
                rows,
                config=self.config.reward_config,
                previous_best_percent=previous_best_percent,
                intended_event_count=len(intended_macro.events),
                executed_event_count=len(executed_macro.events),
            )
            previous_best_percent = max(previous_best_percent, outcome.best_percent)

            result = AttemptResult(
                level_id=self.config.level_id,
                attempt_index=attempt_index,
                human_profile=asdict(self.human_profile),
                seed=attempt_seed,
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
                cleared=outcome.cleared,
                total_reward=reward.total,
                reward_terms=reward.terms,
                intended_event_count=len(intended_macro.events),
                executed_event_count=len(executed_macro.events),
                dropped_event_count=humanized.missed_event_count,
                metadata={
                    "policy_name": self.policy.name,
                    "timing_reference": self.config.timing_reference,
                    "outcome": outcome.to_dict(),
                },
            )
            attempt_results.append(result)
            _write_json(result.to_dict(), attempt_dir / "summary.json")
            self.policy.update(result)
            if self.config.stop_after_first_clear and result.cleared:
                break

        summary = PracticeRunSummary.from_attempts(
            level_id=self.config.level_id,
            attempts=attempt_results,
            metadata={
                "policy_name": self.policy.name,
                "human_profile_name": self.human_profile.name,
                "config": {
                    "attempts": self.config.attempts,
                    "max_tick": self.config.max_tick,
                    "success_percent": self.config.success_percent,
                    "stop_after_first_clear": self.config.stop_after_first_clear,
                    "base_seed": base_seed,
                    "timing_reference": self.config.timing_reference,
                    "reward_config": self.config.reward_config.to_dict(),
                    "metadata": self.config.metadata,
                },
            },
        )
        _write_json(summary.to_dict(), output_dir / "summary.json")
        return summary


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
