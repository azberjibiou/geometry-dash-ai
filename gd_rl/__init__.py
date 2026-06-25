"""RL-practice scaffolding for level-specific Geometry Dash agents."""

from gd_rl.actions import ActionKind, IntendedAction, actions_to_events
from gd_rl.policy import (
    NoInputPolicy,
    PracticeContext,
    PracticePolicy,
    RandomEventPolicy,
    ScriptedEventPolicy,
)
from gd_rl.results import AttemptResult, PracticeRunSummary
from gd_rl.rewards import (
    RewardBreakdown,
    RewardConfig,
    TraceOutcome,
    compute_reward,
    summarize_trace_outcome,
)
from gd_rl.runner import PracticeAttemptExecutor, PracticeRunConfig, PracticeRunner

__all__ = [
    "ActionKind",
    "AttemptResult",
    "IntendedAction",
    "NoInputPolicy",
    "PracticeAttemptExecutor",
    "PracticeContext",
    "PracticePolicy",
    "PracticeRunConfig",
    "PracticeRunSummary",
    "PracticeRunner",
    "RandomEventPolicy",
    "RewardBreakdown",
    "RewardConfig",
    "ScriptedEventPolicy",
    "TraceOutcome",
    "actions_to_events",
    "compute_reward",
    "summarize_trace_outcome",
]
