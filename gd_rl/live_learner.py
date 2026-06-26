"""Tiny neural learner for the live step practice environment."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from gd_rl.actions import ActionKind, IntendedAction
from gd_rl.live_env import LivePracticeObservation, LiveStepResult
from gd_rl.results import AttemptResult

ACTION_KINDS: tuple[ActionKind, ...] = ("no_op", "press", "release")
DesiredInputState = Literal["idle", "hold"]
DESIRED_INPUT_STATES: tuple[DesiredInputState, ...] = ("idle", "hold")
MODE_ORDER = (
    "cube",
    "ship",
    "ufo",
    "ball",
    "wave",
    "robot",
    "spider",
    "swing",
    "unknown",
)


class LiveLearnerError(ValueError):
    """Raised when the tiny live RL learner cannot run."""


class LiveStepEnvLike(Protocol):
    """Small reset/step/save contract used by the learner."""

    def reset(self, *, attempt_index: int | None = None) -> LivePracticeObservation:
        ...

    def step(self, intent: IntendedAction) -> LiveStepResult:
        ...

    def save_attempt(self) -> AttemptResult:
        ...


@dataclass(frozen=True, slots=True)
class LiveObservationEncoderConfig:
    """Feature scaling for compact bridge observations."""

    max_tick: int = 1200
    x_scale: float = 1000.0
    y_scale: float = 500.0
    velocity_scale: float = 20.0
    rotation_scale: float = 360.0

    def __post_init__(self) -> None:
        if self.max_tick <= 0:
            raise LiveLearnerError("max_tick must be positive")
        for field_name in ("x_scale", "y_scale", "velocity_scale", "rotation_scale"):
            if getattr(self, field_name) <= 0.0:
                raise LiveLearnerError(f"{field_name} must be positive")


@dataclass(frozen=True, slots=True)
class NeuralPolicyConfig:
    """Settings for the tiny MLP policy."""

    hidden_size: int = 32
    seed: int = 0
    device: str = "cpu"
    encoder: LiveObservationEncoderConfig = field(
        default_factory=LiveObservationEncoderConfig
    )

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise LiveLearnerError("hidden_size must be positive")


@dataclass(frozen=True, slots=True)
class ReinforceConfig:
    """One small policy-gradient update loop for live practice."""

    attempts: int = 1
    gamma: float = 0.99
    learning_rate: float = 1e-3
    entropy_bonus: float = 0.0
    max_grad_norm: float | None = 1.0
    normalize_returns: bool = False
    deterministic_actions: bool = False
    seed: int = 0
    min_dwell_ticks: int = 4

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise LiveLearnerError("attempts must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise LiveLearnerError("gamma must be between 0 and 1")
        if self.learning_rate <= 0.0:
            raise LiveLearnerError("learning_rate must be positive")
        if self.entropy_bonus < 0.0:
            raise LiveLearnerError("entropy_bonus must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise LiveLearnerError("max_grad_norm must be positive or None")
        if self.min_dwell_ticks < 0:
            raise LiveLearnerError("min_dwell_ticks must be non-negative")


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    """Small actor-critic update loop for closed-loop live practice."""

    attempts: int = 1
    gamma: float = 0.99
    learning_rate: float = 1e-3
    entropy_bonus: float = 0.01
    value_loss_weight: float = 0.5
    max_grad_norm: float | None = 1.0
    normalize_advantages: bool = False
    deterministic_actions: bool = False
    seed: int = 0
    history_length: int = 4
    death_local_window: int = 24
    death_local_penalty: float = 1.0
    input_rate_penalty: float = 0.0
    min_dwell_ticks: int = 4

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise LiveLearnerError("attempts must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise LiveLearnerError("gamma must be between 0 and 1")
        if self.learning_rate <= 0.0:
            raise LiveLearnerError("learning_rate must be positive")
        if self.entropy_bonus < 0.0:
            raise LiveLearnerError("entropy_bonus must be non-negative")
        if self.value_loss_weight < 0.0:
            raise LiveLearnerError("value_loss_weight must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise LiveLearnerError("max_grad_norm must be positive or None")
        if self.history_length < 0:
            raise LiveLearnerError("history_length must be non-negative")
        if self.death_local_window < 0:
            raise LiveLearnerError("death_local_window must be non-negative")
        if self.death_local_penalty < 0.0:
            raise LiveLearnerError("death_local_penalty must be non-negative")
        if self.input_rate_penalty < 0.0:
            raise LiveLearnerError("input_rate_penalty must be non-negative")
        if self.min_dwell_ticks < 0:
            raise LiveLearnerError("min_dwell_ticks must be non-negative")


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Small value-based baseline for closed-loop live practice."""

    attempts: int = 1
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 0.20
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1000
    batch_size: int = 32
    replay_capacity: int = 2048
    warmup_steps: int = 32
    target_update_interval: int = 100
    max_grad_norm: float | None = 1.0
    deterministic_actions: bool = False
    seed: int = 0
    history_length: int = 4
    input_rate_penalty: float = 0.0
    min_dwell_ticks: int = 4
    skip_replay_death_reasons: tuple[str, ...] = ("tick_rewind_reset",)

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise LiveLearnerError("attempts must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise LiveLearnerError("gamma must be between 0 and 1")
        if self.learning_rate <= 0.0:
            raise LiveLearnerError("learning_rate must be positive")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise LiveLearnerError("epsilon_start must be between 0 and 1")
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise LiveLearnerError("epsilon_end must be between 0 and 1")
        if self.epsilon_decay_steps <= 0:
            raise LiveLearnerError("epsilon_decay_steps must be positive")
        if self.batch_size <= 0:
            raise LiveLearnerError("batch_size must be positive")
        if self.replay_capacity <= 0:
            raise LiveLearnerError("replay_capacity must be positive")
        if self.replay_capacity < self.batch_size:
            raise LiveLearnerError("replay_capacity must be at least batch_size")
        if self.warmup_steps < 0:
            raise LiveLearnerError("warmup_steps must be non-negative")
        if self.target_update_interval <= 0:
            raise LiveLearnerError("target_update_interval must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise LiveLearnerError("max_grad_norm must be positive or None")
        if self.history_length < 0:
            raise LiveLearnerError("history_length must be non-negative")
        if self.input_rate_penalty < 0.0:
            raise LiveLearnerError("input_rate_penalty must be non-negative")
        if self.min_dwell_ticks < 0:
            raise LiveLearnerError("min_dwell_ticks must be non-negative")
        if any(not reason for reason in self.skip_replay_death_reasons):
            raise LiveLearnerError("skip_replay_death_reasons cannot contain empty values")


@dataclass(slots=True)
class ButtonStateIntentAdapter:
    """Map desired idle/hold states to intended press/release edges.

    The adapter tracks intended button state instead of live executed input.
    That keeps visual and motor delay from causing repeated press/release
    intents while an already-requested humanized event is still pending.
    """

    intended_input_down: bool = False
    min_dwell_ticks: int = 0
    last_transition_tick: int | None = None

    def __post_init__(self) -> None:
        if self.min_dwell_ticks < 0:
            raise LiveLearnerError("min_dwell_ticks must be non-negative")

    def reset(self, observation: LivePracticeObservation | None = None) -> None:
        """Start a new attempt from the current fresh-reset input state."""

        self.intended_input_down = (
            bool(observation.latest.input_down) if observation is not None else False
        )
        self.last_transition_tick = None

    def intent_for_desired_state(
        self,
        desired_input_state: DesiredInputState,
        *,
        tick: int,
    ) -> IntendedAction:
        """Return the edge needed to reach the desired intended state."""

        if desired_input_state == "hold":
            target_down = True
        elif desired_input_state == "idle":
            target_down = False
        else:
            raise LiveLearnerError(
                f"unknown desired input state {desired_input_state!r}"
            )

        if target_down == self.intended_input_down:
            return IntendedAction.no_op(tick)
        if not self.can_transition(tick):
            return IntendedAction.no_op(tick)

        self.intended_input_down = target_down
        self.last_transition_tick = tick
        if target_down:
            return IntendedAction.press(tick)
        return IntendedAction.release(tick)

    def can_transition(self, tick: int) -> bool:
        """Return whether the dwell window allows a new input edge."""

        return (
            self.last_transition_tick is None
            or tick - self.last_transition_tick >= self.min_dwell_ticks
        )

    @property
    def effective_input_state(self) -> DesiredInputState:
        """Current intended button state after dwell gating."""

        return "hold" if self.intended_input_down else "idle"


@dataclass(frozen=True, slots=True)
class LiveActionHistoryEntry:
    """One recent intended decision for compact delay-aware features."""

    desired_input_state: DesiredInputState
    intent_kind: ActionKind


@dataclass(slots=True)
class LiveActionHistory:
    """Fixed-length intended action history exposed to closed-loop learners."""

    length: int = 4
    entries: list[LiveActionHistoryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.length < 0:
            raise LiveLearnerError("history length must be non-negative")

    def reset(self) -> None:
        self.entries.clear()

    def append(
        self,
        *,
        desired_input_state: DesiredInputState,
        intent_kind: ActionKind,
    ) -> None:
        if self.length == 0:
            return
        self.entries.append(
            LiveActionHistoryEntry(
                desired_input_state=desired_input_state,
                intent_kind=intent_kind,
            )
        )
        if len(self.entries) > self.length:
            del self.entries[: len(self.entries) - self.length]

    def features(self) -> list[float]:
        if self.length == 0:
            return []
        padded: list[LiveActionHistoryEntry | None] = [None] * (
            self.length - len(self.entries)
        )
        padded.extend(self.entries)
        features: list[float] = []
        for entry in padded:
            if entry is None:
                features.extend([0.0] * live_action_history_entry_feature_dim())
                continue
            features.append(1.0 if entry.desired_input_state == "hold" else 0.0)
            features.extend(
                1.0 if entry.intent_kind == kind else 0.0
                for kind in ACTION_KINDS
            )
        return features


@dataclass(slots=True)
class NeuralActionDecision:
    """One policy decision plus tensors needed for REINFORCE."""

    intent: IntendedAction
    action_index: int
    desired_input_state: DesiredInputState
    effective_input_state: DesiredInputState
    dwell_blocked: bool
    desired_holding: bool
    probability: float
    logits: list[float]
    features: list[float]
    log_probability_tensor: Any = field(repr=False)
    entropy_tensor: Any = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": asdict(self.intent),
            "action_index": self.action_index,
            "desired_input_state": self.desired_input_state,
            "effective_input_state": self.effective_input_state,
            "dwell_blocked": self.dwell_blocked,
            "desired_holding": self.desired_holding,
            "probability": self.probability,
            "logits": list(self.logits),
            "features": list(self.features),
        }


@dataclass(slots=True)
class ActorCriticActionDecision:
    """One actor-critic decision plus policy/value tensors."""

    intent: IntendedAction
    action_index: int
    desired_input_state: DesiredInputState
    effective_input_state: DesiredInputState
    dwell_blocked: bool
    desired_holding: bool
    probability: float
    logits: list[float]
    value: float
    features: list[float]
    log_probability_tensor: Any = field(repr=False)
    entropy_tensor: Any = field(repr=False)
    value_tensor: Any = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": asdict(self.intent),
            "action_index": self.action_index,
            "desired_input_state": self.desired_input_state,
            "effective_input_state": self.effective_input_state,
            "dwell_blocked": self.dwell_blocked,
            "desired_holding": self.desired_holding,
            "probability": self.probability,
            "logits": list(self.logits),
            "value": self.value,
            "features": list(self.features),
        }


@dataclass(slots=True)
class DQNActionDecision:
    """One epsilon-greedy DQN decision for desired button state."""

    intent: IntendedAction
    action_index: int
    desired_input_state: DesiredInputState
    effective_input_state: DesiredInputState
    dwell_blocked: bool
    desired_holding: bool
    epsilon: float
    greedy: bool
    q_values: list[float]
    features: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": asdict(self.intent),
            "action_index": self.action_index,
            "desired_input_state": self.desired_input_state,
            "effective_input_state": self.effective_input_state,
            "dwell_blocked": self.dwell_blocked,
            "desired_holding": self.desired_holding,
            "epsilon": self.epsilon,
            "greedy": self.greedy,
            "q_values": list(self.q_values),
            "features": list(self.features),
        }


@dataclass(frozen=True, slots=True)
class ReinforceAttemptSummary:
    """Training metrics for one live attempt."""

    attempt_index: int
    step_count: int
    total_step_reward: float
    loss: float
    policy_loss: float
    entropy: float
    action_counts: dict[str, int]
    effective_action_counts: dict[str, int]
    dwell_blocked_count: int
    intent_counts: dict[str, int]
    attempt_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActorCriticAttemptSummary:
    """Training metrics for one A2C-style live attempt."""

    attempt_index: int
    step_count: int
    total_step_reward: float
    total_training_reward: float
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    mean_value: float
    advantage_stats: dict[str, float]
    action_counts: dict[str, int]
    effective_action_counts: dict[str, int]
    dwell_blocked_count: int
    intent_counts: dict[str, int]
    death_local_stats: dict[str, Any]
    attempt_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReinforceTrainingSummary:
    """Aggregate metrics for a short live REINFORCE run."""

    attempts: list[ReinforceAttemptSummary]
    config: dict[str, Any]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_count": self.attempt_count,
            "config": dict(self.config),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class ActorCriticTrainingSummary:
    """Aggregate metrics for a short live actor-critic run."""

    attempts: list[ActorCriticAttemptSummary]
    config: dict[str, Any]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_count": self.attempt_count,
            "config": dict(self.config),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class DQNAttemptSummary:
    """Training metrics for one tiny DQN live attempt."""

    attempt_index: int
    step_count: int
    total_step_reward: float
    total_training_reward: float
    mean_loss: float | None
    last_loss: float | None
    update_count: int
    replay_size: int
    replay_appended_count: int
    replay_skipped: bool
    replay_skip_reason: str | None
    epsilon_start: float
    epsilon_end: float
    q_value_stats: dict[str, float]
    action_counts: dict[str, int]
    effective_action_counts: dict[str, int]
    dwell_blocked_count: int
    intent_counts: dict[str, int]
    attempt_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DQNTrainingSummary:
    """Aggregate metrics for a short live DQN baseline run."""

    attempts: list[DQNAttemptSummary]
    config: dict[str, Any]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_count": self.attempt_count,
            "config": dict(self.config),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


class TinyLivePolicyNetwork:
    """Small MLP policy that maps compact observations to desired button state."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: NeuralPolicyConfig | None = None,
    ) -> None:
        self.config = config or NeuralPolicyConfig()
        if input_dim <= 0:
            raise LiveLearnerError("input_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = len(DESIRED_INPUT_STATES)
        self.torch = _import_torch()
        _set_seed(self.torch, self.config.seed)
        self.device = self.torch.device(self.config.device)
        self.model = self.torch.nn.Sequential(
            self.torch.nn.Linear(input_dim, self.config.hidden_size),
            self.torch.nn.Tanh(),
            self.torch.nn.Linear(self.config.hidden_size, self.output_dim),
        ).to(self.device)

    @classmethod
    def from_encoder_config(
        cls,
        config: NeuralPolicyConfig | None = None,
    ) -> "TinyLivePolicyNetwork":
        effective_config = config or NeuralPolicyConfig()
        return cls(
            input_dim=live_observation_feature_dim(),
            config=effective_config,
        )

    def logits(self, observation: LivePracticeObservation) -> Any:
        features = encode_live_observation(observation, config=self.config.encoder)
        feature_tensor = self.torch.tensor(
            features,
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return self.model(feature_tensor).squeeze(0)

    def action_probabilities(self, observation: LivePracticeObservation) -> list[float]:
        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.logits(observation), dim=-1)
        return [float(value) for value in probabilities.detach().cpu().tolist()]

    def act(
        self,
        observation: LivePracticeObservation,
        *,
        intent_adapter: ButtonStateIntentAdapter | None = None,
        deterministic: bool = False,
    ) -> NeuralActionDecision:
        features = encode_live_observation(observation, config=self.config.encoder)
        feature_tensor = self.torch.tensor(
            features,
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits = self.model(feature_tensor).squeeze(0)
        distribution = self.torch.distributions.Categorical(logits=logits)
        if deterministic:
            action_index_tensor = self.torch.argmax(logits, dim=-1)
        else:
            action_index_tensor = distribution.sample()
        action_index = int(action_index_tensor.detach().cpu().item())
        desired_input_state = DESIRED_INPUT_STATES[action_index]
        adapter = intent_adapter or _adapter_from_observation(observation)
        intent = adapter.intent_for_desired_state(
            desired_input_state,
            tick=observation.tick,
        )
        effective_input_state = adapter.effective_input_state
        probability = float(
            self.torch.softmax(logits, dim=-1)[action_index].detach().cpu().item()
        )
        return NeuralActionDecision(
            intent=intent,
            action_index=action_index,
            desired_input_state=desired_input_state,
            effective_input_state=effective_input_state,
            dwell_blocked=effective_input_state != desired_input_state,
            desired_holding=desired_input_state == "hold",
            probability=probability,
            logits=[float(value) for value in logits.detach().cpu().tolist()],
            features=features,
            log_probability_tensor=distribution.log_prob(action_index_tensor),
            entropy_tensor=distribution.entropy(),
        )

    def make_optimizer(self, config: ReinforceConfig) -> Any:
        return self.torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)


class TinyLiveActorCriticNetwork:
    """Small shared MLP with actor and value heads for desired button state."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: NeuralPolicyConfig | None = None,
        history_length: int = 4,
    ) -> None:
        self.config = config or NeuralPolicyConfig()
        if input_dim <= 0:
            raise LiveLearnerError("input_dim must be positive")
        if history_length < 0:
            raise LiveLearnerError("history_length must be non-negative")
        self.input_dim = input_dim
        self.history_length = history_length
        self.output_dim = len(DESIRED_INPUT_STATES)
        self.torch = _import_torch()
        _set_seed(self.torch, self.config.seed)
        self.device = self.torch.device(self.config.device)
        self.shared = self.torch.nn.Sequential(
            self.torch.nn.Linear(input_dim, self.config.hidden_size),
            self.torch.nn.Tanh(),
        ).to(self.device)
        self.actor_head = self.torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
        ).to(self.device)
        self.value_head = self.torch.nn.Linear(self.config.hidden_size, 1).to(
            self.device
        )

    @classmethod
    def from_encoder_config(
        cls,
        config: NeuralPolicyConfig | None = None,
        *,
        history_length: int = 4,
    ) -> "TinyLiveActorCriticNetwork":
        effective_config = config or NeuralPolicyConfig()
        return cls(
            input_dim=actor_critic_feature_dim(history_length),
            config=effective_config,
            history_length=history_length,
        )

    def forward_features(self, features: Sequence[float]) -> tuple[Any, Any]:
        feature_tensor = self.torch.tensor(
            list(features),
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        hidden = self.shared(feature_tensor)
        logits = self.actor_head(hidden).squeeze(0)
        value = self.value_head(hidden).squeeze(0).squeeze(-1)
        return logits, value

    def logits_and_value(
        self,
        observation: LivePracticeObservation,
        *,
        history: LiveActionHistory | None = None,
    ) -> tuple[Any, Any]:
        features = encode_actor_critic_observation(
            observation,
            history=history,
            history_length=self.history_length,
            config=self.config.encoder,
        )
        return self.forward_features(features)

    def action_probabilities(
        self,
        observation: LivePracticeObservation,
        *,
        history: LiveActionHistory | None = None,
    ) -> list[float]:
        with self.torch.no_grad():
            logits, _value = self.logits_and_value(observation, history=history)
            probabilities = self.torch.softmax(logits, dim=-1)
        return [float(value) for value in probabilities.detach().cpu().tolist()]

    def act(
        self,
        observation: LivePracticeObservation,
        *,
        intent_adapter: ButtonStateIntentAdapter | None = None,
        history: LiveActionHistory | None = None,
        deterministic: bool = False,
    ) -> ActorCriticActionDecision:
        features = encode_actor_critic_observation(
            observation,
            history=history,
            history_length=self.history_length,
            config=self.config.encoder,
        )
        logits, value_tensor = self.forward_features(features)
        distribution = self.torch.distributions.Categorical(logits=logits)
        if deterministic:
            action_index_tensor = self.torch.argmax(logits, dim=-1)
        else:
            action_index_tensor = distribution.sample()
        action_index = int(action_index_tensor.detach().cpu().item())
        desired_input_state = DESIRED_INPUT_STATES[action_index]
        adapter = intent_adapter or _adapter_from_observation(observation)
        intent = adapter.intent_for_desired_state(
            desired_input_state,
            tick=observation.tick,
        )
        effective_input_state = adapter.effective_input_state
        probability = float(
            self.torch.softmax(logits, dim=-1)[action_index].detach().cpu().item()
        )
        return ActorCriticActionDecision(
            intent=intent,
            action_index=action_index,
            desired_input_state=desired_input_state,
            effective_input_state=effective_input_state,
            dwell_blocked=effective_input_state != desired_input_state,
            desired_holding=desired_input_state == "hold",
            probability=probability,
            logits=[float(value) for value in logits.detach().cpu().tolist()],
            value=float(value_tensor.detach().cpu().item()),
            features=features,
            log_probability_tensor=distribution.log_prob(action_index_tensor),
            entropy_tensor=distribution.entropy(),
            value_tensor=value_tensor,
        )

    def make_optimizer(self, config: ActorCriticConfig) -> Any:
        return self.torch.optim.Adam(
            [
                *self.shared.parameters(),
                *self.actor_head.parameters(),
                *self.value_head.parameters(),
            ],
            lr=config.learning_rate,
        )


@dataclass(frozen=True, slots=True)
class DQNTransition:
    """One transition stored by the tiny DQN replay buffer."""

    features: list[float]
    action_index: int
    reward: float
    next_features: list[float]
    done: bool


class DQNReplayBuffer:
    """Small in-memory replay buffer for short diagnostic live runs."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise LiveLearnerError("replay buffer capacity must be positive")
        self.capacity = capacity
        self._items: deque[DQNTransition] = deque(maxlen=capacity)

    def append(self, transition: DQNTransition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[DQNTransition]:
        if batch_size <= 0:
            raise LiveLearnerError("batch_size must be positive")
        return random.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)


class TinyLiveDQNNetwork:
    """Small MLP Q-network for desired idle/hold state values."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: NeuralPolicyConfig | None = None,
        history_length: int = 4,
    ) -> None:
        self.config = config or NeuralPolicyConfig()
        if input_dim <= 0:
            raise LiveLearnerError("input_dim must be positive")
        if history_length < 0:
            raise LiveLearnerError("history_length must be non-negative")
        self.input_dim = input_dim
        self.history_length = history_length
        self.output_dim = len(DESIRED_INPUT_STATES)
        self.torch = _import_torch()
        _set_seed(self.torch, self.config.seed)
        self.device = self.torch.device(self.config.device)
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.sync_target()

    @classmethod
    def from_encoder_config(
        cls,
        config: NeuralPolicyConfig | None = None,
        *,
        history_length: int = 4,
    ) -> "TinyLiveDQNNetwork":
        effective_config = config or NeuralPolicyConfig()
        return cls(
            input_dim=dqn_feature_dim(history_length),
            config=effective_config,
            history_length=history_length,
        )

    def _build_network(self) -> Any:
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(self.input_dim, self.config.hidden_size),
            self.torch.nn.Tanh(),
            self.torch.nn.Linear(self.config.hidden_size, self.output_dim),
        )

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def q_values_from_features(self, features: Sequence[float]) -> Any:
        feature_tensor = self.torch.tensor(
            list(features),
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return self.q_network(feature_tensor).squeeze(0)

    def q_values(
        self,
        observation: LivePracticeObservation,
        *,
        history: LiveActionHistory | None = None,
    ) -> list[float]:
        features = encode_dqn_observation(
            observation,
            history=history,
            history_length=self.history_length,
            config=self.config.encoder,
        )
        with self.torch.no_grad():
            values = self.q_values_from_features(features)
        return [float(value) for value in values.detach().cpu().tolist()]

    def act(
        self,
        observation: LivePracticeObservation,
        *,
        intent_adapter: ButtonStateIntentAdapter | None = None,
        history: LiveActionHistory | None = None,
        epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> DQNActionDecision:
        features = encode_dqn_observation(
            observation,
            history=history,
            history_length=self.history_length,
            config=self.config.encoder,
        )
        q_values_tensor = self.q_values_from_features(features)
        q_values = [float(value) for value in q_values_tensor.detach().cpu().tolist()]
        greedy_index = int(q_values_tensor.argmax(dim=-1).detach().cpu().item())
        if deterministic or random.random() >= epsilon:
            action_index = greedy_index
            greedy = True
        else:
            action_index = random.randrange(self.output_dim)
            greedy = action_index == greedy_index

        desired_input_state = DESIRED_INPUT_STATES[action_index]
        adapter = intent_adapter or _adapter_from_observation(observation)
        intent = adapter.intent_for_desired_state(
            desired_input_state,
            tick=observation.tick,
        )
        effective_input_state = adapter.effective_input_state
        return DQNActionDecision(
            intent=intent,
            action_index=action_index,
            desired_input_state=desired_input_state,
            effective_input_state=effective_input_state,
            dwell_blocked=effective_input_state != desired_input_state,
            desired_holding=desired_input_state == "hold",
            epsilon=0.0 if deterministic else epsilon,
            greedy=greedy,
            q_values=q_values,
            features=features,
        )

    def make_optimizer(self, config: DQNConfig) -> Any:
        return self.torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
        )


def encode_live_observation(
    observation: LivePracticeObservation,
    *,
    config: LiveObservationEncoderConfig | None = None,
) -> list[float]:
    """Encode only the delayed policy-visible observation into fixed features."""

    effective_config = config or LiveObservationEncoderConfig()
    source = observation.policy_observation
    visible = 1.0 if source is not None else 0.0
    if source is None:
        return [0.0] * live_observation_feature_dim()

    mode_features = [1.0 if source.mode == mode else 0.0 for mode in MODE_ORDER]
    if not any(mode_features):
        mode_features[-1] = 1.0
    return [
        visible,
        _clamp(source.tick / effective_config.max_tick),
        _clamp(source.percent / 100.0),
        source.x / effective_config.x_scale,
        source.y / effective_config.y_scale,
        source.x_vel / effective_config.velocity_scale,
        source.y_vel / effective_config.velocity_scale,
        source.rotation / effective_config.rotation_scale,
        1.0 if source.input_down else 0.0,
        1.0 if source.dead else 0.0,
        1.0 if source.completed else 0.0,
        1.0 if source.gravity == "reverse" else 0.0,
        *mode_features,
    ]


def live_observation_feature_dim() -> int:
    return 12 + len(MODE_ORDER)


def encode_actor_critic_observation(
    observation: LivePracticeObservation,
    *,
    history: LiveActionHistory | None = None,
    history_length: int = 4,
    config: LiveObservationEncoderConfig | None = None,
) -> list[float]:
    """Encode delayed bridge observation plus recent intended action history."""

    if history_length < 0:
        raise LiveLearnerError("history_length must be non-negative")
    base_features = encode_live_observation(observation, config=config)
    if history is None:
        history_features = [0.0] * live_action_history_feature_dim(history_length)
    else:
        if history.length != history_length:
            raise LiveLearnerError(
                "history length must match actor-critic encoder history_length"
            )
        history_features = history.features()
    return [*base_features, *history_features]


def live_action_history_entry_feature_dim() -> int:
    return 1 + len(ACTION_KINDS)


def live_action_history_feature_dim(history_length: int) -> int:
    if history_length < 0:
        raise LiveLearnerError("history_length must be non-negative")
    return history_length * live_action_history_entry_feature_dim()


def actor_critic_feature_dim(history_length: int = 4) -> int:
    return live_observation_feature_dim() + live_action_history_feature_dim(
        history_length
    )


def encode_dqn_observation(
    observation: LivePracticeObservation,
    *,
    history: LiveActionHistory | None = None,
    history_length: int = 4,
    config: LiveObservationEncoderConfig | None = None,
) -> list[float]:
    """Encode the same compact delayed observation/history state for DQN."""

    return encode_actor_critic_observation(
        observation,
        history=history,
        history_length=history_length,
        config=config,
    )


def dqn_feature_dim(history_length: int = 4) -> int:
    return actor_critic_feature_dim(history_length)


def run_reinforce_attempt(
    env: LiveStepEnvLike,
    policy: TinyLivePolicyNetwork,
    optimizer: Any,
    *,
    attempt_index: int,
    config: ReinforceConfig | None = None,
) -> ReinforceAttemptSummary:
    """Run one live episode and apply one REINFORCE update."""

    effective_config = config or ReinforceConfig()
    observation = env.reset(attempt_index=attempt_index)
    rewards: list[float] = []
    log_probs: list[Any] = []
    entropies: list[Any] = []
    action_counts = {state: 0 for state in DESIRED_INPUT_STATES}
    effective_action_counts = {state: 0 for state in DESIRED_INPUT_STATES}
    dwell_blocked_count = 0
    intent_counts = {kind: 0 for kind in ACTION_KINDS}
    intent_adapter = ButtonStateIntentAdapter(
        min_dwell_ticks=effective_config.min_dwell_ticks
    )
    intent_adapter.reset(observation)
    last_step: LiveStepResult | None = None

    while True:
        decision = policy.act(
            observation,
            intent_adapter=intent_adapter,
            deterministic=effective_config.deterministic_actions,
        )
        action_counts[decision.desired_input_state] += 1
        effective_action_counts[decision.effective_input_state] += 1
        if decision.dwell_blocked:
            dwell_blocked_count += 1
        intent_counts[decision.intent.kind] += 1
        last_step = env.step(decision.intent)
        rewards.append(float(last_step.reward))
        log_probs.append(decision.log_probability_tensor)
        entropies.append(decision.entropy_tensor)
        observation = last_step.observation
        if last_step.done:
            break

    returns = _discounted_returns(
        policy.torch,
        rewards,
        gamma=effective_config.gamma,
        normalize=effective_config.normalize_returns,
        device=policy.device,
    )
    log_prob_tensor = policy.torch.stack(log_probs)
    entropy_tensor = policy.torch.stack(entropies)
    policy_loss_tensor = -(log_prob_tensor * returns).sum()
    entropy_tensor_sum = entropy_tensor.sum()
    loss_tensor = policy_loss_tensor - (
        effective_config.entropy_bonus * entropy_tensor_sum
    )

    optimizer.zero_grad(set_to_none=True)
    loss_tensor.backward()
    if effective_config.max_grad_norm is not None:
        policy.torch.nn.utils.clip_grad_norm_(
            policy.model.parameters(),
            effective_config.max_grad_norm,
        )
    optimizer.step()

    if (
        last_step is not None
        and isinstance(last_step.info.get("attempt_result"), dict)
    ):
        attempt_result = dict(last_step.info["attempt_result"])
    else:
        attempt_result = env.save_attempt().to_dict()

    return ReinforceAttemptSummary(
        attempt_index=attempt_index,
        step_count=len(rewards),
        total_step_reward=sum(rewards),
        loss=float(loss_tensor.detach().cpu().item()),
        policy_loss=float(policy_loss_tensor.detach().cpu().item()),
        entropy=float(entropy_tensor_sum.detach().cpu().item()),
        action_counts={kind: int(count) for kind, count in action_counts.items()},
        effective_action_counts={
            kind: int(count) for kind, count in effective_action_counts.items()
        },
        dwell_blocked_count=dwell_blocked_count,
        intent_counts={kind: int(count) for kind, count in intent_counts.items()},
        attempt_result=attempt_result,
    )


def run_reinforce_training(
    env: LiveStepEnvLike,
    policy: TinyLivePolicyNetwork,
    *,
    config: ReinforceConfig | None = None,
    summary_path: str | Path | None = None,
) -> ReinforceTrainingSummary:
    """Run a short REINFORCE practice loop."""

    effective_config = config or ReinforceConfig()
    random.seed(effective_config.seed)
    policy.torch.manual_seed(effective_config.seed)
    optimizer = policy.make_optimizer(effective_config)
    attempts = [
        run_reinforce_attempt(
            env,
            policy,
            optimizer,
            attempt_index=attempt_index,
            config=effective_config,
        )
        for attempt_index in range(1, effective_config.attempts + 1)
    ]
    summary = ReinforceTrainingSummary(
        attempts=attempts,
        config={
            **asdict(effective_config),
            "policy": {
                "input_dim": policy.input_dim,
                "output_dim": policy.output_dim,
                "desired_input_states": list(DESIRED_INPUT_STATES),
                "intent_action_kinds": list(ACTION_KINDS),
                "hidden_size": policy.config.hidden_size,
                "device": policy.config.device,
                "encoder": asdict(policy.config.encoder),
            },
        },
    )
    if summary_path is not None:
        _write_json(summary.to_dict(), Path(summary_path))
    return summary


def run_actor_critic_attempt(
    env: LiveStepEnvLike,
    policy: TinyLiveActorCriticNetwork,
    optimizer: Any,
    *,
    attempt_index: int,
    config: ActorCriticConfig | None = None,
) -> ActorCriticAttemptSummary:
    """Run one live episode and apply one actor-critic update."""

    effective_config = config or ActorCriticConfig()
    observation = env.reset(attempt_index=attempt_index)
    step_rewards: list[float] = []
    training_rewards: list[float] = []
    log_probs: list[Any] = []
    entropies: list[Any] = []
    values: list[Any] = []
    trajectory_steps: list[dict[str, Any]] = []
    action_counts = {state: 0 for state in DESIRED_INPUT_STATES}
    effective_action_counts = {state: 0 for state in DESIRED_INPUT_STATES}
    dwell_blocked_count = 0
    intent_counts = {kind: 0 for kind in ACTION_KINDS}
    intent_adapter = ButtonStateIntentAdapter(
        min_dwell_ticks=effective_config.min_dwell_ticks
    )
    intent_adapter.reset(observation)
    history = LiveActionHistory(length=effective_config.history_length)
    last_step: LiveStepResult | None = None

    while True:
        decision = policy.act(
            observation,
            intent_adapter=intent_adapter,
            history=history,
            deterministic=effective_config.deterministic_actions,
        )
        action_counts[decision.desired_input_state] += 1
        effective_action_counts[decision.effective_input_state] += 1
        if decision.dwell_blocked:
            dwell_blocked_count += 1
        intent_counts[decision.intent.kind] += 1
        last_step = env.step(decision.intent)
        base_reward = float(last_step.reward)
        rate_penalty = (
            -effective_config.input_rate_penalty
            if decision.intent.kind != "no_op"
            else 0.0
        )
        step_reward = base_reward + rate_penalty
        step_rewards.append(base_reward)
        training_rewards.append(step_reward)
        log_probs.append(decision.log_probability_tensor)
        entropies.append(decision.entropy_tensor)
        values.append(decision.value_tensor)
        trajectory_steps.append(
            _trajectory_step_summary(
                step_index=len(step_rewards),
                observation=observation,
                decision=decision,
                env_reward=base_reward,
                input_rate_penalty=rate_penalty,
                training_reward=step_reward,
            )
        )
        history.append(
            desired_input_state=decision.effective_input_state,
            intent_kind=decision.intent.kind,
        )
        observation = last_step.observation
        if last_step.done:
            break

    attempt_result = _attempt_result_from_last_step(env, last_step)
    death_local_stats = _apply_death_local_feedback(
        training_rewards,
        attempt_result=attempt_result,
        trajectory_steps=trajectory_steps,
        config=effective_config,
    )
    returns = _discounted_returns(
        policy.torch,
        training_rewards,
        gamma=effective_config.gamma,
        normalize=False,
        device=policy.device,
    )
    value_tensor = policy.torch.stack(values).view(-1)
    log_prob_tensor = policy.torch.stack(log_probs)
    entropy_tensor = policy.torch.stack(entropies)
    advantages = returns - value_tensor.detach()
    if effective_config.normalize_advantages and advantages.numel() > 1:
        advantage_std = advantages.std(unbiased=False)
        if float(advantage_std.detach().cpu().item()) > 1e-8:
            advantages = (advantages - advantages.mean()) / advantage_std

    policy_loss_tensor = -(log_prob_tensor * advantages).sum()
    value_loss_tensor = policy.torch.nn.functional.mse_loss(
        value_tensor,
        returns,
        reduction="sum",
    )
    entropy_tensor_sum = entropy_tensor.sum()
    loss_tensor = (
        policy_loss_tensor
        + effective_config.value_loss_weight * value_loss_tensor
        - effective_config.entropy_bonus * entropy_tensor_sum
    )

    optimizer.zero_grad(set_to_none=True)
    loss_tensor.backward()
    if effective_config.max_grad_norm is not None:
        policy.torch.nn.utils.clip_grad_norm_(
            [
                *policy.shared.parameters(),
                *policy.actor_head.parameters(),
                *policy.value_head.parameters(),
            ],
            effective_config.max_grad_norm,
        )
    optimizer.step()

    return ActorCriticAttemptSummary(
        attempt_index=attempt_index,
        step_count=len(step_rewards),
        total_step_reward=sum(step_rewards),
        total_training_reward=sum(training_rewards),
        loss=float(loss_tensor.detach().cpu().item()),
        policy_loss=float(policy_loss_tensor.detach().cpu().item()),
        value_loss=float(value_loss_tensor.detach().cpu().item()),
        entropy=float(entropy_tensor_sum.detach().cpu().item()),
        mean_value=float(value_tensor.detach().mean().cpu().item()),
        advantage_stats=_tensor_stats(policy.torch, advantages),
        action_counts={kind: int(count) for kind, count in action_counts.items()},
        effective_action_counts={
            kind: int(count) for kind, count in effective_action_counts.items()
        },
        dwell_blocked_count=dwell_blocked_count,
        intent_counts={kind: int(count) for kind, count in intent_counts.items()},
        death_local_stats=death_local_stats,
        attempt_result=attempt_result,
    )


def run_actor_critic_training(
    env: LiveStepEnvLike,
    policy: TinyLiveActorCriticNetwork,
    *,
    config: ActorCriticConfig | None = None,
    summary_path: str | Path | None = None,
) -> ActorCriticTrainingSummary:
    """Run a short actor-critic closed-loop practice session."""

    effective_config = config or ActorCriticConfig()
    random.seed(effective_config.seed)
    policy.torch.manual_seed(effective_config.seed)
    optimizer = policy.make_optimizer(effective_config)
    attempts = [
        run_actor_critic_attempt(
            env,
            policy,
            optimizer,
            attempt_index=attempt_index,
            config=effective_config,
        )
        for attempt_index in range(1, effective_config.attempts + 1)
    ]
    summary = ActorCriticTrainingSummary(
        attempts=attempts,
        config={
            **asdict(effective_config),
            "algorithm": "tiny_actor_critic",
            "policy": {
                "input_dim": policy.input_dim,
                "output_dim": policy.output_dim,
                "desired_input_states": list(DESIRED_INPUT_STATES),
                "intent_action_kinds": list(ACTION_KINDS),
                "hidden_size": policy.config.hidden_size,
                "device": policy.config.device,
                "encoder": asdict(policy.config.encoder),
                "history_length": policy.history_length,
                "history_feature_dim": live_action_history_feature_dim(
                    policy.history_length
                ),
            },
        },
    )
    if summary_path is not None:
        _write_json(summary.to_dict(), Path(summary_path))
    return summary


def run_dqn_attempt(
    env: LiveStepEnvLike,
    policy: TinyLiveDQNNetwork,
    optimizer: Any,
    replay_buffer: DQNReplayBuffer,
    *,
    attempt_index: int,
    config: DQNConfig | None = None,
    global_step_start: int = 0,
) -> DQNAttemptSummary:
    """Run one live episode and apply online replay-based DQN updates."""

    effective_config = config or DQNConfig()
    observation = env.reset(attempt_index=attempt_index)
    step_rewards: list[float] = []
    training_rewards: list[float] = []
    losses: list[float] = []
    selected_q_values: list[float] = []
    action_counts = {state: 0 for state in DESIRED_INPUT_STATES}
    effective_action_counts = {state: 0 for state in DESIRED_INPUT_STATES}
    dwell_blocked_count = 0
    intent_counts = {kind: 0 for kind in ACTION_KINDS}
    pending_transitions: list[DQNTransition] = []
    intent_adapter = ButtonStateIntentAdapter(
        min_dwell_ticks=effective_config.min_dwell_ticks
    )
    intent_adapter.reset(observation)
    history = LiveActionHistory(length=effective_config.history_length)
    update_count = 0
    replay_appended_count = 0
    epsilon_first: float | None = None
    epsilon_last = 0.0
    last_step: LiveStepResult | None = None

    while True:
        epsilon = _linear_epsilon(
            effective_config,
            global_step_start + len(pending_transitions),
        )
        if epsilon_first is None:
            epsilon_first = epsilon
        epsilon_last = epsilon
        decision = policy.act(
            observation,
            intent_adapter=intent_adapter,
            history=history,
            epsilon=epsilon,
            deterministic=effective_config.deterministic_actions,
        )
        action_counts[decision.desired_input_state] += 1
        effective_action_counts[decision.effective_input_state] += 1
        if decision.dwell_blocked:
            dwell_blocked_count += 1
        intent_counts[decision.intent.kind] += 1
        selected_q_values.append(decision.q_values[decision.action_index])

        last_step = env.step(decision.intent)
        base_reward = float(last_step.reward)
        rate_penalty = (
            -effective_config.input_rate_penalty
            if decision.intent.kind != "no_op"
            else 0.0
        )
        step_reward = base_reward + rate_penalty
        step_rewards.append(base_reward)
        training_rewards.append(step_reward)

        history.append(
            desired_input_state=decision.effective_input_state,
            intent_kind=decision.intent.kind,
        )
        next_features = encode_dqn_observation(
            last_step.observation,
            history=history,
            history_length=effective_config.history_length,
            config=policy.config.encoder,
        )
        pending_transitions.append(
            DQNTransition(
                features=decision.features,
                action_index=decision.action_index,
                reward=step_reward,
                next_features=next_features,
                done=last_step.done,
            )
        )

        observation = last_step.observation
        if last_step.done:
            break

    attempt_result = _attempt_result_from_last_step(env, last_step)
    replay_skip_reason = _dqn_replay_skip_reason(
        last_step,
        config=effective_config,
    )
    if replay_skip_reason is None:
        committed_step = global_step_start
        for transition in pending_transitions:
            replay_buffer.append(transition)
            replay_appended_count += 1
            committed_step += 1
            loss = _optimize_dqn(
                policy,
                optimizer,
                replay_buffer,
                config=effective_config,
            )
            if loss is not None:
                losses.append(loss)
                update_count += 1
            if committed_step % effective_config.target_update_interval == 0:
                policy.sync_target()

    return DQNAttemptSummary(
        attempt_index=attempt_index,
        step_count=len(step_rewards),
        total_step_reward=sum(step_rewards),
        total_training_reward=sum(training_rewards),
        mean_loss=(sum(losses) / len(losses)) if losses else None,
        last_loss=losses[-1] if losses else None,
        update_count=update_count,
        replay_size=len(replay_buffer),
        replay_appended_count=replay_appended_count,
        replay_skipped=replay_skip_reason is not None,
        replay_skip_reason=replay_skip_reason,
        epsilon_start=epsilon_first if epsilon_first is not None else 0.0,
        epsilon_end=epsilon_last,
        q_value_stats=_float_stats(selected_q_values),
        action_counts={kind: int(count) for kind, count in action_counts.items()},
        effective_action_counts={
            kind: int(count) for kind, count in effective_action_counts.items()
        },
        dwell_blocked_count=dwell_blocked_count,
        intent_counts={kind: int(count) for kind, count in intent_counts.items()},
        attempt_result=attempt_result,
    )


def run_dqn_training(
    env: LiveStepEnvLike,
    policy: TinyLiveDQNNetwork,
    *,
    config: DQNConfig | None = None,
    summary_path: str | Path | None = None,
) -> DQNTrainingSummary:
    """Run a short DQN baseline practice session."""

    effective_config = config or DQNConfig()
    random.seed(effective_config.seed)
    policy.torch.manual_seed(effective_config.seed)
    optimizer = policy.make_optimizer(effective_config)
    replay_buffer = DQNReplayBuffer(effective_config.replay_capacity)
    global_step = 0
    attempts: list[DQNAttemptSummary] = []
    for attempt_index in range(1, effective_config.attempts + 1):
        attempt = run_dqn_attempt(
            env,
            policy,
            optimizer,
            replay_buffer,
            attempt_index=attempt_index,
            config=effective_config,
            global_step_start=global_step,
        )
        attempts.append(attempt)
        global_step += attempt.replay_appended_count

    summary = DQNTrainingSummary(
        attempts=attempts,
        config={
            **asdict(effective_config),
            "algorithm": "tiny_dqn",
            "policy": {
                "input_dim": policy.input_dim,
                "output_dim": policy.output_dim,
                "desired_input_states": list(DESIRED_INPUT_STATES),
                "intent_action_kinds": list(ACTION_KINDS),
                "hidden_size": policy.config.hidden_size,
                "device": policy.config.device,
                "encoder": asdict(policy.config.encoder),
                "history_length": policy.history_length,
                "history_feature_dim": live_action_history_feature_dim(
                    policy.history_length
                ),
            },
        },
    )
    if summary_path is not None:
        _write_json(summary.to_dict(), Path(summary_path))
    return summary


def _attempt_result_from_last_step(
    env: LiveStepEnvLike,
    last_step: LiveStepResult | None,
) -> dict[str, Any]:
    if (
        last_step is not None
        and isinstance(last_step.info.get("attempt_result"), dict)
    ):
        return dict(last_step.info["attempt_result"])
    return env.save_attempt().to_dict()


def _dqn_replay_skip_reason(
    last_step: LiveStepResult | None,
    *,
    config: DQNConfig,
) -> str | None:
    if last_step is None:
        return None
    death_reason = last_step.observation.latest.death_reason
    if death_reason in config.skip_replay_death_reasons:
        return f"death_reason:{death_reason}"
    return None


def _trajectory_step_summary(
    *,
    step_index: int,
    observation: LivePracticeObservation,
    decision: ActorCriticActionDecision,
    env_reward: float,
    input_rate_penalty: float,
    training_reward: float,
) -> dict[str, Any]:
    policy_observation = observation.policy_observation
    return {
        "step_index": step_index,
        "latest_tick": observation.latest.tick,
        "policy_tick": (
            policy_observation.tick if policy_observation is not None else None
        ),
        "policy_visible": policy_observation is not None,
        "policy_percent": (
            policy_observation.percent if policy_observation is not None else None
        ),
        "policy_y": policy_observation.y if policy_observation is not None else None,
        "policy_y_vel": (
            policy_observation.y_vel if policy_observation is not None else None
        ),
        "policy_input_down": (
            policy_observation.input_down
            if policy_observation is not None
            else None
        ),
        "desired_input_state": decision.desired_input_state,
        "effective_input_state": decision.effective_input_state,
        "dwell_blocked": decision.dwell_blocked,
        "intent_kind": decision.intent.kind,
        "value": decision.value,
        "env_reward": env_reward,
        "input_rate_penalty": input_rate_penalty,
        "death_local_penalty": 0.0,
        "training_reward": training_reward,
    }


def _apply_death_local_feedback(
    training_rewards: list[float],
    *,
    attempt_result: dict[str, Any],
    trajectory_steps: list[dict[str, Any]],
    config: ActorCriticConfig,
) -> dict[str, Any]:
    death_tick = attempt_result.get("death_tick")
    cleared = bool(attempt_result.get("cleared", False))
    stats: dict[str, Any] = {
        "applied": False,
        "window_size": config.death_local_window,
        "affected_step_count": 0,
        "penalty_total": 0.0,
        "death_tick": death_tick,
        "death_percent": attempt_result.get("death_percent"),
        "reason": None,
    }
    if cleared:
        stats["reason"] = "cleared_attempt"
        return stats
    if death_tick is None:
        stats["reason"] = "no_death_tick"
        return stats
    if config.death_local_window == 0 or config.death_local_penalty == 0.0:
        stats["reason"] = "disabled"
        return stats
    if not training_rewards:
        stats["reason"] = "empty_trajectory"
        return stats

    affected_count = min(config.death_local_window, len(training_rewards))
    weight_total = affected_count * (affected_count + 1) / 2.0
    penalties = [
        -config.death_local_penalty * ((index + 1) / weight_total)
        for index in range(affected_count)
    ]
    start_index = len(training_rewards) - affected_count
    for offset, penalty in enumerate(penalties):
        reward_index = start_index + offset
        training_rewards[reward_index] += penalty
        trajectory_steps[reward_index]["death_local_penalty"] = penalty
        trajectory_steps[reward_index]["training_reward"] += penalty

    recent_steps = trajectory_steps[start_index:]
    stats.update(
        {
            "applied": True,
            "affected_step_count": affected_count,
            "penalty_total": sum(penalties),
            "start_step_index": recent_steps[0]["step_index"],
            "end_step_index": recent_steps[-1]["step_index"],
            "latest_tick_range": [
                recent_steps[0]["latest_tick"],
                recent_steps[-1]["latest_tick"],
            ],
            "policy_tick_range": [
                recent_steps[0]["policy_tick"],
                recent_steps[-1]["policy_tick"],
            ],
            "desired_input_state_counts": _count_recent(
                recent_steps,
                "desired_input_state",
                DESIRED_INPUT_STATES,
            ),
            "effective_input_state_counts": _count_recent(
                recent_steps,
                "effective_input_state",
                DESIRED_INPUT_STATES,
            ),
            "dwell_blocked_count": sum(
                1 for step in recent_steps if step.get("dwell_blocked")
            ),
            "intent_counts": _count_recent(recent_steps, "intent_kind", ACTION_KINDS),
            "average_policy_y": _mean_optional(
                step["policy_y"] for step in recent_steps
            ),
            "average_policy_y_vel": _mean_optional(
                step["policy_y_vel"] for step in recent_steps
            ),
            "recent_steps": _compact_recent_steps(recent_steps[-5:]),
            "reason": "terminal_death",
        }
    )
    return stats


def _count_recent(
    steps: Sequence[dict[str, Any]],
    key: str,
    expected_values: Sequence[str],
) -> dict[str, int]:
    counts = {value: 0 for value in expected_values}
    for step in steps:
        value = step.get(key)
        if value in counts:
            counts[value] += 1
    return counts


def _mean_optional(values: Sequence[float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _compact_recent_steps(steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "step_index",
        "latest_tick",
        "policy_tick",
        "policy_percent",
        "policy_y",
        "policy_y_vel",
        "policy_input_down",
        "desired_input_state",
        "effective_input_state",
        "dwell_blocked",
        "intent_kind",
        "env_reward",
        "input_rate_penalty",
        "death_local_penalty",
        "training_reward",
    )
    return [{key: step.get(key) for key in keys} for step in steps]


def _linear_epsilon(config: DQNConfig, global_step: int) -> float:
    if config.deterministic_actions:
        return 0.0
    progress = _clamp(global_step / config.epsilon_decay_steps)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def _optimize_dqn(
    policy: TinyLiveDQNNetwork,
    optimizer: Any,
    replay_buffer: DQNReplayBuffer,
    *,
    config: DQNConfig,
) -> float | None:
    ready_count = max(config.batch_size, config.warmup_steps)
    if len(replay_buffer) < ready_count:
        return None

    batch = replay_buffer.sample(config.batch_size)
    torch = policy.torch
    feature_tensor = torch.tensor(
        [transition.features for transition in batch],
        dtype=torch.float32,
        device=policy.device,
    )
    action_tensor = torch.tensor(
        [transition.action_index for transition in batch],
        dtype=torch.int64,
        device=policy.device,
    ).unsqueeze(1)
    reward_tensor = torch.tensor(
        [transition.reward for transition in batch],
        dtype=torch.float32,
        device=policy.device,
    )
    next_feature_tensor = torch.tensor(
        [transition.next_features for transition in batch],
        dtype=torch.float32,
        device=policy.device,
    )
    done_tensor = torch.tensor(
        [1.0 if transition.done else 0.0 for transition in batch],
        dtype=torch.float32,
        device=policy.device,
    )

    predicted_q = policy.q_network(feature_tensor).gather(1, action_tensor).squeeze(1)
    with torch.no_grad():
        next_q = policy.target_network(next_feature_tensor).max(dim=1).values
        target_q = reward_tensor + config.gamma * (1.0 - done_tensor) * next_q

    loss_tensor = torch.nn.functional.smooth_l1_loss(predicted_q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss_tensor.backward()
    if config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            policy.q_network.parameters(),
            config.max_grad_norm,
        )
    optimizer.step()
    return float(loss_tensor.detach().cpu().item())


def _float_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "std": variance**0.5,
    }


def _tensor_stats(torch: Any, tensor: Any) -> dict[str, float]:
    del torch
    detached = tensor.detach()
    if detached.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(detached.mean().cpu().item()),
        "min": float(detached.min().cpu().item()),
        "max": float(detached.max().cpu().item()),
        "std": float(detached.std(unbiased=False).cpu().item()),
    }


def _discounted_returns(
    torch: Any,
    rewards: Sequence[float],
    *,
    gamma: float,
    normalize: bool,
    device: Any,
) -> Any:
    returns: list[float] = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = float(reward) + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    if normalize and tensor.numel() > 1:
        std = tensor.std(unbiased=False)
        if float(std.detach().cpu().item()) > 1e-8:
            tensor = (tensor - tensor.mean()) / std
    return tensor


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise LiveLearnerError(
            "PyTorch is required for the live neural learner. Install torch "
            "or keep using scripted/random policies."
        ) from exc
    return torch


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _adapter_from_observation(
    observation: LivePracticeObservation,
) -> ButtonStateIntentAdapter:
    adapter = ButtonStateIntentAdapter()
    adapter.reset(observation)
    return adapter


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _write_json(data: dict[str, Any], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
