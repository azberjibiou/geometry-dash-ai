"""Tiny neural learner for the live step practice environment."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from gd_rl.actions import ActionKind, IntendedAction
from gd_rl.live_env import LivePracticeObservation, LiveStepResult
from gd_rl.results import AttemptResult

ACTION_KINDS: tuple[ActionKind, ...] = ("no_op", "press", "release")
DesiredButtonState = Literal["up", "down"]
DESIRED_BUTTON_STATES: tuple[DesiredButtonState, ...] = ("up", "down")
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


@dataclass(slots=True)
class ButtonStateIntentAdapter:
    """Map desired button states to intended press/release edges.

    The adapter tracks intended button state instead of live executed input.
    That keeps visual and motor delay from causing repeated press/release
    intents while an already-requested humanized event is still pending.
    """

    intended_input_down: bool = False

    def reset(self, observation: LivePracticeObservation | None = None) -> None:
        """Start a new attempt from the current fresh-reset input state."""

        self.intended_input_down = (
            bool(observation.latest.input_down) if observation is not None else False
        )

    def intent_for_desired_state(
        self,
        desired_button_state: DesiredButtonState,
        *,
        tick: int,
    ) -> IntendedAction:
        """Return the edge needed to reach the desired intended state."""

        if desired_button_state == "down":
            if self.intended_input_down:
                return IntendedAction.no_op(tick)
            self.intended_input_down = True
            return IntendedAction.press(tick)
        if desired_button_state == "up":
            if not self.intended_input_down:
                return IntendedAction.no_op(tick)
            self.intended_input_down = False
            return IntendedAction.release(tick)
        raise LiveLearnerError(
            f"unknown desired button state {desired_button_state!r}"
        )


@dataclass(slots=True)
class NeuralActionDecision:
    """One policy decision plus tensors needed for REINFORCE."""

    intent: IntendedAction
    action_index: int
    desired_button_state: DesiredButtonState
    desired_input_down: bool
    probability: float
    logits: list[float]
    features: list[float]
    log_probability_tensor: Any = field(repr=False)
    entropy_tensor: Any = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": asdict(self.intent),
            "action_index": self.action_index,
            "desired_button_state": self.desired_button_state,
            "desired_input_down": self.desired_input_down,
            "probability": self.probability,
            "logits": list(self.logits),
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
    intent_counts: dict[str, int]
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
        self.output_dim = len(DESIRED_BUTTON_STATES)
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
        desired_button_state = DESIRED_BUTTON_STATES[action_index]
        adapter = intent_adapter or _adapter_from_observation(observation)
        intent = adapter.intent_for_desired_state(
            desired_button_state,
            tick=observation.tick,
        )
        probability = float(
            self.torch.softmax(logits, dim=-1)[action_index].detach().cpu().item()
        )
        return NeuralActionDecision(
            intent=intent,
            action_index=action_index,
            desired_button_state=desired_button_state,
            desired_input_down=desired_button_state == "down",
            probability=probability,
            logits=[float(value) for value in logits.detach().cpu().tolist()],
            features=features,
            log_probability_tensor=distribution.log_prob(action_index_tensor),
            entropy_tensor=distribution.entropy(),
        )

    def make_optimizer(self, config: ReinforceConfig) -> Any:
        return self.torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)


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
    action_counts = {state: 0 for state in DESIRED_BUTTON_STATES}
    intent_counts = {kind: 0 for kind in ACTION_KINDS}
    intent_adapter = ButtonStateIntentAdapter()
    intent_adapter.reset(observation)
    last_step: LiveStepResult | None = None

    while True:
        decision = policy.act(
            observation,
            intent_adapter=intent_adapter,
            deterministic=effective_config.deterministic_actions,
        )
        action_counts[decision.desired_button_state] += 1
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
                "desired_button_states": list(DESIRED_BUTTON_STATES),
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
