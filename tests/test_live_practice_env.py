import json
from pathlib import Path

import pytest

import gd_rl.live_env as live_env_module
from gd_env import BridgeDiagnostic, BridgeObservation
from gd_human_model import Event, HumanProfile
from gd_rl import IntendedAction, LivePracticeEnv, LivePracticeEnvConfig
from gd_trace import load_macro_json, load_trace_jsonl


class FakeLiveGeodeClient:
    def __init__(self, observations: list[BridgeObservation]) -> None:
        if not observations:
            raise ValueError("observations must be non-empty")
        self.observations = observations
        self.index = 0
        self.connected = False
        self.closed = False
        self.reset_reasons: list[str] = []
        self.sent_events: list[Event] = []

    def connect(self) -> "FakeLiveGeodeClient":
        self.connected = True
        return self

    def close(self) -> None:
        self.closed = True

    def reset_attempt(
        self,
        reason: str = "requested",
        *,
        max_observations: int = 600,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        self.index = 0
        self.reset_reasons.append(reason)
        if diagnostics is not None:
            diagnostics.append(
                BridgeDiagnostic(
                    kind="fake_reset",
                    tick=0,
                    data={"reason": reason, "max_observations": max_observations},
                )
            )
        return self.observations[0]

    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        self.index += 1
        if self.index >= len(self.observations):
            raise EOFError("fake observation stream exhausted")
        if diagnostics is not None and self.sent_events:
            diagnostics.append(
                BridgeDiagnostic(
                    kind="fake_live_action_applied",
                    tick=self.observations[self.index].tick,
                    data={"event_count": len(self.sent_events)},
                )
            )
        return self.observations[self.index]

    def send_event(self, event: Event) -> None:
        self.sent_events.append(event)


def test_live_env_humanizes_intent_before_dispatching_input(tmp_path: Path) -> None:
    fake_client = FakeLiveGeodeClient(
        [_observation(tick, percent=float(tick * 10)) for tick in range(4)]
    )
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="local_step_fixture",
            output_dir=tmp_path,
            max_steps=4,
            success_percent=100.0,
            action_horizon_ticks=1,
        ),
        human_profile=_profile(visual_delay_frames=0, motor_delay_frames=2),
        client_factory=lambda: fake_client,
    )

    with env:
        initial = env.reset(attempt_index=1)
        assert initial.tick == 0
        assert initial.policy_observation == initial.latest

        first_step = env.step(IntendedAction.press(999))
        assert first_step.observation.tick == 1
        assert fake_client.sent_events == []

        second_step = env.step(IntendedAction.no_op(999))
        assert second_step.observation.tick == 2
        assert fake_client.sent_events == [Event(2, "press")]

        result = env.save_attempt()

    attempt_dir = tmp_path / "attempt_001"
    intended_macro = load_macro_json(attempt_dir / "policy_intended_events.json")
    executed_macro = load_macro_json(attempt_dir / "human_executed_events.json")
    details = json.loads(
        (attempt_dir / "humanization_details.json").read_text(encoding="utf-8")
    )

    assert fake_client.connected is True
    assert fake_client.closed is True
    assert intended_macro.events == [Event(0, "press")]
    assert executed_macro.events == [Event(2, "press")]
    assert details["event_results"][0]["requested_action"]["tick"] == 999
    assert details["event_results"][0]["live_decision_tick"] == 0
    assert result.intended_event_count == 1
    assert result.executed_event_count == 1
    assert result.trace_path.endswith("trace.jsonl")
    diagnostics = json.loads(
        (attempt_dir / "geode_diagnostics.json").read_text(encoding="utf-8")
    )
    assert any(
        diagnostic["kind"] == "fake_live_action_applied"
        for diagnostic in diagnostics["diagnostics"]
    )


def test_live_env_returns_delayed_policy_observation_and_persists_terminal_attempt(
    tmp_path: Path,
) -> None:
    fake_client = FakeLiveGeodeClient(
        [
            _observation(0, percent=0.0),
            _observation(1, percent=20.0),
            _observation(2, percent=40.0),
            _observation(3, percent=100.0, completed=True),
        ]
    )
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="local_step_fixture",
            output_dir=tmp_path,
            max_steps=5,
            success_percent=100.0,
        ),
        human_profile=_profile(visual_delay_frames=2, motor_delay_frames=0),
        client_factory=lambda: fake_client,
    )

    with env:
        initial = env.reset(attempt_index=1)
        assert initial.policy_observation is None

        first_step = env.step(IntendedAction.no_op(0))
        assert first_step.observation.policy_observation is None

        second_step = env.step(IntendedAction.no_op(1))
        assert second_step.observation.policy_observation is not None
        assert second_step.observation.policy_observation.tick == 0

        terminal_step = env.step(IntendedAction.no_op(2))
        assert terminal_step.done is True
        assert terminal_step.info["attempt_result"]["cleared"] is True

    attempt_dir = tmp_path / "attempt_001"
    trace = load_trace_jsonl(attempt_dir / "trace.jsonl")
    summary = json.loads((attempt_dir / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (attempt_dir / "geode_diagnostics.json").read_text(encoding="utf-8")
    )

    assert [row.tick for row in trace] == [0, 1, 2, 3]
    assert summary["cleared"] is True
    assert summary["metadata"]["executor"] == "geode_live_step"
    assert diagnostics["executor"] == "geode_live_step"


def test_live_env_only_waits_after_clear_not_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(live_env_module.time, "sleep", sleeps.append)

    death_client = FakeLiveGeodeClient(
        [
            _observation(0, percent=0.0),
            _observation(1, percent=20.0, dead=True),
        ]
    )
    death_env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="death_fixture",
            output_dir=tmp_path / "death",
            post_terminal_delay_seconds=5.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: death_client,
    )
    with death_env:
        death_env.reset(attempt_index=1)
        death_step = death_env.step(IntendedAction.no_op(0))

    assert death_step.done is True
    assert death_step.info["attempt_result"]["cleared"] is False
    assert sleeps == []

    clear_client = FakeLiveGeodeClient(
        [
            _observation(0, percent=0.0),
            _observation(1, percent=100.0, completed=True),
        ]
    )
    clear_env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="clear_fixture",
            output_dir=tmp_path / "clear",
            post_terminal_delay_seconds=5.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: clear_client,
    )
    with clear_env:
        clear_env.reset(attempt_index=1)
        clear_step = clear_env.step(IntendedAction.no_op(0))

    assert clear_step.done is True
    assert clear_step.info["attempt_result"]["cleared"] is True
    assert sleeps == [5.0]


def test_live_env_treats_tick_rewind_as_terminal_death(tmp_path: Path) -> None:
    fake_client = FakeLiveGeodeClient(
        [
            _observation(0, percent=0.0),
            _observation(1, percent=10.0),
            _observation(0, percent=0.0),
        ]
    )
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="rewind_fixture",
            output_dir=tmp_path,
            max_steps=5,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )

    with env:
        env.reset(attempt_index=1)
        first_step = env.step(IntendedAction.no_op(0))
        rewind_step = env.step(IntendedAction.no_op(1))

    attempt_dir = tmp_path / "attempt_001"
    trace = load_trace_jsonl(attempt_dir / "trace.jsonl")
    diagnostics = json.loads(
        (attempt_dir / "geode_diagnostics.json").read_text(encoding="utf-8")
    )

    assert first_step.done is False
    assert rewind_step.done is True
    assert rewind_step.observation.latest.tick == 1
    assert rewind_step.observation.latest.dead is True
    assert rewind_step.info["attempt_result"]["death_tick"] == 1
    assert [row.tick for row in trace] == [0, 1]
    assert trace[-1].dead is True
    assert trace[-1].death_reason == "tick_rewind_reset"
    assert any(
        diagnostic["kind"] == "live_tick_rewind_terminal"
        for diagnostic in diagnostics["diagnostics"]
    )


def test_live_env_start_guard_rejects_wrong_start(tmp_path: Path) -> None:
    fake_client = FakeLiveGeodeClient(
        [_observation(0, x=100.0, percent=20.0)]
    )
    env = LivePracticeEnv(
        config=LivePracticeEnvConfig(
            level_id="local_step_fixture",
            output_dir=tmp_path,
            require_start_percent_max=2.0,
            require_start_x_max=50.0,
        ),
        human_profile=_profile(),
        client_factory=lambda: fake_client,
    )

    with pytest.raises(ValueError, match="fresh start check failed"):
        env.reset(attempt_index=1)


def _profile(
    *,
    visual_delay_frames: int = 0,
    motor_delay_frames: int = 0,
) -> HumanProfile:
    return HumanProfile(
        name="DeterministicTest",
        visual_delay_frames=visual_delay_frames,
        motor_delay_frames=motor_delay_frames,
        base_press_std_frames=0.0,
        base_release_std_frames=0.0,
        close_amp=0.0,
        close_tau=1.0,
        long_amp=0.0,
        long_tau=1.0,
        error_rho=0.0,
        miss_prob_base=0.0,
        miss_prob_close_amp=0.0,
        miss_prob_close_tau=1.0,
        random_seed=0,
    )


def _observation(
    tick: int,
    *,
    x: float | None = None,
    percent: float,
    dead: bool = False,
    completed: bool = False,
    input_down: bool = False,
) -> BridgeObservation:
    return BridgeObservation(
        tick=tick,
        x=float(tick if x is None else x),
        y=0.0,
        y_vel=0.0,
        mode="cube",
        gravity="normal",
        percent=percent,
        dead=dead,
        input_down=input_down,
        completed=completed,
        x_vel=1.0,
    )
