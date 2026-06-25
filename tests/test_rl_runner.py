from pathlib import Path

from gd_human_model import Event
from gd_rl import PracticeRunConfig, PracticeRunner, ScriptedEventPolicy
from gd_rl.synthetic import SyntheticTraceExecutor
from gd_trace import load_macro_json, load_trace_jsonl
from gd_trace.trace_schema import TraceRow
from tests.test_macro_humanizer import quiet_profile
from tests.test_trace_io import make_row


def row(tick: int, *, percent: float = 0.0, dead: bool = False) -> TraceRow:
    data = make_row(tick, percent=percent).to_dict()
    data["dead"] = dead
    return TraceRow.from_mapping(data)


def test_practice_runner_persists_intended_executed_trace_and_summaries(
    tmp_path: Path,
) -> None:
    profile = quiet_profile()
    policy = ScriptedEventPolicy([Event(10, "press"), Event(20, "release")])
    executor = SyntheticTraceExecutor(
        traces=[
            [row(0), row(60, percent=25.0, dead=True)],
            [row(0), row(120, percent=100.0)],
        ]
    )
    config = PracticeRunConfig(
        level_id="tiny-local-fixture",
        attempts=2,
        output_dir=tmp_path / "practice",
        success_percent=100.0,
        base_seed=50,
    )

    summary = PracticeRunner(
        policy=policy,
        executor=executor,
        human_profile=profile,
        config=config,
    ).run()

    assert summary.attempt_count == 2
    assert summary.clears == 1
    assert summary.deaths == 1
    assert summary.attempts_to_first_clear == 2
    assert summary.best_percent == 100.0
    assert (config.output_dir / "summary.json").exists()

    attempt_dir = config.output_dir / "attempt_001"
    intended_path = attempt_dir / "policy_intended_events.json"
    executed_path = attempt_dir / "human_executed_events.json"
    humanization_path = attempt_dir / "humanization_details.json"
    trace_path = attempt_dir / "trace.jsonl"
    attempt_summary_path = attempt_dir / "summary.json"

    assert intended_path.exists()
    assert executed_path.exists()
    assert humanization_path.exists()
    assert trace_path.exists()
    assert attempt_summary_path.exists()

    intended = load_macro_json(intended_path)
    executed = load_macro_json(executed_path)
    trace = load_trace_jsonl(trace_path)

    assert intended.metadata["kind"] == "policy_intent"
    assert executed.metadata["kind"] == "human_executed_input"
    assert intended.events == [Event(10, "press"), Event(20, "release")]
    assert executed.events == intended.events
    assert trace[-1].percent == 25.0
    assert summary.attempts[0].intended_event_count == 2
    assert summary.attempts[0].executed_event_count == 2
    assert summary.attempts[0].dropped_event_count == 0


def test_practice_runner_can_stop_after_first_clear(tmp_path: Path) -> None:
    profile = quiet_profile()
    policy = ScriptedEventPolicy([Event(10, "press"), Event(20, "release")])
    executor = SyntheticTraceExecutor(
        traces=[
            [row(0), row(60, percent=100.0)],
            [row(0), row(60, percent=25.0, dead=True)],
        ]
    )
    config = PracticeRunConfig(
        level_id="tiny-local-fixture",
        attempts=2,
        output_dir=tmp_path / "practice",
        success_percent=100.0,
        stop_after_first_clear=True,
        base_seed=50,
    )

    summary = PracticeRunner(
        policy=policy,
        executor=executor,
        human_profile=profile,
        config=config,
    ).run()

    assert summary.attempt_count == 1
    assert summary.clears == 1
    assert (config.output_dir / "attempt_001" / "summary.json").exists()
    assert not (config.output_dir / "attempt_002").exists()
