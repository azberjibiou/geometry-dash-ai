import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from gd_env import BridgeDiagnostic, BridgeObservation
from gd_human_model import Event
from gd_rl.geode_executor import (
    GeodeExecutorConfig,
    GeodePracticeExecutor,
    _is_terminal_trace,
    _validate_start_observation,
    _validate_trace_progress,
)
from gd_trace import Macro, TraceRow
from tests.test_trace_io import make_row


class FakeGeodeClient:
    def __init__(
        self,
        *,
        initial_observation: BridgeObservation | None = None,
        rows: Sequence[TraceRow] | None = None,
    ) -> None:
        self.initial_observation = initial_observation or BridgeObservation(
            tick=0,
            x=0.0,
            y=0.0,
            y_vel=0.0,
            mode="cube",
            gravity="normal",
            percent=0.0,
            dead=False,
            input_down=False,
        )
        self.rows = list(rows or [make_row(0), make_row(10, percent=5.0)])
        self.connected = False
        self.closed = False
        self.loaded_events: list[Event] = []
        self.loaded_metadata: dict[str, Any] = {}
        self.stop_percent: float | None = None

    def connect(self) -> "FakeGeodeClient":
        self.connected = True
        return self

    def close(self) -> None:
        self.closed = True

    def load_macro(
        self,
        events: Sequence[Event],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.loaded_events = list(events)
        self.loaded_metadata = dict(metadata or {})

    def reset_attempt(
        self,
        reason: str = "requested",
        *,
        max_observations: int = 600,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        if diagnostics is not None:
            diagnostics.append(
                BridgeDiagnostic(
                    kind="reset",
                    tick=0,
                    data={"reason": reason, "max_observations": max_observations},
                )
            )
        return self.initial_observation

    def run_loaded_macro(
        self,
        *,
        max_observations: int,
        fps: int = 240,
        cbf: bool = False,
        physics_bypass: bool = False,
        trace_path: str | Path | None = None,
        initial_observation: BridgeObservation | None = None,
        diagnostics: list[BridgeDiagnostic] | None = None,
        stop_percent: float | None = None,
    ) -> list[TraceRow]:
        self.stop_percent = stop_percent
        if diagnostics is not None:
            diagnostics.append(
                BridgeDiagnostic(
                    kind="macro_complete",
                    tick=self.rows[-1].tick,
                    data={"max_observations": max_observations},
                )
            )
        return self.rows


def test_geode_executor_loads_macro_runs_trace_and_saves_diagnostics(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeodeClient(rows=[make_row(0), make_row(30, percent=100.0)])
    executor = GeodePracticeExecutor(
        GeodeExecutorConfig(stop_on_success=True, success_percent=100.0),
        client_factory=lambda: fake_client,
    )
    macro = Macro(
        events=[Event(10, "press"), Event(20, "release")],
        metadata={"kind": "human_executed_input"},
    )

    with executor:
        rows = executor.run_attempt(
            attempt_index=1,
            executed_macro=macro,
            attempt_dir=tmp_path,
            metadata={"level_id": "tiny"},
        )

    assert fake_client.connected is True
    assert fake_client.closed is True
    assert fake_client.loaded_events == macro.events
    assert fake_client.loaded_metadata["executor"] == "geode_queued_macro"
    assert fake_client.stop_percent == 100.0
    assert rows[-1].percent == 100.0

    diagnostics_path = tmp_path / "geode_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["executor"] == "geode_queued_macro"
    assert [item["kind"] for item in diagnostics["diagnostics"]] == [
        "reset",
        "macro_complete",
    ]


def test_start_guard_rejects_wrong_live_level() -> None:
    observation = BridgeObservation(
        tick=0,
        x=100.0,
        y=0.0,
        y_vel=0.0,
        mode="cube",
        gravity="normal",
        percent=20.0,
        dead=False,
        input_down=False,
    )

    with pytest.raises(ValueError, match="fresh start check failed"):
        _validate_start_observation(
            observation,
            attempt_index=1,
            require_percent_max=2.0,
            require_x_max=50.0,
        )


def test_progress_guard_rejects_wrong_live_level() -> None:
    rows = [make_row(0, percent=0.0), make_row(120, percent=0.5)]

    with pytest.raises(ValueError, match="progress check failed"):
        _validate_trace_progress(
            rows,
            attempt_index=1,
            require_tick=120,
            require_percent_min=10.0,
        )


def test_terminal_trace_detects_death_or_success() -> None:
    dead_row = make_row(10, percent=50.0).to_dict()
    dead_row["dead"] = True

    assert _is_terminal_trace(
        [make_row(0), make_row(10, percent=100.0)],
        success_percent=100.0,
    )
    assert _is_terminal_trace(
        [make_row(0), TraceRow.from_mapping(dead_row)],
        success_percent=100.0,
    )
    assert not _is_terminal_trace(
        [make_row(0), make_row(10, percent=50.0)],
        success_percent=100.0,
    )
