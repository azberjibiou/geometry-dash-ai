"""Probe simple live macros and optionally save the first deterministic death."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_env import GeometryDashClient
from gd_env.protocol import BridgeDiagnostic
from gd_human_model import Event
from gd_trace.macro_schema import Macro
from gd_trace.replay_check import ReplayCheckSummary, summarize_replay_check
from gd_trace.save_trace import save_macro_json
from gd_trace.trace_schema import TraceRow


@dataclass(frozen=True, slots=True)
class CandidateMacro:
    """A candidate macro plus the metadata to save if it proves useful."""

    name: str
    events: list[Event]
    metadata: dict[str, object]

    def to_macro(self) -> Macro:
        return Macro(events=self.events, metadata=self.metadata)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Try simple queued macros against the live Geode bridge and report "
            "whether any produce deterministic deaths."
        )
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-observations", type=int, default=2400)
    parser.add_argument("--reset-wait-observations", type=int, default=600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29430)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--cbf", action="store_true")
    parser.add_argument("--physics-bypass", action="store_true")
    parser.add_argument(
        "--save-first",
        type=Path,
        help=(
            "Save the first deterministic death macro to this path. The file is "
            "written only after a live candidate passes."
        ),
    )
    args = parser.parse_args(argv)

    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.max_observations <= 0:
        parser.error("--max-observations must be positive")

    results = []
    first_deterministic_death: CandidateMacro | None = None

    with GeometryDashClient(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        for candidate in candidate_macros(args.max_observations):
            traces, diagnostics_by_trial = run_candidate(
                client,
                candidate,
                trials=args.trials,
                max_observations=args.max_observations,
                reset_wait_observations=args.reset_wait_observations,
                fps=args.fps,
                cbf=args.cbf,
                physics_bypass=args.physics_bypass,
            )
            summary = summarize_replay_check(
                traces,
                candidate.events,
                diagnostics_by_trial=diagnostics_by_trial,
            )
            deterministic_death = is_deterministic_death(summary)
            if deterministic_death and first_deterministic_death is None:
                first_deterministic_death = candidate
            results.append(
                {
                    "name": candidate.name,
                    "deterministic_death": deterministic_death,
                    "summary": summary.to_dict(),
                }
            )

    saved_path = None
    if args.save_first is not None and first_deterministic_death is not None:
        save_macro_json(first_deterministic_death.to_macro(), args.save_first)
        saved_path = str(args.save_first)

    print(
        json.dumps(
            {
                "saved_path": saved_path,
                "first_deterministic_death": (
                    first_deterministic_death.name
                    if first_deterministic_death is not None
                    else None
                ),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if first_deterministic_death is not None else 1


def candidate_macros(max_observations: int) -> list[CandidateMacro]:
    """Return conservative candidate macros for a local/offline test level."""

    hold_release_tick = max(1, max_observations - 1)
    repeated_clicks = [
        event
        for index in range(40)
        for event in (
            Event(20 + index * 12, "press"),
            Event(25 + index * 12, "release"),
        )
    ]
    return [
        CandidateMacro(
            name="no_input_until_death",
            events=[],
            metadata={
                "description": "No-input death candidate for deterministic replay checks.",
                "level_name": "local_offline_test",
                "probe_max_observations": max_observations,
            },
        ),
        CandidateMacro(
            name="hold_from_start",
            events=[Event(0, "press"), Event(hold_release_tick, "release")],
            metadata={
                "description": "Held p1 input from attempt start until late release.",
                "level_name": "local_offline_test",
                "probe_max_observations": max_observations,
            },
        ),
        CandidateMacro(
            name="single_jump_then_no_input",
            events=[Event(20, "press"), Event(30, "release")],
            metadata={
                "description": "One early jump followed by no input.",
                "level_name": "local_offline_test",
                "probe_max_observations": max_observations,
            },
        ),
        CandidateMacro(
            name="early_repeated_clicks",
            events=repeated_clicks,
            metadata={
                "description": "Dense early clicks for deterministic death probing.",
                "level_name": "local_offline_test",
                "probe_max_observations": max_observations,
            },
        ),
    ]


def run_candidate(
    client: GeometryDashClient,
    candidate: CandidateMacro,
    *,
    trials: int,
    max_observations: int,
    reset_wait_observations: int,
    fps: int,
    cbf: bool,
    physics_bypass: bool,
) -> tuple[list[list[TraceRow]], list[list[dict[str, object]]]]:
    """Run one candidate macro several times through mod-side queued replay."""

    traces: list[list[TraceRow]] = []
    diagnostics_by_trial: list[list[dict[str, object]]] = []
    client.load_macro(candidate.events, metadata=candidate.metadata)

    for trial_index in range(trials):
        diagnostics: list[BridgeDiagnostic] = []
        initial_observation = client.reset_attempt(
            f"death_probe_{candidate.name}_{trial_index + 1}",
            max_observations=reset_wait_observations,
            diagnostics=diagnostics,
        )
        rows = client.run_loaded_macro(
            max_observations=max_observations,
            fps=fps,
            cbf=cbf,
            physics_bypass=physics_bypass,
            initial_observation=initial_observation,
            diagnostics=diagnostics,
        )
        traces.append(rows)
        diagnostics_by_trial.append([diagnostic.to_dict() for diagnostic in diagnostics])

    return traces, diagnostics_by_trial


def is_deterministic_death(summary: ReplayCheckSummary) -> bool:
    """Return true when every trial dies at the same aligned trace tick."""

    return (
        bool(summary.death_ticks)
        and all(tick is not None for tick in summary.death_ticks)
        and summary.death_tick_std == 0.0
        and summary.x_position_max_diff == 0.0
        and summary.y_position_max_diff == 0.0
        and summary.input_state_mismatch_ticks == 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
