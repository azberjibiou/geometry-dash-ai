"""Synthetic practice executors used for tests and local smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from gd_trace import Macro, TraceRow


@dataclass(frozen=True, slots=True)
class SyntheticTraceExecutor:
    """Return prebuilt traces, one per attempt.

    This executor never represents Geometry Dash physics. It exists so reward,
    logging, humanization, and summary behavior can be tested without a live
    game process.
    """

    traces: Sequence[Sequence[TraceRow]]

    def run_attempt(
        self,
        *,
        attempt_index: int,
        executed_macro: Macro,
        attempt_dir: Path,
        metadata: dict[str, Any],
    ) -> Sequence[TraceRow]:
        if attempt_index < 1 or attempt_index > len(self.traces):
            raise IndexError("synthetic trace not available for attempt")
        return self.traces[attempt_index - 1]
