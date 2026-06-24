"""Trace and macro file formats for Geometry Dash experiments."""

from gd_trace.click_window import ClickWindow, analyze_click_windows
from gd_trace.compare_trace import TraceComparison, compare_traces
from gd_trace.humanized_run import (
    HumanizedAttemptSummary,
    HumanizedRunSummary,
    NumericDistribution,
    summarize_humanized_attempts,
)
from gd_trace.load_trace import load_macro_json, load_trace_jsonl
from gd_trace.macro_schema import Macro, MacroSchemaError
from gd_trace.replay_check import (
    InputLatencySummary,
    MacroApplicationSummary,
    ObservedInputTransition,
    ReplayCheckSummary,
    detect_input_transitions,
    summarize_macro_applications,
    summarize_input_latency,
    summarize_replay_check,
)
from gd_trace.save_trace import save_macro_json, save_trace_jsonl
from gd_trace.trace_schema import TraceRow, TraceSchemaError

__all__ = [
    "ClickWindow",
    "HumanizedAttemptSummary",
    "HumanizedRunSummary",
    "InputLatencySummary",
    "MacroApplicationSummary",
    "Macro",
    "MacroSchemaError",
    "NumericDistribution",
    "ObservedInputTransition",
    "ReplayCheckSummary",
    "TraceComparison",
    "TraceRow",
    "TraceSchemaError",
    "analyze_click_windows",
    "compare_traces",
    "detect_input_transitions",
    "load_macro_json",
    "load_trace_jsonl",
    "save_macro_json",
    "save_trace_jsonl",
    "summarize_humanized_attempts",
    "summarize_input_latency",
    "summarize_macro_applications",
    "summarize_replay_check",
]
