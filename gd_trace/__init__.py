"""Trace and macro file formats for Geometry Dash experiments."""

from gd_trace.click_window import ClickWindow, analyze_click_windows
from gd_trace.compare_trace import TraceComparison, compare_traces
from gd_trace.load_trace import load_macro_json, load_trace_jsonl
from gd_trace.macro_schema import Macro, MacroSchemaError
from gd_trace.save_trace import save_macro_json, save_trace_jsonl
from gd_trace.trace_schema import TraceRow, TraceSchemaError

__all__ = [
    "ClickWindow",
    "Macro",
    "MacroSchemaError",
    "TraceComparison",
    "TraceRow",
    "TraceSchemaError",
    "analyze_click_windows",
    "compare_traces",
    "load_macro_json",
    "load_trace_jsonl",
    "save_macro_json",
    "save_trace_jsonl",
]
