"""Canonical per-tick trace row schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class TraceSchemaError(ValueError):
    """Raised when trace data does not match the canonical schema."""


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One recorded Geometry Dash physics tick."""

    tick: int
    time_ms: float
    input_down: bool
    x: float
    y: float
    x_vel: float
    y_vel: float
    rotation: float
    mode: str
    gravity: str
    percent: float
    dead: bool
    death_reason: str | None
    fps: int
    cbf: bool
    physics_bypass: bool

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise TraceSchemaError("tick must be non-negative")
        if self.time_ms < 0:
            raise TraceSchemaError("time_ms must be non-negative")
        if self.fps <= 0:
            raise TraceSchemaError("fps must be positive")
        if not 0.0 <= self.percent <= 100.0:
            raise TraceSchemaError("percent must be between 0 and 100")
        if not self.mode:
            raise TraceSchemaError("mode must be non-empty")
        if not self.gravity:
            raise TraceSchemaError("gravity must be non-empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TraceRow":
        """Build and validate a row from parsed JSON data."""

        return cls(
            tick=_as_int(data, "tick"),
            time_ms=_as_float(data, "time_ms"),
            input_down=_as_bool(data, "input_down"),
            x=_as_float(data, "x"),
            y=_as_float(data, "y"),
            x_vel=_as_float(data, "x_vel"),
            y_vel=_as_float(data, "y_vel"),
            rotation=_as_float(data, "rotation"),
            mode=_as_str(data, "mode"),
            gravity=_as_str(data, "gravity"),
            percent=_as_float(data, "percent"),
            dead=_as_bool(data, "dead"),
            death_reason=_as_optional_str(data, "death_reason"),
            fps=_as_int(data, "fps"),
            cbf=_as_bool(data, "cbf"),
            physics_bypass=_as_bool(data, "physics_bypass"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable row data."""

        return asdict(self)


def validate_trace_sequence(rows: list[TraceRow]) -> None:
    """Validate ordering and fixed run settings across a trace."""

    previous_tick: int | None = None
    fps: int | None = None
    cbf: bool | None = None
    physics_bypass: bool | None = None

    for index, row in enumerate(rows):
        if previous_tick is not None and row.tick <= previous_tick:
            raise TraceSchemaError(
                f"trace ticks must be strictly increasing at row {index}"
            )
        previous_tick = row.tick

        if fps is None:
            fps = row.fps
            cbf = row.cbf
            physics_bypass = row.physics_bypass
            continue

        if row.fps != fps:
            raise TraceSchemaError(f"fps changed at row {index}")
        if row.cbf != cbf:
            raise TraceSchemaError(f"cbf changed at row {index}")
        if row.physics_bypass != physics_bypass:
            raise TraceSchemaError(f"physics_bypass changed at row {index}")


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise TraceSchemaError(f"missing required field: {key}")
    return data[key]


def _as_int(data: Mapping[str, Any], key: str) -> int:
    value = _required(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceSchemaError(f"{key} must be an int")
    return value


def _as_float(data: Mapping[str, Any], key: str) -> float:
    value = _required(data, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceSchemaError(f"{key} must be a number")
    return float(value)


def _as_bool(data: Mapping[str, Any], key: str) -> bool:
    value = _required(data, key)
    if not isinstance(value, bool):
        raise TraceSchemaError(f"{key} must be a bool")
    return value


def _as_str(data: Mapping[str, Any], key: str) -> str:
    value = _required(data, key)
    if not isinstance(value, str):
        raise TraceSchemaError(f"{key} must be a string")
    return value


def _as_optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = _required(data, key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TraceSchemaError(f"{key} must be a string or null")
    return value
