"""JSON-line protocol shared by Python and the future Geode mod."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from gd_human_model.events import Event
from gd_trace.macro_schema import event_from_mapping, event_to_dict
from gd_trace.trace_schema import TraceRow

PROTOCOL_VERSION = 1
MessageType = Literal["observation", "action", "reset", "ack", "error"]


class ProtocolError(ValueError):
    """Raised when a bridge message is malformed or unexpected."""


@dataclass(frozen=True, slots=True)
class BridgeObservation:
    """Minimal observation sent by the Geode mod once per physics tick."""

    tick: int
    x: float
    y: float
    y_vel: float
    mode: str
    gravity: str
    percent: float
    dead: bool
    input_down: bool
    x_vel: float = 0.0
    rotation: float = 0.0
    death_reason: str | None = None

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ProtocolError("observation.tick must be non-negative")
        if not 0.0 <= self.percent <= 100.0:
            raise ProtocolError("observation.percent must be between 0 and 100")
        if not self.mode:
            raise ProtocolError("observation.mode must be non-empty")
        if not self.gravity:
            raise ProtocolError("observation.gravity must be non-empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BridgeObservation":
        return cls(
            tick=_as_int(data, "tick"),
            x=_as_float(data, "x"),
            y=_as_float(data, "y"),
            y_vel=_as_float(data, "y_vel"),
            mode=_as_str(data, "mode"),
            gravity=_as_str(data, "gravity"),
            percent=_as_float(data, "percent"),
            dead=_as_bool(data, "dead"),
            input_down=_as_bool(data, "input_down"),
            x_vel=_as_float(data, "x_vel", default=0.0),
            rotation=_as_float(data, "rotation", default=0.0),
            death_reason=_as_optional_str(data, "death_reason", default=None),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_trace_row(
        self,
        *,
        fps: int,
        cbf: bool,
        physics_bypass: bool,
    ) -> TraceRow:
        """Convert a bridge observation to the Phase 2 trace schema."""

        return TraceRow(
            tick=self.tick,
            time_ms=self.tick * 1000.0 / fps,
            input_down=self.input_down,
            x=self.x,
            y=self.y,
            x_vel=self.x_vel,
            y_vel=self.y_vel,
            rotation=self.rotation,
            mode=self.mode,
            gravity=self.gravity,
            percent=self.percent,
            dead=self.dead,
            death_reason=self.death_reason,
            fps=fps,
            cbf=cbf,
            physics_bypass=physics_bypass,
        )


@dataclass(frozen=True, slots=True)
class ResetCommand:
    """Request that the mod restart the current attempt."""

    reason: str = "requested"

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason}


@dataclass(frozen=True, slots=True)
class AckMessage:
    """Positive response from the mod or dummy server."""

    tick: int | None
    message: str


@dataclass(frozen=True, slots=True)
class ErrorMessage:
    """Error response from the mod or dummy server."""

    message: str


def observation_message(observation: BridgeObservation) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "observation",
        "observation": observation.to_dict(),
    }


def action_message(event: Event) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "action",
        "event": event_to_dict(event),
    }


def reset_message(reason: str = "requested") -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "reset",
        "reason": reason,
    }


def ack_message(message: str, tick: int | None = None) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "ack",
        "tick": tick,
        "message": message,
    }


def error_message(message: str) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "error",
        "message": message,
    }


def encode_message(message: Mapping[str, Any]) -> str:
    """Encode one protocol message as a newline-terminated JSON string."""

    _validate_message_envelope(message)
    return json.dumps(message, separators=(",", ":"), sort_keys=True) + "\n"


def decode_message(line: str) -> BridgeObservation | Event | ResetCommand | AckMessage | ErrorMessage:
    """Decode one newline-delimited JSON protocol message."""

    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON message: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    _validate_message_envelope(data)

    message_type = data["type"]
    if message_type == "observation":
        observation_data = data.get("observation")
        if not isinstance(observation_data, Mapping):
            raise ProtocolError("observation message must contain observation object")
        return BridgeObservation.from_mapping(observation_data)
    if message_type == "action":
        return event_from_mapping(data.get("event"))
    if message_type == "reset":
        return ResetCommand(reason=_as_str(data, "reason", default="requested"))
    if message_type == "ack":
        tick = data.get("tick")
        if tick is not None and (isinstance(tick, bool) or not isinstance(tick, int)):
            raise ProtocolError("ack.tick must be an int or null")
        return AckMessage(tick=tick, message=_as_str(data, "message"))
    if message_type == "error":
        return ErrorMessage(message=_as_str(data, "message"))

    raise ProtocolError(f"unsupported message type: {message_type}")


def _validate_message_envelope(message: Mapping[str, Any]) -> None:
    version = message.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    message_type = message.get("type")
    if message_type not in ("observation", "action", "reset", "ack", "error"):
        raise ProtocolError("message.type is invalid")


def _required(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in data:
        return data[key]
    if default is not None:
        return default
    raise ProtocolError(f"missing required field: {key}")


def _as_int(data: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = _required(data, key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an int")
    return value


def _as_float(data: Mapping[str, Any], key: str, default: float | None = None) -> float:
    value = _required(data, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{key} must be a number")
    return float(value)


def _as_bool(data: Mapping[str, Any], key: str) -> bool:
    value = _required(data, key)
    if not isinstance(value, bool):
        raise ProtocolError(f"{key} must be a bool")
    return value


def _as_str(data: Mapping[str, Any], key: str, default: str | None = None) -> str:
    value = _required(data, key, default)
    if not isinstance(value, str):
        raise ProtocolError(f"{key} must be a string")
    return value


def _as_optional_str(
    data: Mapping[str, Any],
    key: str,
    default: str | None = None,
) -> str | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProtocolError(f"{key} must be a string or null")
    return value
