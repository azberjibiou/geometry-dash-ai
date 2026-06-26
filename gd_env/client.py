"""Blocking Python client for the Geometry Dash bridge protocol."""

from __future__ import annotations

import socket
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Mapping, TextIO

from gd_human_model.events import Event, sort_events
from gd_trace.save_trace import save_trace_jsonl
from gd_trace.trace_schema import TraceRow

from gd_env.protocol import (
    AckMessage,
    BridgeDiagnostic,
    BridgeObservation,
    ErrorMessage,
    ProtocolError,
    action_message,
    decode_message,
    encode_message,
    load_macro_message,
    reset_message,
)


class GeometryDashClient:
    """Small JSON-line client for a Geode mod or compatible dummy server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 29430,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self._socket: socket.socket | None = None
        self._reader: TextIO | None = None
        self._writer: TextIO | None = None

    def __enter__(self) -> "GeometryDashClient":
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def connect(self) -> "GeometryDashClient":
        if self._socket is not None:
            return self

        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_seconds)
        sock.settimeout(self.timeout_seconds)
        self._socket = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")
        self._writer = sock.makefile("w", encoding="utf-8", newline="\n")
        return self

    def close(self) -> None:
        for stream in (self._reader, self._writer):
            if stream is not None:
                stream.close()
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._reader = None
        self._writer = None

    def send_event(self, event: Event) -> None:
        """Send a press/release event to be applied by the mod."""

        self._send(action_message(event))

    def load_macro(
        self,
        events: Iterable[Event],
        *,
        metadata: Mapping[str, Any] | None = None,
        max_messages: int = 600,
    ) -> AckMessage:
        """Load a complete macro into the mod and wait for acknowledgement."""

        self._send(load_macro_message(list(events), metadata=metadata))

        for _ in range(max_messages):
            message = self._receive()
            if isinstance(message, ErrorMessage):
                raise ProtocolError(message.message)
            if isinstance(message, AckMessage) and message.message == "macro loaded":
                return message

        raise TimeoutError("did not receive macro loaded acknowledgement")

    def send_reset(self, reason: str = "requested") -> None:
        """Request restart/reset for the current attempt."""

        self._send(reset_message(reason))

    def reset_attempt(
        self,
        reason: str = "requested",
        *,
        max_observations: int = 600,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        """Request a reset and return the first observation from the fresh attempt."""

        self.send_reset(reason)
        saw_reset_ack = False

        for _ in range(max_observations):
            message = self._receive()
            if isinstance(message, ErrorMessage):
                raise ProtocolError(message.message)
            if isinstance(message, AckMessage):
                if message.message == "reset queued":
                    saw_reset_ack = True
                continue
            if isinstance(message, BridgeDiagnostic):
                if diagnostics is not None:
                    diagnostics.append(message)
                continue
            if not isinstance(message, BridgeObservation):
                continue
            if saw_reset_ack and message.tick == 0:
                return message

        raise TimeoutError("did not receive a fresh tick-0 observation after reset")

    def receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        """Read messages until the next observation arrives."""

        return self._receive_observation(diagnostics=diagnostics)

    def run_scripted_events(
        self,
        events: Iterable[Event],
        *,
        max_observations: int,
        fps: int = 240,
        cbf: bool = False,
        physics_bypass: bool = False,
        trace_path: str | Path | None = None,
        initial_observation: BridgeObservation | None = None,
        stop_percent: float | None = None,
    ) -> list[TraceRow]:
        """Receive observations, send due scripted events, and optionally save a trace."""

        sorted_events = sort_events(events)
        next_event_index = 0
        trace: list[TraceRow] = []

        for observation_index in range(max_observations):
            if observation_index == 0 and initial_observation is not None:
                observation = initial_observation
            else:
                observation = self.receive_observation()
            trace.append(
                observation.to_trace_row(
                    fps=fps,
                    cbf=cbf,
                    physics_bypass=physics_bypass,
                )
            )

            while (
                next_event_index < len(sorted_events)
                and sorted_events[next_event_index].tick <= observation.tick
            ):
                self.send_event(sorted_events[next_event_index])
                next_event_index += 1

            if observation.dead or _reached_stop_percent(observation, stop_percent):
                break

        if trace_path is not None:
            save_trace_jsonl(trace, trace_path)
        return trace

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
        """Collect a trace while the mod plays its pre-loaded macro."""

        trace: list[TraceRow] = []

        for observation_index in range(max_observations):
            if observation_index == 0 and initial_observation is not None:
                observation = initial_observation
            else:
                observation = self._receive_observation(diagnostics=diagnostics)
            trace.append(
                observation.to_trace_row(
                    fps=fps,
                    cbf=cbf,
                    physics_bypass=physics_bypass,
                )
            )

            if observation.dead or _reached_stop_percent(observation, stop_percent):
                break

        if trace_path is not None:
            save_trace_jsonl(trace, trace_path)
        return trace

    def _send(self, message: dict[str, object]) -> None:
        if self._writer is None:
            raise RuntimeError("client is not connected")
        self._writer.write(encode_message(message))
        self._writer.flush()

    def _receive(self) -> object:
        if self._reader is None:
            raise RuntimeError("client is not connected")
        line = self._reader.readline()
        if line == "":
            raise EOFError("bridge connection closed")
        return decode_message(line)

    def _receive_observation(
        self,
        *,
        diagnostics: list[BridgeDiagnostic] | None = None,
    ) -> BridgeObservation:
        while True:
            message = self._receive()
            if isinstance(message, BridgeObservation):
                return message
            if isinstance(message, BridgeDiagnostic):
                if diagnostics is not None:
                    diagnostics.append(message)
                continue
            if isinstance(message, ErrorMessage):
                raise ProtocolError(message.message)


def _reached_stop_percent(
    observation: BridgeObservation,
    stop_percent: float | None,
) -> bool:
    if stop_percent is None:
        return False
    return observation.percent >= stop_percent
