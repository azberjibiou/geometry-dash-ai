"""A tiny TCP dummy server that speaks the bridge protocol for tests."""

from __future__ import annotations

import select
import socket
import threading
import time
from types import TracebackType

from gd_human_model.events import Event

from gd_env.protocol import (
    BridgeObservation,
    LoadMacroCommand,
    ResetCommand,
    action_message,
    ack_message,
    decode_message,
    diagnostic_message,
    encode_message,
    error_message,
    observation_message,
)


class DummyGeometryDashServer:
    """Single-client fake bridge server.

    This is not a Geometry Dash simulator. It only proves that the Python bridge
    can exchange observations, actions, resets, and traces over the planned wire
    protocol.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        max_ticks: int = 120,
        tick_interval_seconds: float = 0.002,
    ) -> None:
        self.host = host
        self.requested_port = port
        self.max_ticks = max_ticks
        self.tick_interval_seconds = tick_interval_seconds
        self.port: int | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None

    def __enter__(self) -> "DummyGeometryDashServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("dummy server did not start")

    def stop(self) -> None:
        self._stop.set()
        if self._server_socket is not None:
            self._server_socket.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def _run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.requested_port))
            server_socket.listen(1)
            server_socket.settimeout(0.2)
            self._server_socket = server_socket
            self.port = int(server_socket.getsockname()[1])
            self._ready.set()

            try:
                connection, _ = server_socket.accept()
            except OSError:
                return
            with connection:
                connection.setblocking(False)
                self._serve_client(connection)

    def _serve_client(self, connection: socket.socket) -> None:
        tick = 0
        input_down = False
        pending_actions: list[Event] = []
        loaded_macro: list[Event] = []
        macro_loaded = False
        macro_active = False
        next_macro_event_index = 0
        buffer = b""

        while not self._stop.is_set() and tick < self.max_ticks:
            for event in pending_actions:
                if event.player == "p1":
                    input_down = event.kind == "press"
            pending_actions.clear()

            while (
                macro_active
                and next_macro_event_index < len(loaded_macro)
                and loaded_macro[next_macro_event_index].tick <= tick
            ):
                event = loaded_macro[next_macro_event_index]
                if event.player == "p1":
                    input_down = event.kind == "press"
                self._send(
                    connection,
                    diagnostic_message(
                        "macro_event_applied",
                        tick=tick,
                        data={
                            "event_index": next_macro_event_index,
                            "intended_tick": event.tick,
                            "applied_tick": tick,
                            "kind": event.kind,
                            "player": event.player,
                        },
                    ),
                )
                next_macro_event_index += 1

            observation = BridgeObservation(
                tick=tick,
                x=float(tick),
                y=10.0 if input_down else 0.0,
                y_vel=1.0 if input_down else 0.0,
                mode="cube",
                gravity="normal",
                percent=min(100.0, tick * 100.0 / max(1, self.max_ticks - 1)),
                dead=False,
                input_down=input_down,
                x_vel=1.0,
            )
            self._send(connection, observation_message(observation))

            time.sleep(self.tick_interval_seconds)
            buffer, reset_requested, new_actions, new_macro = self._drain_commands(
                connection,
                buffer,
                tick,
            )
            if new_macro is not None:
                loaded_macro = new_macro
                macro_loaded = True
                macro_active = False
                next_macro_event_index = 0
            if reset_requested:
                tick = 0
                input_down = False
                pending_actions.clear()
                macro_active = macro_loaded
                next_macro_event_index = 0
                continue
            pending_actions.extend(new_actions)
            tick += 1

    def _drain_commands(
        self,
        connection: socket.socket,
        buffer: bytes,
        current_tick: int,
    ) -> tuple[bytes, bool, list[Event], list[Event] | None]:
        reset_requested = False
        actions: list[Event] = []
        loaded_macro: list[Event] | None = None

        while True:
            readable, _, _ = select.select([connection], [], [], 0.0)
            if not readable:
                break
            try:
                chunk = connection.recv(4096)
            except BlockingIOError:
                break
            except OSError:
                return b"", reset_requested, actions, loaded_macro
            if not chunk:
                return b"", reset_requested, actions, loaded_macro
            buffer += chunk

            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                if not raw_line.strip():
                    continue
                try:
                    message = decode_message(raw_line.decode("utf-8"))
                except Exception as exc:  # pragma: no cover - defensive server path.
                    self._send(connection, error_message(str(exc)))
                    continue

                if isinstance(message, Event):
                    actions.append(message)
                    self._send(connection, ack_message("action queued", tick=current_tick))
                elif isinstance(message, LoadMacroCommand):
                    loaded_macro = message.events
                    self._send(connection, ack_message("macro loaded", tick=current_tick))
                elif isinstance(message, ResetCommand):
                    reset_requested = True
                    self._send(connection, ack_message("reset queued", tick=current_tick))
                else:
                    self._send(connection, error_message("unexpected client message"))

        return buffer, reset_requested, actions, loaded_macro

    @staticmethod
    def _send(connection: socket.socket, message: dict[str, object]) -> None:
        try:
            connection.sendall(encode_message(message).encode("utf-8"))
        except OSError:
            pass
