import pytest

from gd_env.protocol import (
    BridgeObservation,
    BridgeDiagnostic,
    LoadMacroCommand,
    ProtocolError,
    action_message,
    decode_message,
    diagnostic_message,
    encode_message,
    load_macro_message,
    observation_message,
    reset_message,
)
from gd_human_model import Event


def test_observation_message_roundtrip_and_trace_conversion() -> None:
    observation = BridgeObservation(
        tick=12,
        x=100.0,
        y=20.0,
        y_vel=-2.0,
        mode="cube",
        gravity="normal",
        percent=5.5,
        dead=False,
        input_down=True,
        completed=True,
        x_vel=8.0,
        rotation=45.0,
    )

    decoded = decode_message(encode_message(observation_message(observation)))

    assert decoded == observation
    assert isinstance(decoded, BridgeObservation)
    trace_row = decoded.to_trace_row(fps=240, cbf=False, physics_bypass=False)
    assert decoded.completed is True
    assert trace_row.tick == 12
    assert trace_row.time_ms == 50.0
    assert trace_row.input_down is True


def test_action_and_reset_messages_roundtrip() -> None:
    event = Event(10, "press")

    assert decode_message(encode_message(action_message(event))) == event
    reset = decode_message(encode_message(reset_message("death")))
    assert reset.reason == "death"  # type: ignore[attr-defined]


def test_load_macro_message_roundtrip_sorts_events() -> None:
    decoded = decode_message(
        encode_message(
            load_macro_message(
                [Event(20, "release"), Event(10, "press")],
                metadata={"level_name": "test"},
            )
        )
    )

    assert isinstance(decoded, LoadMacroCommand)
    assert decoded.events == [Event(10, "press"), Event(20, "release")]
    assert decoded.metadata == {"level_name": "test"}


def test_diagnostic_message_roundtrip() -> None:
    decoded = decode_message(
        encode_message(
            diagnostic_message(
                "macro_event_applied",
                tick=12,
                data={"event_index": 0, "intended_tick": 10, "applied_tick": 12},
            )
        )
    )

    assert isinstance(decoded, BridgeDiagnostic)
    assert decoded.kind == "macro_event_applied"
    assert decoded.tick == 12
    assert decoded.data["applied_tick"] == 12


def test_protocol_rejects_bad_version() -> None:
    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        decode_message('{"version":2,"type":"reset","reason":"test"}')
