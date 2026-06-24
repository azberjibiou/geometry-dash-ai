import pytest

from gd_env.protocol import (
    BridgeObservation,
    ProtocolError,
    action_message,
    decode_message,
    encode_message,
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
        x_vel=8.0,
        rotation=45.0,
    )

    decoded = decode_message(encode_message(observation_message(observation)))

    assert decoded == observation
    assert isinstance(decoded, BridgeObservation)
    trace_row = decoded.to_trace_row(fps=240, cbf=False, physics_bypass=False)
    assert trace_row.tick == 12
    assert trace_row.time_ms == 50.0
    assert trace_row.input_down is True


def test_action_and_reset_messages_roundtrip() -> None:
    event = Event(10, "press")

    assert decode_message(encode_message(action_message(event))) == event
    reset = decode_message(encode_message(reset_message("death")))
    assert reset.reason == "death"  # type: ignore[attr-defined]


def test_protocol_rejects_bad_version() -> None:
    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        decode_message('{"version":2,"type":"reset","reason":"test"}')
