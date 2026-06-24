from gd_env import DummyGeometryDashServer, GeometryDashClient
from gd_human_model import Event
from gd_trace.load_trace import load_trace_jsonl


def test_client_receives_observations_and_sends_action() -> None:
    with DummyGeometryDashServer(max_ticks=30, tick_interval_seconds=0.005) as server:
        assert server.port is not None
        with GeometryDashClient(port=server.port, timeout_seconds=2.0) as client:
            first = client.receive_observation()
            assert first.tick == 0
            assert first.input_down is False

            client.send_event(Event(first.tick, "press"))

            observations = [client.receive_observation() for _ in range(6)]
            assert any(observation.input_down for observation in observations)


def test_client_reset_restarts_dummy_attempt() -> None:
    with DummyGeometryDashServer(max_ticks=40, tick_interval_seconds=0.005) as server:
        assert server.port is not None
        with GeometryDashClient(port=server.port, timeout_seconds=2.0) as client:
            assert client.receive_observation().tick == 0
            assert client.receive_observation().tick >= 1

            client.send_reset("test")

            restarted = None
            for _ in range(10):
                observation = client.receive_observation()
                if observation.tick == 0:
                    restarted = observation
                    break

            assert restarted is not None
            assert restarted.input_down is False


def test_client_reset_attempt_returns_fresh_tick_zero() -> None:
    with DummyGeometryDashServer(max_ticks=40, tick_interval_seconds=0.005) as server:
        assert server.port is not None
        with GeometryDashClient(port=server.port, timeout_seconds=2.0) as client:
            assert client.receive_observation().tick == 0
            assert client.receive_observation().tick >= 1

            restarted = client.reset_attempt("test", max_observations=20)

            assert restarted.tick == 0
            assert restarted.input_down is False


def test_run_scripted_events_saves_trace(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trace_path = tmp_path / "dummy_trace.jsonl"

    with DummyGeometryDashServer(max_ticks=30, tick_interval_seconds=0.005) as server:
        assert server.port is not None
        with GeometryDashClient(port=server.port, timeout_seconds=2.0) as client:
            rows = client.run_scripted_events(
                [Event(1, "press"), Event(5, "release")],
                max_observations=12,
                trace_path=trace_path,
            )

    loaded_rows = load_trace_jsonl(trace_path)
    assert loaded_rows == rows
    assert len(rows) == 12
    assert any(row.input_down for row in rows)


def test_client_loads_macro_and_collects_mod_side_replay_diagnostics() -> None:
    with DummyGeometryDashServer(max_ticks=30, tick_interval_seconds=0.005) as server:
        assert server.port is not None
        with GeometryDashClient(port=server.port, timeout_seconds=2.0) as client:
            ack = client.load_macro([Event(2, "press"), Event(5, "release")])
            assert ack.message == "macro loaded"

            diagnostics = []
            initial = client.reset_attempt("queued_macro", diagnostics=diagnostics)
            rows = client.run_loaded_macro(
                max_observations=10,
                initial_observation=initial,
                diagnostics=diagnostics,
            )

    assert any(row.input_down for row in rows)
    assert rows[2].input_down is True
    assert rows[5].input_down is False
    assert [diagnostic.kind for diagnostic in diagnostics] == [
        "macro_event_applied",
        "macro_event_applied",
    ]
    assert diagnostics[0].data["intended_tick"] == 2
    assert diagnostics[0].data["applied_tick"] == 2
