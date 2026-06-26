import pytest

from gd_capture import CaptureRegion, FrameCaptureRecord, save_manifest_jsonl
from gd_human_model import Event
from gd_imitation import (
    DatasetConfig,
    DatasetPrepError,
    build_imitation_samples,
    load_imitation_samples,
    split_imitation_samples,
)
from gd_trace import Macro, TraceRow, save_macro_json, save_trace_jsonl


def test_samples_align_frames_trace_state_and_event_labels() -> None:
    records = [_record(tick=tick, percent=0.0) for tick in range(5)]
    rows = [
        _row(tick=tick, percent=50.0 + tick, input_down=(tick == 2))
        for tick in range(5)
    ]
    events = [Event(3, "press"), Event(4, "release")]

    samples = build_imitation_samples(
        records,
        rows,
        events,
        config=DatasetConfig(frame_stack_size=1),
    )

    by_tick = {sample.tick: sample for sample in samples}
    assert by_tick[2].progress == 52.0
    assert by_tick[2].input_down is True
    assert by_tick[2].target_input_down is True
    assert by_tick[3].press_event is True
    assert by_tick[3].release_event is False
    assert by_tick[4].press_event is False
    assert by_tick[4].release_event is True


def test_delayed_label_shift_uses_future_event_tick() -> None:
    records = [_record(tick=tick) for tick in range(8)]
    rows = [_row(tick=tick) for tick in range(8)]
    events = [Event(5, "press")]

    samples = build_imitation_samples(
        records,
        rows,
        events,
        config=DatasetConfig(frame_stack_size=1, label_shift_frames=2),
    )

    by_tick = {sample.tick: sample for sample in samples}
    assert by_tick[3].label_tick == 5
    assert by_tick[3].target_input_down is False
    assert by_tick[3].press_event is True
    assert by_tick[5].label_tick == 7
    assert by_tick[5].press_event is False
    assert 6 not in by_tick
    assert 7 not in by_tick


def test_frame_windowing_uses_previous_captured_frames() -> None:
    records = [_record(tick=tick) for tick in (0, 2, 4, 6)]
    rows = [_row(tick=tick) for tick in range(7)]

    samples = build_imitation_samples(
        records,
        rows,
        [],
        config=DatasetConfig(frame_stack_size=3),
    )

    assert [sample.tick for sample in samples] == [4, 6]
    assert samples[0].frame_ticks == (0, 2, 4)
    assert samples[0].frame_paths == (
        "frames/frame_000000.bmp",
        "frames/frame_000002.bmp",
        "frames/frame_000004.bmp",
    )
    assert samples[1].frame_ticks == (2, 4, 6)


def test_event_label_radius_handles_strided_captures() -> None:
    records = [_record(tick=tick) for tick in (0, 5, 10, 15)]
    rows = [_row(tick=tick) for tick in range(16)]
    events = [Event(11, "press")]

    samples = build_imitation_samples(
        records,
        rows,
        events,
        config=DatasetConfig(
            frame_stack_size=1,
            event_label_radius_frames=1,
        ),
    )

    by_tick = {sample.tick: sample for sample in samples}
    assert by_tick[10].label_tick == 10
    assert by_tick[10].press_event is True
    assert by_tick[10].events == (Event(11, "press"),)


def test_split_samples_is_deterministic() -> None:
    samples = build_imitation_samples(
        [_record(tick=tick) for tick in range(10)],
        [_row(tick=tick) for tick in range(10)],
        [],
        config=DatasetConfig(frame_stack_size=1),
    )

    contiguous = split_imitation_samples(samples, validation_fraction=0.3)
    first_shuffle = split_imitation_samples(
        samples,
        validation_fraction=0.3,
        shuffle=True,
        seed=123,
    )
    second_shuffle = split_imitation_samples(
        samples,
        validation_fraction=0.3,
        shuffle=True,
        seed=123,
    )

    assert contiguous.validation_indices == (7, 8, 9)
    assert first_shuffle.validation_indices == second_shuffle.validation_indices
    assert first_shuffle.train_indices == second_shuffle.train_indices
    assert first_shuffle.to_dict()["validation_count"] == 3


def test_load_imitation_samples_from_artifact_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest_path = tmp_path / "manifest.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    macro_path = tmp_path / "macro.json"
    save_manifest_jsonl([_record(tick=tick) for tick in range(4)], manifest_path)
    save_trace_jsonl([_row(tick=tick) for tick in range(4)], trace_path)
    save_macro_json(Macro(events=[Event(2, "press")]), macro_path)

    samples = load_imitation_samples(
        manifest_path,
        trace_path,
        macro_path,
        config=DatasetConfig(frame_stack_size=2, label_shift_frames=1),
    )

    assert [sample.tick for sample in samples] == [1, 2]
    assert samples[0].frame_ticks == (0, 1)
    assert samples[0].label_tick == 2
    assert samples[0].press_event is True


def test_missing_trace_tick_is_rejected() -> None:
    records = [_record(tick=0), _record(tick=1), _record(tick=2)]
    rows = [_row(tick=0), _row(tick=2)]

    with pytest.raises(DatasetPrepError, match="frame tick 1"):
        build_imitation_samples(
            records,
            rows,
            [],
            config=DatasetConfig(frame_stack_size=1),
        )


def test_unmatched_frame_ticks_can_be_dropped() -> None:
    records = [_record(tick=0), _record(tick=1), _record(tick=2)]
    rows = [_row(tick=0), _row(tick=2)]

    samples = build_imitation_samples(
        records,
        rows,
        [],
        config=DatasetConfig(
            frame_stack_size=1,
            drop_unmatched_frame_ticks=True,
        ),
    )

    assert [sample.tick for sample in samples] == [0, 2]


def _record(
    *,
    tick: int,
    percent: float = 0.0,
    input_down: bool = False,
) -> FrameCaptureRecord:
    return FrameCaptureRecord(
        frame_path=f"frames/frame_{tick:06d}.bmp",
        tick=tick,
        time_ms=tick * 1000.0 / 240,
        percent=percent,
        x=float(tick),
        y=2.0,
        x_vel=1.0,
        y_vel=0.0,
        rotation=0.0,
        mode="cube",
        gravity="normal",
        input_down=input_down,
        dead=False,
        completed=False,
        death_reason=None,
        capture_width=2,
        capture_height=2,
        capture_region=CaptureRegion(0, 0, 2, 2).to_dict(),
        window_title="Geometry Dash",
    )


def _row(
    *,
    tick: int,
    percent: float = 0.0,
    input_down: bool = False,
) -> TraceRow:
    return TraceRow(
        tick=tick,
        time_ms=tick * 1000.0 / 240,
        input_down=input_down,
        x=float(tick),
        y=2.0,
        x_vel=1.0,
        y_vel=0.0,
        rotation=0.0,
        mode="cube",
        gravity="normal",
        percent=percent,
        dead=False,
        death_reason=None,
        fps=240,
        cbf=False,
        physics_bypass=False,
    )
