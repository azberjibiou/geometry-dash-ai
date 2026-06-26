import json

from gd_capture import CaptureRegion, FrameCaptureRecord, save_manifest_jsonl
from gd_human_model import Event
from gd_trace import Macro, TraceRow, save_macro_json, save_trace_jsonl
from scripts.prepare_imitation_dataset import main


def test_prepare_imitation_dataset_writes_samples_split_and_summary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest_path = tmp_path / "manifest.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    macro_path = tmp_path / "macro.json"
    output_dir = tmp_path / "dataset"
    save_manifest_jsonl([_record(tick=tick) for tick in range(6)], manifest_path)
    save_trace_jsonl(
        [_row(tick=tick, input_down=(3 <= tick < 4)) for tick in range(6)],
        trace_path,
    )
    save_macro_json(
        Macro(events=[Event(3, "press"), Event(4, "release")]),
        macro_path,
    )

    exit_code = main(
        [
            "--manifest-jsonl",
            str(manifest_path),
            "--trace-jsonl",
            str(trace_path),
            "--macro-json",
            str(macro_path),
            "--output-dir",
            str(output_dir),
            "--frame-stack-size",
            "2",
            "--label-shift-frames",
            "1",
            "--validation-fraction",
            "0.25",
        ]
    )

    assert exit_code == 0
    samples_path = output_dir / "samples.jsonl"
    split_path = output_dir / "split.json"
    summary_path = output_dir / "summary.json"
    samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [sample["tick"] for sample in samples] == [1, 2, 3, 4]
    assert [sample["tick"] for sample in samples if sample["press_event"]] == [2]
    assert [sample["tick"] for sample in samples if sample["release_event"]] == [3]
    assert [sample["tick"] for sample in samples if sample["target_input_down"]] == [2]
    assert split["train_count"] == 3
    assert split["validation_count"] == 1
    assert summary["summary"]["sample_count"] == 4
    assert summary["summary"]["press_label_count"] == 1
    assert summary["summary"]["release_label_count"] == 1
    assert summary["summary"]["target_input_down_count"] == 1
    assert summary["config"]["label_shift_frames"] == 1


def test_prepare_imitation_dataset_rejects_empty_dataset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest_path = tmp_path / "manifest.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    macro_path = tmp_path / "macro.json"
    save_manifest_jsonl([_record(tick=0)], manifest_path)
    save_trace_jsonl([_row(tick=0)], trace_path)
    save_macro_json(Macro(events=[]), macro_path)

    exit_code = main(
        [
            "--manifest-jsonl",
            str(manifest_path),
            "--trace-jsonl",
            str(trace_path),
            "--macro-json",
            str(macro_path),
            "--frame-stack-size",
            "2",
        ]
    )

    assert exit_code == 1


def _record(*, tick: int) -> FrameCaptureRecord:
    return FrameCaptureRecord(
        frame_path=f"frames/frame_{tick:06d}.bmp",
        tick=tick,
        time_ms=tick * 1000.0 / 240,
        percent=float(tick),
        x=float(tick),
        y=2.0,
        x_vel=1.0,
        y_vel=0.0,
        rotation=0.0,
        mode="cube",
        gravity="normal",
        input_down=False,
        dead=False,
        completed=False,
        death_reason=None,
        capture_width=2,
        capture_height=2,
        capture_region=CaptureRegion(0, 0, 2, 2).to_dict(),
        window_title="Geometry Dash",
    )


def _row(*, tick: int, input_down: bool = False) -> TraceRow:
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
        percent=float(tick),
        dead=False,
        death_reason=None,
        fps=240,
        cbf=False,
        physics_bypass=False,
    )
