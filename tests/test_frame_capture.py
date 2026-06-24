import pytest

from gd_capture import (
    CapturedFrame,
    CaptureRegion,
    FrameCaptureRecord,
    inspect_bmp,
    load_manifest_jsonl,
    save_manifest_jsonl,
    validate_frame_manifest,
    write_bmp,
)
from gd_capture.manifest import ManifestError
from gd_env import BridgeObservation


def test_manifest_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest_path = tmp_path / "manifest.jsonl"
    region = CaptureRegion(left=10, top=20, width=2, height=2)
    observation = _make_observation(tick=7, percent=12.5, input_down=True)
    records = [
        FrameCaptureRecord.from_observation(
            observation,
            frame_path="frames/frame_000001.bmp",
            capture_width=2,
            capture_height=2,
            capture_region=region,
            fps=240,
            window_title="Geometry Dash",
        )
    ]

    save_manifest_jsonl(records, manifest_path)

    assert load_manifest_jsonl(manifest_path) == records


def test_bmp_inspection_reports_dimensions_and_stats(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frame_path = tmp_path / "frame.bmp"

    write_bmp(frame_path, _varied_frame())

    stats = inspect_bmp(frame_path)
    assert stats.width == 2
    assert stats.height == 2
    assert stats.brightness_variance > 1.0
    assert len(stats.sha256) == 64


def test_frame_manifest_validation_accepts_changing_frames(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames_dir = tmp_path / "frames"
    first_frame = frames_dir / "first.bmp"
    second_frame = frames_dir / "second.bmp"
    write_bmp(first_frame, _varied_frame())
    write_bmp(second_frame, _other_varied_frame())
    records = [
        _record(tick=0, frame_path="frames/first.bmp"),
        _record(tick=1, frame_path="frames/second.bmp"),
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    save_manifest_jsonl(records, manifest_path)

    summary = validate_frame_manifest(manifest_path, base_dir=tmp_path)

    assert summary.ok
    assert summary.frame_count == 2
    assert summary.width_values == [2]
    assert summary.height_values == [2]
    assert summary.unique_hash_count == 2


def test_frame_manifest_validation_flags_bad_frames(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames_dir = tmp_path / "frames"
    write_bmp(frames_dir / "blank.bmp", _solid_frame())
    write_bmp(frames_dir / "wrong_size.bmp", _varied_frame())
    records = [
        _record(tick=0, frame_path="frames/blank.bmp"),
        FrameCaptureRecord(
            frame_path="frames/wrong_size.bmp",
            tick=1,
            time_ms=1000.0 / 240,
            percent=0.0,
            x=0.0,
            y=0.0,
            x_vel=0.0,
            y_vel=0.0,
            rotation=0.0,
            mode="cube",
            gravity="normal",
            input_down=False,
            dead=False,
            completed=False,
            death_reason=None,
            capture_width=3,
            capture_height=2,
            capture_region=CaptureRegion(0, 0, 3, 2).to_dict(),
            window_title="Geometry Dash",
        ),
        _record(tick=2, frame_path="frames/missing.bmp"),
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    save_manifest_jsonl(records, manifest_path)

    summary = validate_frame_manifest(
        manifest_path,
        base_dir=tmp_path,
        require_change=False,
    )

    assert not summary.ok
    assert {issue.code for issue in summary.issues} == {
        "blank_frame",
        "width_mismatch",
        "missing_frame",
    }


def test_frame_manifest_validation_flags_static_hashes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames_dir = tmp_path / "frames"
    write_bmp(frames_dir / "first.bmp", _varied_frame())
    write_bmp(frames_dir / "second.bmp", _varied_frame())
    manifest_path = tmp_path / "manifest.jsonl"
    save_manifest_jsonl(
        [
            _record(tick=0, frame_path="frames/first.bmp"),
            _record(tick=1, frame_path="frames/second.bmp"),
        ],
        manifest_path,
    )

    summary = validate_frame_manifest(manifest_path, base_dir=tmp_path)

    assert not summary.ok
    assert "static_hashes" in {issue.code for issue in summary.issues}


def test_manifest_rejects_decreasing_ticks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest_path = tmp_path / "manifest.jsonl"
    save_manifest_jsonl(
        [
            _record(tick=2, frame_path="frames/late.bmp"),
            _record(tick=1, frame_path="frames/early.bmp"),
        ],
        manifest_path,
    )

    with pytest.raises(ManifestError, match="non-decreasing"):
        load_manifest_jsonl(manifest_path)


def _record(*, tick: int, frame_path: str) -> FrameCaptureRecord:
    return FrameCaptureRecord.from_observation(
        _make_observation(tick=tick),
        frame_path=frame_path,
        capture_width=2,
        capture_height=2,
        capture_region=CaptureRegion(0, 0, 2, 2),
        fps=240,
        window_title="Geometry Dash",
    )


def _make_observation(
    *,
    tick: int,
    percent: float = 0.0,
    input_down: bool = False,
) -> BridgeObservation:
    return BridgeObservation(
        tick=tick,
        x=1.0,
        y=2.0,
        y_vel=0.0,
        mode="cube",
        gravity="normal",
        percent=percent,
        dead=False,
        input_down=input_down,
    )


def _varied_frame() -> CapturedFrame:
    return _frame_from_rgb(
        [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
        ]
    )


def _other_varied_frame() -> CapturedFrame:
    return _frame_from_rgb(
        [
            (255, 255, 0),
            (0, 0, 255),
            (10, 20, 30),
            (200, 30, 90),
        ]
    )


def _solid_frame() -> CapturedFrame:
    return _frame_from_rgb([(0, 0, 0)] * 4)


def _frame_from_rgb(pixels: list[tuple[int, int, int]]) -> CapturedFrame:
    bgra = bytearray()
    for red, green, blue in pixels:
        bgra.extend((blue, green, red, 255))
    region = CaptureRegion(0, 0, 2, 2)
    return CapturedFrame(width=2, height=2, bgra=bytes(bgra), region=region)
