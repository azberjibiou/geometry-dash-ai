import json

import pytest

from gd_capture import CapturedFrame, CaptureRegion, write_bmp
from gd_human_model import Event
from gd_imitation import (
    GrayscaleFrame,
    ImageDatasetConfig,
    ImageDatasetError,
    ImitationSample,
    load_image_dataset,
    load_prepared_image_dataset,
    read_bmp_grayscale,
    read_bmp_grayscale_resized,
    resize_grayscale_frame,
)


def test_read_bmp_grayscale_returns_top_down_normalized_pixels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "frame.bmp"
    _write_bmp_rows(
        path,
        [
            [(0, 0, 0), (255, 255, 255)],
            [(255, 0, 0), (0, 255, 0)],
        ],
    )

    frame = read_bmp_grayscale(path)

    assert frame.width == 2
    assert frame.height == 2
    assert frame.pixels[0][0] == pytest.approx(0.0)
    assert frame.pixels[0][1] == pytest.approx(1.0)
    assert frame.pixels[1][0] == pytest.approx(0.299)
    assert frame.pixels[1][1] == pytest.approx(0.587)


def test_resize_grayscale_frame_uses_nearest_neighbor() -> None:
    frame = GrayscaleFrame(
        width=2,
        height=2,
        pixels=((0.0, 0.5), (0.75, 1.0)),
    )

    resized = resize_grayscale_frame(frame, width=4, height=4)

    assert resized.width == 4
    assert resized.height == 4
    assert resized.pixels[0] == pytest.approx((0.0, 0.0, 0.5, 0.5))
    assert resized.pixels[2] == pytest.approx((0.75, 0.75, 1.0, 1.0))


def test_read_bmp_grayscale_resized_matches_resize_result(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "frame.bmp"
    _write_bmp_rows(
        path,
        [
            [(0, 0, 0), (255, 255, 255)],
            [(255, 0, 0), (0, 255, 0)],
        ],
    )

    resized_directly = read_bmp_grayscale_resized(path, width=4, height=4)
    resized_after_read = resize_grayscale_frame(
        read_bmp_grayscale(path),
        width=4,
        height=4,
    )

    assert resized_directly == resized_after_read


def test_load_image_dataset_reads_frame_stacks_features_and_labels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture_dir = tmp_path / "capture"
    frames_dir = capture_dir / "frames"
    _write_bmp_rows(frames_dir / "first.bmp", [[(0, 0, 0), (255, 255, 255)]])
    _write_bmp_rows(frames_dir / "second.bmp", [[(255, 0, 0), (0, 255, 0)]])
    samples_path = tmp_path / "samples.jsonl"
    _write_samples(
        samples_path,
        [
            _sample(
                frame_paths=("frames/first.bmp", "frames/second.bmp"),
                progress=25.0,
                input_down=True,
                press_event=True,
                release_event=False,
            )
        ],
    )

    loaded = load_image_dataset(
        samples_path,
        frame_base_dir=capture_dir,
        config=ImageDatasetConfig(image_width=2, image_height=1),
    )

    assert len(loaded) == 1
    assert loaded[0].frame_shape == (2, 1, 2)
    assert loaded[0].scalar_features == pytest.approx((0.25, 1.0))
    assert loaded[0].labels == pytest.approx((1.0, 0.0))
    assert loaded[0].frame_stack[0].pixels[0] == pytest.approx((0.0, 1.0))
    assert loaded[0].frame_stack[1].pixels[0] == pytest.approx((0.299, 0.587))


def test_load_image_dataset_can_use_target_input_down_labels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture_dir = tmp_path / "capture"
    frames_dir = capture_dir / "frames"
    _write_bmp_rows(frames_dir / "frame.bmp", [[(0, 0, 0)]])
    samples_path = tmp_path / "samples.jsonl"
    _write_samples(
        samples_path,
        [
            _sample(
                frame_paths=("frames/frame.bmp",),
                input_down=False,
                target_input_down=True,
            )
        ],
    )

    loaded = load_image_dataset(
        samples_path,
        frame_base_dir=capture_dir,
        config=ImageDatasetConfig(
            image_width=1,
            image_height=1,
            label_mode="target_input_down",
        ),
    )

    assert loaded[0].scalar_features == pytest.approx((0.0, 0.0))
    assert loaded[0].labels == pytest.approx((1.0,))


def test_load_image_dataset_can_pad_short_frame_stacks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture_dir = tmp_path / "capture"
    frames_dir = capture_dir / "frames"
    _write_bmp_rows(frames_dir / "only.bmp", [[(255, 255, 255)]])
    samples_path = tmp_path / "samples.jsonl"
    _write_samples(samples_path, [_sample(frame_paths=("frames/only.bmp",))])

    loaded = load_image_dataset(
        samples_path,
        frame_base_dir=capture_dir,
        config=ImageDatasetConfig(
            image_width=1,
            image_height=1,
            required_frame_stack_size=4,
        ),
    )

    assert loaded[0].frame_shape == (4, 1, 1)
    assert [frame.pixels[0][0] for frame in loaded[0].frame_stack] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )


def test_load_prepared_image_dataset_uses_summary_manifest_parent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture_dir = tmp_path / "capture"
    frames_dir = capture_dir / "frames"
    _write_bmp_rows(frames_dir / "frame.bmp", [[(255, 255, 255)]])
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_samples(
        dataset_dir / "samples.jsonl",
        [_sample(frame_paths=("frames/frame.bmp",))],
    )
    (dataset_dir / "summary.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "manifest_jsonl": str(capture_dir / "manifest.jsonl"),
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_prepared_image_dataset(
        dataset_dir,
        config=ImageDatasetConfig(image_width=1, image_height=1),
    )

    assert loaded[0].frame_shape == (1, 1, 1)
    assert loaded[0].frame_stack[0].pixels[0][0] == pytest.approx(1.0)


def test_read_bmp_grayscale_rejects_invalid_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.bmp"
    path.write_bytes(b"not a bmp")

    with pytest.raises(ImageDatasetError, match="BMP"):
        read_bmp_grayscale(path)


def _sample(
    *,
    frame_paths: tuple[str, ...],
    progress: float = 0.0,
    input_down: bool = False,
    target_input_down: bool | None = None,
    press_event: bool = False,
    release_event: bool = False,
) -> ImitationSample:
    return ImitationSample(
        index=0,
        tick=10,
        label_tick=10,
        frame_ticks=tuple(range(len(frame_paths))),
        frame_paths=frame_paths,
        progress=progress,
        input_down=input_down,
        target_input_down=input_down
        if target_input_down is None
        else target_input_down,
        x=10.0,
        y=0.0,
        x_vel=1.0,
        y_vel=0.0,
        rotation=0.0,
        mode="cube",
        gravity="normal",
        dead=False,
        death_reason=None,
        fps=240,
        cbf=False,
        physics_bypass=False,
        press_event=press_event,
        release_event=release_event,
        events=(Event(10, "press"),) if press_event else (),
    )


def _write_samples(path, samples: list[ImitationSample]) -> None:  # type: ignore[no-untyped-def]
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for index, sample in enumerate(samples):
            data = sample.to_dict()
            data["index"] = index
            file.write(json.dumps(data, separators=(",", ":"), sort_keys=True))
            file.write("\n")


def _write_bmp_rows(path, rows: list[list[tuple[int, int, int]]]) -> None:  # type: ignore[no-untyped-def]
    height = len(rows)
    width = len(rows[0])
    bgra = bytearray()
    for row in reversed(rows):
        for red, green, blue in row:
            bgra.extend((blue, green, red, 255))
    write_bmp(
        path,
        CapturedFrame(
            width=width,
            height=height,
            bgra=bytes(bgra),
            region=CaptureRegion(0, 0, width, height),
        ),
    )
