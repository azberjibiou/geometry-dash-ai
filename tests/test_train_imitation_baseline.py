import json

import pytest

from gd_capture import CapturedFrame, CaptureRegion, write_bmp
from gd_human_model import Event
from gd_imitation import (
    BaselineTrainingConfig,
    ImitationSample,
    binary_classification_metrics,
    load_or_create_split,
    load_prepared_image_dataset,
    train_imitation_baseline,
)
from gd_imitation.baseline import train_baseline_from_dataset_dir
from scripts.train_imitation_baseline import main


def test_binary_metrics_use_none_for_undefined_ratios() -> None:
    metrics = binary_classification_metrics(
        actual=[False, False],
        predicted=[False, False],
    )

    assert metrics["actual_positive_count"] == 0
    assert metrics["predicted_positive_count"] == 0
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None


def test_train_imitation_baseline_writes_metrics_and_predictions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    dataset_dir = _write_tiny_dataset(tmp_path)
    output_dir = tmp_path / "baseline"

    exit_code = main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--image-width",
            "1",
            "--image-height",
            "1",
            "--frame-stack-size",
            "1",
            "--hidden-size",
            "4",
            "--epochs",
            "20",
            "--learning-rate",
            "0.05",
            "--seed",
            "123",
            "--event-window-radius",
            "1",
        ]
    )

    assert exit_code == 0
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.jsonl"
    assert metrics_path.exists()
    assert predictions_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(predictions) == 8
    assert metrics["dataset"]["train_count"] == 6
    assert metrics["dataset"]["validation_count"] == 2
    assert metrics["dataset"]["press_label_count"] == 1
    assert metrics["dataset"]["release_label_count"] == 1
    assert metrics["validation"]["press"]["actual_positive_count"] == 0
    assert metrics["validation"]["press"]["recall"] is None
    assert metrics["training"]["final_loss"] <= metrics["training"]["initial_loss"]
    assert metrics["outputs"]["checkpoint"] is None
    assert set(metrics["top_event_ticks"]) == {"press", "release"}
    assert {row["split"] for row in predictions} == {"train", "validation"}


def test_training_is_deterministic_for_fixed_seed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    dataset_dir = _write_tiny_dataset(tmp_path)
    config = BaselineTrainingConfig(
        image_width=1,
        image_height=1,
        frame_stack_size=1,
        hidden_size=4,
        epochs=15,
        learning_rate=0.05,
        seed=77,
    )
    samples = load_prepared_image_dataset(
        dataset_dir,
        config=config_to_image_dataset(config),
    )
    split = load_or_create_split(dataset_dir, samples, config=config)

    first = train_imitation_baseline(
        samples,
        train_indices=split["train_indices"],
        validation_indices=split["validation_indices"],
        config=config,
    )
    second = train_imitation_baseline(
        samples,
        train_indices=split["train_indices"],
        validation_indices=split["validation_indices"],
        config=config,
    )

    assert first["metrics"]["training"]["final_loss"] == pytest.approx(
        second["metrics"]["training"]["final_loss"]
    )
    assert _flat_probabilities(first["predictions"]) == pytest.approx(
        _flat_probabilities(second["predictions"])
    )


def test_train_baseline_from_dataset_dir_can_save_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    dataset_dir = _write_tiny_dataset(tmp_path)
    output_dir = tmp_path / "with_checkpoint"

    result = train_baseline_from_dataset_dir(
        dataset_dir,
        output_dir,
        config=BaselineTrainingConfig(
            image_width=1,
            image_height=1,
            frame_stack_size=1,
            hidden_size=4,
            epochs=5,
            learning_rate=0.05,
            seed=1,
            save_checkpoint=True,
        ),
    )

    assert result["checkpoint_path"] == output_dir / "model.pt"
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions.jsonl").exists()


def config_to_image_dataset(config: BaselineTrainingConfig):  # type: ignore[no-untyped-def]
    from gd_imitation import ImageDatasetConfig

    return ImageDatasetConfig(
        image_width=config.image_width,
        image_height=config.image_height,
        required_frame_stack_size=config.frame_stack_size,
        progress_scale=config.progress_scale,
    )


def _write_tiny_dataset(root) -> object:  # type: ignore[no-untyped-def]
    capture_dir = root / "capture"
    frames_dir = capture_dir / "frames"
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()

    samples: list[ImitationSample] = []
    for index in range(8):
        color = (index * 10, index * 10, index * 10)
        if index == 2:
            color = (255, 0, 0)
        elif index == 5:
            color = (0, 255, 0)
        frame_path = frames_dir / f"frame_{index:06d}.bmp"
        _write_bmp_rows(frame_path, [[color]])
        samples.append(
            _sample(
                index=index,
                frame_path=f"frames/frame_{index:06d}.bmp",
                press_event=index == 2,
                release_event=index == 5,
            )
        )

    with (dataset_dir / "samples.jsonl").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for sample in samples:
            file.write(json.dumps(sample.to_dict(), separators=(",", ":"), sort_keys=True))
            file.write("\n")

    (dataset_dir / "split.json").write_text(
        json.dumps(
            {
                "train_indices": [0, 1, 2, 3, 4, 5],
                "validation_indices": [6, 7],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (dataset_dir / "summary.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "manifest_jsonl": str(capture_dir / "manifest.jsonl"),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _sample(
    *,
    index: int,
    frame_path: str,
    press_event: bool,
    release_event: bool,
) -> ImitationSample:
    events = []
    if press_event:
        events.append(Event(index, "press"))
    if release_event:
        events.append(Event(index, "release"))
    return ImitationSample(
        index=index,
        tick=index,
        label_tick=index,
        frame_ticks=(index,),
        frame_paths=(frame_path,),
        progress=float(index),
        input_down=False,
        x=float(index),
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
        events=tuple(events),
    )


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


def _flat_probabilities(predictions) -> list[float]:  # type: ignore[no-untyped-def]
    values: list[float] = []
    for row in predictions:
        values.append(row["probabilities"]["press_event"])
        values.append(row["probabilities"]["release_event"])
    return values
