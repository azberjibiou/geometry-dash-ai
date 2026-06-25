"""Tiny Phase 7 imitation-learning baseline."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from gd_imitation.dataset import split_imitation_samples
from gd_imitation.image_dataset import (
    ImageDatasetConfig,
    ImageInputSample,
    load_prepared_image_dataset,
)


class ImitationBaselineError(ValueError):
    """Raised when the tiny imitation baseline cannot be trained."""


@dataclass(frozen=True, slots=True)
class BaselineTrainingConfig:
    """Configuration for the first tiny image-backed imitation baseline."""

    image_width: int = 84
    image_height: int = 84
    frame_stack_size: int = 4
    progress_scale: float = 100.0
    hidden_size: int = 64
    epochs: int = 300
    learning_rate: float = 0.03
    weight_decay: float = 0.0
    threshold: float = 0.5
    seed: int = 0
    validation_fraction: float = 0.2
    shuffle_split: bool = False
    use_split_json: bool = True
    event_window_radius: int = 5
    device: str = "cpu"
    save_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.image_width <= 0:
            raise ImitationBaselineError("image_width must be positive")
        if self.image_height <= 0:
            raise ImitationBaselineError("image_height must be positive")
        if self.frame_stack_size <= 0:
            raise ImitationBaselineError("frame_stack_size must be positive")
        if self.progress_scale <= 0.0:
            raise ImitationBaselineError("progress_scale must be positive")
        if self.hidden_size <= 0:
            raise ImitationBaselineError("hidden_size must be positive")
        if self.epochs <= 0:
            raise ImitationBaselineError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ImitationBaselineError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ImitationBaselineError("weight_decay must be non-negative")
        if not 0.0 <= self.threshold <= 1.0:
            raise ImitationBaselineError("threshold must be between 0 and 1")
        if not 0.0 <= self.validation_fraction <= 1.0:
            raise ImitationBaselineError(
                "validation_fraction must be between 0 and 1"
            )
        if self.event_window_radius < 0:
            raise ImitationBaselineError("event_window_radius must be non-negative")


def train_baseline_from_dataset_dir(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    frame_base_dir: str | Path | None = None,
    config: BaselineTrainingConfig | None = None,
) -> dict[str, Any]:
    """Load prepared samples, train the tiny baseline, and write artifacts."""

    effective_config = config or BaselineTrainingConfig()
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    samples = load_prepared_image_dataset(
        dataset_path,
        frame_base_dir=frame_base_dir,
        config=ImageDatasetConfig(
            image_width=effective_config.image_width,
            image_height=effective_config.image_height,
            required_frame_stack_size=effective_config.frame_stack_size,
            progress_scale=effective_config.progress_scale,
        ),
    )
    split = load_or_create_split(
        dataset_path,
        samples,
        config=effective_config,
    )
    result = train_imitation_baseline(
        samples,
        train_indices=split["train_indices"],
        validation_indices=split["validation_indices"],
        config=effective_config,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "metrics.json"
    predictions_path = output_path / "predictions.jsonl"
    checkpoint_path = output_path / "model.pt"

    metrics = dict(result["metrics"])
    metrics["outputs"] = {
        "metrics_json": str(metrics_path),
        "predictions_jsonl": str(predictions_path),
        "checkpoint": str(checkpoint_path)
        if effective_config.save_checkpoint
        else None,
    }
    _write_json(metrics, metrics_path)
    _write_predictions_jsonl(result["predictions"], predictions_path)
    if effective_config.save_checkpoint:
        torch = _import_torch()
        torch.save(result["checkpoint"], checkpoint_path)

    return {
        "metrics": metrics,
        "predictions": result["predictions"],
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "checkpoint_path": checkpoint_path
        if effective_config.save_checkpoint
        else None,
    }


def predict_baseline_from_checkpoint(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    frame_base_dir: str | Path | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run a saved tiny baseline checkpoint over a prepared dataset."""

    torch = _import_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ImitationBaselineError("checkpoint must contain an object")

    config_data = checkpoint.get("config")
    if not isinstance(config_data, Mapping):
        raise ImitationBaselineError("checkpoint is missing config")
    try:
        config = BaselineTrainingConfig(**dict(config_data))
    except TypeError as exc:
        raise ImitationBaselineError(f"checkpoint config is invalid: {exc}") from exc
    config = replace(config, device=device)

    samples = load_prepared_image_dataset(
        dataset_dir,
        frame_base_dir=frame_base_dir,
        config=ImageDatasetConfig(
            image_width=config.image_width,
            image_height=config.image_height,
            required_frame_stack_size=config.frame_stack_size,
            progress_scale=config.progress_scale,
        ),
    )
    if not samples:
        raise ImitationBaselineError("dataset contains no samples")

    features = [_feature_vector(sample) for sample in samples]
    input_dim = _validate_feature_vectors(features)
    expected_input_dim = checkpoint.get("input_dim")
    if expected_input_dim != input_dim:
        raise ImitationBaselineError(
            f"checkpoint input_dim {expected_input_dim} does not match "
            f"dataset input_dim {input_dim}"
        )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ImitationBaselineError("checkpoint is missing model_state_dict")

    split = load_or_create_split(dataset_dir, samples, config=config)
    model = _build_model(
        torch,
        input_dim=input_dim,
        hidden_size=config.hidden_size,
    ).to(torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
        device=torch.device(device),
    )
    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.sigmoid(logits)

    return _prediction_rows(
        samples,
        logits=logits.detach().cpu().tolist(),
        probabilities=probabilities.detach().cpu().tolist(),
        train_indices=set(split["train_indices"]),
        validation_indices=set(split["validation_indices"]),
        threshold=config.threshold,
    )


def load_or_create_split(
    dataset_dir: str | Path,
    samples: Sequence[ImageInputSample],
    *,
    config: BaselineTrainingConfig,
) -> dict[str, tuple[int, ...]]:
    """Load ``split.json`` when present or create a deterministic fallback."""

    sample_count = len(samples)
    if sample_count == 0:
        raise ImitationBaselineError("dataset contains no samples")

    split_path = Path(dataset_dir) / "split.json"
    if config.use_split_json and split_path.exists():
        with split_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, Mapping):
            raise ImitationBaselineError("split.json must contain an object")
        train_indices = _read_index_tuple(data, "train_indices", sample_count)
        validation_indices = _read_index_tuple(
            data,
            "validation_indices",
            sample_count,
        )
    else:
        split = split_imitation_samples(
            [sample.sample for sample in samples],
            validation_fraction=config.validation_fraction,
            shuffle=config.shuffle_split,
            seed=config.seed,
        )
        train_indices = split.train_indices
        validation_indices = split.validation_indices

    _validate_split_indices(train_indices, validation_indices, sample_count)
    if not train_indices:
        raise ImitationBaselineError("training split contains no samples")
    return {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }


def train_imitation_baseline(
    samples: Sequence[ImageInputSample],
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    config: BaselineTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train a tiny PyTorch MLP over flattened frame stacks plus scalars."""

    effective_config = config or BaselineTrainingConfig()
    torch = _import_torch()
    _set_seed(torch, effective_config.seed)

    sample_list = list(samples)
    if not sample_list:
        raise ImitationBaselineError("dataset contains no samples")
    train_index_tuple = tuple(train_indices)
    validation_index_tuple = tuple(validation_indices)
    _validate_split_indices(
        train_index_tuple,
        validation_index_tuple,
        len(sample_list),
    )
    if not train_index_tuple:
        raise ImitationBaselineError("training split contains no samples")

    features = [_feature_vector(sample) for sample in sample_list]
    input_dim = _validate_feature_vectors(features)
    labels = [list(sample.labels) for sample in sample_list]
    device = torch.device(effective_config.device)

    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    train_tensor = torch.tensor(
        train_index_tuple,
        dtype=torch.long,
        device=device,
    )
    validation_tensor = torch.tensor(
        validation_index_tuple,
        dtype=torch.long,
        device=device,
    )

    model = _build_model(
        torch,
        input_dim=input_dim,
        hidden_size=effective_config.hidden_size,
    ).to(device)
    pos_weight = _positive_weights(torch, label_tensor[train_tensor])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=effective_config.learning_rate,
        weight_decay=effective_config.weight_decay,
    )

    train_loss_history: list[float] = []
    for _epoch in range(effective_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(feature_tensor[train_tensor])
        loss = criterion(logits, label_tensor[train_tensor])
        loss.backward()
        optimizer.step()
        train_loss_history.append(float(loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        all_logits = model(feature_tensor)
        all_probabilities = torch.sigmoid(all_logits)
        train_loss = _subset_loss(
            torch,
            criterion,
            all_logits,
            label_tensor,
            train_tensor,
        )
        validation_loss = _subset_loss(
            torch,
            criterion,
            all_logits,
            label_tensor,
            validation_tensor,
        )

    predictions = _prediction_rows(
        sample_list,
        logits=all_logits.detach().cpu().tolist(),
        probabilities=all_probabilities.detach().cpu().tolist(),
        train_indices=set(train_index_tuple),
        validation_indices=set(validation_index_tuple),
        threshold=effective_config.threshold,
    )
    metrics = _metrics_report(
        samples=sample_list,
        predictions=predictions,
        train_indices=train_index_tuple,
        validation_indices=validation_index_tuple,
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_loss_history=train_loss_history,
        input_dim=input_dim,
        config=effective_config,
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "checkpoint": {
            "backend": "torch",
            "config": asdict(effective_config),
            "input_dim": input_dim,
            "model_state_dict": model.state_dict(),
        },
    }


def binary_classification_metrics(
    *,
    actual: Sequence[bool],
    predicted: Sequence[bool],
) -> dict[str, Any]:
    """Return JSON-safe binary metrics with undefined ratios as ``None``."""

    if len(actual) != len(predicted):
        raise ImitationBaselineError("actual and predicted lengths differ")
    true_positive = sum(1 for a, p in zip(actual, predicted) if a and p)
    false_positive = sum(1 for a, p in zip(actual, predicted) if not a and p)
    true_negative = sum(1 for a, p in zip(actual, predicted) if not a and not p)
    false_negative = sum(1 for a, p in zip(actual, predicted) if a and not p)
    actual_positive_count = true_positive + false_negative
    predicted_positive_count = true_positive + false_positive
    total = len(actual)

    precision = (
        true_positive / predicted_positive_count
        if predicted_positive_count
        else None
    )
    recall = (
        true_positive / actual_positive_count
        if actual_positive_count
        else None
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )

    return {
        "count": total,
        "actual_positive_count": actual_positive_count,
        "predicted_positive_count": predicted_positive_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _build_model(torch: Any, *, input_dim: int, hidden_size: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 2),
    )


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImitationBaselineError(
            "PyTorch is required for the neural imitation baseline. "
            "Install torch or use a dependency-free baseline."
        ) from exc
    return torch


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _feature_vector(sample: ImageInputSample) -> list[float]:
    values: list[float] = []
    for frame in sample.frame_stack:
        for row in frame.pixels:
            values.extend(row)
    values.extend(sample.scalar_features)
    return values


def _validate_feature_vectors(features: Sequence[Sequence[float]]) -> int:
    if not features:
        raise ImitationBaselineError("dataset contains no samples")
    input_dim = len(features[0])
    if input_dim == 0:
        raise ImitationBaselineError("feature vectors are empty")
    for index, feature in enumerate(features):
        if len(feature) != input_dim:
            raise ImitationBaselineError(
                f"sample {index} has feature length {len(feature)}; "
                f"expected {input_dim}"
            )
    return input_dim


def _positive_weights(torch: Any, train_labels: Any) -> Any:
    positive_counts = train_labels.sum(dim=0)
    negative_counts = train_labels.shape[0] - positive_counts
    ones = torch.ones_like(positive_counts)
    return torch.where(positive_counts > 0, negative_counts / positive_counts, ones)


def _subset_loss(
    torch: Any,
    criterion: Any,
    logits: Any,
    labels: Any,
    indices: Any,
) -> float | None:
    if indices.numel() == 0:
        return None
    with torch.no_grad():
        return float(criterion(logits[indices], labels[indices]).cpu().item())


def _prediction_rows(
    samples: Sequence[ImageInputSample],
    *,
    logits: Sequence[Sequence[float]],
    probabilities: Sequence[Sequence[float]],
    train_indices: set[int],
    validation_indices: set[int],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, (sample, sample_logits, sample_probs) in enumerate(
        zip(samples, logits, probabilities)
    ):
        split = (
            "train"
            if position in train_indices
            else "validation"
            if position in validation_indices
            else "unused"
        )
        press_probability = float(sample_probs[0])
        release_probability = float(sample_probs[1])
        rows.append(
            {
                "position": position,
                "index": sample.sample.index,
                "tick": sample.sample.tick,
                "label_tick": sample.sample.label_tick,
                "split": split,
                "progress": sample.sample.progress,
                "input_down": sample.sample.input_down,
                "labels": {
                    "press_event": bool(sample.labels[0]),
                    "release_event": bool(sample.labels[1]),
                },
                "logits": {
                    "press_event": float(sample_logits[0]),
                    "release_event": float(sample_logits[1]),
                },
                "probabilities": {
                    "press_event": press_probability,
                    "release_event": release_probability,
                },
                "predicted": {
                    "press_event": press_probability >= threshold,
                    "release_event": release_probability >= threshold,
                },
            }
        )
    return rows


def _metrics_report(
    *,
    samples: Sequence[ImageInputSample],
    predictions: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    train_loss: float | None,
    validation_loss: float | None,
    train_loss_history: Sequence[float],
    input_dim: int,
    config: BaselineTrainingConfig,
) -> dict[str, Any]:
    train_index_set = set(train_indices)
    validation_index_set = set(validation_indices)
    all_indices = tuple(range(len(samples)))
    return {
        "backend": "torch",
        "config": asdict(config),
        "dataset": {
            "sample_count": len(samples),
            "train_count": len(train_indices),
            "validation_count": len(validation_indices),
            "input_dim": input_dim,
            "press_label_count": sum(1 for sample in samples if sample.labels[0]),
            "release_label_count": sum(1 for sample in samples if sample.labels[1]),
            "first_tick": samples[0].sample.tick if samples else None,
            "last_tick": samples[-1].sample.tick if samples else None,
        },
        "training": {
            "initial_loss": train_loss_history[0] if train_loss_history else None,
            "final_loss": train_loss_history[-1] if train_loss_history else None,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        "train": _subset_metrics(predictions, train_index_set),
        "validation": _subset_metrics(predictions, validation_index_set),
        "all": _subset_metrics(predictions, set(all_indices)),
        "predicted_event_ticks": {
            "press": [
                int(row["tick"])
                for row in predictions
                if row["predicted"]["press_event"]
            ],
            "release": [
                int(row["tick"])
                for row in predictions
                if row["predicted"]["release_event"]
            ],
        },
        "top_event_ticks": {
            "press": _top_event_ticks(predictions, "press_event"),
            "release": _top_event_ticks(predictions, "release_event"),
        },
        "labeled_event_ticks": {
            "press": [
                sample.sample.tick for sample in samples if sample.labels[0]
            ],
            "release": [
                sample.sample.tick for sample in samples if sample.labels[1]
            ],
        },
        "event_neighborhoods": _event_neighborhoods(
            predictions,
            radius=config.event_window_radius,
        ),
    }


def _subset_metrics(
    predictions: Sequence[Mapping[str, Any]],
    positions: set[int],
) -> dict[str, Any]:
    rows = [row for row in predictions if int(row["position"]) in positions]
    return {
        "count": len(rows),
        "press": binary_classification_metrics(
            actual=[bool(row["labels"]["press_event"]) for row in rows],
            predicted=[bool(row["predicted"]["press_event"]) for row in rows],
        ),
        "release": binary_classification_metrics(
            actual=[bool(row["labels"]["release_event"]) for row in rows],
            predicted=[bool(row["predicted"]["release_event"]) for row in rows],
        ),
    }


def _event_neighborhoods(
    predictions: Sequence[Mapping[str, Any]],
    *,
    radius: int,
) -> list[dict[str, Any]]:
    by_tick = {int(row["tick"]): row for row in predictions}
    neighborhoods: list[dict[str, Any]] = []
    for event_name in ("press_event", "release_event"):
        label_name = "press" if event_name == "press_event" else "release"
        event_ticks = [
            int(row["tick"])
            for row in predictions
            if row["labels"][event_name]
        ]
        for tick in event_ticks:
            rows: list[dict[str, Any]] = []
            for nearby_tick in range(tick - radius, tick + radius + 1):
                row = by_tick.get(nearby_tick)
                if row is None:
                    continue
                rows.append(
                    {
                        "tick": int(row["tick"]),
                        "label": bool(row["labels"][event_name]),
                        "probability": float(row["probabilities"][event_name]),
                        "predicted": bool(row["predicted"][event_name]),
                    }
                )
            neighborhoods.append(
                {
                    "event": label_name,
                    "tick": tick,
                    "rows": rows,
                }
            )
    return neighborhoods


def _top_event_ticks(
    predictions: Sequence[Mapping[str, Any]],
    event_name: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        predictions,
        key=lambda row: float(row["probabilities"][event_name]),
        reverse=True,
    )
    return [
        {
            "tick": int(row["tick"]),
            "probability": float(row["probabilities"][event_name]),
            "label": bool(row["labels"][event_name]),
            "predicted": bool(row["predicted"][event_name]),
        }
        for row in ranked[:limit]
    ]


def _read_index_tuple(
    data: Mapping[str, Any],
    key: str,
    sample_count: int,
) -> tuple[int, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ImitationBaselineError(f"split.json {key} must be a list")
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ImitationBaselineError(f"split.json {key} must contain ints")
        if item < 0 or item >= sample_count:
            raise ImitationBaselineError(
                f"split.json {key} index {item} is out of range"
            )
        indices.append(item)
    return tuple(indices)


def _validate_split_indices(
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    sample_count: int,
) -> None:
    seen: set[int] = set()
    for split_name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
    ):
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ImitationBaselineError(f"{split_name} index must be an int")
            if index < 0 or index >= sample_count:
                raise ImitationBaselineError(
                    f"{split_name} index {index} is out of range"
                )
            if index in seen:
                raise ImitationBaselineError(
                    f"sample index {index} appears in multiple splits"
                )
            seen.add(index)


def _write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_predictions_jsonl(
    predictions: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in predictions:
            file.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            file.write("\n")
