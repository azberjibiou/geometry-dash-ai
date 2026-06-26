"""Train the first tiny Phase 7 imitation-learning baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_imitation.baseline import (
    BaselineTrainingConfig,
    ImitationBaselineError,
    train_baseline_from_dataset_dir,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny PyTorch imitation baseline on prepared image-backed "
            "samples."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--frame-base-dir", type=Path)
    parser.add_argument(
        "--target",
        choices=("events", "target-input-down"),
        default="events",
        help="train press/release event labels or future input_down state labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to DATASET_PARENT/imitation_baseline_<timestamp>",
    )
    parser.add_argument("--image-width", type=int, default=84)
    parser.add_argument("--image-height", type=int, default=84)
    parser.add_argument("--frame-stack-size", type=int, default=4)
    parser.add_argument("--progress-scale", type=float, default=100.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument(
        "--ignore-split-json",
        action="store_true",
        help="create a split from CLI options instead of using dataset split.json",
    )
    parser.add_argument("--event-window-radius", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = BaselineTrainingConfig(
            target=_target_value(args.target),
            image_width=args.image_width,
            image_height=args.image_height,
            frame_stack_size=args.frame_stack_size,
            progress_scale=args.progress_scale,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            shuffle_split=args.shuffle_split,
            use_split_json=not args.ignore_split_json,
            event_window_radius=args.event_window_radius,
            device=args.device,
            save_checkpoint=args.save_checkpoint,
        )
    except ImitationBaselineError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or (
        args.dataset_dir.parent
        / f"imitation_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    try:
        result = train_baseline_from_dataset_dir(
            args.dataset_dir,
            output_dir,
            frame_base_dir=args.frame_base_dir,
            config=config,
        )
    except (ImitationBaselineError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metrics = result["metrics"]
    output = {
        "output_dir": str(output_dir),
        "metrics_json": str(result["metrics_path"]),
        "predictions_jsonl": str(result["predictions_path"]),
        "checkpoint": str(result["checkpoint_path"])
        if result["checkpoint_path"] is not None
        else None,
        "target": metrics["dataset"]["target"],
        "sample_count": metrics["dataset"]["sample_count"],
        "train_count": metrics["dataset"]["train_count"],
        "validation_count": metrics["dataset"]["validation_count"],
        "initial_loss": metrics["training"]["initial_loss"],
        "final_loss": metrics["training"]["final_loss"],
    }
    for optional_key in (
        "predicted_event_ticks",
        "top_event_ticks",
        "labeled_event_ticks",
        "predicted_state_ticks",
        "labeled_state_ticks",
    ):
        if optional_key in metrics:
            output[optional_key] = metrics[optional_key]
    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _target_value(raw: str) -> str:
    if raw == "target-input-down":
        return "target_input_down"
    return raw


if __name__ == "__main__":
    raise SystemExit(main())
