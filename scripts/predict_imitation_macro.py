"""Decode imitation baseline predictions into a playable macro."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_imitation import (  # noqa: E402
    DecoderConfig,
    EventDecoderError,
    decode_predictions,
    event_level_metrics,
    load_prediction_jsonl,
    predict_baseline_from_checkpoint,
)
from gd_trace import Macro, save_macro_json  # noqa: E402
from gd_trace.macro_schema import event_to_dict  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Turn per-tick imitation policy probabilities into a schema-valid "
            "Geometry Dash macro."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--predictions-jsonl",
        type=Path,
        help="baseline predictions.jsonl to decode",
    )
    source_group.add_argument(
        "--checkpoint",
        type=Path,
        help="saved tiny baseline checkpoint to run before decoding",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="prepared dataset directory; required with --checkpoint",
    )
    parser.add_argument("--frame-base-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--press-threshold", type=float, default=0.5)
    parser.add_argument("--release-threshold", type=float, default=0.5)
    parser.add_argument("--non-max-radius-frames", type=int, default=4)
    parser.add_argument("--min-event-spacing-frames", type=int, default=2)
    parser.add_argument("--initial-input-down", action="store_true")
    parser.add_argument("--player", choices=("p1", "p2"), default="p1")
    parser.add_argument("--match-tolerance-frames", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for --checkpoint inference",
    )
    args = parser.parse_args(argv)

    if args.checkpoint is not None and args.dataset_dir is None:
        parser.error("--dataset-dir is required with --checkpoint")

    try:
        config = DecoderConfig(
            press_threshold=args.press_threshold,
            release_threshold=args.release_threshold,
            non_max_radius_frames=args.non_max_radius_frames,
            min_event_spacing_frames=args.min_event_spacing_frames,
            initial_input_down=args.initial_input_down,
            player=args.player,
        )
        predictions = _load_or_predict_rows(args)
        events = decode_predictions(predictions, config=config)
        metrics = event_level_metrics(
            predictions,
            events,
            match_tolerance_frames=args.match_tolerance_frames,
            top_k=args.top_k,
        )
    except (EventDecoderError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or _default_output_dir(args)
    macro_path = output_dir / "predicted_macro.json"
    summary_path = output_dir / "prediction_summary.json"
    decoded_events_path = output_dir / "decoded_events.jsonl"

    macro = Macro(
        events=events,
        metadata={
            "generated_by": "scripts/predict_imitation_macro.py",
            "decoder_config": config.to_dict(),
            "source": _source_metadata(args),
        },
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_macro_json(macro, macro_path)
        _write_decoded_events_jsonl(events, decoded_events_path)
        _write_json(
            {
                "inputs": _source_metadata(args),
                "outputs": {
                    "predicted_macro_json": str(macro_path),
                    "prediction_summary_json": str(summary_path),
                    "decoded_events_jsonl": str(decoded_events_path),
                },
                "decoder_config": config.to_dict(),
                "prediction_row_count": len(predictions),
                "decoded_event_count": len(events),
                "decoded_events": [event_to_dict(event) for event in events],
                "event_metrics": metrics,
            },
            summary_path,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "predicted_macro_json": str(macro_path),
                "prediction_summary_json": str(summary_path),
                "decoded_events_jsonl": str(decoded_events_path),
                "decoded_event_count": len(events),
                "decoded_events": [event_to_dict(event) for event in events],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_or_predict_rows(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    if args.predictions_jsonl is not None:
        return load_prediction_jsonl(args.predictions_jsonl)
    return predict_baseline_from_checkpoint(
        args.dataset_dir,
        args.checkpoint,
        frame_base_dir=args.frame_base_dir,
        device=args.device,
    )


def _default_output_dir(args: argparse.Namespace) -> Path:
    if args.predictions_jsonl is not None:
        return args.predictions_jsonl.parent / "predicted_macro"
    return args.checkpoint.parent / "predicted_macro"


def _source_metadata(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "predictions_jsonl": str(args.predictions_jsonl)
        if args.predictions_jsonl is not None
        else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
        "frame_base_dir": str(args.frame_base_dir)
        if args.frame_base_dir is not None
        else None,
    }


def _write_decoded_events_jsonl(events: Sequence[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for event in events:
            file.write(
                json.dumps(
                    event_to_dict(event),
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            file.write("\n")


def _write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
