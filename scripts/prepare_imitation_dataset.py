"""Prepare Phase 7 imitation-learning samples from captured gameplay artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gd_imitation import (
    DatasetConfig,
    ImitationSample,
    load_imitation_samples,
    split_imitation_samples,
)
from gd_trace import Macro, load_macro_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a frame manifest, same-run trace, and macro into aligned "
            "supervised imitation-learning samples."
        )
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--macro-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to MANIFEST_PARENT/imitation_dataset",
    )
    parser.add_argument("--frame-stack-size", type=int, default=4)
    parser.add_argument("--label-shift-frames", type=int, default=0)
    parser.add_argument("--event-label-radius-frames", type=int, default=0)
    parser.add_argument("--player", choices=("p1", "p2"), default="p1")
    parser.add_argument("--include-partial-windows", action="store_true")
    parser.add_argument(
        "--strict-label-horizon",
        action="store_true",
        help="fail when a shifted label tick is beyond the trace instead of dropping it",
    )
    parser.add_argument(
        "--drop-unmatched-frame-ticks",
        action="store_true",
        help="drop frame rows whose ticks are absent from the trace",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
        config = DatasetConfig(
            frame_stack_size=args.frame_stack_size,
            label_shift_frames=args.label_shift_frames,
            event_label_radius_frames=args.event_label_radius_frames,
            player=args.player,
            include_partial_windows=args.include_partial_windows,
            drop_incomplete_label_horizon=not args.strict_label_horizon,
            drop_unmatched_frame_ticks=args.drop_unmatched_frame_ticks,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or args.manifest_jsonl.parent / "imitation_dataset"
    samples_path = output_dir / "samples.jsonl"
    split_path = output_dir / "split.json"
    summary_path = output_dir / "summary.json"

    try:
        macro = load_macro_json(args.macro_json)
        samples = load_imitation_samples(
            args.manifest_jsonl,
            args.trace_jsonl,
            args.macro_json,
            config=config,
        )
        if not samples:
            raise ValueError("dataset contains no samples")
        split = split_imitation_samples(
            samples,
            validation_fraction=args.validation_fraction,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_samples_jsonl(samples, samples_path)
    _write_json(split.to_dict(), split_path)

    summary = {
        "inputs": {
            "manifest_jsonl": str(args.manifest_jsonl),
            "trace_jsonl": str(args.trace_jsonl),
            "macro_json": str(args.macro_json),
        },
        "outputs": {
            "samples_jsonl": str(samples_path),
            "split_json": str(split_path),
            "summary_json": str(summary_path),
        },
        "config": asdict(config),
        "split": split.to_dict(),
        "summary": _summarize_samples(samples, macro),
    }
    _write_json(summary, summary_path)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "samples_jsonl": str(samples_path),
                "split_json": str(split_path),
                "summary_json": str(summary_path),
                "sample_count": len(samples),
                "train_count": len(split.train),
                "validation_count": len(split.validation),
                "press_label_count": sum(1 for sample in samples if sample.press_event),
                "release_label_count": sum(
                    1 for sample in samples if sample.release_event
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.frame_stack_size <= 0:
        raise ValueError("--frame-stack-size must be positive")
    if args.label_shift_frames < 0:
        raise ValueError("--label-shift-frames must be non-negative")
    if args.event_label_radius_frames < 0:
        raise ValueError("--event-label-radius-frames must be non-negative")
    if not 0.0 <= args.validation_fraction <= 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")


def _write_samples_jsonl(samples: Sequence[ImitationSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps(sample.to_dict(), separators=(",", ":"), sort_keys=True))
            file.write("\n")


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _summarize_samples(
    samples: Sequence[ImitationSample],
    macro: Macro,
) -> dict[str, Any]:
    sample_ticks = [sample.tick for sample in samples]
    label_ticks = [sample.label_tick for sample in samples]
    press_label_ticks = [sample.tick for sample in samples if sample.press_event]
    release_label_ticks = [sample.tick for sample in samples if sample.release_event]
    frame_stack_sizes = sorted({len(sample.frame_paths) for sample in samples})

    return {
        "sample_count": len(samples),
        "first_tick": sample_ticks[0],
        "last_tick": sample_ticks[-1],
        "first_label_tick": label_ticks[0],
        "last_label_tick": label_ticks[-1],
        "frame_stack_sizes": frame_stack_sizes,
        "sample_tick_step_values": _step_values(sample_ticks),
        "label_tick_step_values": _step_values(label_ticks),
        "macro_event_count": len(macro.events),
        "press_label_count": len(press_label_ticks),
        "release_label_count": len(release_label_ticks),
        "press_label_ticks": press_label_ticks,
        "release_label_ticks": release_label_ticks,
        "positive_label_count": sum(
            1 for sample in samples if sample.press_event or sample.release_event
        ),
        "input_down_count": sum(1 for sample in samples if sample.input_down),
        "target_input_down_count": sum(
            1 for sample in samples if sample.target_input_down
        ),
        "dead_count": sum(1 for sample in samples if sample.dead),
        "min_progress": min(sample.progress for sample in samples),
        "max_progress": max(sample.progress for sample in samples),
    }


def _step_values(values: Sequence[int]) -> list[int]:
    return sorted({right - left for left, right in zip(values, values[1:])})


if __name__ == "__main__":
    raise SystemExit(main())
