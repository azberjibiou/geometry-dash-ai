"""Capture visible Geometry Dash frames aligned to Geode bridge observations."""

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

from gd_capture import (
    CaptureRegion,
    CaptureSource,
    FrameCaptureRecord,
    list_visible_windows,
    validate_frame_manifest,
    write_bmp,
)
from gd_env import GeometryDashClient


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture visible Geometry Dash frames and align them with Geode "
            "bridge observations."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29430)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="capture one frame every N bridge observations",
    )
    parser.add_argument(
        "--window-title",
        default="Geometry Dash",
        help="visible window title substring to capture when --region is omitted",
    )
    parser.add_argument(
        "--region",
        type=_parse_region,
        help="absolute screen capture region as LEFT,TOP,WIDTH,HEIGHT",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="request a fresh level attempt before capturing",
    )
    parser.add_argument(
        "--reset-wait-observations",
        type=int,
        default=600,
        help="maximum messages to wait for a fresh tick-0 observation after reset",
    )
    parser.add_argument(
        "--stop-on-death",
        action="store_true",
        help="stop after capturing an observation where dead=true",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip manifest/frame validation after capture",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="list visible windows and exit",
    )
    args = parser.parse_args(argv)

    if args.list_windows:
        windows = [window.to_dict() for window in list_visible_windows()]
        print(json.dumps(windows, indent=2, sort_keys=True))
        return 0

    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.reset_wait_observations <= 0:
        parser.error("--reset-wait-observations must be positive")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("artifacts") / f"frame_capture_{run_id}"
    frames_dir = output_dir / "frames"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    source = CaptureSource(
        window_title=_optional_text(args.window_title),
        region=args.region,
    )
    records: list[FrameCaptureRecord] = []

    with GeometryDashClient(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        initial_observation = (
            client.reset_attempt(
                "frame_capture",
                max_observations=args.reset_wait_observations,
            )
            if args.reset
            else None
        )

        observation_index = 0
        while len(records) < args.max_frames:
            if observation_index == 0 and initial_observation is not None:
                observation = initial_observation
            else:
                observation = client.receive_observation()

            if observation_index % args.stride == 0:
                frame = source.capture()
                frame_name = (
                    f"frame_{len(records) + 1:06d}_tick_{observation.tick:06d}.bmp"
                )
                frame_relative_path = Path("frames") / frame_name
                write_bmp(output_dir / frame_relative_path, frame)
                records.append(
                    FrameCaptureRecord.from_observation(
                        observation,
                        frame_path=frame_relative_path.as_posix(),
                        capture_width=frame.width,
                        capture_height=frame.height,
                        capture_region=frame.region,
                        fps=args.fps,
                        window_title=_optional_text(args.window_title),
                    )
                )

            observation_index += 1
            if args.stop_on_death and observation.dead:
                break

    _write_manifest(records, manifest_path)

    validation_document = None
    validation_ok = True
    if not args.no_validate:
        validation = validate_frame_manifest(manifest_path, base_dir=output_dir)
        validation_document = validation.to_dict()
        validation_ok = validation.ok

    summary = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "frame_count": len(records),
        "capture_source": source.describe(),
        "settings": {
            "host": args.host,
            "port": args.port,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "stride": args.stride,
            "reset": args.reset,
            "stop_on_death": args.stop_on_death,
        },
        "validation": validation_document,
    }
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest_jsonl": str(manifest_path),
                "summary_json": str(summary_path),
                "frame_count": len(records),
                "validation_ok": validation_ok,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation_ok else 1


def _write_manifest(records: list[FrameCaptureRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)
            )
            file.write("\n")


def _parse_region(value: str) -> CaptureRegion:
    try:
        return CaptureRegion.from_string(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


if __name__ == "__main__":
    raise SystemExit(main())
