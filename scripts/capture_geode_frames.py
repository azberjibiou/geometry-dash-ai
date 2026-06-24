"""Capture visible Geometry Dash frames aligned to Geode bridge observations."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    ScreenCaptureError,
    activate_window,
    foreground_window,
    foreground_window_matches,
    list_visible_windows,
    save_manifest_jsonl,
    validate_frame_manifest,
    write_bmp,
)
from gd_env import BridgeObservation, GeometryDashClient
from gd_trace import load_macro_json


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
        "--start-capture-tick",
        type=int,
        default=0,
        help=(
            "discard observations before this attempt tick instead of saving "
            "frames for them"
        ),
    )
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
        "--activate-window",
        action="store_true",
        help="try to bring --window-title to the foreground before capturing",
    )
    parser.add_argument(
        "--require-foreground",
        action="store_true",
        help="fail if --window-title is not the foreground window before each capture",
    )
    parser.add_argument(
        "--window-activation-wait-seconds",
        type=float,
        default=0.5,
        help="wait this long after --activate-window before validating foreground",
    )
    parser.add_argument(
        "--region",
        type=_parse_region,
        help="absolute screen capture region as LEFT,TOP,WIDTH,HEIGHT",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--macro-json",
        type=Path,
        help=(
            "load this macro into the Geode mod, reset the level, and capture "
            "frames while the queued macro replays"
        ),
    )
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
        "--pre-reset-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "wait before sending reset; useful after a prior clear so GD can "
            "finish its completion UI transition"
        ),
    )
    parser.add_argument(
        "--post-terminal-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "wait after death/success/completion before exiting, so a following "
            "run does not reset immediately after the terminal event"
        ),
    )
    parser.add_argument(
        "--stop-on-death",
        action="store_true",
        help="stop after capturing an observation where dead=true",
    )
    parser.add_argument("--success-percent", type=float, default=100.0)
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="stop once observation.percent reaches --success-percent",
    )
    parser.add_argument(
        "--stop-before-completion",
        action="store_true",
        help=(
            "stop on completed=true before saving that frame, useful for "
            "gameplay-only captures that should exclude the result screen"
        ),
    )
    parser.add_argument(
        "--require-start-percent-max",
        type=float,
        help="fail if the fresh tick-0 observation starts above this percent",
    )
    parser.add_argument(
        "--require-start-x-max",
        type=float,
        help="fail if the fresh tick-0 observation starts beyond this x position",
    )
    parser.add_argument(
        "--require-progress-tick",
        type=int,
        help="tick used with --require-progress-percent-min to verify the expected level",
    )
    parser.add_argument(
        "--require-progress-percent-min",
        type=float,
        help="minimum percent required at --require-progress-tick",
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
    if args.window_activation_wait_seconds < 0.0:
        parser.error("--window-activation-wait-seconds must be non-negative")
    if args.start_capture_tick < 0:
        parser.error("--start-capture-tick must be non-negative")
    if args.reset_wait_observations <= 0:
        parser.error("--reset-wait-observations must be positive")
    if args.pre_reset_delay_seconds < 0.0:
        parser.error("--pre-reset-delay-seconds must be non-negative")
    if args.post_terminal_delay_seconds < 0.0:
        parser.error("--post-terminal-delay-seconds must be non-negative")
    if not 0.0 <= args.success_percent <= 100.0:
        parser.error("--success-percent must be between 0 and 100")
    if (
        args.require_start_percent_max is not None
        and not 0.0 <= args.require_start_percent_max <= 100.0
    ):
        parser.error("--require-start-percent-max must be between 0 and 100")
    if args.require_start_x_max is not None and args.require_start_x_max < 0.0:
        parser.error("--require-start-x-max must be non-negative")
    if args.require_progress_tick is not None and args.require_progress_tick < 0:
        parser.error("--require-progress-tick must be non-negative")
    if (
        args.require_progress_percent_min is not None
        and not 0.0 <= args.require_progress_percent_min <= 100.0
    ):
        parser.error("--require-progress-percent-min must be between 0 and 100")
    if (args.require_progress_tick is None) != (
        args.require_progress_percent_min is None
    ):
        parser.error(
            "--require-progress-tick and --require-progress-percent-min must be used together"
        )

    macro = load_macro_json(args.macro_json) if args.macro_json is not None else None
    should_reset = args.reset or macro is not None

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
    try:
        _prepare_target_window(
            window_title=_optional_text(args.window_title),
            activate=args.activate_window,
            require_foreground=args.require_foreground,
            wait_seconds=args.window_activation_wait_seconds,
        )
    except ScreenCaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    records: list[FrameCaptureRecord] = []
    observations: list[BridgeObservation] = []
    stop_reason = "max_frames"

    with GeometryDashClient(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        if macro is not None:
            macro_metadata = dict(macro.metadata)
            macro_metadata.setdefault("capture_mode", "queued_frame_capture")
            client.load_macro(macro.events, metadata=macro_metadata)
        elif args.reset:
            client.load_macro([], metadata={"capture_mode": "frame_capture_clear"})

        initial_observation = (
            _delayed_reset_attempt(
                client,
                delay_seconds=args.pre_reset_delay_seconds,
                max_observations=args.reset_wait_observations,
            )
            if should_reset
            else None
        )
        if initial_observation is not None:
            try:
                _validate_start_observation(
                    initial_observation,
                    require_percent_max=args.require_start_percent_max,
                    require_x_max=args.require_start_x_max,
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1

        observation_index = 0
        while len(records) < args.max_frames:
            if observation_index == 0 and initial_observation is not None:
                observation = initial_observation
            else:
                observation = client.receive_observation()
            observations.append(observation)

            pre_capture_terminal_reason = _pre_capture_terminal_reason(
                observation,
                stop_before_completion=args.stop_before_completion,
            )
            if pre_capture_terminal_reason is not None:
                stop_reason = pre_capture_terminal_reason
                break

            terminal_reason = _terminal_reason(
                observation,
                stop_on_death=args.stop_on_death,
                stop_on_success=args.stop_on_success,
                success_percent=args.success_percent,
            )
            should_capture = (
                observation.tick >= args.start_capture_tick
                and observation_index % args.stride == 0
            )
            should_capture = should_capture or terminal_reason is not None
            if should_capture:
                records.append(
                    _capture_record(
                        observation,
                        source=source,
                        output_dir=output_dir,
                        frame_index=len(records) + 1,
                        fps=args.fps,
                        window_title=_optional_text(args.window_title),
                        require_foreground=args.require_foreground,
                    )
                )

            observation_index += 1
            if terminal_reason is not None:
                stop_reason = terminal_reason
                break

        if stop_reason != "max_frames" and args.post_terminal_delay_seconds > 0.0:
            time.sleep(args.post_terminal_delay_seconds)

    try:
        _validate_observation_progress(
            observations,
            require_tick=args.require_progress_tick,
            require_percent_min=args.require_progress_percent_min,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_manifest_jsonl(records, manifest_path)

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
        "observation_count": len(observations),
        "macro": (
            {
                "path": str(args.macro_json),
                "metadata": macro.metadata,
                "event_count": len(macro.events),
            }
            if macro is not None
            else None
        ),
        "capture_source": source.describe(),
        "settings": {
            "host": args.host,
            "port": args.port,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "start_capture_tick": args.start_capture_tick,
            "stride": args.stride,
            "activate_window": args.activate_window,
            "require_foreground": args.require_foreground,
            "window_activation_wait_seconds": args.window_activation_wait_seconds,
            "reset": should_reset,
            "requested_reset": args.reset,
            "pre_reset_delay_seconds": args.pre_reset_delay_seconds,
            "post_terminal_delay_seconds": args.post_terminal_delay_seconds,
            "stop_on_death": args.stop_on_death,
            "success_percent": args.success_percent,
            "stop_on_success": args.stop_on_success,
            "stop_before_completion": args.stop_before_completion,
            "require_start_percent_max": args.require_start_percent_max,
            "require_start_x_max": args.require_start_x_max,
            "require_progress_tick": args.require_progress_tick,
            "require_progress_percent_min": args.require_progress_percent_min,
        },
        "result": _summarize_observations(observations, stop_reason=stop_reason),
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


def _prepare_target_window(
    *,
    window_title: str | None,
    activate: bool,
    require_foreground: bool,
    wait_seconds: float,
) -> None:
    if not window_title:
        return
    if activate:
        activate_window(window_title)
        if wait_seconds > 0.0:
            time.sleep(wait_seconds)
    if require_foreground and not foreground_window_matches(window_title):
        window = foreground_window()
        foreground_title = window.title if window is not None else "<none>"
        raise ScreenCaptureError(
            f"target window {window_title!r} is not foreground; "
            f"foreground is {foreground_title!r}"
        )


def _delayed_reset_attempt(
    client: GeometryDashClient,
    *,
    delay_seconds: float,
    max_observations: int,
) -> BridgeObservation:
    if delay_seconds > 0.0:
        time.sleep(delay_seconds)
    return client.reset_attempt(
        "frame_capture",
        max_observations=max_observations,
    )


def _capture_record(
    observation: BridgeObservation,
    *,
    source: CaptureSource,
    output_dir: Path,
    frame_index: int,
    fps: int,
    window_title: str | None,
    require_foreground: bool,
) -> FrameCaptureRecord:
    if require_foreground and window_title and not foreground_window_matches(window_title):
        window = foreground_window()
        foreground_title = window.title if window is not None else "<none>"
        raise ScreenCaptureError(
            f"target window {window_title!r} is not foreground; "
            f"foreground is {foreground_title!r}"
        )
    frame = source.capture()
    frame_name = f"frame_{frame_index:06d}_tick_{observation.tick:06d}.bmp"
    frame_relative_path = Path("frames") / frame_name
    write_bmp(output_dir / frame_relative_path, frame)
    return FrameCaptureRecord.from_observation(
        observation,
        frame_path=frame_relative_path.as_posix(),
        capture_width=frame.width,
        capture_height=frame.height,
        capture_region=frame.region,
        fps=fps,
        window_title=window_title,
    )


def _validate_start_observation(
    observation: BridgeObservation,
    *,
    require_percent_max: float | None,
    require_x_max: float | None,
) -> None:
    failures = []
    if (
        require_percent_max is not None
        and observation.percent > require_percent_max
    ):
        failures.append(
            f"percent {observation.percent:.3f} > {require_percent_max:.3f}"
        )
    if require_x_max is not None and observation.x > require_x_max:
        failures.append(f"x {observation.x:.3f} > {require_x_max:.3f}")
    if failures:
        raise ValueError("fresh start check failed: " + "; ".join(failures))


def _validate_observation_progress(
    observations: list[BridgeObservation],
    *,
    require_tick: int | None,
    require_percent_min: float | None,
) -> None:
    if require_tick is None or require_percent_min is None:
        return

    observation = next(
        (candidate for candidate in observations if candidate.tick >= require_tick),
        None,
    )
    if observation is None:
        raise ValueError(f"capture ended before progress guard tick {require_tick}")
    if observation.percent < require_percent_min:
        raise ValueError(
            f"progress check failed at tick {observation.tick}: "
            f"percent {observation.percent:.3f} < {require_percent_min:.3f}"
        )


def _terminal_reason(
    observation: BridgeObservation,
    *,
    stop_on_death: bool,
    stop_on_success: bool,
    success_percent: float,
) -> str | None:
    if stop_on_death and observation.dead:
        return "death"
    if stop_on_success and observation.percent >= success_percent:
        return "success"
    return None


def _pre_capture_terminal_reason(
    observation: BridgeObservation,
    *,
    stop_before_completion: bool,
) -> str | None:
    if stop_before_completion and observation.completed:
        return "completed"
    return None


def _summarize_observations(
    observations: list[BridgeObservation],
    *,
    stop_reason: str,
) -> dict[str, object]:
    if not observations:
        return {
            "stop_reason": stop_reason,
            "first_tick": None,
            "last_tick": None,
            "start_percent": None,
            "final_percent": None,
            "dead": False,
            "completed": False,
        }

    first = observations[0]
    last = observations[-1]
    return {
        "stop_reason": stop_reason,
        "first_tick": first.tick,
        "last_tick": last.tick,
        "start_percent": first.percent,
        "final_percent": last.percent,
        "dead": last.dead,
        "completed": last.completed,
    }


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
    try:
        raise SystemExit(main())
    except ScreenCaptureError as exc:
        print(f"error: screen capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (OSError, TimeoutError, EOFError) as exc:
        print(f"error: bridge communication failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
