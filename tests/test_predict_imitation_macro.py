import json

import pytest

from gd_human_model import Event
from gd_trace import load_macro_json
from scripts.predict_imitation_macro import main


def test_predict_imitation_macro_writes_macro_summary_and_decoded_events(
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    predictions_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "predicted"
    _write_predictions(
        predictions_path,
        press={18: 0.6, 20: 0.95, 22: 0.7},
        release={30: 0.9},
        press_labels={20},
        release_labels={30},
    )

    exit_code = main(
        [
            "--predictions-jsonl",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--non-max-radius-frames",
            "4",
            "--min-event-spacing-frames",
            "2",
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    macro_path = output_dir / "predicted_macro.json"
    summary_path = output_dir / "prediction_summary.json"
    decoded_events_path = output_dir / "decoded_events.jsonl"

    macro = load_macro_json(macro_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decoded_events = [
        json.loads(line)
        for line in decoded_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert macro.events == [Event(20, "press"), Event(30, "release")]
    assert macro.metadata["decoder_config"]["press_threshold"] == 0.5
    assert decoded_events == [
        {"kind": "press", "player": "p1", "tick": 20},
        {"kind": "release", "player": "p1", "tick": 30},
    ]
    assert summary["decoded_event_count"] == 2
    assert summary["event_metrics"]["press"]["missed_labeled_events"] == []
    assert printed["decoded_events"] == decoded_events


def test_predict_imitation_macro_requires_dataset_dir_for_checkpoint(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as exc_info:
        main(["--checkpoint", "model.pt"])

    assert exc_info.value.code == 2
    assert "--dataset-dir is required" in capsys.readouterr().err


def _write_predictions(
    path,
    *,
    press: dict[int, float],
    release: dict[int, float],
    press_labels: set[int],
    release_labels: set[int],
) -> None:  # type: ignore[no-untyped-def]
    last_tick = max([0, *press.keys(), *release.keys(), *press_labels, *release_labels])
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for tick in range(last_tick + 1):
            file.write(
                json.dumps(
                    {
                        "position": tick,
                        "index": tick,
                        "tick": tick,
                        "split": "train",
                        "progress": float(tick),
                        "input_down": 20 <= tick < 30,
                        "probabilities": {
                            "press_event": press.get(tick, 0.05),
                            "release_event": release.get(tick, 0.05),
                        },
                        "labels": {
                            "press_event": tick in press_labels,
                            "release_event": tick in release_labels,
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            file.write("\n")
