import json

from gd_human_model import Event
from gd_trace import load_macro_json
from scripts.predict_imitation_state_macro import main


def test_predict_imitation_state_macro_writes_transition_macro(
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    predictions_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "predicted"
    _write_state_predictions(
        predictions_path,
        probabilities={
            0: 0.1,
            1: 0.9,
            2: 0.8,
            3: 0.2,
        },
        labels={
            0: False,
            1: True,
            2: True,
            3: False,
        },
    )

    exit_code = main(
        [
            "--predictions-jsonl",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    macro = load_macro_json(output_dir / "predicted_macro.json")
    summary = json.loads(
        (output_dir / "prediction_summary.json").read_text(encoding="utf-8")
    )

    assert macro.events == [Event(101, "press"), Event(103, "release")]
    assert macro.metadata["kind"] == "predicted_target_input_down_transitions"
    assert summary["state_metrics"]["state"]["accuracy"] == 1.0
    assert printed["decoded_events"] == [
        {"kind": "press", "player": "p1", "tick": 101},
        {"kind": "release", "player": "p1", "tick": 103},
    ]


def _write_state_predictions(
    path,
    *,
    probabilities: dict[int, float],
    labels: dict[int, bool],
) -> None:  # type: ignore[no-untyped-def]
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for tick in sorted(probabilities):
            file.write(
                json.dumps(
                    {
                        "position": tick,
                        "index": tick,
                        "tick": tick,
                        "label_tick": tick + 100,
                        "split": "train",
                        "progress": float(tick),
                        "input_down": tick > 0,
                        "target_input_down": labels[tick],
                        "probabilities": {
                            "target_input_down": probabilities[tick],
                        },
                        "labels": {
                            "target_input_down": labels[tick],
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            file.write("\n")
