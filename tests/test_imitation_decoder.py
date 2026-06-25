import pytest

from gd_human_model import Event
from gd_imitation import (
    DecoderConfig,
    EventDecoderError,
    decode_predictions,
    event_level_metrics,
    select_event_candidates,
)


def test_decoder_emits_single_clean_press_and_release() -> None:
    predictions = _rows(
        press={20: 0.9},
        release={30: 0.8},
        press_labels={20},
        release_labels={30},
    )

    events = decode_predictions(predictions)
    metrics = event_level_metrics(predictions, events)

    assert events == [Event(20, "press"), Event(30, "release")]
    assert metrics["press"]["precision"] == pytest.approx(1.0)
    assert metrics["press"]["recall"] == pytest.approx(1.0)
    assert metrics["release"]["within_1_frame"]["matched_count"] == 1


def test_non_max_suppression_collapses_nearby_duplicate_peaks() -> None:
    predictions = _rows(
        press={18: 0.65, 20: 0.95, 22: 0.75},
        release={30: 0.9},
    )

    candidates = select_event_candidates(
        predictions,
        config=DecoderConfig(non_max_radius_frames=4),
    )
    events = decode_predictions(
        predictions,
        config=DecoderConfig(non_max_radius_frames=4),
    )

    assert [(candidate.tick, candidate.kind) for candidate in candidates] == [
        (20, "press"),
        (30, "release"),
    ]
    assert events == [Event(20, "press"), Event(30, "release")]


def test_decoder_skips_invalid_release_before_press() -> None:
    predictions = _rows(
        press={15: 0.9},
        release={10: 0.9, 25: 0.9},
    )

    events = decode_predictions(predictions)

    assert events == [Event(15, "press"), Event(25, "release")]


def test_decoder_skips_invalid_press_while_already_holding() -> None:
    predictions = _rows(
        press={5: 0.9, 10: 0.9},
        release={20: 0.9},
    )

    events = decode_predictions(predictions)

    assert events == [Event(5, "press"), Event(20, "release")]


def test_decoder_keeps_separated_repeated_clicks() -> None:
    predictions = _rows(
        press={5: 0.9, 20: 0.85},
        release={10: 0.9, 25: 0.8},
    )

    events = decode_predictions(predictions)

    assert events == [
        Event(5, "press"),
        Event(10, "release"),
        Event(20, "press"),
        Event(25, "release"),
    ]


def test_decoder_rejects_duplicate_prediction_ticks() -> None:
    predictions = [
        _row(1, press_probability=0.6),
        _row(1, release_probability=0.6),
    ]

    with pytest.raises(EventDecoderError, match="duplicate prediction tick 1"):
        decode_predictions(predictions)


def test_decoder_config_rejects_invalid_values() -> None:
    with pytest.raises(EventDecoderError, match="press_threshold"):
        DecoderConfig(press_threshold=True)  # type: ignore[arg-type]

    with pytest.raises(EventDecoderError, match="non_max_radius_frames must be an int"):
        DecoderConfig(non_max_radius_frames=False)  # type: ignore[arg-type]


def _rows(
    *,
    press: dict[int, float],
    release: dict[int, float],
    press_labels: set[int] | None = None,
    release_labels: set[int] | None = None,
) -> list[dict[str, object]]:
    press_labels = press_labels or set()
    release_labels = release_labels or set()
    last_tick = max(
        [0, *press.keys(), *release.keys(), *press_labels, *release_labels]
    )
    return [
        _row(
            tick,
            press_probability=press.get(tick, 0.05),
            release_probability=release.get(tick, 0.05),
            press_label=tick in press_labels,
            release_label=tick in release_labels,
        )
        for tick in range(last_tick + 1)
    ]


def _row(
    tick: int,
    *,
    press_probability: float = 0.05,
    release_probability: float = 0.05,
    press_label: bool = False,
    release_label: bool = False,
) -> dict[str, object]:
    return {
        "position": tick,
        "index": tick,
        "tick": tick,
        "split": "train",
        "probabilities": {
            "press_event": press_probability,
            "release_event": release_probability,
        },
        "labels": {
            "press_event": press_label,
            "release_event": release_label,
        },
    }
