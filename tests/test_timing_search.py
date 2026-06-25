import random

from gd_human_model import Event
from gd_rl import (
    CandidateEvaluation,
    TimingCandidate,
    TimingEventWindow,
    TimingSearchConfig,
    load_timing_windows_json,
    run_timing_search,
    sample_candidate,
    update_windows_from_elites,
)


def windows() -> list[TimingEventWindow]:
    return [
        TimingEventWindow(
            name="press",
            kind="press",
            mean_tick=50.0,
            std_tick=10.0,
            min_tick=20,
            max_tick=80,
        ),
        TimingEventWindow(
            name="release",
            kind="release",
            mean_tick=65.0,
            std_tick=10.0,
            min_tick=55,
            max_tick=95,
        ),
    ]


def test_sample_candidate_produces_ordered_events_within_windows() -> None:
    rng = random.Random(1)

    candidate = sample_candidate(
        windows=windows(),
        rng=rng,
        generation_index=2,
        population_index=3,
        min_event_spacing=2,
    )

    assert candidate.candidate_id == "g002_c003"
    assert candidate.event_names == ["press", "release"]
    assert candidate.events[0].kind == "press"
    assert candidate.events[1].kind == "release"
    assert 20 <= candidate.events[0].tick <= 80
    assert 55 <= candidate.events[1].tick <= 95
    assert candidate.events[1].tick >= candidate.events[0].tick + 2


def test_update_windows_moves_distribution_toward_elites() -> None:
    current_windows = windows()
    elites = [
        CandidateEvaluation(
            candidate=_candidate([60, 74]),
            score=10.0,
        ),
        CandidateEvaluation(
            candidate=_candidate([62, 76]),
            score=9.0,
        ),
    ]

    updated = update_windows_from_elites(
        windows=current_windows,
        elites=elites,
        min_std_tick=2.0,
        max_std_tick=80.0,
        update_smoothing=0.0,
    )

    assert updated[0].mean_tick == 61.0
    assert updated[1].mean_tick == 75.0
    assert updated[0].std_tick == 2.0
    assert updated[1].std_tick == 2.0


def test_run_timing_search_tracks_best_candidate() -> None:
    config = TimingSearchConfig(
        level_id="synthetic",
        event_windows=[
            TimingEventWindow(
                name="press",
                kind="press",
                mean_tick=20.0,
                std_tick=12.0,
                min_tick=0,
                max_tick=100,
            )
        ],
        generations=2,
        population_size=6,
        elite_fraction=0.5,
        seed=7,
        update_smoothing=0.0,
    )

    def evaluator(candidate):  # type: ignore[no-untyped-def]
        tick = candidate.events[0].tick
        return CandidateEvaluation(
            candidate=candidate,
            score=100.0 - abs(50 - tick),
            summary={"tick": tick},
        )

    result = run_timing_search(config=config, evaluator=evaluator)

    assert len(result.generations) == 2
    assert result.best_evaluation is not None
    assert result.best_evaluation.score > 60.0
    assert result.final_windows[0].mean_tick > 20.0


def test_load_timing_windows_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "windows.json"
    path.write_text(
        """{
  "metadata": {"level_name": "tiny"},
  "events": [
    {
      "name": "press",
      "kind": "press",
      "mean_tick": 10,
      "std_tick": 3,
      "min_tick": 0,
      "max_tick": 20
    }
  ]
}
""",
        encoding="utf-8",
    )

    event_windows, metadata = load_timing_windows_json(path)

    assert metadata == {"level_name": "tiny"}
    assert event_windows == [
        TimingEventWindow(
            name="press",
            kind="press",
            mean_tick=10.0,
            std_tick=3.0,
            min_tick=0,
            max_tick=20,
        )
    ]


def _candidate(ticks: list[int]):
    return TimingCandidate(
        candidate_id="fixed",
        generation_index=0,
        population_index=0,
        event_names=["press", "release"],
        events=[
            Event(tick=ticks[0], kind="press"),
            Event(tick=ticks[1], kind="release"),
        ],
    )
