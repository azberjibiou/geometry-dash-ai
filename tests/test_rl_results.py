from gd_rl import AttemptResult, PracticeRunSummary


def attempt(
    index: int,
    *,
    final_percent: float,
    best_percent: float | None = None,
    death_tick: int | None = None,
    cleared: bool = False,
    reward: float = 0.0,
) -> AttemptResult:
    return AttemptResult(
        level_id="level-a",
        attempt_index=index,
        human_profile={"name": "Test"},
        seed=100 + index,
        trace_path=f"attempt_{index}/trace.jsonl",
        intended_events_path=f"attempt_{index}/policy_intended_events.json",
        executed_events_path=f"attempt_{index}/human_executed_events.json",
        humanization_path=f"attempt_{index}/humanization_details.json",
        row_count=10,
        playtime_seconds=1.0,
        final_percent=final_percent,
        best_percent=best_percent if best_percent is not None else final_percent,
        death_tick=death_tick,
        death_percent=final_percent if death_tick is not None else None,
        cleared=cleared,
        total_reward=reward,
        reward_terms={"progress_delta": reward},
        intended_event_count=2,
        executed_event_count=2,
        dropped_event_count=0,
    )


def test_practice_run_summary_aggregates_attempts() -> None:
    attempts = [
        attempt(1, final_percent=30.0, death_tick=100, reward=30.0),
        attempt(2, final_percent=100.0, cleared=True, reward=150.0),
        attempt(3, final_percent=45.0, death_tick=120, reward=40.0),
    ]

    summary = PracticeRunSummary.from_attempts(
        level_id="level-a",
        attempts=attempts,
    )

    assert summary.attempt_count == 3
    assert summary.clears == 1
    assert summary.deaths == 2
    assert summary.clear_rate == 1 / 3
    assert summary.attempts_to_first_clear == 2
    assert summary.playtime_to_first_clear_seconds == 2.0
    assert summary.average_final_percent == (30.0 + 100.0 + 45.0) / 3
    assert summary.best_percent == 100.0
    assert summary.total_reward == 220.0
    assert summary.reward_curve == [30.0, 150.0, 40.0]
    assert summary.death_tick_histogram == {"100": 1, "120": 1}
    assert summary.death_percent_histogram == {"30": 1, "45": 1}
