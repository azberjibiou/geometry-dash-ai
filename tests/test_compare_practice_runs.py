import json

from scripts.compare_practice_runs import (
    format_table,
    load_comparison_row,
    main,
)


def write_summary(
    run_dir,  # type: ignore[no-untyped-def]
    *,
    level_id: str = "triple_spike_fixture",
    policy_name: str = "scripted_events:triple_spike_jump.json",
    human_profile_name: str = "Advanced",
    attempts: int = 3,
    clears: int = 2,
    deaths: int = 1,
) -> None:
    summary = {
        "level_id": level_id,
        "attempt_count": attempts,
        "clears": clears,
        "deaths": deaths,
        "clear_rate": clears / attempts,
        "average_final_percent": 88.5,
        "best_percent": 100.0,
        "total_reward": 345.25,
        "reward_curve": [95.25, 125.0, 125.0],
        "death_tick_histogram": {"206": 1},
        "death_percent_histogram": {"21": 1},
        "attempts": [
            {
                "human_profile": {"name": human_profile_name},
                "metadata": {"policy_name": policy_name},
            }
        ],
        "metadata": {
            "policy_name": policy_name,
            "human_profile_name": human_profile_name,
        },
    }
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def test_load_comparison_row_reads_practice_summary_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "scripted_run"
    write_summary(run_dir)

    row = load_comparison_row(run_dir)

    assert row.run_path == str(run_dir)
    assert row.level_id == "triple_spike_fixture"
    assert row.policy_name == "scripted_events:triple_spike_jump.json"
    assert row.human_profile_name == "Advanced"
    assert row.attempts == 3
    assert row.clears == 2
    assert row.clear_rate == 2 / 3
    assert row.average_final_percent == 88.5
    assert row.best_percent == 100.0
    assert row.deaths == 1
    assert row.total_reward == 345.25
    assert row.reward_curve == [95.25, 125.0, 125.0]
    assert row.death_histogram == {
        "tick": {"206": 1},
        "percent": {"21": 1},
    }


def test_format_table_includes_requested_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "no_input_run"
    write_summary(
        run_dir,
        policy_name="no_input",
        human_profile_name="Beginner",
        clears=0,
        deaths=3,
    )

    table = format_table([load_comparison_row(run_dir / "summary.json")])

    assert "run_path" in table
    assert "policy_name" in table
    assert "human_profile_name" in table
    assert "reward_curve" in table
    assert "death_histogram" in table
    assert "no_input" in table
    assert "Beginner" in table
    assert '{"percent":{"21":1},"tick":{"206":1}}' in table


def test_main_writes_json_comparison(tmp_path) -> None:  # type: ignore[no-untyped-def]
    first_run = tmp_path / "first"
    second_run = tmp_path / "second"
    output_path = tmp_path / "comparison.json"
    write_summary(first_run, policy_name="no_input", clears=0, deaths=3)
    write_summary(second_run, policy_name="random_events", clears=1, deaths=2)

    exit_code = main(
        [
            str(first_run),
            str(second_run / "summary.json"),
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert [row["policy_name"] for row in rows] == ["no_input", "random_events"]
    assert rows[0]["run_path"] == str(first_run)
    assert rows[1]["run_path"] == str(second_run)
    assert rows[0]["attempts"] == 3
