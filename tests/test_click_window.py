from gd_human_model import Event
from gd_trace.click_window import analyze_click_windows


def test_analyze_click_windows_expands_events_by_radius() -> None:
    windows = analyze_click_windows(
        [Event(1, "press"), Event(10, "release")],
        radius_frames=3,
    )

    assert windows[0].start_tick == 0
    assert windows[0].end_tick == 4
    assert windows[0].width_frames == 5
    assert windows[1].start_tick == 7
    assert windows[1].end_tick == 13
    assert windows[1].width_frames == 7
