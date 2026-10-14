"""Reset-aware behaviour of TargetEvents comparisons."""

import pandas as pd  # type: ignore
import pytest

from ts_shape.events.production.target_tracking import TargetEvents


@pytest.fixture
def reset_target_data():
    """A shift_1 counter that resets mid-shift.

    Values: 0,100,200,[reset]0,150 within 06:00-14:00 (shift_1).
    naive last-first = 150 ; reset-aware = 350 ; 1 reset.
    """
    times = pd.date_range("2024-01-15 07:00:00", periods=5, freq="1h")
    values = [0, 100, 200, 0, 150]
    rows = [
        {"systime": t, "uuid": "target_counter", "value_integer": v}
        for t, v in zip(times, values)
    ]
    return pd.DataFrame(rows)


def test_compare_to_target_default_under_reports(reset_target_data):
    tracker = TargetEvents(reset_target_data)
    result = tracker.compare_to_target("target_counter", targets={"shift_1": 300})
    assert "resets" not in result.columns
    assert result["actual"].iloc[0] == 150


def test_compare_to_target_handles_resets(reset_target_data):
    tracker = TargetEvents(reset_target_data)
    result = tracker.compare_to_target(
        "target_counter", targets={"shift_1": 300}, handle_resets=True
    )
    assert "resets" in result.columns
    assert result["actual"].iloc[0] == 350
    assert result["resets"].iloc[0] == 1
    assert result["status"].iloc[0] == "on_target"


def test_achievement_summary_default_under_reports(reset_target_data):
    tracker = TargetEvents(reset_target_data)
    result = tracker.target_achievement_summary("target_counter", daily_target=300)
    assert "resets" not in result.columns
    assert result["actual"].iloc[0] == 150


def test_achievement_summary_handles_resets(reset_target_data):
    tracker = TargetEvents(reset_target_data)
    result = tracker.target_achievement_summary(
        "target_counter", daily_target=300, handle_resets=True
    )
    assert "resets" in result.columns
    assert result["actual"].iloc[0] == 350
    assert result["resets"].iloc[0] == 1
