"""Tests for TargetTracking module."""

import pandas as pd
import pytest

from ts_shape.events.production.target_tracking import TargetTracking


def _make_counter_df():
    """Create sample production counter data across 3 days."""
    rows = []
    for day in range(1, 4):
        base = (day - 1) * 1500
        for hour in range(6, 22):
            t = pd.Timestamp(f"2024-01-0{day} {hour:02d}:00:00")
            rows.append(
                {
                    "systime": t,
                    "uuid": "prod_counter",
                    "value_integer": base + (hour - 6) * 90 + (10 if day == 2 else 0),
                }
            )
    return pd.DataFrame(rows)


class TestTargetTracking:

    def test_compare_to_target(self):
        df = _make_counter_df()
        tracker = TargetTracking(df)
        result = tracker.compare_to_target(
            metric_uuid="prod_counter",
            targets={"shift_1": 700, "shift_2": 700, "shift_3": 700},
        )
        assert not result.empty
        assert "status" in result.columns
        assert set(result["status"].unique()).issubset(
            {"on_target", "warning", "below_target"}
        )

    def test_target_achievement_summary(self):
        df = _make_counter_df()
        tracker = TargetTracking(df)
        result = tracker.target_achievement_summary(
            metric_uuid="prod_counter",
            daily_target=1300,
        )
        assert not result.empty
        assert "achievement_pct" in result.columns

    def test_target_hit_rate(self):
        df = _make_counter_df()
        tracker = TargetTracking(df)
        result = tracker.target_hit_rate(
            metric_uuid="prod_counter",
            daily_target=1300,
        )
        assert "total_days" in result
        assert "hit_rate_pct" in result
        assert result["total_days"] > 0

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer"])
        tracker = TargetTracking(df)
        result = tracker.compare_to_target(
            metric_uuid="x",
            targets={"shift_1": 100},
        )
        assert result.empty

    def test_target_hit_rate_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer"])
        tracker = TargetTracking(df)
        result = tracker.target_hit_rate(metric_uuid="x", daily_target=100)
        assert result["total_days"] == 0
