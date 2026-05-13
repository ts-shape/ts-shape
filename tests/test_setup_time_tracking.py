"""Tests for SetupTimeTracking module."""

import pandas as pd
import pytest

from ts_shape.events.production.setup_time_tracking import SetupTimeTracking


def _make_setup_df():
    """Create sample data with machine state including setup periods."""
    times = pd.date_range("2024-01-01 06:00", periods=24, freq="30min")
    rows = []
    # Machine state: Running → Setup → Running → Setup → Running ...
    states = (
        ["Running"] * 4
        + ["Setup"] * 2
        + ["Running"] * 6
        + ["Setup"] * 3
        + ["Running"] * 5
        + ["Setup"] * 2
        + ["Running"] * 2
    )
    for t, state in zip(times, states):
        rows.append({"systime": t, "uuid": "machine_state", "value_string": state})
    # Part numbers: changes after each setup
    parts = ["PART_A"] * 6 + ["PART_B"] * 12 + ["PART_C"] * 6
    for t, part in zip(times, parts):
        rows.append({"systime": t, "uuid": "part_number", "value_string": part})
    return pd.DataFrame(rows)


class TestSetupTimeTracking:

    def test_setup_durations(self):
        df = _make_setup_df()
        tracker = SetupTimeTracking(df)
        result = tracker.setup_durations(state_uuid="machine_state")
        assert not result.empty
        assert "duration_minutes" in result.columns
        assert "date" in result.columns
        assert "shift" in result.columns
        assert all(result["duration_minutes"] > 0)

    def test_setup_by_product(self):
        df = _make_setup_df()
        tracker = SetupTimeTracking(df)
        result = tracker.setup_by_product(
            state_uuid="machine_state",
            part_id_uuid="part_number",
        )
        assert not result.empty
        assert "from_product" in result.columns
        assert "to_product" in result.columns
        assert "avg_minutes" in result.columns
        assert "count" in result.columns

    def test_setup_statistics(self):
        df = _make_setup_df()
        tracker = SetupTimeTracking(df)
        result = tracker.setup_statistics(state_uuid="machine_state")
        assert not result.empty
        assert len(result) == 1
        assert "total_setups" in result.columns
        assert "avg_minutes" in result.columns
        assert "pct_of_available_time" in result.columns
        assert result["total_setups"].iloc[0] > 0

    def test_setup_trend(self):
        df = _make_setup_df()
        tracker = SetupTimeTracking(df)
        result = tracker.setup_trend(state_uuid="machine_state", window="1D")
        assert not result.empty
        assert "period" in result.columns
        assert "avg_setup_minutes" in result.columns
        assert "setup_count" in result.columns

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_string"])
        tracker = SetupTimeTracking(df)
        assert tracker.setup_durations(state_uuid="x").empty
        assert tracker.setup_by_product(state_uuid="x", part_id_uuid="y").empty
        assert tracker.setup_statistics(state_uuid="x").empty
        assert tracker.setup_trend(state_uuid="x").empty

    def test_no_setup_in_data(self):
        """All running, no setup events → empty results."""
        times = pd.date_range("2024-01-01 06:00", periods=6, freq="1h")
        rows = [
            {"systime": t, "uuid": "state", "value_string": "Running"} for t in times
        ]
        df = pd.DataFrame(rows)
        tracker = SetupTimeTracking(df)
        assert tracker.setup_durations(state_uuid="state").empty
