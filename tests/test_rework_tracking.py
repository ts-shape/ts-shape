"""Tests for ReworkTracking module."""

import pandas as pd
import pytest

from ts_shape.events.production.rework_tracking import ReworkTracking


def _make_rework_df():
    """Create sample rework data."""
    times = pd.date_range("2024-01-01 06:00", periods=12, freq="1h")
    rows = []
    # Rework counter (monotonic)
    for i, t in enumerate(times):
        rows.append(
            {"systime": t, "uuid": "rework_counter", "value_integer": 5 + i * 2}
        )
    # Total production counter (monotonic, higher numbers)
    for i, t in enumerate(times):
        rows.append(
            {"systime": t, "uuid": "total_counter", "value_integer": 100 + i * 20}
        )
    # Reason codes
    reasons = [
        "Dimension",
        "Dimension",
        "Surface",
        "Surface",
        "Dimension",
        "Tooling",
        "Surface",
        "Dimension",
        "Dimension",
        "Surface",
        "Tooling",
        "Dimension",
    ]
    for t, reason in zip(times, reasons):
        rows.append({"systime": t, "uuid": "rework_reason", "value_string": reason})
    # Part numbers
    parts = ["PART_A"] * 6 + ["PART_B"] * 6
    for t, part in zip(times, parts):
        rows.append({"systime": t, "uuid": "part_number", "value_string": part})
    return pd.DataFrame(rows)


class TestReworkTracking:

    def test_rework_by_shift(self):
        df = _make_rework_df()
        tracker = ReworkTracking(df)
        result = tracker.rework_by_shift(rework_uuid="rework_counter")
        assert not result.empty
        assert "rework_count" in result.columns
        assert result["rework_count"].sum() > 0

    def test_rework_by_reason(self):
        df = _make_rework_df()
        tracker = ReworkTracking(df)
        result = tracker.rework_by_reason(
            rework_uuid="rework_counter",
            reason_uuid="rework_reason",
        )
        assert not result.empty
        assert "reason" in result.columns
        assert "pct_of_total" in result.columns

    def test_rework_rate(self):
        df = _make_rework_df()
        tracker = ReworkTracking(df)
        result = tracker.rework_rate(
            rework_uuid="rework_counter",
            total_production_uuid="total_counter",
        )
        assert not result.empty
        assert "rework_rate_pct" in result.columns
        assert "total_produced" in result.columns
        assert "rework_count" in result.columns

    def test_rework_cost(self):
        df = _make_rework_df()
        tracker = ReworkTracking(df)
        result = tracker.rework_cost(
            rework_uuid="rework_counter",
            part_id_uuid="part_number",
            rework_costs={"PART_A": 25.00, "PART_B": 15.00},
        )
        assert not result.empty
        assert "total_cost" in result.columns
        assert result["total_cost"].sum() > 0

    def test_rework_trend(self):
        df = _make_rework_df()
        tracker = ReworkTracking(df)
        result = tracker.rework_trend(rework_uuid="rework_counter")
        assert not result.empty
        assert "period" in result.columns
        assert "rework_count" in result.columns

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer"])
        tracker = ReworkTracking(df)
        assert tracker.rework_by_shift(rework_uuid="x").empty
        assert tracker.rework_by_reason(rework_uuid="x", reason_uuid="y").empty
        assert tracker.rework_rate(rework_uuid="x", total_production_uuid="y").empty
