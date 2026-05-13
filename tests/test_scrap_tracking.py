"""Tests for ScrapTracking module."""

import pandas as pd

from ts_shape.events.production.scrap_tracking import ScrapTracking


def _make_scrap_df():
    """Create sample scrap data."""
    times = pd.date_range("2024-01-01 06:00", periods=12, freq="1h")
    rows = []
    # Scrap counter (monotonic)
    for i, t in enumerate(times):
        rows.append(
            {"systime": t, "uuid": "scrap_weight", "value_double": 10.0 + i * 2.5}
        )
    # Reason codes
    reasons = [
        "Material",
        "Material",
        "Tool",
        "Tool",
        "Material",
        "Operator",
        "Tool",
        "Material",
        "Material",
        "Tool",
        "Operator",
        "Material",
    ]
    for t, reason in zip(times, reasons):
        rows.append({"systime": t, "uuid": "scrap_reason", "value_string": reason})
    # Part numbers
    parts = ["PART_A"] * 6 + ["PART_B"] * 6
    for t, part in zip(times, parts):
        rows.append({"systime": t, "uuid": "part_number", "value_string": part})
    return pd.DataFrame(rows)


class TestScrapTracking:

    def test_scrap_by_shift(self):
        df = _make_scrap_df()
        tracker = ScrapTracking(df)
        result = tracker.scrap_by_shift(scrap_uuid="scrap_weight")
        assert not result.empty
        assert "scrap_quantity" in result.columns
        assert result["scrap_quantity"].sum() > 0

    def test_scrap_by_reason(self):
        df = _make_scrap_df()
        tracker = ScrapTracking(df)
        result = tracker.scrap_by_reason(
            scrap_uuid="scrap_weight",
            reason_uuid="scrap_reason",
        )
        assert not result.empty
        assert "reason" in result.columns
        assert "pct_of_total" in result.columns

    def test_scrap_cost(self):
        df = _make_scrap_df()
        tracker = ScrapTracking(df)
        result = tracker.scrap_cost(
            scrap_uuid="scrap_weight",
            part_id_uuid="part_number",
            material_costs={"PART_A": 12.50, "PART_B": 8.75},
        )
        assert not result.empty
        assert "total_cost" in result.columns
        assert result["total_cost"].sum() > 0

    def test_scrap_trend(self):
        df = _make_scrap_df()
        tracker = ScrapTracking(df)
        result = tracker.scrap_trend(scrap_uuid="scrap_weight")
        assert not result.empty
        assert "period" in result.columns

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        tracker = ScrapTracking(df)
        assert tracker.scrap_by_shift(scrap_uuid="x").empty
        assert tracker.scrap_by_reason(scrap_uuid="x", reason_uuid="y").empty
