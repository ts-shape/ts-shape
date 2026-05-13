"""Tests for OperatorPerformanceTracking module."""

import pandas as pd

from ts_shape.events.production.operator_performance import OperatorPerformanceTracking


def _make_operator_df():
    """Create sample data with operator IDs and production counters."""
    times = pd.date_range("2024-01-01 06:00", periods=16, freq="30min")
    rows = []
    # Operator signal: Alice for first 8 (shift_1), Bob for next 8 (shift_2 starts at 14:00)
    operators = ["Alice"] * 8 + ["Bob"] * 8
    for t, op in zip(times, operators):
        rows.append({"systime": t, "uuid": "operator_id", "value_string": op})
    # Production counter (monotonic)
    for i, t in enumerate(times):
        rows.append(
            {"systime": t, "uuid": "part_counter", "value_integer": 100 + i * 10}
        )
    # OK parts counter
    for i, t in enumerate(times):
        rows.append({"systime": t, "uuid": "ok_parts", "value_integer": 90 + i * 9})
    # NOK parts counter
    for i, t in enumerate(times):
        rows.append({"systime": t, "uuid": "nok_parts", "value_integer": 10 + i * 1})
    return pd.DataFrame(rows)


class TestOperatorPerformanceTracking:

    def test_production_by_operator(self):
        df = _make_operator_df()
        tracker = OperatorPerformanceTracking(df)
        result = tracker.production_by_operator(
            operator_uuid="operator_id",
            counter_uuid="part_counter",
        )
        assert not result.empty
        assert "operator" in result.columns
        assert "total_produced" in result.columns
        assert "shifts_worked" in result.columns
        assert "avg_per_shift" in result.columns
        assert set(result["operator"].tolist()).issubset({"Alice", "Bob"})

    def test_operator_efficiency(self):
        df = _make_operator_df()
        tracker = OperatorPerformanceTracking(df)
        result = tracker.operator_efficiency(
            operator_uuid="operator_id",
            counter_uuid="part_counter",
            target_per_shift=100,
        )
        assert not result.empty
        assert "efficiency_pct" in result.columns
        assert "target" in result.columns

    def test_quality_by_operator(self):
        df = _make_operator_df()
        tracker = OperatorPerformanceTracking(df)
        result = tracker.quality_by_operator(
            operator_uuid="operator_id",
            ok_uuid="ok_parts",
            nok_uuid="nok_parts",
        )
        assert not result.empty
        assert "operator" in result.columns
        assert "ok_count" in result.columns
        assert "nok_count" in result.columns
        assert "first_pass_yield_pct" in result.columns

    def test_operator_comparison(self):
        df = _make_operator_df()
        tracker = OperatorPerformanceTracking(df)
        result = tracker.operator_comparison(
            operator_uuid="operator_id",
            counter_uuid="part_counter",
        )
        assert not result.empty
        assert "rank" in result.columns
        assert "pct_of_best" in result.columns
        # Best operator should have rank 1 and 100% of best
        assert result.iloc[0]["rank"] == 1
        assert result.iloc[0]["pct_of_best"] == 100.0

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_string", "value_integer"])
        tracker = OperatorPerformanceTracking(df)
        assert tracker.production_by_operator("x", "y").empty
        assert tracker.operator_efficiency("x", "y", 100).empty
        assert tracker.quality_by_operator("x", "y", "z").empty
        assert tracker.operator_comparison("x", "y").empty
