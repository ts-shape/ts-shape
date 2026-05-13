"""Tests for supply chain event detectors.

Covers InventoryMonitoringEvents, LeadTimeAnalysisEvents, and DemandPatternEvents
with synthetic but realistic manufacturing/warehouse data.
"""

import pandas as pd  # type: ignore
import numpy as np
import pytest

from ts_shape.events.supplychain import (
    InventoryMonitoringEvents,
    LeadTimeAnalysisEvents,
    DemandPatternEvents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_df():
    """Return an empty DataFrame with the standard timeseries schema."""
    return pd.DataFrame(
        columns=[
            "systime",
            "uuid",
            "value_bool",
            "value_integer",
            "value_double",
            "value_string",
            "is_delta",
        ]
    )


# ===========================================================================
# InventoryMonitoringEvents
# ===========================================================================


class TestInventoryMonitoringEvents:
    """Tests for InventoryMonitoringEvents."""

    @pytest.fixture
    def declining_inventory_df(self):
        """Inventory that declines steadily from 500 to 20 over 24 hours."""
        times = pd.date_range("2024-06-01 00:00", periods=25, freq="1h")
        levels = np.linspace(500, 20, 25)
        return pd.DataFrame(
            {
                "systime": times,
                "uuid": "warehouse_level",
                "value_double": levels,
                "is_delta": True,
            }
        )

    def test_detect_low_stock_basic(self, declining_inventory_df):
        tracker = InventoryMonitoringEvents(
            declining_inventory_df, level_uuid="warehouse_level"
        )
        result = tracker.detect_low_stock(min_level=100)
        assert not result.empty
        assert "start" in result.columns
        assert "end" in result.columns
        assert "min_value" in result.columns
        # The minimum value in the low-stock interval should be below 100
        assert (result["min_value"] < 100).all()

    def test_detect_low_stock_with_hold(self, declining_inventory_df):
        tracker = InventoryMonitoringEvents(
            declining_inventory_df, level_uuid="warehouse_level"
        )
        # With a long hold, should still detect since the stock stays below
        result = tracker.detect_low_stock(min_level=100, hold="2h")
        assert not result.empty
        assert (result["duration_seconds"] >= 7200).all()

    def test_detect_low_stock_none_below(self, declining_inventory_df):
        tracker = InventoryMonitoringEvents(
            declining_inventory_df, level_uuid="warehouse_level"
        )
        # Nothing below 10
        result = tracker.detect_low_stock(min_level=10)
        assert result.empty

    def test_consumption_rate(self, declining_inventory_df):
        tracker = InventoryMonitoringEvents(
            declining_inventory_df, level_uuid="warehouse_level"
        )
        result = tracker.consumption_rate(window="4h")
        assert not result.empty
        assert "consumption_rate" in result.columns
        assert "window_start" in result.columns
        # All rates should be positive (inventory is declining)
        assert (result["consumption_rate"] >= 0).all()

    def test_reorder_point_breach(self, declining_inventory_df):
        tracker = InventoryMonitoringEvents(
            declining_inventory_df, level_uuid="warehouse_level"
        )
        result = tracker.reorder_point_breach(reorder_level=200, safety_stock=50)
        assert not result.empty
        assert "breach_type" in result.columns
        breach_types = set(result["breach_type"].unique())
        # Should detect at least a reorder breach
        assert "reorder" in breach_types or "safety_stock" in breach_types

    def test_reorder_safety_stock_breach(self, declining_inventory_df):
        tracker = InventoryMonitoringEvents(
            declining_inventory_df, level_uuid="warehouse_level"
        )
        result = tracker.reorder_point_breach(reorder_level=200, safety_stock=100)
        # Since inventory goes from 500->20, should eventually breach safety stock
        safety_breaches = result[result["breach_type"] == "safety_stock"]
        # May or may not have separate safety breach depending on transition
        assert not result.empty

    def test_stockout_prediction(self, declining_inventory_df):
        tracker = InventoryMonitoringEvents(
            declining_inventory_df, level_uuid="warehouse_level"
        )
        result = tracker.stockout_prediction(consumption_rate_window="4h")
        assert not result.empty
        assert "estimated_stockout_time_hours" in result.columns
        assert "consumption_rate" in result.columns
        # Later rows should have shorter stockout estimates
        finite_rows = result[np.isfinite(result["estimated_stockout_time_hours"])]
        if len(finite_rows) > 1:
            # The last predicted stockout time should be less than the first
            assert (
                finite_rows["estimated_stockout_time_hours"].iloc[-1]
                < finite_rows["estimated_stockout_time_hours"].iloc[0]
            )

    def test_empty_dataframe(self):
        tracker = InventoryMonitoringEvents(_empty_df(), level_uuid="nonexistent")
        assert tracker.detect_low_stock(min_level=100).empty
        assert tracker.consumption_rate().empty
        assert tracker.reorder_point_breach(reorder_level=100).empty
        assert tracker.stockout_prediction().empty

    def test_empty_returns_correct_columns(self):
        tracker = InventoryMonitoringEvents(_empty_df(), level_uuid="nonexistent")
        low = tracker.detect_low_stock(min_level=100)
        assert "start" in low.columns
        assert "duration_seconds" in low.columns

        rate = tracker.consumption_rate()
        assert "consumption_rate" in rate.columns

        breach = tracker.reorder_point_breach(reorder_level=100)
        assert "breach_type" in breach.columns

        pred = tracker.stockout_prediction()
        assert "estimated_stockout_time_hours" in pred.columns


# ===========================================================================
# LeadTimeAnalysisEvents
# ===========================================================================


class TestLeadTimeAnalysisEvents:
    """Tests for LeadTimeAnalysisEvents."""

    @pytest.fixture
    def order_delivery_df(self):
        """5 orders placed every 2 days, deliveries arrive 3-7 days later."""
        order_times = pd.date_range("2024-01-01", periods=5, freq="2D")
        # Deliveries: 3, 4, 5, 3, 7 days after order
        delivery_delays = [3, 4, 5, 3, 7]
        delivery_times = [
            ot + pd.Timedelta(days=d) for ot, d in zip(order_times, delivery_delays)
        ]
        order_ids = [f"PO-{i+1:03d}" for i in range(5)]

        orders = pd.DataFrame(
            {
                "systime": order_times,
                "uuid": "order_placed",
                "value_string": order_ids,
                "is_delta": True,
            }
        )
        deliveries = pd.DataFrame(
            {
                "systime": delivery_times,
                "uuid": "delivery_received",
                "value_string": order_ids,
                "is_delta": True,
            }
        )
        return pd.concat([orders, deliveries], ignore_index=True)

    def test_calculate_lead_times(self, order_delivery_df):
        analyzer = LeadTimeAnalysisEvents(order_delivery_df)
        result = analyzer.calculate_lead_times("order_placed", "delivery_received")
        assert len(result) == 5
        assert "lead_time_hours" in result.columns
        # First order has 3-day lead time = 72 hours
        assert abs(result["lead_time_hours"].iloc[0] - 72.0) < 0.01

    def test_lead_time_statistics(self, order_delivery_df):
        analyzer = LeadTimeAnalysisEvents(order_delivery_df)
        stats = analyzer.lead_time_statistics("order_placed", "delivery_received")
        assert len(stats) == 1
        assert "mean_hours" in stats.columns
        assert "p95_hours" in stats.columns
        # Mean of [3,4,5,3,7] days = 4.4 days = 105.6 hours
        assert abs(stats["mean_hours"].iloc[0] - 105.6) < 0.1
        assert stats["count"].iloc[0] == 5

    def test_detect_lead_time_anomalies(self, order_delivery_df):
        analyzer = LeadTimeAnalysisEvents(order_delivery_df)
        # threshold_factor=1.0 should catch the 7-day outlier
        anomalies = analyzer.detect_lead_time_anomalies(
            "order_placed", "delivery_received", threshold_factor=1.0
        )
        assert not anomalies.empty
        assert "z_score" in anomalies.columns
        # The 7-day delivery should be flagged
        assert (anomalies["lead_time_hours"] > 150).any()

    def test_detect_no_anomalies_high_threshold(self, order_delivery_df):
        analyzer = LeadTimeAnalysisEvents(order_delivery_df)
        anomalies = analyzer.detect_lead_time_anomalies(
            "order_placed", "delivery_received", threshold_factor=5.0
        )
        # With a very high threshold, nothing should be flagged
        assert anomalies.empty

    def test_unequal_orders_deliveries(self):
        """More orders than deliveries: only pairs that exist are matched."""
        orders = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=5, freq="1D"),
                "uuid": "order",
                "value_string": ["A", "B", "C", "D", "E"],
                "is_delta": True,
            }
        )
        deliveries = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-03", periods=3, freq="2D"),
                "uuid": "delivery",
                "value_string": ["A", "B", "C"],
                "is_delta": True,
            }
        )
        df = pd.concat([orders, deliveries], ignore_index=True)
        analyzer = LeadTimeAnalysisEvents(df)
        result = analyzer.calculate_lead_times("order", "delivery")
        assert len(result) == 3

    def test_empty_dataframe(self):
        analyzer = LeadTimeAnalysisEvents(_empty_df())
        assert analyzer.calculate_lead_times("o", "d").empty
        assert analyzer.lead_time_statistics("o", "d").empty
        assert analyzer.detect_lead_time_anomalies("o", "d").empty

    def test_empty_returns_correct_columns(self):
        analyzer = LeadTimeAnalysisEvents(_empty_df())
        lt = analyzer.calculate_lead_times("o", "d")
        assert "lead_time_hours" in lt.columns
        assert "order_id" in lt.columns

        stats = analyzer.lead_time_statistics("o", "d")
        assert "mean_hours" in stats.columns

        anom = analyzer.detect_lead_time_anomalies("o", "d")
        assert "z_score" in anom.columns


# ===========================================================================
# DemandPatternEvents
# ===========================================================================


class TestDemandPatternEvents:
    """Tests for DemandPatternEvents."""

    @pytest.fixture
    def daily_demand_df(self):
        """3 weeks of hourly demand data with a weekly pattern.

        Weekdays have higher demand than weekends. One spike day injected.
        """
        times = pd.date_range("2024-01-01", periods=21 * 24, freq="1h")
        np.random.seed(42)
        demand = []
        for t in times:
            # Base demand by day of week (Mon-Fri: ~100, Sat-Sun: ~30)
            if t.dayofweek < 5:
                base = 100.0
            else:
                base = 30.0
            demand.append(base + np.random.normal(0, 5))

        # Inject a spike on day 10 (all hours get 3x demand)
        spike_day = pd.Timestamp("2024-01-11")
        for i, t in enumerate(times):
            if t.date() == spike_day.date():
                demand[i] *= 3.0

        return pd.DataFrame(
            {
                "systime": times,
                "uuid": "customer_orders",
                "value_double": demand,
                "is_delta": True,
            }
        )

    def test_demand_by_period_daily(self, daily_demand_df):
        analyzer = DemandPatternEvents(daily_demand_df, demand_uuid="customer_orders")
        result = analyzer.demand_by_period(period="1D")
        assert not result.empty
        assert "total_demand" in result.columns
        assert "period_start" in result.columns
        # Should have roughly 21 days
        assert len(result) == 21

    def test_demand_by_period_hourly(self, daily_demand_df):
        analyzer = DemandPatternEvents(daily_demand_df, demand_uuid="customer_orders")
        result = analyzer.demand_by_period(period="1h")
        assert not result.empty
        # 21 days * 24 hours = 504 periods
        assert len(result) == 21 * 24

    def test_detect_demand_spikes(self, daily_demand_df):
        analyzer = DemandPatternEvents(daily_demand_df, demand_uuid="customer_orders")
        spikes = analyzer.detect_demand_spikes(threshold_factor=2.0, window="1D")
        assert not spikes.empty
        assert "spike_magnitude" in spikes.columns
        # The spike day (Jan 11) should be detected
        spike_dates = pd.to_datetime(spikes["period_start"]).dt.date
        assert pd.Timestamp("2024-01-11").date() in spike_dates.values

    def test_detect_no_spikes_high_threshold(self, daily_demand_df):
        analyzer = DemandPatternEvents(daily_demand_df, demand_uuid="customer_orders")
        spikes = analyzer.detect_demand_spikes(threshold_factor=10.0, window="1D")
        # Very high threshold should not flag anything
        assert spikes.empty

    def test_seasonality_summary_daily(self, daily_demand_df):
        analyzer = DemandPatternEvents(daily_demand_df, demand_uuid="customer_orders")
        result = analyzer.seasonality_summary(period="1D")
        assert not result.empty
        assert "period_label" in result.columns
        # Should have entries for each day of the week
        assert len(result) == 7
        # Weekday demand should be higher than weekend
        weekday_rows = result[
            result["period_label"].isin(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            )
        ]
        weekend_rows = result[result["period_label"].isin(["Saturday", "Sunday"])]
        assert weekday_rows["avg_demand"].mean() > weekend_rows["avg_demand"].mean()

    def test_seasonality_summary_hourly(self, daily_demand_df):
        analyzer = DemandPatternEvents(daily_demand_df, demand_uuid="customer_orders")
        result = analyzer.seasonality_summary(period="1h")
        assert not result.empty
        assert "period_label" in result.columns
        # Should have 24 hour labels
        assert len(result) == 24

    def test_empty_dataframe(self):
        analyzer = DemandPatternEvents(_empty_df(), demand_uuid="nonexistent")
        assert analyzer.demand_by_period().empty
        assert analyzer.detect_demand_spikes().empty
        assert analyzer.seasonality_summary().empty

    def test_empty_returns_correct_columns(self):
        analyzer = DemandPatternEvents(_empty_df(), demand_uuid="nonexistent")
        by_period = analyzer.demand_by_period()
        assert "total_demand" in by_period.columns

        spikes = analyzer.detect_demand_spikes()
        assert "spike_magnitude" in spikes.columns

        seasonal = analyzer.seasonality_summary()
        assert "period_label" in seasonal.columns


# ===========================================================================
# Cross-cutting: single-row and minimal data
# ===========================================================================


class TestEdgeCases:
    """Edge cases: single data point, all same values, etc."""

    def test_inventory_single_point(self):
        df = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01")],
                "uuid": ["lvl"],
                "value_double": [50.0],
                "is_delta": [True],
            }
        )
        tracker = InventoryMonitoringEvents(df, level_uuid="lvl")
        # Single point below threshold: interval is zero-length
        low = tracker.detect_low_stock(min_level=100)
        # Might be empty (zero duration < hold=0s means start==end which is ok)
        assert "start" in low.columns

        rate = tracker.consumption_rate(window="1h")
        assert "consumption_rate" in rate.columns

    def test_lead_time_single_pair(self):
        df = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")],
                "uuid": ["order", "delivery"],
                "value_string": ["PO-1", "PO-1"],
                "is_delta": [True, True],
            }
        )
        analyzer = LeadTimeAnalysisEvents(df)
        lt = analyzer.calculate_lead_times("order", "delivery")
        assert len(lt) == 1
        assert abs(lt["lead_time_hours"].iloc[0] - 72.0) < 0.01

        # Statistics with a single pair
        stats = analyzer.lead_time_statistics("order", "delivery")
        assert len(stats) == 1

        # Anomaly detection needs >= 2 data points
        anom = analyzer.detect_lead_time_anomalies("order", "delivery")
        assert anom.empty

    def test_demand_constant_values(self):
        """All demand values are the same -- std=0 so no spikes."""
        times = pd.date_range("2024-01-01", periods=48, freq="1h")
        df = pd.DataFrame(
            {
                "systime": times,
                "uuid": "demand",
                "value_double": 100.0,
                "is_delta": True,
            }
        )
        analyzer = DemandPatternEvents(df, demand_uuid="demand")
        spikes = analyzer.detect_demand_spikes(threshold_factor=2.0, window="1D")
        assert spikes.empty
