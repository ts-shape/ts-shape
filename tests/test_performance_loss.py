"""Tests for PerformanceLossTracking module."""

import pandas as pd

from ts_shape.events.production.performance_loss import PerformanceLossTracking


def _make_cycle_df(n_cycles=20, base_cycle_time=45.0, jitter=5.0):
    """Create a DataFrame with a monotonic counter incrementing at ~base_cycle_time intervals."""
    import numpy as np

    rng = np.random.RandomState(42)
    times = []
    t = pd.Timestamp("2024-01-01 08:00:00")
    for _ in range(n_cycles):
        times.append(t)
        t += pd.Timedelta(seconds=base_cycle_time + rng.uniform(-jitter, jitter))

    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": "cycle_counter",
            "value_integer": list(range(100, 100 + n_cycles)),
        }
    )
    return df


class TestPerformanceLossTracking:

    def test_performance_by_shift_basic(self):
        df = _make_cycle_df(n_cycles=20, base_cycle_time=45.0)
        tracker = PerformanceLossTracking(df)
        result = tracker.performance_by_shift(
            cycle_uuid="cycle_counter",
            target_cycle_time=45.0,
        )
        assert not result.empty
        assert "performance_pct" in result.columns
        assert "loss_minutes" in result.columns
        # Performance should be close to 100% since avg cycle time ≈ target
        assert result["performance_pct"].iloc[0] > 80

    def test_performance_by_shift_slow(self):
        # Cycles 2x slower than target → ~50% performance
        df = _make_cycle_df(n_cycles=20, base_cycle_time=90.0, jitter=2.0)
        tracker = PerformanceLossTracking(df)
        result = tracker.performance_by_shift(
            cycle_uuid="cycle_counter",
            target_cycle_time=45.0,
        )
        assert not result.empty
        assert result["performance_pct"].iloc[0] < 60

    def test_slow_periods(self):
        df = _make_cycle_df(n_cycles=50, base_cycle_time=60.0, jitter=3.0)
        tracker = PerformanceLossTracking(df)
        result = tracker.slow_periods(
            cycle_uuid="cycle_counter",
            target_cycle_time=45.0,
            threshold_pct=90.0,
            window="30min",
        )
        # With 60s actual vs 45s target, performance ≈ 75%, all windows should be flagged
        assert not result.empty
        assert all(result["performance_pct"] < 90)

    def test_performance_trend(self):
        df = _make_cycle_df(n_cycles=30, base_cycle_time=50.0)
        tracker = PerformanceLossTracking(df)
        result = tracker.performance_trend(
            cycle_uuid="cycle_counter",
            target_cycle_time=45.0,
            window="1D",
        )
        assert not result.empty
        assert "period" in result.columns
        assert "performance_pct" in result.columns

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer"])
        tracker = PerformanceLossTracking(df)
        result = tracker.performance_by_shift(
            cycle_uuid="nonexistent",
            target_cycle_time=45.0,
        )
        assert result.empty

    def test_slow_periods_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer"])
        tracker = PerformanceLossTracking(df)
        result = tracker.slow_periods(
            cycle_uuid="nonexistent",
            target_cycle_time=45.0,
        )
        assert result.empty
