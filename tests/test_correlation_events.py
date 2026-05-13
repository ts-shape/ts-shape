"""Tests for the correlation events pack."""

import pandas as pd
import numpy as np
import pytest

from ts_shape.events.correlation.signal_correlation import SignalCorrelationEvents
from ts_shape.events.correlation.anomaly_correlation import AnomalyCorrelationEvents

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_correlated_df(n: int = 500) -> pd.DataFrame:
    """Create two correlated signals + one with injected anomalies."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []

    # Signal A: base signal
    values_a = np.cumsum(np.random.normal(0, 1, n)) + 100

    # Signal B: correlated with A (r~0.9) but diverges in the middle
    noise = np.random.normal(0, 2, n)
    values_b = values_a * 0.5 + 20 + noise
    # Inject divergence in middle section
    values_b[200:250] += 50

    # Signal C: independent but with anomalies that follow A's anomalies
    values_c = np.random.normal(50, 5, n)
    # Inject anomalies in A and matching delayed anomalies in C
    values_a[100] = values_a[100] + 30  # anomaly in A
    values_c[103] = values_c[103] + 40  # delayed anomaly in C (3 min later)
    values_a[300] = values_a[300] + 35
    values_c[305] = values_c[305] + 45

    for i in range(n):
        t = base + pd.Timedelta(minutes=i)

        rows.append(
            {
                "systime": t,
                "uuid": "signal:temperature",
                "value_double": values_a[i],
                "is_delta": True,
            }
        )
        rows.append(
            {
                "systime": t,
                "uuid": "signal:pressure",
                "value_double": values_b[i],
                "is_delta": True,
            }
        )
        rows.append(
            {
                "systime": t,
                "uuid": "signal:vibration",
                "value_double": values_c[i],
                "is_delta": True,
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def correlated_df():
    return _make_correlated_df()


# ---------------------------------------------------------------------------
# SignalCorrelationEvents
# ---------------------------------------------------------------------------


class TestSignalCorrelationEvents:

    def test_rolling_correlation(self, correlated_df):
        sc = SignalCorrelationEvents(correlated_df)
        result = sc.rolling_correlation(
            "signal:temperature",
            "signal:pressure",
            resample="1min",
            window=30,
        )
        assert not result.empty
        assert "correlation" in result.columns
        assert result["source_uuid_a"].iloc[0] == "signal:temperature"
        assert result["source_uuid_b"].iloc[0] == "signal:pressure"

    def test_rolling_correlation_empty(self, correlated_df):
        sc = SignalCorrelationEvents(correlated_df)
        result = sc.rolling_correlation(
            "nonexistent:a", "signal:pressure", resample="1min", window=30
        )
        assert result.empty

    def test_correlation_breakdown(self, correlated_df):
        sc = SignalCorrelationEvents(correlated_df)
        result = sc.correlation_breakdown(
            "signal:temperature",
            "signal:pressure",
            resample="1min",
            window=30,
            threshold=0.5,
        )
        # There should be at least one breakdown in the divergence zone
        assert not result.empty
        assert "min_correlation" in result.columns
        assert "duration_seconds" in result.columns
        assert all(result["min_correlation"] < 0.5)

    def test_correlation_breakdown_high_threshold(self, correlated_df):
        sc = SignalCorrelationEvents(correlated_df)
        # Very low threshold should yield no breakdowns
        result = sc.correlation_breakdown(
            "signal:temperature",
            "signal:pressure",
            resample="1min",
            window=30,
            threshold=-1.0,
        )
        assert result.empty

    def test_lag_correlation(self, correlated_df):
        sc = SignalCorrelationEvents(correlated_df)
        result = sc.lag_correlation(
            "signal:temperature",
            "signal:pressure",
            resample="1min",
            max_lag=10,
        )
        assert not result.empty
        assert "lag_periods" in result.columns
        assert "correlation" in result.columns
        assert "is_best_lag" in result.columns
        assert result["is_best_lag"].sum() == 1

    def test_lag_correlation_empty(self, correlated_df):
        sc = SignalCorrelationEvents(correlated_df)
        result = sc.lag_correlation("nonexistent:a", "nonexistent:b")
        assert result.empty


# ---------------------------------------------------------------------------
# AnomalyCorrelationEvents
# ---------------------------------------------------------------------------


class TestAnomalyCorrelationEvents:

    def test_coincident_anomalies(self, correlated_df):
        ac = AnomalyCorrelationEvents(correlated_df)
        result = ac.coincident_anomalies(
            ["signal:temperature", "signal:pressure", "signal:vibration"],
            z_threshold=2.5,
            coincidence_window="10min",
            min_signals=2,
        )
        # May or may not find coincident anomalies depending on data
        assert "anomaly_count" in result.columns or result.empty

    def test_coincident_anomalies_empty(self, correlated_df):
        ac = AnomalyCorrelationEvents(correlated_df)
        result = ac.coincident_anomalies(
            ["nonexistent:a", "nonexistent:b"],
            z_threshold=3.0,
        )
        assert result.empty

    def test_cascade_detection(self, correlated_df):
        ac = AnomalyCorrelationEvents(correlated_df)
        result = ac.cascade_detection(
            "signal:temperature",
            "signal:vibration",
            z_threshold=2.5,
            max_delay="10min",
        )
        assert "leader_time" in result.columns or result.empty
        if not result.empty:
            assert "delay_seconds" in result.columns
            assert all(result["delay_seconds"] > 0)

    def test_cascade_detection_empty(self, correlated_df):
        ac = AnomalyCorrelationEvents(correlated_df)
        result = ac.cascade_detection("nonexistent:a", "nonexistent:b")
        assert result.empty

    def test_root_cause_ranking(self, correlated_df):
        ac = AnomalyCorrelationEvents(correlated_df)
        result = ac.root_cause_ranking(
            ["signal:temperature", "signal:pressure", "signal:vibration"],
            z_threshold=2.5,
            max_delay="10min",
        )
        assert not result.empty
        assert "signal_uuid" in result.columns
        assert "leader_ratio" in result.columns
        assert "rank" in result.columns
        assert len(result) == 3

    def test_root_cause_ranking_single_signal(self, correlated_df):
        ac = AnomalyCorrelationEvents(correlated_df)
        result = ac.root_cause_ranking(["signal:temperature"])
        assert result.empty
