import pytest
import pandas as pd
import numpy as np
from ts_shape.events.quality.capability_trending import CapabilityTrendingEvents


@pytest.fixture
def stable_process_df():
    """Process well within spec limits (USL=110, LSL=90, target=100)."""
    np.random.seed(42)
    n = 500
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 100 + np.random.randn() * 1.5,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def degrading_process_df():
    """Process with increasing variance over time (capability degrades)."""
    np.random.seed(42)
    n = 600
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        # Variance increases linearly
        std = 1.0 + (i / n) * 5.0
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 100 + np.random.randn() * std,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def uuid_spec_df():
    """Process data with spec limits provided as UUID signals."""
    np.random.seed(42)
    n = 200
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 100 + np.random.randn() * 2.0,
            }
        )
    # Add spec limit signals
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "upper_spec",
                "value_double": 110.0,
            }
        )
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "lower_spec",
                "value_double": 90.0,
            }
        )
    return pd.DataFrame(rows)


class TestCapabilityOverTime:
    def test_stable_process(self, stable_process_df):
        ct = CapabilityTrendingEvents(
            stable_process_df,
            "sensor_1",
            upper_spec=110.0,
            lower_spec=90.0,
        )
        result = ct.capability_over_time(window="4h")
        assert len(result) > 0
        # Stable process should have high Cpk
        assert result.iloc[0]["cpk"] > 1.0
        assert result.iloc[0]["n_samples"] > 0
        assert "cp" in result.columns

    def test_with_uuid_specs(self, uuid_spec_df):
        ct = CapabilityTrendingEvents(
            uuid_spec_df,
            "sensor_1",
            upper_spec_uuid="upper_spec",
            lower_spec_uuid="lower_spec",
        )
        result = ct.capability_over_time(window="4h")
        assert len(result) > 0
        assert result.iloc[0]["cpk"] > 0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        ct = CapabilityTrendingEvents(
            df,
            "sensor_1",
            upper_spec=110.0,
            lower_spec=90.0,
        )
        result = ct.capability_over_time()
        assert len(result) == 0


class TestDetectCapabilityDrop:
    def test_degrading_process(self, degrading_process_df):
        ct = CapabilityTrendingEvents(
            degrading_process_df,
            "sensor_1",
            upper_spec=110.0,
            lower_spec=90.0,
        )
        result = ct.detect_capability_drop(window="2h", min_cpk=1.33)
        assert len(result) > 0
        # Later windows should trigger alerts
        alerts = result[result["alert"] == True]
        assert len(alerts) > 0

    def test_stable_no_alerts(self, stable_process_df):
        ct = CapabilityTrendingEvents(
            stable_process_df,
            "sensor_1",
            upper_spec=110.0,
            lower_spec=90.0,
        )
        result = ct.detect_capability_drop(window="4h", min_cpk=0.5)
        if len(result) > 0:
            # A very stable process with low threshold should have no alerts
            alerts = result[result["alert"] == True]
            assert len(alerts) == 0


class TestCapabilityForecast:
    def test_degrading_forecast(self, degrading_process_df):
        ct = CapabilityTrendingEvents(
            degrading_process_df,
            "sensor_1",
            upper_spec=110.0,
            lower_spec=90.0,
        )
        result = ct.capability_forecast(window="2h", horizon=5)
        assert len(result) > 0
        assert "trend_slope" in result.columns
        assert "forecast_cpk" in result.columns
        # Degrading process should have negative slope
        assert result.iloc[0]["trend_slope"] < 0

    def test_too_few_windows(self):
        base = pd.Timestamp("2024-01-01")
        rows = [
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "s1",
                "value_double": 100.0 + i * 0.01,
            }
            for i in range(10)
        ]
        df = pd.DataFrame(rows)
        ct = CapabilityTrendingEvents(df, "s1", upper_spec=110.0, lower_spec=90.0)
        result = ct.capability_forecast(window="8h")
        assert len(result) <= 1


class TestYieldEstimate:
    def test_high_yield(self, stable_process_df):
        ct = CapabilityTrendingEvents(
            stable_process_df,
            "sensor_1",
            upper_spec=110.0,
            lower_spec=90.0,
        )
        result = ct.yield_estimate(window="4h")
        assert len(result) > 0
        assert result.iloc[0]["estimated_yield_pct"] > 99.0
        assert result.iloc[0]["dpmo"] < 10000
        assert result.iloc[0]["sigma_level"] > 3.0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        ct = CapabilityTrendingEvents(
            df,
            "sensor_1",
            upper_spec=110.0,
            lower_spec=90.0,
        )
        result = ct.yield_estimate()
        assert len(result) == 0


class TestConstructorValidation:
    def test_missing_upper_spec_raises(self):
        df = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01")],
                "uuid": ["s1"],
                "value_double": [1.0],
            }
        )
        with pytest.raises(ValueError, match="upper_spec"):
            CapabilityTrendingEvents(df, "s1", lower_spec=0.0)

    def test_missing_lower_spec_raises(self):
        df = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01")],
                "uuid": ["s1"],
                "value_double": [1.0],
            }
        )
        with pytest.raises(ValueError, match="lower_spec"):
            CapabilityTrendingEvents(df, "s1", upper_spec=10.0)
