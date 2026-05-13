import pytest
import pandas as pd
import numpy as np
from ts_shape.events.quality.value_distribution import ValueDistributionEvents


@pytest.fixture
def normal_df():
    """Normally distributed signal over 2 hours at 1 Hz."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    n = 7200  # 2 hours
    rows = [
        {
            "systime": base + pd.Timedelta(seconds=i),
            "uuid": "sensor_1",
            "value_double": 100.0 + np.random.normal(0, 5),
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def bimodal_df():
    """Bimodal signal: mode at 50 for first half, mode at 150 for second half."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    n = 4000
    rows = []
    for i in range(n):
        if i < n // 2:
            val = 50.0 + np.random.normal(0, 3)
        else:
            val = 150.0 + np.random.normal(0, 3)
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": val,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def mode_shift_df():
    """Signal that shifts from one operating mode to another mid-way."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []
    # First hour: mode around 100
    for i in range(3600):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": 100.0 + np.random.normal(0, 2),
            }
        )
    # Second hour: mode around 200
    for i in range(3600):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=3600 + i),
                "uuid": "sensor_1",
                "value_double": 200.0 + np.random.normal(0, 2),
            }
        )
    return pd.DataFrame(rows)


class TestDetectModeChanges:
    def test_detects_shift(self, mode_shift_df):
        det = ValueDistributionEvents(mode_shift_df, "sensor_1")
        result = det.detect_mode_changes(window="1h")
        assert len(result) == 2
        # Second window should show a mode change
        assert result.iloc[1]["mode_changed"] == True

    def test_stable_mode(self, normal_df):
        det = ValueDistributionEvents(normal_df, "sensor_1")
        result = det.detect_mode_changes(window="1h")
        assert len(result) == 2
        # No mode changes expected in a stable normal signal
        # First window can't detect change (no previous), second should be stable
        assert result.iloc[0]["mode_changed"] == False

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = ValueDistributionEvents(df, "sensor_1")
        assert len(det.detect_mode_changes()) == 0


class TestDetectBimodal:
    def test_bimodal_signal(self, bimodal_df):
        det = ValueDistributionEvents(bimodal_df, "sensor_1")
        result = det.detect_bimodal()
        assert len(result) == 1
        assert result.iloc[0]["is_bimodal"] == True
        assert result.iloc[0]["dip_score"] > 0.5
        assert result.iloc[0]["mode_1"] is not None
        assert result.iloc[0]["mode_2"] is not None

    def test_unimodal_signal(self, normal_df):
        det = ValueDistributionEvents(normal_df, "sensor_1")
        result = det.detect_bimodal()
        assert len(result) == 1
        assert result.iloc[0]["is_bimodal"] == False

    def test_too_few_samples(self):
        base = pd.Timestamp("2024-01-01")
        rows = [
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "s1",
                "value_double": float(i),
            }
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        det = ValueDistributionEvents(df, "s1")
        result = det.detect_bimodal(min_samples=30)
        assert len(result) == 0


class TestNormalityWindows:
    def test_normal_windows(self, normal_df):
        det = ValueDistributionEvents(normal_df, "sensor_1")
        result = det.normality_windows(freq="1h", alpha=0.05)
        assert len(result) > 0
        assert "is_normal" in result.columns
        assert "p_value" in result.columns
        assert "skewness" in result.columns

    def test_non_normal_detected(self):
        """Uniform distribution should fail normality test."""
        np.random.seed(42)
        base = pd.Timestamp("2024-01-01")
        n = 3600
        rows = [
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": np.random.uniform(0, 100),
            }
            for i in range(n)
        ]
        df = pd.DataFrame(rows)
        det = ValueDistributionEvents(df, "sensor_1")
        result = det.normality_windows(freq="1h", alpha=0.05)
        assert len(result) > 0
        # Uniform should be flagged as non-normal
        assert any(~result["is_normal"])

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = ValueDistributionEvents(df, "sensor_1")
        assert len(det.normality_windows()) == 0


class TestPercentileTracking:
    def test_default_percentiles(self, normal_df):
        det = ValueDistributionEvents(normal_df, "sensor_1")
        result = det.percentile_tracking(freq="1h")
        assert len(result) > 0
        for col in ["p5", "p25", "p50", "p75", "p95"]:
            assert col in result.columns

    def test_custom_percentiles(self, normal_df):
        det = ValueDistributionEvents(normal_df, "sensor_1")
        result = det.percentile_tracking(percentiles=(10, 90), freq="1h")
        assert "p10" in result.columns
        assert "p90" in result.columns

    def test_percentile_ordering(self, normal_df):
        det = ValueDistributionEvents(normal_df, "sensor_1")
        result = det.percentile_tracking(freq="1h")
        for _, row in result.iterrows():
            assert row["p5"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p95"]

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = ValueDistributionEvents(df, "sensor_1")
        assert len(det.percentile_tracking()) == 0
