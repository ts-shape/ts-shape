import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.anomaly_classification import AnomalyClassificationEvents


@pytest.fixture
def spike_df():
    """Normal signal with injected spikes."""
    np.random.seed(42)
    n = 600
    base = pd.Timestamp("2024-01-01")
    times = [base + pd.Timedelta(seconds=i) for i in range(n)]
    values = np.random.randn(n) * 0.1
    # Inject spikes at positions 100 and 300
    values[100] = 50.0
    values[300] = -50.0
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": "sensor_1",
            "value_double": values,
        }
    )


@pytest.fixture
def flatline_df():
    """Signal with a flatline region."""
    n = 300
    base = pd.Timestamp("2024-01-01")
    times = [base + pd.Timedelta(seconds=i) for i in range(n)]
    values = np.random.randn(n) * 0.5
    # Inject flatline from 100 to 200
    values[100:200] = 5.0
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": "sensor_1",
            "value_double": values,
        }
    )


@pytest.fixture
def oscillation_df():
    """Signal with high-frequency oscillation."""
    n = 300
    base = pd.Timestamp("2024-01-01")
    times = [base + pd.Timedelta(seconds=i) for i in range(n)]
    values = np.sin(np.linspace(0, 100 * np.pi, n))  # Very rapid oscillation
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": "sensor_1",
            "value_double": values,
        }
    )


@pytest.fixture
def drift_df():
    """Signal with a sustained upward drift."""
    n = 600
    base = pd.Timestamp("2024-01-01")
    times = [base + pd.Timedelta(seconds=i) for i in range(n)]
    values = np.linspace(0, 10, n) + np.random.randn(n) * 0.01
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": "sensor_1",
            "value_double": values,
        }
    )


class TestDetectFlatline:
    def test_finds_flatline(self, flatline_df):
        detector = AnomalyClassificationEvents(flatline_df, "sensor_1")
        result = detector.detect_flatline(min_duration="10s")
        assert len(result) > 0
        assert result.iloc[0]["stuck_value"] == 5.0

    def test_no_flatline_in_noise(self, spike_df):
        detector = AnomalyClassificationEvents(spike_df, "sensor_1")
        result = detector.detect_flatline(min_duration="10s")
        assert len(result) == 0


class TestDetectOscillation:
    def test_finds_oscillation(self, oscillation_df):
        detector = AnomalyClassificationEvents(oscillation_df, "sensor_1")
        result = detector.detect_oscillation(window="60s", min_crossings=5)
        assert len(result) > 0
        assert result.iloc[0]["crossing_count"] >= 5


class TestDetectDrift:
    def test_finds_drift(self, drift_df):
        detector = AnomalyClassificationEvents(drift_df, "sensor_1")
        result = detector.detect_drift(window="300s", min_slope=0.001)
        assert len(result) > 0
        assert result.iloc[0]["direction"] == "increasing"


class TestClassifyAnomalies:
    def test_classifies_spikes(self, spike_df):
        detector = AnomalyClassificationEvents(spike_df, "sensor_1")
        result = detector.classify_anomalies(window="60s", z_threshold=3.0)
        assert len(result) > 0
        types = result["anomaly_type"].tolist()
        assert "spike" in types

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        detector = AnomalyClassificationEvents(df, "sensor_1")
        result = detector.classify_anomalies()
        assert len(result) == 0
