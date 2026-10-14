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
        result = detector.classify_anomalies(window="60s", n_sigma=3.0)
        assert len(result) > 0
        types = result["anomaly_type"].tolist()
        assert "spike" in types

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        detector = AnomalyClassificationEvents(df, "sensor_1")
        result = detector.classify_anomalies()
        assert len(result) == 0


def _ramp(step_seconds: int, hours: int = 2) -> pd.DataFrame:
    """Ramp of 0.01 units per second sampled every ``step_seconds``."""
    n = hours * 3600 // step_seconds
    t = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        np.arange(n) * step_seconds, unit="s"
    )
    return pd.DataFrame(
        {
            "systime": t,
            "uuid": "sensor_1",
            "value_double": 0.01 * np.arange(n) * step_seconds,
        }
    )


class TestSlopePerSecond:
    def test_detect_drift_slope_independent_of_sampling_rate(self):
        fast = AnomalyClassificationEvents(_ramp(1), "sensor_1")
        slow = AnomalyClassificationEvents(_ramp(10), "sensor_1")
        # Per sample (default): slopes 0.01 vs 0.1 per sample.
        assert fast.detect_drift(min_slope=0.05).empty
        assert not slow.detect_drift(min_slope=0.05).empty
        # Per second: 0.01 units/s for both.
        for det in (fast, slow):
            out = det.detect_drift(min_slope=0.005, slope_per="second")
            assert not out.empty
            assert np.allclose(out["slope"], 0.01)

    def test_classify_anomalies_slope_per_second_details(self):
        # Alternating +-0.3 noise every 10 s with a 0.04/sample
        # (= 0.004/s) ramp in the 00:50 window.
        n = 720
        v = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * 0.3
        v[300:331] += np.arange(31) * 0.04
        t = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n) * 10, unit="s")
        df = pd.DataFrame({"systime": t, "uuid": "sensor_1", "value_double": v})
        det = AnomalyClassificationEvents(df, "sensor_1")
        default = det.classify_anomalies(window="5min")
        per_s = det.classify_anomalies(window="5min", slope_per="second")
        # Regular sampling: the same windows and types are flagged.
        assert list(default["anomaly_type"]) == list(per_s["anomaly_type"])
        d0 = default[default["anomaly_type"] == "drift"]["details"]
        d1 = per_s[per_s["anomaly_type"] == "drift"]["details"]
        assert list(d0) == ["slope=0.040000, direction=increasing"]
        assert list(d1) == ["slope=0.004000, direction=increasing"]

    def test_invalid_slope_per(self):
        det = AnomalyClassificationEvents(_ramp(10), "sensor_1")
        with pytest.raises(ValueError, match="slope_per"):
            det.detect_drift(slope_per="minute")
