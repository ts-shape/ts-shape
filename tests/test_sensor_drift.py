import pytest
import pandas as pd
import numpy as np
from ts_shape.events.quality.sensor_drift import SensorDriftEvents


@pytest.fixture
def drifting_sensor_df():
    """Sensor with linear drift of +0.5 per hour over 24h."""
    np.random.seed(42)
    n = 1440  # 1 sample per minute for 24h
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        drift = (i / 60) * 0.5  # 0.5 per hour
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 100.0 + drift + np.random.randn() * 0.1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sensor_with_reference_df():
    """Sensor + reference signal. Sensor drifts, reference stays constant."""
    np.random.seed(42)
    n = 480  # 8 hours at 1/min
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        drift = (i / 60) * 0.3
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 50.0 + drift + np.random.randn() * 0.05,
            }
        )
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "reference",
                "value_double": 50.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def stable_sensor_df():
    """Stable sensor with no drift."""
    np.random.seed(42)
    n = 480
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.randn() * 0.1,
            }
        )
    return pd.DataFrame(rows)


class TestDetectZeroDrift:
    def test_detects_drift_with_reference(self, sensor_with_reference_df):
        det = SensorDriftEvents(
            sensor_with_reference_df,
            "sensor_1",
            reference_uuid="reference",
        )
        result = det.detect_zero_drift(window="2h")
        assert len(result) > 0
        # Later windows should have higher offset
        offsets = result["mean_offset"].tolist()
        assert offsets[-1] > offsets[0]

    def test_detects_drift_with_float_reference(self, drifting_sensor_df):
        det = SensorDriftEvents(
            drifting_sensor_df,
            "sensor_1",
            reference_value=100.0,
        )
        result = det.detect_zero_drift(window="4h")
        assert len(result) > 0
        offsets = result["mean_offset"].tolist()
        assert offsets[-1] > offsets[0]

    def test_stable_sensor_low_severity(self, stable_sensor_df):
        det = SensorDriftEvents(
            stable_sensor_df,
            "sensor_1",
            reference_value=50.0,
        )
        result = det.detect_zero_drift(window="2h")
        assert len(result) > 0
        assert all(s == "low" for s in result["severity"].tolist())

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = SensorDriftEvents(df, "sensor_1")
        assert len(det.detect_zero_drift()) == 0


class TestDetectSpanDrift:
    def test_with_reference_uuid(self, sensor_with_reference_df):
        det = SensorDriftEvents(
            sensor_with_reference_df,
            "sensor_1",
            reference_uuid="reference",
        )
        result = det.detect_span_drift(window="2h")
        assert len(result) > 0
        # Sensitivity should increase as sensor drifts up
        assert result.iloc[-1]["sensitivity_change_pct"] > 0

    def test_with_float_reference(self, drifting_sensor_df):
        det = SensorDriftEvents(
            drifting_sensor_df,
            "sensor_1",
            reference_value=100.0,
        )
        result = det.detect_span_drift(window="4h")
        assert len(result) > 0

    def test_no_reference_returns_empty(self, stable_sensor_df):
        det = SensorDriftEvents(stable_sensor_df, "sensor_1")
        result = det.detect_span_drift()
        assert len(result) == 0


class TestDriftTrend:
    def test_increasing_trend(self, drifting_sensor_df):
        det = SensorDriftEvents(drifting_sensor_df, "sensor_1")
        result = det.drift_trend(window="4h", metric="mean")
        assert len(result) > 0
        assert result.iloc[0]["direction"] == "increasing"
        assert result.iloc[0]["slope"] > 0
        assert result.iloc[0]["r_squared"] > 0.5

    def test_stable_trend(self, stable_sensor_df):
        det = SensorDriftEvents(stable_sensor_df, "sensor_1")
        result = det.drift_trend(window="2h", metric="mean")
        assert len(result) > 0
        assert result.iloc[0]["direction"] == "stable"

    def test_std_metric(self, drifting_sensor_df):
        det = SensorDriftEvents(drifting_sensor_df, "sensor_1")
        result = det.drift_trend(window="4h", metric="std")
        assert len(result) > 0
        assert "value" in result.columns


class TestCalibrationHealth:
    def test_healthy_sensor(self, stable_sensor_df):
        det = SensorDriftEvents(
            stable_sensor_df,
            "sensor_1",
            reference_value=50.0,
        )
        result = det.calibration_health(window="2h", tolerance=1.0)
        assert len(result) > 0
        assert result.iloc[0]["health_score"] > 80

    def test_drifting_sensor_lower_health(self, sensor_with_reference_df):
        det = SensorDriftEvents(
            sensor_with_reference_df,
            "sensor_1",
            reference_uuid="reference",
        )
        result = det.calibration_health(window="2h", tolerance=1.0)
        assert len(result) > 0
        # Later windows should have lower health as drift increases
        scores = result["health_score"].tolist()
        assert scores[-1] < scores[0]

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = SensorDriftEvents(df, "sensor_1")
        assert len(det.calibration_health()) == 0
