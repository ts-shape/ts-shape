import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.calibration_validation import CalibrationValidationEvents


@pytest.fixture
def drifting_sensor_df():
    """Sensor drifting +0.2 per hour from reference value of 100."""
    np.random.seed(42)
    n = 720  # 12 hours at 1/min
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        drift = (i / 60) * 0.2
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 100.0 + drift + np.random.randn() * 0.02,
            }
        )
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "reference",
                "value_double": 100.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def stable_sensor_df():
    """Stable sensor near reference value."""
    np.random.seed(42)
    n = 480
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 100.0 + np.random.randn() * 0.01,
            }
        )
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "reference",
                "value_double": 100.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def calibration_points():
    """Multi-point calibration data with slight nonlinearity at 75."""
    return {
        0.0: [0.01, -0.02, 0.00, 0.01, -0.01],
        25.0: [25.05, 24.98, 25.02, 25.01, 25.03],
        50.0: [50.02, 49.99, 50.01, 50.00, 49.98],
        75.0: [75.30, 75.28, 75.32, 75.29, 75.31],  # +0.3 bias here
        100.0: [100.01, 99.98, 100.02, 100.00, 99.99],
    }


class TestLinearityAssessment:
    def test_detects_nonlinearity(self, calibration_points):
        result = CalibrationValidationEvents.linearity_assessment(calibration_points)
        assert len(result) == 5
        # The 75.0 point should have the largest linearity error
        row_75 = result[result["reference"] == 75.0].iloc[0]
        assert abs(row_75["linearity_error"]) > abs(result.iloc[0]["linearity_error"])
        assert abs(row_75["bias"]) > 0.2

    def test_perfect_linearity(self):
        points = {0.0: [0.0], 50.0: [50.0], 100.0: [100.0]}
        result = CalibrationValidationEvents.linearity_assessment(points)
        assert len(result) == 3
        for _, row in result.iterrows():
            assert abs(row["linearity_error"]) < 1e-6

    def test_empty(self):
        result = CalibrationValidationEvents.linearity_assessment({})
        assert len(result) == 0


class TestCalibrationCurve:
    def test_linear_fit(self, calibration_points):
        result = CalibrationValidationEvents.calibration_curve(
            calibration_points, degree=1
        )
        assert len(result) == 2  # 2 coefficients for linear
        assert result.iloc[0]["r_squared"] > 0.99

    def test_quadratic_fit(self, calibration_points):
        result = CalibrationValidationEvents.calibration_curve(
            calibration_points, degree=2
        )
        assert len(result) == 3  # 3 coefficients for quadratic

    def test_insufficient_points(self):
        result = CalibrationValidationEvents.calibration_curve({10.0: [10.0]})
        assert len(result) == 0


class TestDetectRecalibrationNeed:
    def test_drifting_triggers_recal(self, drifting_sensor_df):
        cv = CalibrationValidationEvents(
            drifting_sensor_df,
            "sensor_1",
            reference_uuid="reference",
        )
        result = cv.detect_recalibration_need(
            window="2h",
            max_bias=0.5,
            max_drift_rate=0.3,
        )
        assert len(result) > 0
        # Later windows should trigger recalibration
        recals = result[result["recalibration_needed"].astype(bool)]
        assert len(recals) > 0
        assert any("bias" in r for r in recals["reason"].tolist())

    def test_stable_no_recal(self, stable_sensor_df):
        cv = CalibrationValidationEvents(
            stable_sensor_df,
            "sensor_1",
            reference_uuid="reference",
        )
        result = cv.detect_recalibration_need(
            window="2h",
            max_bias=1.0,
            max_drift_rate=0.5,
        )
        if len(result) > 0:
            recals = result[result["recalibration_needed"].astype(bool)]
            assert len(recals) == 0

    def test_with_float_reference(self, drifting_sensor_df):
        # Use only sensor data with a float reference
        sensor_only = drifting_sensor_df[
            drifting_sensor_df["uuid"] == "sensor_1"
        ].copy()
        cv = CalibrationValidationEvents(
            sensor_only,
            "sensor_1",
            reference_value=100.0,
        )
        result = cv.detect_recalibration_need(window="2h", max_bias=0.5)
        assert len(result) > 0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        cv = CalibrationValidationEvents(df, "sensor_1", reference_value=100.0)
        assert len(cv.detect_recalibration_need()) == 0


class TestCalibrationStability:
    def test_stable_high_score(self, stable_sensor_df):
        cv = CalibrationValidationEvents(
            stable_sensor_df,
            "sensor_1",
            reference_uuid="reference",
        )
        result = cv.calibration_stability(window="2h")
        assert len(result) > 0
        assert result.iloc[0]["stability_score"] > 80

    def test_drifting_low_score(self, drifting_sensor_df):
        cv = CalibrationValidationEvents(
            drifting_sensor_df,
            "sensor_1",
            reference_uuid="reference",
        )
        result = cv.calibration_stability(window="2h")
        assert len(result) > 0
        # Later windows should have lower stability
        scores = result["stability_score"].tolist()
        assert scores[-1] < scores[0]

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        cv = CalibrationValidationEvents(df, "sensor_1", reference_value=100.0)
        assert len(cv.calibration_stability()) == 0
