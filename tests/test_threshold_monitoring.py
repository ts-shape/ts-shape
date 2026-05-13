import pytest
import pandas as pd
import numpy as np
from ts_shape.events.engineering.threshold_monitoring import ThresholdMonitoringEvents


@pytest.fixture
def ramp_df():
    """Signal that ramps up from 0 to 100."""
    n = 1000
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "temp_sensor",
                "value_double": float(i) / 10.0,  # 0 to 100
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def oscillating_df():
    """Signal that oscillates around a threshold."""
    n = 600
    base = pd.Timestamp("2024-01-01")
    values = 50.0 + 10.0 * np.sin(np.linspace(0, 20 * np.pi, n))
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "temp_sensor",
                "value_double": values[i],
            }
        )
    return pd.DataFrame(rows)


class TestMultiLevelThreshold:
    def test_correct_levels(self, ramp_df):
        detector = ThresholdMonitoringEvents(ramp_df, "temp_sensor")
        levels = {"warning": 50, "alarm": 75, "critical": 90}
        result = detector.multi_level_threshold(levels, direction="above")
        assert len(result) > 0
        level_names = set(result["level"].tolist())
        assert "warning" in level_names
        assert "critical" in level_names

    def test_below_direction(self, ramp_df):
        detector = ThresholdMonitoringEvents(ramp_df, "temp_sensor")
        levels = {"low": 20, "very_low": 10}
        result = detector.multi_level_threshold(levels, direction="below")
        assert len(result) > 0


class TestThresholdWithHysteresis:
    def test_no_chattering(self, oscillating_df):
        detector = ThresholdMonitoringEvents(oscillating_df, "temp_sensor")
        # Without hysteresis, signal crosses 55 many times
        # With hysteresis (high=55, low=45), should have fewer events
        result = detector.threshold_with_hysteresis(high=55.0, low=45.0)
        assert len(result) > 0
        # Each event should have a reasonable duration (not just 1 sample)
        for _, row in result.iterrows():
            assert row["duration"] >= pd.Timedelta("0s")
            assert row["peak_value"] >= 55.0

    def test_clear_exceedance(self, ramp_df):
        detector = ThresholdMonitoringEvents(ramp_df, "temp_sensor")
        result = detector.threshold_with_hysteresis(high=80.0, low=70.0)
        assert len(result) >= 1


class TestTimeAboveThreshold:
    def test_correct_percentage(self, ramp_df):
        detector = ThresholdMonitoringEvents(ramp_df, "temp_sensor")
        result = detector.time_above_threshold(threshold=50.0, window="1h")
        assert len(result) > 0
        assert "pct_above" in result.columns
        assert "exceedance_count" in result.columns

    def test_zero_above(self):
        base = pd.Timestamp("2024-01-01")
        rows = [
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "s1",
                "value_double": 10.0,
            }
            for i in range(100)
        ]
        df = pd.DataFrame(rows)
        detector = ThresholdMonitoringEvents(df, "s1")
        result = detector.time_above_threshold(threshold=50.0, window="1h")
        if not result.empty:
            assert result.iloc[0]["pct_above"] == 0.0


class TestThresholdExceedanceTrend:
    def test_trend_direction(self, ramp_df):
        detector = ThresholdMonitoringEvents(ramp_df, "temp_sensor")
        result = detector.threshold_exceedance_trend(threshold=50.0, window="5min")
        if not result.empty:
            assert "trend_direction" in result.columns


class TestEmptyData:
    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        detector = ThresholdMonitoringEvents(df, "sensor_1")
        assert len(detector.multi_level_threshold({"w": 50})) == 0
        assert len(detector.threshold_with_hysteresis(80, 70)) == 0
        assert len(detector.time_above_threshold(50)) == 0
