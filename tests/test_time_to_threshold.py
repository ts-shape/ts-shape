import pandas as pd
import pytest

from ts_shape.events.engineering.time_to_threshold import TimeToThresholdEvents


@pytest.fixture
def rising_df():
    """Linearly rising signal: 0 to 100 over 1000 seconds."""
    base = pd.Timestamp("2024-01-01")
    rows = [
        {
            "systime": base + pd.Timedelta(seconds=i),
            "uuid": "sensor_1",
            "value_double": float(i) * 0.1,  # 0 to 100 over 1000s
        }
        for i in range(1001)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def falling_df():
    """Linearly falling signal: 100 to 0 over 1000 seconds."""
    base = pd.Timestamp("2024-01-01")
    rows = [
        {
            "systime": base + pd.Timedelta(seconds=i),
            "uuid": "sensor_1",
            "value_double": 100.0 - float(i) * 0.1,
        }
        for i in range(1001)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def flat_df():
    """Flat signal at 50."""
    base = pd.Timestamp("2024-01-01")
    rows = [
        {
            "systime": base + pd.Timedelta(seconds=i),
            "uuid": "sensor_1",
            "value_double": 50.0,
        }
        for i in range(500)
    ]
    return pd.DataFrame(rows)


class TestTimeToThreshold:
    def test_rising_toward_threshold(self, rising_df):
        det = TimeToThresholdEvents(rising_df, "sensor_1")
        result = det.time_to_threshold(threshold=150.0, lookback="1000s")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["slope_per_second"] > 0
        assert row["estimated_seconds"] is not None
        assert row["estimated_seconds"] > 0
        assert row["direction"] == "rising"
        # At 0.1/s, current value ~100, need 50 more → ~500s
        assert 400 < row["estimated_seconds"] < 600

    def test_falling_toward_threshold(self, falling_df):
        det = TimeToThresholdEvents(falling_df, "sensor_1")
        result = det.time_to_threshold(threshold=-50.0, lookback="1000s")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["slope_per_second"] < 0
        assert row["estimated_seconds"] is not None
        assert row["estimated_seconds"] > 0

    def test_moving_away_from_threshold(self, rising_df):
        det = TimeToThresholdEvents(rising_df, "sensor_1")
        # Threshold below current value, signal is rising
        result = det.time_to_threshold(threshold=-10.0, lookback="1000s")
        assert len(result) == 1
        # Should indicate not reachable
        assert result.iloc[0]["estimated_seconds"] is None

    def test_flat_signal(self, flat_df):
        det = TimeToThresholdEvents(flat_df, "sensor_1")
        result = det.time_to_threshold(threshold=100.0, lookback="500s")
        assert len(result) == 1
        assert result.iloc[0]["direction"] == "flat"
        assert result.iloc[0]["estimated_seconds"] is None

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = TimeToThresholdEvents(df, "sensor_1")
        assert len(det.time_to_threshold(threshold=100)) == 0


class TestTimeToThresholdWindows:
    def test_windowed_estimates(self, rising_df):
        det = TimeToThresholdEvents(rising_df, "sensor_1")
        result = det.time_to_threshold_windows(
            threshold=150.0, window="500s", lookback="500s"
        )
        assert len(result) > 0
        assert "estimated_seconds" in result.columns
        assert "confidence" in result.columns

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = TimeToThresholdEvents(df, "sensor_1")
        assert len(det.time_to_threshold_windows(threshold=100)) == 0


class TestRemainingUsefulRange:
    def test_rising_toward_upper(self, rising_df):
        det = TimeToThresholdEvents(rising_df, "sensor_1")
        result = det.remaining_useful_range(
            lower_limit=0, upper_limit=150, lookback="1000s"
        )
        assert len(result) == 1
        row = result.iloc[0]
        assert row["nearest_bound"] == "upper"
        assert row["estimated_seconds"] > 0

    def test_falling_toward_lower(self, falling_df):
        det = TimeToThresholdEvents(falling_df, "sensor_1")
        result = det.remaining_useful_range(
            lower_limit=-50, upper_limit=200, lookback="1000s"
        )
        assert len(result) == 1
        row = result.iloc[0]
        assert row["nearest_bound"] == "lower"
        assert row["estimated_seconds"] > 0

    def test_flat_signal(self, flat_df):
        det = TimeToThresholdEvents(flat_df, "sensor_1")
        result = det.remaining_useful_range(
            lower_limit=0, upper_limit=100, lookback="500s"
        )
        assert len(result) == 1
        assert result.iloc[0]["nearest_bound"] == "none"

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = TimeToThresholdEvents(df, "sensor_1")
        assert len(det.remaining_useful_range(0, 100)) == 0


class TestCrossingForecast:
    def test_multiple_thresholds(self, rising_df):
        det = TimeToThresholdEvents(rising_df, "sensor_1")
        result = det.crossing_forecast(
            thresholds={"warning": 120, "alarm": 150, "critical": 200},
            lookback="1000s",
        )
        assert len(result) == 3
        assert all(result["reachable"])
        # Warning should be reached sooner than alarm
        warning_secs = result[result["threshold_name"] == "warning"][
            "estimated_seconds"
        ].iloc[0]
        alarm_secs = result[result["threshold_name"] == "alarm"][
            "estimated_seconds"
        ].iloc[0]
        assert warning_secs < alarm_secs

    def test_unreachable_thresholds(self, rising_df):
        det = TimeToThresholdEvents(rising_df, "sensor_1")
        # Signal is rising, these are below current value
        result = det.crossing_forecast(
            thresholds={"floor": -10},
            lookback="1000s",
        )
        assert len(result) == 1
        assert not result.iloc[0]["reachable"]

    def test_flat_signal(self, flat_df):
        det = TimeToThresholdEvents(flat_df, "sensor_1")
        result = det.crossing_forecast(
            thresholds={"target": 100},
            lookback="500s",
        )
        assert len(result) == 1
        assert not result.iloc[0]["reachable"]

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = TimeToThresholdEvents(df, "sensor_1")
        assert len(det.crossing_forecast(thresholds={"x": 100})) == 0
