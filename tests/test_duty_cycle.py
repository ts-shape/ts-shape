import pytest
import pandas as pd
from ts_shape.events.production.duty_cycle import DutyCycleEvents


@pytest.fixture
def fifty_pct_df():
    """50% duty cycle: alternating 10s on, 10s off."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(3600):  # 1 hour
        state = (i % 20) < 10  # 10s on, 10s off
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "motor_1",
                "value_bool": state,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def high_cycling_df():
    """Rapidly cycling signal: 1s on, 1s off."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(3600):
        state = (i % 2) == 0
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "motor_1",
                "value_bool": state,
            }
        )
    return pd.DataFrame(rows)


class TestDutyCyclePerWindow:
    def test_fifty_percent(self, fifty_pct_df):
        detector = DutyCycleEvents(fifty_pct_df, "motor_1")
        result = detector.duty_cycle_per_window(window="1h")
        assert len(result) > 0
        assert abs(result.iloc[0]["duty_cycle_pct"] - 50.0) < 5

    def test_columns(self, fifty_pct_df):
        detector = DutyCycleEvents(fifty_pct_df, "motor_1")
        result = detector.duty_cycle_per_window(window="1h")
        assert "on_time" in result.columns
        assert "off_time" in result.columns
        assert "duty_cycle_pct" in result.columns


class TestOnOffIntervals:
    def test_interval_count(self, fifty_pct_df):
        detector = DutyCycleEvents(fifty_pct_df, "motor_1")
        intervals = detector.on_off_intervals()
        assert len(intervals) > 0
        states = intervals["state"].unique()
        assert "on" in states
        assert "off" in states

    def test_interval_durations(self, fifty_pct_df):
        detector = DutyCycleEvents(fifty_pct_df, "motor_1")
        intervals = detector.on_off_intervals()
        # Most intervals should be about 10s (with possible edge effects)
        median_dur = intervals["duration"].median()
        assert median_dur >= pd.Timedelta("5s")


class TestCycleCount:
    def test_transition_count(self, fifty_pct_df):
        detector = DutyCycleEvents(fifty_pct_df, "motor_1")
        counts = detector.cycle_count(window="1h")
        assert len(counts) > 0
        assert counts.iloc[0]["total_transitions"] > 0
        # 10s on, 10s off = 180 full cycles per hour → ~360 transitions
        assert counts.iloc[0]["total_transitions"] > 100

    def test_rapid_cycling_higher(self, high_cycling_df):
        detector = DutyCycleEvents(high_cycling_df, "motor_1")
        counts = detector.cycle_count(window="1h")
        # 1s on, 1s off → 1800 transitions per hour
        assert counts.iloc[0]["total_transitions"] > 500


class TestExcessiveCycling:
    def test_flags_rapid_cycling(self, high_cycling_df):
        detector = DutyCycleEvents(high_cycling_df, "motor_1")
        result = detector.excessive_cycling(max_transitions=100, window="1h")
        assert len(result) > 0
        assert result.iloc[0]["transition_count"] > 100

    def test_no_excessive_with_high_threshold(self, fifty_pct_df):
        detector = DutyCycleEvents(fifty_pct_df, "motor_1")
        result = detector.excessive_cycling(max_transitions=10000, window="1h")
        assert len(result) == 0


class TestEmptyData:
    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_bool"])
        detector = DutyCycleEvents(df, "motor_1")
        assert len(detector.on_off_intervals()) == 0
        assert len(detector.duty_cycle_per_window()) == 0
        assert len(detector.cycle_count()) == 0
