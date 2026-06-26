import numpy as np
import pandas as pd
import pytest

from ts_shape.events.engineering.rate_of_change import RateOfChangeEvents


@pytest.fixture
def step_df():
    """Signal with a sudden step change."""
    n = 200
    base = pd.Timestamp("2024-01-01")
    values = np.full(n, 50.0)
    values[100] = 100.0  # Sudden jump
    values[101:] = 100.0  # Stays at new level
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "pressure",
                "value_double": values[i],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def ramp_df():
    """Gradually increasing signal."""
    n = 300
    base = pd.Timestamp("2024-01-01")
    values = np.linspace(0, 30, n)  # 0.1 units/second
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "pressure",
                "value_double": values[i],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def stable_df():
    """Stable signal with minor noise."""
    np.random.seed(42)
    n = 200
    base = pd.Timestamp("2024-01-01")
    values = 50.0 + np.random.randn(n) * 0.001
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "pressure",
                "value_double": values[i],
            }
        )
    return pd.DataFrame(rows)


class TestDetectRapidChange:
    def test_finds_step(self, step_df):
        detector = RateOfChangeEvents(step_df, "pressure")
        result = detector.detect_rapid_change(threshold=10.0)
        assert len(result) > 0
        assert result.iloc[0]["max_rate"] > 10.0

    def test_gradual_ramp_not_flagged(self, ramp_df):
        detector = RateOfChangeEvents(ramp_df, "pressure")
        # Rate is ~0.1 units/s, threshold is 10 → should not flag
        result = detector.detect_rapid_change(threshold=10.0)
        assert len(result) == 0

    def test_direction(self, step_df):
        detector = RateOfChangeEvents(step_df, "pressure")
        result = detector.detect_rapid_change(threshold=10.0)
        if len(result) > 0:
            assert result.iloc[0]["direction"] == "increasing"


class TestRateStatistics:
    def test_statistics_columns(self, ramp_df):
        detector = RateOfChangeEvents(ramp_df, "pressure")
        stats = detector.rate_statistics(window="5min")
        assert len(stats) > 0
        assert "mean_rate" in stats.columns
        assert "std_rate" in stats.columns
        assert "max_rate" in stats.columns

    def test_stable_signal_low_rate(self, stable_df):
        detector = RateOfChangeEvents(stable_df, "pressure")
        stats = detector.rate_statistics(window="5min")
        assert len(stats) > 0
        assert stats.iloc[0]["mean_rate"] < 1.0


class TestDetectStepChanges:
    def test_finds_step(self, step_df):
        detector = RateOfChangeEvents(step_df, "pressure")
        steps = detector.detect_step_changes(min_delta=40.0)
        assert len(steps) >= 1
        assert steps.iloc[0]["delta"] == 50.0
        assert steps.iloc[0]["value_before"] == 50.0
        assert steps.iloc[0]["value_after"] == 100.0

    def test_no_steps_in_ramp(self, ramp_df):
        detector = RateOfChangeEvents(ramp_df, "pressure")
        steps = detector.detect_step_changes(min_delta=10.0)
        assert len(steps) == 0


class TestEmptyData:
    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        detector = RateOfChangeEvents(df, "pressure")
        assert len(detector.detect_rapid_change(threshold=1.0)) == 0
        assert len(detector.rate_statistics()) == 0
        assert len(detector.detect_step_changes(min_delta=1.0)) == 0
