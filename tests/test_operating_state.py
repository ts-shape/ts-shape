import numpy as np
import pandas as pd
import pytest

from ts_shape.events.production.operating_state import OperatingStateEvents


@pytest.fixture
def two_state_df():
    """Signal with two clear states: low (~10) and high (~90)."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []
    # Low state: 0-99s
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": 10.0 + np.random.normal(0, 1),
            }
        )
    # High state: 100-299s
    for i in range(200):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=100 + i),
                "uuid": "sensor_1",
                "value_double": 90.0 + np.random.normal(0, 1),
            }
        )
    # Low state again: 300-399s
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=300 + i),
                "uuid": "sensor_1",
                "value_double": 10.0 + np.random.normal(0, 1),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def three_state_df():
    """Signal with three states: low (~10), mid (~50), high (~90)."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": 10.0 + np.random.normal(0, 1),
            }
        )
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=100 + i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.normal(0, 1),
            }
        )
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=200 + i),
                "uuid": "sensor_1",
                "value_double": 90.0 + np.random.normal(0, 1),
            }
        )
    return pd.DataFrame(rows)


class TestInferStates:
    def test_two_states(self, two_state_df):
        det = OperatingStateEvents(two_state_df, "sensor_1")
        result = det.infer_states(n_states=2)
        assert len(result) >= 2
        labels = set(result["state_label"].tolist())
        assert "low" in labels
        assert "high" in labels

    def test_three_states(self, three_state_df):
        det = OperatingStateEvents(three_state_df, "sensor_1")
        result = det.infer_states(n_states=3)
        assert len(result) >= 3
        labels = set(result["state_label"].tolist())
        assert "low" in labels
        assert "high" in labels

    def test_min_duration_filter(self, two_state_df):
        det = OperatingStateEvents(two_state_df, "sensor_1")
        # Without filter
        all_intervals = det.infer_states(n_states=2, min_duration="0s")
        # With filter - should remove short intervals
        filtered = det.infer_states(n_states=2, min_duration="50s")
        assert len(filtered) <= len(all_intervals)

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = OperatingStateEvents(df, "sensor_1")
        assert len(det.infer_states()) == 0


class TestStateDurationStats:
    def test_stats_per_state(self, two_state_df):
        det = OperatingStateEvents(two_state_df, "sensor_1")
        stats = det.state_duration_statistics(n_states=2)
        assert len(stats) >= 1
        assert "mean_duration" in stats.columns
        assert "count" in stats.columns
        assert all(stats["mean_duration"] >= 0)

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = OperatingStateEvents(df, "sensor_1")
        assert len(det.state_duration_statistics()) == 0


class TestUnusualStateSequence:
    def test_detects_transitions(self, two_state_df):
        det = OperatingStateEvents(two_state_df, "sensor_1")
        result = det.unusual_state_sequence(n_states=2)
        assert len(result) > 0
        assert "is_unusual" in result.columns
        assert "from_label" in result.columns

    def test_with_expected_order(self, three_state_df):
        det = OperatingStateEvents(three_state_df, "sensor_1")
        # Expected: low -> mid -> high; anything else is unusual
        result = det.unusual_state_sequence(n_states=3, expected_order=[0, 1, 2])
        assert len(result) > 0

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = OperatingStateEvents(df, "sensor_1")
        assert len(det.unusual_state_sequence()) == 0


class TestDwellTimeViolation:
    def test_detects_long_dwell(self, two_state_df):
        det = OperatingStateEvents(two_state_df, "sensor_1")
        # High state is 200s long, set max at 100s
        violations = det.dwell_time_violation(state=1, max_duration="100s", n_states=2)
        assert len(violations) >= 1
        assert all(violations["excess_seconds"] > 0)

    def test_no_violations(self, two_state_df):
        det = OperatingStateEvents(two_state_df, "sensor_1")
        # Very generous max duration — no violations
        violations = det.dwell_time_violation(state=1, max_duration="1h", n_states=2)
        assert len(violations) == 0

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = OperatingStateEvents(df, "sensor_1")
        assert len(det.dwell_time_violation(state=0, max_duration="1h")) == 0
