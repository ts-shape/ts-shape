import pytest
import pandas as pd
import numpy as np
from ts_shape.events.quality.signal_quality import SignalQualityEvents


@pytest.fixture
def regular_df():
    """Regular 1s sampling with a 30s gap."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    # First segment: 0-100s
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": float(i),
            }
        )
    # Gap of 30 seconds (100s to 130s missing)
    # Second segment: 130-230s
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=130 + i),
                "uuid": "sensor_1",
                "value_double": float(130 + i),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def irregular_df():
    """Irregularly sampled signal."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    intervals = np.random.exponential(scale=2.0, size=100)
    cumulative = np.cumsum(intervals)
    rows = []
    for i, t in enumerate(cumulative):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=t),
                "uuid": "sensor_1",
                "value_double": np.sin(t),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def out_of_range_df():
    """Signal with known out-of-range values."""
    base = pd.Timestamp("2024-01-01")
    n = 200
    values = np.full(n, 50.0)
    # Out of range: above 100 from index 50-70
    values[50:70] = 120.0
    # Out of range: below 0 from index 150-160
    values[150:160] = -10.0
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": values[i],
            }
        )
    return pd.DataFrame(rows)


class TestDetectMissingData:
    def test_finds_gap(self, regular_df):
        detector = SignalQualityEvents(regular_df, "sensor_1")
        gaps = detector.detect_missing_data(expected_freq="1s", tolerance_factor=2.0)
        assert len(gaps) == 1
        assert gaps.iloc[0]["gap_duration"] >= pd.Timedelta("25s")

    def test_no_gaps(self):
        base = pd.Timestamp("2024-01-01")
        rows = [
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "s1",
                "value_double": float(i),
            }
            for i in range(100)
        ]
        df = pd.DataFrame(rows)
        detector = SignalQualityEvents(df, "s1")
        gaps = detector.detect_missing_data(expected_freq="1s")
        assert len(gaps) == 0


class TestSamplingRegularity:
    def test_regular_signal(self):
        # Use a truly regular signal (no gaps)
        base = pd.Timestamp("2024-01-01")
        rows = [
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "s1",
                "value_double": float(i),
            }
            for i in range(200)
        ]
        df = pd.DataFrame(rows)
        detector = SignalQualityEvents(df, "s1")
        reg = detector.sampling_regularity(window="1h")
        assert len(reg) > 0
        # Perfectly regular 1s sampling should have high regularity
        assert reg.iloc[0]["regularity_score"] > 0.5

    def test_irregular_signal(self, irregular_df):
        detector = SignalQualityEvents(irregular_df, "sensor_1")
        reg = detector.sampling_regularity(window="1h")
        assert len(reg) > 0
        # Exponential intervals → lower regularity
        assert reg.iloc[0]["regularity_score"] < 0.9


class TestDetectOutOfRange:
    def test_finds_out_of_range(self, out_of_range_df):
        detector = SignalQualityEvents(out_of_range_df, "sensor_1")
        oor = detector.detect_out_of_range(min_value=0, max_value=100)
        assert len(oor) == 2
        directions = set(oor["direction"].tolist())
        assert "above" in directions
        assert "below" in directions

    def test_all_in_range(self, regular_df):
        detector = SignalQualityEvents(regular_df, "sensor_1")
        oor = detector.detect_out_of_range(min_value=-1000, max_value=1000)
        assert len(oor) == 0


class TestDataCompleteness:
    def test_completeness_with_gap(self, regular_df):
        detector = SignalQualityEvents(regular_df, "sensor_1")
        comp = detector.data_completeness(window="1h", expected_freq="1s")
        assert len(comp) > 0
        # With a 30s gap in 230s of data, completeness should be < 100%
        assert comp.iloc[0]["completeness_pct"] < 100


class TestEmptyData:
    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        detector = SignalQualityEvents(df, "sensor_1")
        assert len(detector.detect_missing_data()) == 0
        assert len(detector.sampling_regularity()) == 0
        assert len(detector.detect_out_of_range(0, 100)) == 0
        assert len(detector.data_completeness()) == 0
