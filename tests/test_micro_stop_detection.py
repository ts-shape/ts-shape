import pytest
import pandas as pd
from ts_shape.events.production.micro_stop_detection import MicroStopEvents


@pytest.fixture
def micro_stop_df():
    """Boolean signal with known micro-stops and longer stops."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    t = 0
    # Pattern: 60s run, 5s micro-stop, 60s run, 5s micro-stop, 60s run, 120s long stop, 60s run
    patterns = [
        (60, True),
        (5, False),  # run then micro-stop
        (60, True),
        (5, False),  # run then micro-stop
        (60, True),
        (120, False),  # run then long stop
        (60, True),
        (3, False),  # run then micro-stop
        (60, True),
    ]
    for duration, state in patterns:
        for i in range(duration):
            rows.append(
                {
                    "systime": base + pd.Timedelta(seconds=t),
                    "uuid": "machine_1",
                    "value_bool": state,
                }
            )
            t += 1
    return pd.DataFrame(rows)


class TestDetectMicroStops:
    def test_finds_micro_stops(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        stops = detector.detect_micro_stops(max_duration="30s")
        # Should find the 5s and 3s stops but not the 120s stop
        assert len(stops) == 3
        for _, row in stops.iterrows():
            assert row["duration"] <= pd.Timedelta("30s")

    def test_excludes_long_stops(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        stops = detector.detect_micro_stops(max_duration="10s")
        for _, row in stops.iterrows():
            assert row["duration"] <= pd.Timedelta("10s")

    def test_min_duration_filter(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        stops = detector.detect_micro_stops(max_duration="30s", min_duration="4s")
        # Should find only the 5s stops, not the 3s one
        assert all(row["duration"] >= pd.Timedelta("4s") for _, row in stops.iterrows())

    def test_preceding_run_duration(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        stops = detector.detect_micro_stops(max_duration="30s")
        # First micro-stop should have ~60s preceding run
        if len(stops) > 0:
            assert stops.iloc[0]["preceding_run_duration"] >= pd.Timedelta("50s")


class TestMicroStopFrequency:
    def test_frequency_per_window(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        freq = detector.micro_stop_frequency(window="10min", max_duration="30s")
        assert len(freq) > 0
        assert "count" in freq.columns
        assert "total_lost_time" in freq.columns


class TestMicroStopImpact:
    def test_impact_columns(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        impact = detector.micro_stop_impact(window="10min", max_duration="30s")
        assert "availability_loss_pct" in impact.columns
        if not impact.empty:
            assert impact.iloc[0]["availability_loss_pct"] >= 0


class TestMicroStopPatterns:
    def test_hour_grouping(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        patterns = detector.micro_stop_patterns(hour_grouping=True, max_duration="30s")
        assert "hour" in patterns.columns
        assert "avg_count" in patterns.columns

    def test_shift_grouping(self, micro_stop_df):
        detector = MicroStopEvents(micro_stop_df, "machine_1")
        patterns = detector.micro_stop_patterns(hour_grouping=False, max_duration="30s")
        assert "shift" in patterns.columns


class TestEmptyData:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_bool"])
        detector = MicroStopEvents(df, "machine_1")
        assert len(detector.detect_micro_stops()) == 0
        assert len(detector.micro_stop_frequency()) == 0
