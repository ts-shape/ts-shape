import pytest
import pandas as pd
import numpy as np
from ts_shape.events.production.bottleneck_detection import BottleneckDetectionEvents


@pytest.fixture
def three_station_df():
    """Three stations: A at 90% util, B at 60%, C at 30%."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(3600):  # 1 hour of 1s data
        t = base + pd.Timedelta(seconds=i)
        # Station A: 90% running (9 on, 1 off repeating)
        rows.append({"systime": t, "uuid": "station_a", "value_bool": (i % 10) < 9})
        # Station B: 60% running
        rows.append({"systime": t, "uuid": "station_b", "value_bool": (i % 10) < 6})
        # Station C: 30% running
        rows.append({"systime": t, "uuid": "station_c", "value_bool": (i % 10) < 3})
    return pd.DataFrame(rows)


@pytest.fixture
def shifting_bottleneck_df():
    """Station A is bottleneck first half, Station B second half."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(7200):  # 2 hours
        t = base + pd.Timedelta(seconds=i)
        if i < 3600:
            a_running = (i % 10) < 9  # 90%
            b_running = (i % 10) < 5  # 50%
        else:
            a_running = (i % 10) < 4  # 40%
            b_running = (i % 10) < 9  # 90%
        rows.append({"systime": t, "uuid": "station_a", "value_bool": a_running})
        rows.append({"systime": t, "uuid": "station_b", "value_bool": b_running})
    return pd.DataFrame(rows)


class TestStationUtilization:
    def test_correct_utilization(self, three_station_df):
        detector = BottleneckDetectionEvents(three_station_df)
        util = detector.station_utilization(
            ["station_a", "station_b", "station_c"], window="1h"
        )
        assert len(util) > 0
        a_util = util[util["uuid"] == "station_a"]["utilization_pct"].values[0]
        b_util = util[util["uuid"] == "station_b"]["utilization_pct"].values[0]
        c_util = util[util["uuid"] == "station_c"]["utilization_pct"].values[0]
        assert a_util > b_util > c_util
        assert abs(a_util - 90) < 5

    def test_empty_uuid(self, three_station_df):
        detector = BottleneckDetectionEvents(three_station_df)
        util = detector.station_utilization(["nonexistent"], window="1h")
        assert len(util) == 0


class TestDetectBottleneck:
    def test_highest_utilization_is_bottleneck(self, three_station_df):
        detector = BottleneckDetectionEvents(three_station_df)
        bottlenecks = detector.detect_bottleneck(
            ["station_a", "station_b", "station_c"], window="1h"
        )
        assert len(bottlenecks) > 0
        assert bottlenecks.iloc[0]["bottleneck_uuid"] == "station_a"


class TestShiftingBottleneck:
    def test_bottleneck_shift_detected(self, shifting_bottleneck_df):
        detector = BottleneckDetectionEvents(shifting_bottleneck_df)
        shifts = detector.shifting_bottleneck(["station_a", "station_b"], window="1h")
        assert len(shifts) >= 1
        # Should shift from station_a to station_b
        assert shifts.iloc[0]["from_uuid"] == "station_a"
        assert shifts.iloc[0]["to_uuid"] == "station_b"


class TestThroughputConstraintSummary:
    def test_summary_structure(self, three_station_df):
        detector = BottleneckDetectionEvents(three_station_df)
        summary = detector.throughput_constraint_summary(
            ["station_a", "station_b", "station_c"], window="1h"
        )
        assert "bottleneck_counts" in summary
        assert "most_frequent_bottleneck" in summary
        assert "avg_utilization_per_station" in summary
        assert summary["most_frequent_bottleneck"] == "station_a"
