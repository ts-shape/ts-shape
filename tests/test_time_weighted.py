"""Theme D: time-weighted utilisation for bottleneck + duty cycle."""

import pandas as pd

from ts_shape.events.production.bottleneck_detection import BottleneckDetectionEvents
from ts_shape.events.production.duty_cycle import DutyCycleEvents
from ts_shape.events.production.line_throughput import LineThroughputEvents


def test_station_utilization_time_weighted():
    # Station is ON at 09:00 and OFF at 09:59 (change-encoded). It was up ~98%
    # of the hour, but a sample-count mean of [1, 0] reads 50%.
    df = pd.DataFrame(
        [
            {
                "uuid": "s1",
                "systime": pd.Timestamp("2024-01-01 09:00:00"),
                "value_bool": True,
            },
            {
                "uuid": "s1",
                "systime": pd.Timestamp("2024-01-01 09:59:00"),
                "value_bool": False,
            },
            {
                "uuid": "s1",
                "systime": pd.Timestamp("2024-01-01 10:00:00"),
                "value_bool": False,
            },
        ]
    )
    bd = BottleneckDetectionEvents(df)

    naive = bd.station_utilization(["s1"], window="1h", time_weighted=False)
    weighted = bd.station_utilization(["s1"], window="1h")  # default: weighted

    first_naive = naive[naive["start"] == pd.Timestamp("2024-01-01 09:00:00")]
    first_weighted = weighted[weighted["start"] == pd.Timestamp("2024-01-01 09:00:00")]
    assert first_naive["utilization_pct"].iloc[0] == 50.0
    assert first_weighted["utilization_pct"].iloc[0] > 95.0


def test_duty_cycle_per_window_time_weighted():
    df = pd.DataFrame(
        [
            {
                "uuid": "sig",
                "systime": pd.Timestamp("2024-01-01 09:00:00"),
                "value_bool": True,
            },
            {
                "uuid": "sig",
                "systime": pd.Timestamp("2024-01-01 09:59:00"),
                "value_bool": False,
            },
            {
                "uuid": "sig",
                "systime": pd.Timestamp("2024-01-01 10:00:00"),
                "value_bool": False,
            },
        ]
    )
    dc = DutyCycleEvents(df, "sig")
    naive = dc.duty_cycle_by_window(window="1h", time_weighted=False)
    weighted = dc.duty_cycle_by_window(window="1h")  # default: weighted
    assert naive["duty_cycle_pct"].iloc[0] == 50.0
    assert weighted["duty_cycle_pct"].iloc[0] > 95.0


def test_cycle_quality_check_zero_expected_returns_empty():
    # Explicit zero expected cycle time must not divide by zero.
    times = pd.date_range("2024-01-01 08:00:00", periods=5, freq="45s")
    rows = []
    for t in times:
        rows.append({"uuid": "c", "systime": t, "value_bool": False})
        rows.append(
            {
                "uuid": "c",
                "systime": t + pd.Timedelta(milliseconds=1),
                "value_bool": True,
            }
        )
    df = pd.DataFrame(rows)
    lt = LineThroughputEvents(df)
    out = lt.cycle_quality_check("c", expected_cycle_time=0.0)
    assert out.empty
    assert "deviation_pct" in out.columns
