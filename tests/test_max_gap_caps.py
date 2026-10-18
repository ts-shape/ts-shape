"""Theme C: max-gap caps exclude durations that span data gaps / stoppages."""

import pandas as pd

from ts_shape.events.production.cycle_time_tracking import CycleTimeEvents
from ts_shape.events.production.downtime_tracking import DowntimeEvents
from ts_shape.events.production.performance_loss import PerformanceLossEvents


def test_downtime_by_shift_max_gap_excludes_logger_outage():
    # Running at 08:00, then a ~6h logging gap, then Running again.
    rows = [
        ("2024-01-01 08:00:00", "Running"),
        ("2024-01-01 14:00:00", "Running"),  # 6h gap booked as uptime by default
        ("2024-01-01 14:05:00", "Running"),
    ]
    df = pd.DataFrame(
        [
            {"uuid": "state", "systime": pd.Timestamp(t), "value_string": v}
            for t, v in rows
        ]
    )
    dt = DowntimeEvents(df)

    naive = dt.downtime_by_shift("state")
    capped = dt.downtime_by_shift("state", max_gap="15min")
    # The 6h gap is excluded, so total tracked time drops dramatically.
    assert naive["total_minutes"].sum() > 300
    assert capped["total_minutes"].sum() < 15


def test_cycle_time_max_cycle_gap_drops_idle_interval():
    # Rising-edge cycle triggers with a 2h idle gap between completions.
    times = [
        "2024-01-01 08:00:00",
        "2024-01-01 08:00:45",
        "2024-01-01 08:01:30",
        "2024-01-01 10:01:30",  # 2h later
    ]
    rows = []
    for t in times:
        rows.append({"uuid": "trig", "systime": pd.Timestamp(t), "value_bool": False})
        rows.append(
            {
                "uuid": "trig",
                "systime": pd.Timestamp(t) + pd.Timedelta(milliseconds=1),
                "value_bool": True,
            }
        )
    df = pd.DataFrame(rows)
    ct = CycleTimeEvents(df)

    naive = ct.cycle_time_by_part("part", "trig")  # no part signal -> UNKNOWN
    capped = ct.cycle_time_by_part("part", "trig", max_cycle_gap="5min")
    assert naive["cycle_time_seconds"].max() > 3600  # the 2h "cycle"
    assert capped["cycle_time_seconds"].max() < 300  # idle interval dropped


def test_performance_by_shift_max_cycle_gap():
    # A counter incrementing every 45s with one big lunch gap.
    base = pd.Timestamp("2024-01-01 08:00:00")
    times = [base + pd.Timedelta(seconds=45 * i) for i in range(5)]
    times += [t + pd.Timedelta(hours=1) for t in times]  # shifted batch after 1h gap
    times = sorted(times)
    rows = [
        {"uuid": "c", "systime": t, "value_integer": i} for i, t in enumerate(times)
    ]
    df = pd.DataFrame(rows)
    pl = PerformanceLossEvents(df)

    naive = pl.performance_by_shift("c", target_cycle_time=45.0)
    capped = pl.performance_by_shift("c", target_cycle_time=45.0, max_cycle_gap="5min")
    # Excluding the 1h gap raises the reported performance (less phantom loss).
    assert capped["performance_pct"].iloc[0] > naive["performance_pct"].iloc[0]


def test_cycle_time_max_cycle_gap_forwarded_by_derived_methods():
    # 30 cycles at 45 s, then a 2 h idle gap, then 5 more cycles.
    base = pd.Timestamp("2024-01-01 08:00:00")
    times = [base + pd.Timedelta(seconds=45 * i) for i in range(30)]
    restart = times[-1] + pd.Timedelta(hours=2)
    times += [restart + pd.Timedelta(seconds=45 * i) for i in range(5)]
    rows = []
    for t in times:
        rows.append({"uuid": "trig", "systime": t, "value_bool": False})
        rows.append(
            {
                "uuid": "trig",
                "systime": t + pd.Timedelta(milliseconds=1),
                "value_bool": True,
            }
        )
    ct = CycleTimeEvents(pd.DataFrame(rows))

    stats = ct.cycle_time_statistics("part", "trig")
    capped = ct.cycle_time_statistics("part", "trig", max_cycle_gap="5min")
    assert stats["max_seconds"].iloc[0] > 3600
    assert capped["max_seconds"].iloc[0] < 300
    assert capped["cycle_count"].iloc[0] == stats["cycle_count"].iloc[0] - 1

    assert len(ct.detect_slow_cycles("part", "trig")) == 1
    assert ct.detect_slow_cycles("part", "trig", max_cycle_gap="5min").empty

    trend = ct.cycle_time_trend("part", "trig", "UNKNOWN", window_size=5)
    trend_capped = ct.cycle_time_trend(
        "part", "trig", "UNKNOWN", window_size=5, max_cycle_gap="5min"
    )
    assert trend["cycle_time_seconds"].max() > 3600
    assert trend_capped["cycle_time_seconds"].max() < 300

    hourly = ct.hourly_cycle_time_summary("part", "trig")
    hourly_capped = ct.hourly_cycle_time_summary("part", "trig", max_cycle_gap="5min")
    assert hourly["max_cycle_time_seconds"].max() > 3600
    assert hourly_capped["max_cycle_time_seconds"].max() < 300
