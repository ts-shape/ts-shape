"""Theme C: max-gap caps exclude durations that span data gaps / stoppages."""

import pandas as pd  # type: ignore

from ts_shape.events.production.cycle_time_tracking import CycleTimeTracking
from ts_shape.events.production.downtime_tracking import DowntimeTracking
from ts_shape.events.production.performance_loss import PerformanceLossTracking


def test_downtime_by_shift_max_gap_excludes_logger_outage():
    # Running at 08:00, then a ~6h logging gap, then Running again.
    rows = [
        ("2024-01-01 08:00:00", "Running"),
        ("2024-01-01 14:00:00", "Running"),  # 6h gap booked as uptime by default
        ("2024-01-01 14:05:00", "Running"),
    ]
    df = pd.DataFrame(
        [{"uuid": "state", "systime": pd.Timestamp(t), "value_string": v} for t, v in rows]
    )
    dt = DowntimeTracking(df)

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
    ct = CycleTimeTracking(df)

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
        {"uuid": "c", "systime": t, "value_integer": i}
        for i, t in enumerate(times)
    ]
    df = pd.DataFrame(rows)
    pl = PerformanceLossTracking(df)

    naive = pl.performance_by_shift("c", target_cycle_time=45.0)
    capped = pl.performance_by_shift("c", target_cycle_time=45.0, max_cycle_gap="5min")
    # Excluding the 1h gap raises the reported performance (less phantom loss).
    assert capped["performance_pct"].iloc[0] > naive["performance_pct"].iloc[0]
