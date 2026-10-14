"""Regression test: downtime_by_reason must not book running time as downtime."""

import pandas as pd  # type: ignore

from ts_shape.events.production.downtime_tracking import DowntimeEvents


def _frame():
    # State timeline: Stopped(reasonA) 08:00 -> Running 08:10 -> Stopped(reasonB)
    # 09:00 -> Running 09:05. Reason A's real stop is 10 min (08:00->08:10), not
    # 60 min (08:00->09:00, which would wrongly include the running span).
    rows = [
        ("state", "2024-01-01 08:00:00", "Stopped"),
        ("state", "2024-01-01 08:10:00", "Running"),
        ("state", "2024-01-01 09:00:00", "Stopped"),
        ("state", "2024-01-01 09:05:00", "Running"),
        ("reason", "2024-01-01 08:00:00", "ReasonA"),
        ("reason", "2024-01-01 09:00:00", "ReasonB"),
    ]
    return pd.DataFrame(
        [{"uuid": u, "systime": pd.Timestamp(t), "value_string": v} for u, t, v in rows]
    )


def test_downtime_by_reason_excludes_running_time():
    dt = DowntimeEvents(_frame())
    result = dt.downtime_by_reason("state", "reason", stopped_value="Stopped")
    by_reason = result.set_index("reason")["total_minutes"].to_dict()
    # ReasonA lasted 08:00 -> 08:10 = 10 minutes (NOT 60).
    assert by_reason["ReasonA"] == 10.0
    # ReasonB lasted 09:00 -> 09:05 = 5 minutes.
    assert by_reason["ReasonB"] == 5.0
