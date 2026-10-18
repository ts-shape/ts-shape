"""Theme E: line-balancing missing-station denominator and alarm boundary."""

import pandas as pd

from ts_shape.events.production.alarm_management import AlarmManagementEvents
from ts_shape.events.production.line_balancing import LineBalancingEvents


def _cycle_rising_edges(uuid, times):
    """Build rising-edge cycle-completion triggers (bool going False->True)."""
    rows = []
    for t in times:
        rows.append({"uuid": uuid, "systime": pd.Timestamp(t), "value_bool": False})
        rows.append(
            {
                "uuid": uuid,
                "systime": pd.Timestamp(t) + pd.Timedelta(milliseconds=1),
                "value_bool": True,
            }
        )
    return rows


def test_line_balancing_require_all_stations_penalises_dead_station():
    # S1 and S2 cycle in the 08:00 window; S3 is completely stopped (no cycles).
    rows = []
    rows += _cycle_rising_edges(
        "s1", ["2024-01-01 08:00:10", "2024-01-01 08:01:00", "2024-01-01 08:01:50"]
    )
    rows += _cycle_rising_edges(
        "s2", ["2024-01-01 08:00:20", "2024-01-01 08:01:10", "2024-01-01 08:02:00"]
    )
    # S3 present in the frame (known station) but only a single reading, no cycle.
    rows.append(
        {
            "uuid": "s3",
            "systime": pd.Timestamp("2024-01-01 08:00:00"),
            "value_bool": False,
        }
    )
    df = pd.DataFrame(rows)

    lb = LineBalancingEvents(df, station_uuids={"s1": "S1", "s2": "S2", "s3": "S3"})

    naive = lb.balance_metrics(window="1h")
    # Only 2 stations counted; line looks nearly perfectly balanced.
    assert (naive["station_count"] == 2).all()

    fixed = lb.balance_metrics(window="1h", require_all_stations=True)
    assert (fixed["station_count"] == 3).all()
    assert "stations_missing" in fixed.columns
    assert (fixed["stations_missing"] == 1).all()
    # Denominator now includes the dead station -> efficiency drops below naive.
    assert (
        fixed["balance_efficiency_pct"].iloc[0]
        < naive["balance_efficiency_pct"].iloc[0]
    )


def test_alarm_frequency_initial_active_flag():
    df = pd.DataFrame(
        [
            {
                "uuid": "alarm",
                "systime": pd.Timestamp("2024-01-01 08:00:00"),
                "value_bool": True,
            },
            {
                "uuid": "alarm",
                "systime": pd.Timestamp("2024-01-01 08:10:00"),
                "value_bool": True,
            },
            {
                "uuid": "alarm",
                "systime": pd.Timestamp("2024-01-01 08:20:00"),
                "value_bool": False,
            },
        ]
    )
    am = AlarmManagementEvents(df, alarm_uuid="alarm")

    default = am.alarm_frequency(window="1h")
    assert default["alarm_count"].sum() == 1  # leading active counted

    fixed = am.alarm_frequency(window="1h", include_initial=False)
    assert fixed["alarm_count"].sum() == 0  # pre-existing alarm not counted
