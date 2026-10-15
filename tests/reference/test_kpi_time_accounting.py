"""Period KPIs on hand-computed examples: time split at period boundaries.

ISO 22400-2 (Automation systems and integration -- Key performance
indicators for manufacturing operations management, Part 2) defines the
time-based KPIs on the time elements *of the period* being reported:

* availability = APT / PBT (actual production time over planned busy time),
* performance  = (PQ x pICT) / APT (produced quantity times planned unit
  cycle time over actual production time),
* quality      = GQ / PQ (good quantity over produced quantity),
* OEE          = availability x performance x quality.

A state that lasts from 07:00 to 15:00 is therefore 7 h of the 06-14 shift
and 1 h of the 14-22 shift, and a part counted at 06:00 that was produced
between 05:00 and 06:00 belongs to the night shift. The tests below build
small signals whose KPIs follow from that definition by hand.

Default shift grid: shift_1 06-14, shift_2 14-22, shift_3 22-06 (an
overnight shift belongs to the production day it starts on).
"""

import warnings

import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.production.downtime_tracking import DowntimeEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
from ts_shape.events.production.oee_calculator import OEEEvents
from ts_shape.events.production.quality_tracking import QualityEvents
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents
from ts_shape.events.production.shift_handover import ShiftHandoverEvents
from ts_shape.utils.shifts import DEFAULT_SHIFTS
from ts_shape.utils.time_accounting import (
    counter_by_period,
    grouper_periods,
    reading_intervals,
    shift_periods,
    spread_increments,
    time_by_period,
)

DAY = pd.Timestamp("2025-01-02")


def _ts(hhmm: str, day: pd.Timestamp = DAY) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {hhmm}")


# ---------------------------------------------------------------------------
# A1: a change-encoded run signal across shift / hour boundaries
#
#   Run @05:00, Stop @07:00, Run @15:00, data end @16:00
#
#   shift_3 (Jan 1, 22-06): 05-06 up                    -> 60 up
#   shift_1 (06-14):        06-07 up, 07-14 down        -> 60 up, 420 down
#   shift_2 (14-22):        14-15 down, 15-16 up        -> 60 down, 60 up
#   hour 07:00:             07-08 down                  -> 60 down
# ---------------------------------------------------------------------------


def _change_encoded_run() -> pd.DataFrame:
    rows = [("05:00", True), ("07:00", False), ("15:00", True), ("16:00", True)]
    return pd.DataFrame(
        {
            "systime": [_ts(t) for t, _ in rows],
            "uuid": "state",
            "value_bool": [v for _, v in rows],
            "value_string": ["Running" if v else "Stopped" for _, v in rows],
        }
    )


def _by_shift(out: pd.DataFrame) -> dict:
    return {(str(r["date"]), r["shift"]): r for _, r in out.iterrows()}


def test_downtime_by_shift_splits_at_shift_boundaries():
    out = _by_shift(DowntimeEvents(_change_encoded_run()).downtime_by_shift("state"))
    assert set(out) == {
        ("2025-01-01", "shift_3"),
        ("2025-01-02", "shift_1"),
        ("2025-01-02", "shift_2"),
    }
    night = out[("2025-01-01", "shift_3")]
    assert night["uptime_minutes"] == pytest.approx(60)
    assert night["downtime_minutes"] == pytest.approx(0)
    morning = out[("2025-01-02", "shift_1")]
    assert morning["uptime_minutes"] == pytest.approx(60)
    assert morning["downtime_minutes"] == pytest.approx(420)
    assert morning["availability_pct"] == pytest.approx(12.5)
    late = out[("2025-01-02", "shift_2")]
    assert late["uptime_minutes"] == pytest.approx(60)
    assert late["downtime_minutes"] == pytest.approx(60)
    assert late["availability_pct"] == pytest.approx(50.0)
    # Seconds mirror minutes and add up to the observed time.
    assert sum(r["total_seconds"] for r in out.values()) == 11 * 3600


def test_downtime_by_shift_calendar_dates_split_the_night_shift():
    out = _by_shift(
        DowntimeEvents(_change_encoded_run()).downtime_by_shift(
            "state", production_day=False
        )
    )
    # 05-06 is on Jan 2 by calendar date.
    assert out[("2025-01-02", "shift_3")]["uptime_minutes"] == pytest.approx(60)


def test_downtime_by_shift_max_gap_excluded_time_is_split():
    out = _by_shift(
        DowntimeEvents(_change_encoded_run()).downtime_by_shift("state", max_gap="3h")
    )
    # The 07-15 stop is longer than 3 h: "no data", 7 h in shift_1, 1 h in shift_2.
    assert out[("2025-01-02", "shift_1")]["excluded_seconds"] == 7 * 3600
    assert out[("2025-01-02", "shift_1")]["downtime_seconds"] == 0
    assert out[("2025-01-02", "shift_2")]["excluded_seconds"] == 3600


def test_availability_trend_hourly_books_each_hour_its_own_time():
    out = DowntimeEvents(_change_encoded_run()).availability_trend("state", window="1h")
    by_hour = out.set_index("period")
    assert list(by_hour.index) == [_ts(f"{h:02d}:00") for h in range(5, 16)]
    assert by_hour.loc[_ts("07:00"), "downtime_minutes"] == pytest.approx(60)
    assert by_hour.loc[_ts("07:00"), "uptime_minutes"] == pytest.approx(0)
    assert by_hour.loc[_ts("05:00"), "uptime_minutes"] == pytest.approx(60)
    assert by_hour.loc[_ts("06:00"), "uptime_minutes"] == pytest.approx(60)
    assert by_hour.loc[_ts("15:00"), "availability_pct"] == pytest.approx(100)
    assert (by_hour["uptime_seconds"] + by_hour["downtime_seconds"] == 3600).all()


def test_shift_handover_availability_splits_at_shift_boundaries():
    out = _by_shift(
        ShiftHandoverEvents(_change_encoded_run())._availability_by_shift(
            "state", "Running", "value_string"
        )
    )
    assert out[("2025-01-01", "shift_3")]["availability_pct"] == pytest.approx(100)
    assert out[("2025-01-02", "shift_1")]["availability_pct"] == pytest.approx(12.5)
    assert out[("2025-01-02", "shift_1")]["downtime_seconds"] == 420 * 60
    assert out[("2025-01-02", "shift_2")]["availability_pct"] == pytest.approx(50)


def test_runtime_by_window_splits_at_window_boundaries():
    rt = RuntimeAccountingEvents(_change_encoded_run(), "state")
    out = rt.runtime_by_window(window="8h").set_index("start")
    # 00-08: run 05-07 (2 h), idle 07-08 (1 h) -> 2/3.
    first = out.loc[_ts("00:00")]
    assert first["run_seconds"] == 2 * 3600
    assert first["utilization_pct"] == pytest.approx(200 / 3)
    # 08-16: idle 08-15 (7 h), run 15-16 (1 h) -> 1/8.
    second = out.loc[_ts("08:00")]
    assert second["run_seconds"] == 3600
    assert second["utilization_pct"] == pytest.approx(12.5)
    hourly = rt.operating_hours_meter(window="1h").set_index("start")
    assert hourly.loc[_ts("06:00"), "run_seconds"] == 3600
    assert hourly.loc[_ts("07:00"), "run_seconds"] == 0
    assert hourly["cumulative_run_hours"].iloc[-1] == pytest.approx(3)


def test_oee_availability_splits_at_midnight():
    rows = [("2025-01-01 22:00", True), ("2025-01-02 02:00", False)]
    rows += [("2025-01-02 04:00", False)]
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime([t for t, _ in rows]),
            "uuid": "run",
            "value_bool": [v for _, v in rows],
        }
    )
    out = OEEEvents(df).availability("run").set_index("start")
    assert out.loc[pd.Timestamp("2025-01-01"), "run_seconds"] == 2 * 3600
    assert out.loc[pd.Timestamp("2025-01-01"), "availability_pct"] == 100
    assert out.loc[pd.Timestamp("2025-01-02"), "run_seconds"] == 2 * 3600
    assert out.loc[pd.Timestamp("2025-01-02"), "planned_seconds"] == 4 * 3600
    assert out.loc[pd.Timestamp("2025-01-02"), "availability_pct"] == 50


def test_micro_stop_impact_splits_run_and_stop_time():
    # Run 10:00:00-10:59:50, 20 s stop across 11:00, run until 12:00.
    t = [_ts("10:00:00"), _ts("10:59:50"), _ts("11:00:10"), _ts("12:00:00")]
    df = pd.DataFrame(
        {"systime": t, "uuid": "run", "value_bool": [True, False, True, True]}
    )
    out = MicroStopEvents(df, "run").micro_stop_impact(window="1h")
    by_hour = out.set_index("start")
    assert by_hour.loc[_ts("10:00"), "total_micro_stop_time"] == 10
    assert by_hour.loc[_ts("10:00"), "total_run_time"] == 3590
    assert by_hour.loc[_ts("11:00"), "total_micro_stop_time"] == 10
    assert by_hour.loc[_ts("11:00"), "total_run_time"] == 3590


# ---------------------------------------------------------------------------
# A2: OEE across midnight
#
#   Night shift 22:00-06:00 always running, part counter read hourly with
#   60 parts per hour, ideal (planned unit) cycle time 60 s.
#
#   Jan 1 (22-24): APT 2 h = planned busy time, PQ 120 -> A 100 %, P 100 %
#   Jan 2 (00-06): APT 6 h = planned busy time, PQ 360 -> A 100 %, P 100 %
# ---------------------------------------------------------------------------

NIGHT = pd.date_range("2025-01-01 22:00", "2025-01-02 06:00", freq="1h")


def _night(encoding: str) -> pd.DataFrame:
    if encoding == "change":
        state = pd.DataFrame(
            {
                "systime": [NIGHT[0], NIGHT[-1]],
                "uuid": "run",
                "value_bool": [True, False],
            }
        )
    else:
        state = pd.DataFrame(
            {
                "systime": NIGHT,
                "uuid": "run",
                "value_bool": [True] * (len(NIGHT) - 1) + [False],
            }
        )
    parts = pd.DataFrame(
        {
            "systime": NIGHT,
            "uuid": "parts",
            "value_integer": [60 * i for i in range(len(NIGHT))],
        }
    )
    nok = pd.DataFrame(
        {
            "systime": NIGHT,
            "uuid": "nok",
            "value_integer": [6 * i for i in range(len(NIGHT))],
        }
    )
    return pd.concat([state, parts, nok], ignore_index=True)


@pytest.mark.parametrize("encoding", ["change", "hourly"])
def test_oee_night_shift_is_100_pct_on_both_days(encoding):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        out = OEEEvents(_night(encoding)).oee("run", "parts", 60.0)
    out = out.set_index("start")
    assert list(out.index) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]
    assert list(out["actual_parts"]) == [120.0, 360.0]
    assert list(out["run_seconds"]) == [7200.0, 21600.0]
    for col in ("availability_pct", "performance_pct", "oee_pct"):
        assert out[col].tolist() == pytest.approx([100.0, 100.0])


def test_quality_by_shift_books_parts_to_the_shift_that_made_them():
    out = _by_shift(QualityEvents(_night("hourly")).nok_by_shift("parts", "nok"))
    night = out[("2025-01-01", "shift_3")]
    # All 480 OK and 48 NOK parts were made 22:00-06:00, including the hour
    # reported by the 06:00 reading.
    assert night["ok_parts"] == 480
    assert night["nok_parts"] == 48
    morning = out[("2025-01-02", "shift_1")]
    assert morning["ok_parts"] == 0
    assert morning["nok_parts"] == 0


def test_counter_increment_across_a_boundary_is_shared_by_time():
    # 30 parts between 13:50 and 14:10: 15 in each shift.
    t = pd.Series([_ts("13:50"), _ts("14:10")])
    ok = pd.DataFrame({"systime": t, "uuid": "ok", "value_integer": [100, 130]})
    nok = pd.DataFrame({"systime": t, "uuid": "nok", "value_integer": [5, 5]})
    df = pd.concat([ok, nok], ignore_index=True)
    out = _by_shift(QualityEvents(df).nok_by_shift("ok", "nok"))
    assert out[("2025-01-02", "shift_1")]["ok_parts"] == pytest.approx(15)
    assert out[("2025-01-02", "shift_2")]["ok_parts"] == pytest.approx(15)


def test_shift_handover_counter_increment_across_a_boundary_is_shared_by_time():
    # 30 parts between 13:50 and 14:10: 15 in each shift (same accounting as
    # QualityEvents / OEEEvents).
    t = pd.Series([_ts("13:50"), _ts("14:10")])
    df = pd.DataFrame({"systime": t, "uuid": "prod", "value_integer": [100, 130]})
    out = _by_shift(ShiftHandoverEvents(df)._counter_by_shift("prod", "value_integer"))
    assert out[("2025-01-02", "shift_1")]["quantity"] == pytest.approx(15)
    assert out[("2025-01-02", "shift_2")]["quantity"] == pytest.approx(15)
    # The night shift keeps the parts its last hour produced.
    night = _by_shift(
        ShiftHandoverEvents(_night("hourly"))._counter_by_shift(
            "parts", "value_integer"
        )
    )
    assert night[("2025-01-01", "shift_3")]["quantity"] == 480
    assert night[("2025-01-02", "shift_1")]["quantity"] == 0


def test_oee_day_without_state_rows_has_unknown_availability():
    # State only on Jan 1; the counter keeps running on Jan 2.
    state = pd.DataFrame(
        {
            "systime": [
                _ts("08:00", pd.Timestamp("2025-01-01")),
                _ts("16:00", pd.Timestamp("2025-01-01")),
            ],
            "uuid": "run",
            "value_bool": [True, True],
        }
    )
    t = pd.date_range("2025-01-02 08:00", "2025-01-02 10:00", freq="1h")
    parts = pd.DataFrame({"systime": t, "uuid": "parts", "value_integer": [0, 60, 120]})
    oee = OEEEvents(pd.concat([state, parts], ignore_index=True))
    perf = oee.performance("parts", 60.0, run_state_uuid="run").set_index("start")
    assert pd.isna(perf.loc[pd.Timestamp("2025-01-02"), "run_seconds"])
    assert pd.isna(perf.loc[pd.Timestamp("2025-01-02"), "performance_pct"])
    out = oee.oee("run", "parts", 60.0).set_index("start")
    assert pd.isna(out.loc[pd.Timestamp("2025-01-02"), "availability_pct"])
    assert pd.isna(out.loc[pd.Timestamp("2025-01-02"), "oee_pct"])
    assert out.loc[pd.Timestamp("2025-01-02"), "actual_parts"] == 120
    assert out.loc[pd.Timestamp("2025-01-01"), "availability_pct"] == 100


def test_oee_performance_above_100_is_kept_and_warned():
    # 120 parts in 1 h of run time at an ideal 60 s/part: P = 200 %.
    state = pd.DataFrame(
        {
            "systime": [_ts("08:00"), _ts("09:00")],
            "uuid": "run",
            "value_bool": [True, False],
        }
    )
    parts = pd.DataFrame(
        {
            "systime": [_ts("08:00"), _ts("09:00")],
            "uuid": "parts",
            "value_integer": [0, 120],
        }
    )
    oee = OEEEvents(pd.concat([state, parts], ignore_index=True))
    with pytest.warns(DataQualityWarning, match="2025-01-02"):
        perf = oee.performance("parts", 60.0, run_state_uuid="run")
    assert perf["performance_pct"].iloc[0] == pytest.approx(200.0)


def test_oee_availability_above_100_is_kept_and_warned():
    state = pd.DataFrame(
        {
            "systime": [_ts("08:00"), _ts("18:00")],
            "uuid": "run",
            "value_bool": [True, False],
        }
    )
    with pytest.warns(DataQualityWarning, match="availability_pct above 100"):
        out = OEEEvents(state).availability("run", planned_time_hours=8)
    assert out["availability_pct"].iloc[0] == pytest.approx(125.0)


def test_oee_single_day_reference():
    # Planned 08:00-16:00 (8 h), running 08:00-14:00 (6 h), 300 parts at an
    # ideal 60 s, 15 rejects: A = 6/8 = 75 %, P = 300*60/21600 = 83.33 %,
    # Q = 285/300 = 95 %, OEE = 0.75 * 0.8333 * 0.95 = 59.375 %.
    state = pd.DataFrame(
        {
            "systime": [_ts("08:00"), _ts("14:00"), _ts("16:00")],
            "uuid": "run",
            "value_bool": [True, False, False],
        }
    )
    t = pd.date_range(_ts("08:00"), _ts("14:00"), freq="1h")
    total = pd.DataFrame(
        {"systime": t, "uuid": "tot", "value_integer": [50 * i for i in range(7)]}
    )
    rej = pd.DataFrame(
        {"systime": [t[0], t[-1]], "uuid": "rej", "value_integer": [0, 15]}
    )
    df = pd.concat([state, total, rej], ignore_index=True)
    out = OEEEvents(df).oee("run", "tot", 60.0, total_uuid="tot", reject_uuid="rej")
    row = out.iloc[0]
    assert len(out) == 1
    assert row["availability_pct"] == pytest.approx(75.0)
    assert row["performance_pct"] == pytest.approx(250 / 3)
    assert row["quality_pct"] == pytest.approx(95.0)
    assert row["oee_pct"] == pytest.approx(59.375)


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


def test_time_by_period_and_spread_increments():
    df = _change_encoded_run()
    iv = reading_intervals(
        df["systime"], df["value_bool"].map({True: "up", False: "down"})
    )
    periods = shift_periods(
        df["systime"].iloc[0], df["systime"].iloc[-1], DEFAULT_SHIFTS
    )
    out = time_by_period(iv, periods, keys=["date", "shift"], states=["up", "down"])
    assert out["up"].tolist() == [3600.0, 3600.0, 3600.0]
    assert out["down"].tolist() == [0.0, 25200.0, 3600.0]

    t = pd.Series(NIGHT)
    days = grouper_periods(t.iloc[0], t.iloc[-1], "1D")
    inc = pd.Series([0.0] + [60.0] * (len(t) - 1))
    spread = spread_increments(t, inc, days, keys=["period"])
    assert spread["quantity"].tolist() == [120.0, 360.0]


def test_counter_by_period_reset_is_counted_where_it_happens():
    t = pd.Series(
        pd.to_datetime(["2025-01-02 13:00", "2025-01-02 13:30", "2025-01-02 14:30"])
    )
    values = pd.Series([10, 40, 20])  # reset before 14:30: +20 after it
    periods = shift_periods(t.iloc[0], t.iloc[-1], DEFAULT_SHIFTS)
    keys = pd.DataFrame(
        {"date": [DAY.date()] * 3, "shift": ["shift_1", "shift_1", "shift_2"]}
    )
    out = counter_by_period(t, values, periods, keys, keys=["date", "shift"])
    by_shift = out.set_index("shift")
    # 30 parts 13:00-13:30; 20 parts 13:30-14:30, half in each shift.
    assert by_shift.loc["shift_1", "quantity"] == pytest.approx(40)
    assert by_shift.loc["shift_2", "quantity"] == pytest.approx(10)
    assert by_shift.loc["shift_2", "resets"] == 1


def test_shift_periods_dst_night_shift_is_7_hours():
    tz = "Europe/Berlin"
    p = shift_periods(
        pd.Timestamp("2025-03-29 23:00", tz=tz),
        pd.Timestamp("2025-03-30 05:00", tz=tz),
        DEFAULT_SHIFTS,
    )
    night = p[p["shift"] == "shift_3"].iloc[0]
    assert (night["end"] - night["start"]) == pd.Timedelta(hours=7)
