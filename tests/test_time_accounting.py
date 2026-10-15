"""ts_shape.utils.time_accounting: the time model behind the shift and OEE reports."""

import pandas as pd
import pytest

from ts_shape.utils.time_accounting import (
    EXCLUDED,
    counter_by_period,
    floor_periods,
    grouper_periods,
    reading_intervals,
    shift_periods,
    split_intervals,
    spread_increments,
    time_by_period,
)

SHIFTS = {
    "day": ("06:00", "14:00"),
    "late": ("14:00", "22:00"),
    "night": ("22:00", "06:00"),
}


def _ts(*values):
    return pd.Series(pd.to_datetime(list(values)))


# --------------------------------------------------------------------------
# reading_intervals
# --------------------------------------------------------------------------


def test_reading_intervals_hold_each_state_until_the_next_reading():
    out = reading_intervals(
        _ts("2024-01-01 06:00", "2024-01-01 06:10", "2024-01-01 08:00"),
        pd.Series([True, False, True]),
        max_gap="1h",
    )
    assert out["end"].tolist() == list(
        pd.to_datetime(["2024-01-01 06:10", "2024-01-01 08:00"])
    )
    # 06:10 -> 08:00 is longer than max_gap: a logging gap, not idle time
    assert out["state"].tolist() == [True, EXCLUDED]


@pytest.mark.parametrize("n", [0, 1])
def test_reading_intervals_need_two_readings(n):
    times = _ts(*["2024-01-01 06:00"] * n) if n else pd.Series(pd.to_datetime([]))
    out = reading_intervals(times, pd.Series([True] * n))
    assert out.empty
    assert list(out.columns) == ["start", "end", "state"]


# --------------------------------------------------------------------------
# period builders
# --------------------------------------------------------------------------


def test_shift_periods_production_day_keeps_the_night_in_one_window():
    out = shift_periods(
        pd.Timestamp("2024-01-01 21:00"), pd.Timestamp("2024-01-02 07:00"), SHIFTS
    )
    night = out[out["shift"] == "night"]
    assert len(night) == 1
    assert night["start"].iloc[0] == pd.Timestamp("2024-01-01 22:00")
    assert night["end"].iloc[0] == pd.Timestamp("2024-01-02 06:00")
    assert str(night["date"].iloc[0]) == "2024-01-01"


def test_shift_periods_calendar_dates_cut_the_night_at_midnight():
    out = shift_periods(
        pd.Timestamp("2024-01-01 21:00"),
        pd.Timestamp("2024-01-02 07:00"),
        SHIFTS,
        production_day=False,
    )
    night = out[out["shift"] == "night"]
    assert len(night) == 2
    assert night["end"].iloc[0] == pd.Timestamp("2024-01-02 00:00")


def test_shift_periods_label_uncovered_time_unknown():
    out = shift_periods(
        pd.Timestamp("2024-01-01 05:00"),
        pd.Timestamp("2024-01-01 15:00"),
        {"day": ("06:00", "14:00")},
    )
    assert set(out["shift"]) == {"day", "unknown"}


def test_shift_periods_keep_the_timezone():
    tz = "Europe/Berlin"
    out = shift_periods(
        pd.Timestamp("2024-03-30 12:00", tz=tz),
        pd.Timestamp("2024-03-31 12:00", tz=tz),
        SHIFTS,
    )
    assert str(out["start"].dt.tz) == tz
    night = out[out["shift"] == "night"].iloc[0]
    # the DST night is one hour short
    assert night["end"] - night["start"] == pd.Timedelta("7h")


@pytest.mark.parametrize(
    "t0,t1",
    [
        (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")),
        (pd.NaT, pd.Timestamp("2024-01-01")),
    ],
)
def test_period_builders_on_an_empty_range(t0, t1):
    assert list(shift_periods(t0, t1, SHIFTS).columns) == [
        "date",
        "shift",
        "start",
        "end",
    ]
    assert shift_periods(t0, t1, SHIFTS).empty
    assert grouper_periods(t0, t1, "1h").empty
    assert floor_periods(t0, t1, "1h").empty


def test_grouper_periods_true_bounds():
    out = grouper_periods(
        pd.Timestamp("2024-01-01 06:30"), pd.Timestamp("2024-01-01 08:10"), "1h"
    )
    assert out["start"].tolist() == list(
        pd.to_datetime(["2024-01-01 06:00", "2024-01-01 07:00", "2024-01-01 08:00"])
    )
    assert (out["end"] - out["start"]).eq(pd.Timedelta("1h")).all()


def test_floor_periods_last_window_ends_one_window_later():
    out = floor_periods(
        pd.Timestamp("2024-01-01 06:20"), pd.Timestamp("2024-01-01 06:50"), "15min"
    )
    assert out["start"].tolist() == list(
        pd.to_datetime(["2024-01-01 06:15", "2024-01-01 06:30", "2024-01-01 06:45"])
    )
    assert out["end"].iloc[-1] == pd.Timestamp("2024-01-01 07:00")


def test_floor_periods_daily_windows():
    out = floor_periods(
        pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-03 01:00"), "1D"
    )
    assert len(out) == 3
    assert out["end"].iloc[-1] == pd.Timestamp("2024-01-04")


# --------------------------------------------------------------------------
# splitting intervals over periods
# --------------------------------------------------------------------------

PERIODS = pd.DataFrame(
    {
        "shift": ["late", "day"],  # unsorted on purpose
        "start": pd.to_datetime(["2024-01-01 14:00", "2024-01-01 06:00"]),
        "end": pd.to_datetime(["2024-01-01 22:00", "2024-01-01 14:00"]),
    }
)


def test_split_intervals_cuts_at_period_bounds():
    intervals = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01 12:00", "2024-01-01 15:00"]),
            "end": pd.to_datetime(["2024-01-01 16:00", "2024-01-01 15:00"]),
            "state": ["run", "idle"],
        }
    )
    out = split_intervals(intervals, PERIODS, keys=["shift"])
    # the zero-length interval has no positive overlap
    assert out["shift"].tolist() == ["day", "late"]
    assert out["state"].tolist() == ["run", "run"]
    assert out["seconds"].tolist() == [7200.0, 7200.0]
    assert out["fraction"].tolist() == [0.5, 0.5]


def test_split_intervals_empty_keeps_the_columns():
    intervals = pd.DataFrame(
        {
            "start": pd.Series(dtype="datetime64[ns]"),
            "end": pd.Series(dtype="datetime64[ns]"),
            "state": pd.Series(dtype=object),
        }
    )
    out = split_intervals(intervals, PERIODS, keys=["shift"])
    assert out.empty
    assert list(out.columns) == ["shift", "state", "seconds", "fraction"]


def test_time_by_period_sums_states_and_ignores_others():
    intervals = pd.DataFrame(
        {
            "start": pd.to_datetime(
                ["2024-01-01 13:00", "2024-01-01 14:30", "2024-01-01 15:00"]
            ),
            "end": pd.to_datetime(
                ["2024-01-01 14:30", "2024-01-01 15:00", "2024-01-01 16:00"]
            ),
            "state": ["run", "idle", EXCLUDED],
        }
    )
    out = time_by_period(
        intervals, PERIODS, keys=["shift"], states=["run", "idle", "setup"]
    )
    assert out["shift"].tolist() == ["day", "late"]
    assert out["run"].tolist() == [3600.0, 1800.0]
    assert out["idle"].tolist() == [0.0, 1800.0]
    assert out["setup"].tolist() == [0.0, 0.0]


def test_time_by_period_no_overlap_is_empty():
    intervals = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-02-01"]),
            "end": pd.to_datetime(["2024-02-02"]),
            "state": ["run"],
        }
    )
    out = time_by_period(intervals, PERIODS, keys=["shift"], states=["run"])
    assert out.empty
    assert list(out.columns) == ["shift", "start", "end", "run"]


# --------------------------------------------------------------------------
# counters over periods
# --------------------------------------------------------------------------


def test_spread_increments_share_by_time():
    times = _ts("2024-01-01 13:00", "2024-01-01 15:00")
    out = spread_increments(times, pd.Series([5.0, 20.0]), PERIODS, keys=["shift"])
    # the first reading's increment is a point in "day"; 20 parts over
    # 13:00-15:00 are shared half and half
    assert out.set_index("shift")["quantity"].to_dict() == {"day": 15.0, "late": 10.0}


def test_spread_increments_without_readings_or_overlap():
    empty = spread_increments(
        pd.Series(pd.to_datetime([])),
        pd.Series([], dtype=float),
        PERIODS,
        keys=["shift"],
    )
    assert empty.empty and list(empty.columns) == ["shift", "quantity"]
    outside = spread_increments(
        _ts("2024-02-01"), pd.Series([3.0]), PERIODS, keys=["shift"]
    )
    assert outside.empty and list(outside.columns) == ["shift", "quantity"]


def test_counter_by_period_counts_resets_where_they_are_read():
    times = _ts(
        "2024-01-01 07:00", "2024-01-01 13:00", "2024-01-01 15:00", "2024-01-01 21:00"
    )
    values = pd.Series([100.0, 160.0, 10.0, None])
    reading_keys = pd.DataFrame({"shift": ["day", "day", "late", "late"]})
    out = counter_by_period(times, values, PERIODS, reading_keys, keys=["shift"])
    rows = out.set_index("shift")
    assert rows.loc["late", "resets"] == 1
    assert rows.loc["day", "resets"] == 0
    assert rows["quantity"].sum() == pytest.approx(
        70.0
    )  # 60 before the reset, 10 after


def test_counter_by_period_without_values():
    out = counter_by_period(
        _ts("2024-01-01 07:00"),
        pd.Series([None]),
        PERIODS,
        pd.DataFrame({"shift": ["day"]}),
        keys=["shift"],
    )
    assert out.empty
    assert list(out.columns) == ["shift", "quantity", "resets"]
