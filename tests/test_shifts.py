"""Tests for the shift assignment / production-day helper."""

import pandas as pd  # type: ignore

from ts_shape.utils.shifts import (
    DEFAULT_SHIFTS,
    assign_shift,
    production_dates,
)


def test_assign_shift_day_and_overnight():
    assert assign_shift(pd.Timestamp("2024-01-01 08:00"), DEFAULT_SHIFTS) == "shift_1"
    assert assign_shift(pd.Timestamp("2024-01-01 15:00"), DEFAULT_SHIFTS) == "shift_2"
    assert assign_shift(pd.Timestamp("2024-01-01 23:00"), DEFAULT_SHIFTS) == "shift_3"
    assert assign_shift(pd.Timestamp("2024-01-02 03:00"), DEFAULT_SHIFTS) == "shift_3"


def test_production_dates_overnight_attributed_to_start_day():
    times = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 23:00",  # night shift, evening -> Jan 1
                "2024-01-02 03:00",  # night shift, small hours -> Jan 1 (prev day)
                "2024-01-02 08:00",  # day shift -> Jan 2
            ]
        )
    )
    shifts = times.apply(lambda t: assign_shift(t, DEFAULT_SHIFTS))
    dates = production_dates(times, shifts, DEFAULT_SHIFTS)
    assert list(dates) == [
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-01-02").date(),
    ]


def test_production_dates_day_only_unchanged():
    day_shifts = {"shift_1": ("06:00", "14:00"), "shift_2": ("14:00", "22:00")}
    times = pd.Series(pd.to_datetime(["2024-01-01 08:00", "2024-01-01 15:00"]))
    shifts = times.apply(lambda t: assign_shift(t, day_shifts))
    dates = production_dates(times, shifts, day_shifts)
    assert list(dates) == [d for d in times.dt.date]


# ---------------------------------------------------------------------------
# Vectorized (Series) path
# ---------------------------------------------------------------------------

import numpy as np  # type: ignore  # noqa: E402
import pytest  # noqa: E402

_ODD_SHIFTS = {
    "early": ("05:30", "13:30"),
    "late": ("13:30", "21:45:30"),
    "night": ("21:45:30", "05:30"),
}


def _scalar(times, defs):
    return [assign_shift(t, defs) for t in times]


@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Berlin"])
@pytest.mark.parametrize("defs", [DEFAULT_SHIFTS, _ODD_SHIFTS])
def test_series_matches_scalar(tz, defs):
    rng = np.random.default_rng(0)
    # Spans both 2024 DST changes; random sub-second offsets hit the edges.
    base = pd.Timestamp("2024-03-30", tz=tz)
    offsets = pd.to_timedelta(rng.integers(0, 220 * 86400 * 10**6, 400), unit="us")
    times = pd.Series(base + offsets)
    edges = pd.Series(
        pd.to_datetime(
            [
                "2024-03-31 05:59:59.999999",
                "2024-03-31 06:00:00.000000",
                "2024-10-27 21:45:30.000000",
            ]
        )
    )
    if tz is not None:
        edges = edges.dt.tz_localize(tz)
    times = pd.concat([times, edges], ignore_index=True)
    got = assign_shift(times, defs)
    assert list(got) == _scalar(times, defs)
    assert got.index.equals(times.index)


def test_series_uses_local_wall_clock():
    times = pd.Series(pd.to_datetime(["2024-07-01 05:00"]).tz_localize("UTC"))
    berlin = times.dt.tz_convert("Europe/Berlin")  # 07:00 local
    assert list(assign_shift(times, DEFAULT_SHIFTS)) == ["shift_3"]
    assert list(assign_shift(berlin, DEFAULT_SHIFTS)) == ["shift_1"]


def test_series_first_definition_wins_and_unknown():
    defs = {"a": ("06:00", "12:00"), "b": ("08:00", "14:00")}
    times = pd.Series(pd.to_datetime(["2024-01-01 09:00", "2024-01-01 20:00"]))
    assert list(assign_shift(times, defs)) == ["a", "unknown"]


def test_series_full_day_shift():
    # start == end is an overnight shift covering the whole day.
    defs = {"all": ("06:00", "06:00")}
    times = pd.Series(pd.to_datetime(["2024-01-01 05:59", "2024-01-01 06:00"]))
    assert list(assign_shift(times, defs)) == ["all", "all"]


def test_series_nat_is_unknown():
    # The per-row version raised "NaTType does not support time".
    times = pd.Series(pd.to_datetime(["2024-01-01 08:00", None]))
    assert list(assign_shift(times, DEFAULT_SHIFTS)) == ["shift_1", "unknown"]


def test_series_empty_and_index_preserved():
    empty = pd.Series(pd.to_datetime([]))
    assert assign_shift(empty, DEFAULT_SHIFTS).empty
    times = pd.Series(pd.to_datetime(["2024-01-01 08:00"]), index=[42])
    assert assign_shift(times, DEFAULT_SHIFTS).index.tolist() == [42]


def test_production_dates_tz_aware_dst():
    times = pd.Series(
        pd.to_datetime(["2024-03-30 23:00", "2024-03-31 04:00", "2024-03-31 07:00"])
    ).dt.tz_localize("Europe/Berlin")
    shifts = assign_shift(times, DEFAULT_SHIFTS)
    assert list(shifts) == ["shift_3", "shift_3", "shift_1"]
    dates = production_dates(times, shifts, DEFAULT_SHIFTS)
    assert [str(d) for d in dates] == ["2024-03-30", "2024-03-30", "2024-03-31"]
