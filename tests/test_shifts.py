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
