"""Shift assignment and production-day helpers.

Production shifts are grouped by ``(date, shift)``. A shift that crosses
midnight -- e.g. a night shift ``22:00 -> 06:00`` -- breaks the naive
``timestamp.date()`` grouping: a reading at 23:00 and a reading at 03:00 the
*next* morning belong to the **same** physical shift, but they carry two
different calendar dates. Worse, within a single calendar date the night-shift
group then spans 00:00-06:00 *and* 22:00-23:59 with an 16-hour hole, so a
``last - first`` over that group sweeps in the whole day's production.

The fix is a **production day**: the post-midnight portion of an overnight
shift is attributed to the shift's *start* date, so each ``(date, shift)`` group
is one contiguous interval. :func:`production_dates` computes it; it is opt-in
at the call sites so day-only schedules are unaffected.
"""

import datetime as _dt

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_SHIFTS", "assign_shift", "production_dates"]

DEFAULT_SHIFTS: dict[str, tuple[str, str]] = {
    "shift_1": ("06:00", "14:00"),
    "shift_2": ("14:00", "22:00"),
    "shift_3": ("22:00", "06:00"),
}


def _bounds(
    shift_definitions: dict[str, tuple[str, str]],
) -> dict[str, tuple[_dt.time, _dt.time]]:
    return {
        name: (pd.to_datetime(start).time(), pd.to_datetime(end).time())
        for name, (start, end) in shift_definitions.items()
    }


def assign_shift(
    timestamp: pd.Timestamp | _dt.datetime | pd.Series,
    shift_definitions: dict[str, tuple[str, str]],
) -> str | pd.Series:
    """Return the shift a timestamp falls in, or ``"unknown"``.

    An overnight shift (start time >= end time, e.g. ``22:00 -> 06:00``) matches
    timestamps at or after its start *or* before its end. Shifts are tried in
    the order of ``shift_definitions``; the first match wins.

    Pass a datetime Series to label every row at once (vectorized). The local
    wall-clock time is used for tz-aware data, and a missing timestamp (NaT)
    is labeled ``"unknown"``.

    Args:
        timestamp: A single timestamp, or a datetime Series.
        shift_definitions: Mapping of shift name to ``(start, end)`` strings.

    Returns:
        The shift name, or a Series of shift names aligned to ``timestamp``.
    """
    bounds = _bounds(shift_definitions)
    if isinstance(timestamp, pd.Series):
        return _assign_shift_series(timestamp, bounds)
    t = timestamp.time()
    for name, (start, end) in bounds.items():
        if start < end:
            if start <= t < end:
                return name
        else:
            if t >= start or t < end:
                return name
    return "unknown"


def _time_us(t: _dt.time) -> int:
    return ((t.hour * 60 + t.minute) * 60 + t.second) * 1_000_000 + t.microsecond


def _time_of_day_us(timestamps: pd.Series) -> np.ndarray:
    """Local wall-clock time of day in microseconds (NaN for NaT).

    Matches ``Timestamp.time()``: nanoseconds are dropped and tz-aware values
    use their local time.
    """
    times = timestamps
    if not pd.api.types.is_datetime64_any_dtype(times.dtype):
        times = pd.to_datetime(times)
    dt = times.dt
    us: np.ndarray = (
        ((dt.hour.astype("float64") * 60 + dt.minute) * 60 + dt.second) * 1_000_000
        + dt.microsecond
    ).to_numpy(dtype="float64", na_value=np.nan)
    return us


def _assign_shift_series(
    timestamps: pd.Series, bounds: dict[str, tuple[_dt.time, _dt.time]]
) -> pd.Series:
    tod = _time_of_day_us(timestamps)
    labels = np.full(len(tod), "unknown", dtype=object)
    free = ~np.isnan(tod)
    for name, (start, end) in bounds.items():
        if not free.any():
            break
        s, e = _time_us(start), _time_us(end)
        if start < end:
            hit = (tod >= s) & (tod < e)
        else:
            hit = (tod >= s) | (tod < e)
        hit &= free
        labels[hit] = name
        free &= ~hit
    return pd.Series(labels, index=timestamps.index, dtype=object)


def _is_overnight(start: _dt.time, end: _dt.time) -> bool:
    return start >= end


def production_dates(
    timestamps: pd.Series,
    shifts: pd.Series,
    shift_definitions: dict[str, tuple[str, str]],
) -> pd.Series:
    """Production date for each timestamp given its assigned shift.

    For a reading in the post-midnight portion of an overnight shift (its time
    is before the shift's end), the production date is the *previous* calendar
    day -- the shift's start date. All other readings keep their calendar date.

    Args:
        timestamps: Datetime Series.
        shifts: Shift name per timestamp (from :func:`assign_shift`), aligned to
            ``timestamps``.
        shift_definitions: Mapping of shift name to ``(start, end)`` strings.

    Returns:
        A Series of :class:`datetime.date`, aligned to ``timestamps``.
    """
    bounds = _bounds(shift_definitions)
    times = pd.to_datetime(timestamps)
    calendar = times.dt.date
    # A reading belongs to the previous day when its shift is overnight and its
    # time is before that shift's end (the small hours after midnight).
    # Subtract one *calendar* day from the local date, not 24 hours: on
    # tz-aware data the day after a DST change is 23 or 25 hours away, and
    # ``time - 24h`` would land on the wrong date (e.g. 00:30 on the day
    # after the spring change -> two days back).
    wall = times.dt.tz_localize(None) if times.dt.tz is not None else times
    shifted = (wall.dt.normalize() - pd.Timedelta(days=1)).dt.date
    tod = _time_of_day_us(times)
    names = shifts.to_numpy()
    mask = np.zeros(len(times), dtype=bool)
    for name, (start, end) in bounds.items():
        if not _is_overnight(start, end):
            continue
        mask |= (names == name) & (tod < _time_us(end))
    is_post_midnight = pd.Series(mask, index=times.index)
    return calendar.where(~is_post_midnight, shifted)
