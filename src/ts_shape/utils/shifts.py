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

import pandas as pd  # type: ignore

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
    timestamp: pd.Timestamp, shift_definitions: dict[str, tuple[str, str]]
) -> str:
    """Return the shift a timestamp falls in, or ``"unknown"``.

    An overnight shift (start time >= end time, e.g. ``22:00 -> 06:00``) matches
    timestamps at or after its start *or* before its end.
    """
    t = timestamp.time()
    for name, (start, end) in _bounds(shift_definitions).items():
        if start < end:
            if start <= t < end:
                return name
        else:
            if t >= start or t < end:
                return name
    return "unknown"


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
    shifted = (times - pd.Timedelta(days=1)).dt.date
    is_post_midnight = pd.Series(False, index=times.index)
    for name, (start, end) in bounds.items():
        if not _is_overnight(start, end):
            continue
        mask = (shifts.to_numpy() == name) & (times.dt.time < end).to_numpy()
        is_post_midnight |= pd.Series(mask, index=times.index)
    return calendar.where(~is_post_midnight, shifted)
