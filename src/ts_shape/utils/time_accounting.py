"""Time and quantity accounting per period (shift, hour, day, ...).

KPIs such as availability, utilization or performance are ratios of
durations and quantities *within a period*. ISO 22400-2 defines them on the
time that actually falls inside the period: a stop that starts at 07:00 and
lasts until 15:00 contributes 7 h of down time to the 06:00-14:00 shift and
1 h to the 14:00-22:00 shift -- not 8 h to the shift it started in.

Booking a whole interval to the period of its *start reading* is a common
shortcut that breaks every period-level KPI on change-encoded data (one row
per transition) and on any interval that crosses a boundary. This module does
the accounting correctly:

* :func:`reading_intervals` -- one ``[t_i, t_{i+1})`` interval per reading,
  the reading's state held until the next one (over-long gaps can be marked
  as "no data").
* :func:`shift_periods`, :func:`grouper_periods`, :func:`floor_periods` --
  the period grid covering a time range: shift windows (overnight shifts and
  the production day included), ``pd.Grouper`` / ``resample`` bins, or
  ``floor``-aligned windows. Wall-clock bounds are localized in the data's
  timezone, so a DST day lasts 23 h or 25 h.
* :func:`split_intervals` -- each interval cut at the period boundaries.
* :func:`time_by_period` -- seconds per ``(period, state)``.
* :func:`spread_increments` -- a counter increment between two readings
  spread over ``(t_prev, t]`` in proportion to time, so the parts produced
  across a boundary are shared between the periods they were produced in.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from ts_shape.utils.frames import localize_in, to_ns, window_offset
from ts_shape.utils.shifts import assign_shift, production_dates

__all__ = [
    "EXCLUDED",
    "reading_intervals",
    "shift_periods",
    "grouper_periods",
    "floor_periods",
    "split_intervals",
    "time_by_period",
    "spread_increments",
    "counter_by_period",
]

#: State label :func:`reading_intervals` gives an over-long gap ("no data").
EXCLUDED = "__excluded__"


def reading_intervals(
    times: pd.Series,
    states: pd.Series,
    *,
    max_gap: Any = None,
) -> pd.DataFrame:
    """One interval per reading: its state held until the next reading.

    Args:
        times: Reading times of one signal, sorted.
        states: State per reading, aligned with ``times``.
        max_gap: Optional cap (``"15min"``, ``Timedelta``). An interval longer
            than this is a logging gap: its ``state`` becomes :data:`EXCLUDED`.

    Returns:
        DataFrame with ``start``, ``end``, ``state``. The last reading has no
        successor and opens no interval.
    """
    t = pd.Series(pd.to_datetime(times).to_numpy(), dtype=times.dtype)
    s = pd.Series(states.to_numpy(), dtype=object)
    if len(t) < 2:
        return pd.DataFrame(
            {
                "start": pd.Series(dtype=t.dtype),
                "end": pd.Series(dtype=t.dtype),
                "state": pd.Series(dtype=object),
            }
        )
    start = t.iloc[:-1].reset_index(drop=True)
    end = t.iloc[1:].reset_index(drop=True)
    state = s.iloc[:-1].reset_index(drop=True)
    if max_gap is not None:
        over = (end - start) > pd.to_timedelta(max_gap)
        state = state.where(~over.to_numpy(), EXCLUDED)
    return pd.DataFrame({"start": start, "end": end, "state": state})


def _empty_periods(keys: Sequence[str], tz: Any) -> pd.DataFrame:
    dtype = f"datetime64[ns, {tz}]" if tz is not None else "datetime64[ns]"
    cols: dict[str, pd.Series] = {k: pd.Series(dtype=object) for k in keys}
    cols["start"] = pd.Series(dtype=dtype)
    cols["end"] = pd.Series(dtype=dtype)
    return pd.DataFrame(cols)


def _tz(ts: pd.Timestamp) -> Any:
    return ts.tz if ts.tzinfo is not None else None


def shift_periods(
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    shift_definitions: Mapping[str, tuple[str, str]],
    *,
    production_day: bool = True,
) -> pd.DataFrame:
    """Shift windows covering ``[t0, t1]``, labeled like the readings would be.

    Each window is labeled with the ``shift`` that
    :func:`~ts_shape.utils.shifts.assign_shift` gives a reading in it
    (``"unknown"`` for time no shift covers) and the ``date`` that
    :func:`~ts_shape.utils.shifts.production_dates` gives it
    (``production_day=True``: an overnight shift is one window on its start
    date) or the calendar date (``production_day=False``: an overnight shift
    is cut at midnight into two windows). Wall-clock bounds are localized in
    the timezone of ``t0``.

    Args:
        t0: Start of the range.
        t1: End of the range.
        shift_definitions: ``{name: ("HH:MM", "HH:MM")}``.
        production_day: Label by production day (default) or calendar date.

    Returns:
        DataFrame with ``date``, ``shift``, ``start``, ``end`` -- contiguous,
        sorted windows overlapping ``[t0, t1]``.
    """
    tz = _tz(t0)
    keys = ["date", "shift"]
    if pd.isna(t0) or pd.isna(t1) or t1 < t0:
        return _empty_periods(keys, tz)
    times_of_day = {_dt.time(0, 0)}
    for start, end in shift_definitions.values():
        times_of_day.add(pd.to_datetime(start).time())
        times_of_day.add(pd.to_datetime(end).time())
    bounds = sorted(times_of_day)
    local0 = t0.tz_localize(None) if tz is not None else t0
    local1 = t1.tz_localize(None) if tz is not None else t1
    first = local0.date() - _dt.timedelta(days=1)
    last = local1.date() + _dt.timedelta(days=1)
    walls: list[pd.Timestamp] = []
    day = first
    while day <= last:
        walls.extend(
            localize_in(pd.Timestamp(_dt.datetime.combine(day, b)), tz) for b in bounds
        )
        day += _dt.timedelta(days=1)
    starts = pd.Series(walls[:-1])
    ends = pd.Series(walls[1:])
    keep = (ends > starts) & (ends > t0) & (starts <= t1)
    starts = starts[keep].reset_index(drop=True)
    ends = ends[keep].reset_index(drop=True)
    if starts.empty:
        return _empty_periods(keys, tz)
    shift = assign_shift(starts, dict(shift_definitions))
    assert isinstance(shift, pd.Series)
    if production_day:
        date = production_dates(starts, shift, dict(shift_definitions))
    else:
        date = starts.dt.date
    seg = pd.DataFrame({"date": date, "shift": shift, "start": starts, "end": ends})
    # Merge consecutive segments that carry the same label.
    label = seg["date"].astype(str) + "\x00" + seg["shift"].astype(str)
    run = (label != label.shift()).cumsum()
    out = seg.groupby(run, sort=True).agg(
        date=("date", "first"),
        shift=("shift", "first"),
        start=("start", "first"),
        end=("end", "last"),
    )
    return out.reset_index(drop=True)


def grouper_periods(t0: pd.Timestamp, t1: pd.Timestamp, window: Any) -> pd.DataFrame:
    """The ``pd.Grouper(freq=window)`` bins covering ``[t0, t1]``.

    Args:
        t0: Start of the range.
        t1: End of the range.
        window: Resample frequency (``"1h"``, ``"1D"``, ``"W"``, ...).

    Returns:
        DataFrame with ``period`` (the resample label), ``start`` and ``end``
        (the true bin bounds, calendar-correct on DST days and for
        right-labeled frequencies), sorted.
    """
    from ts_shape.events._windows import window_bounds

    tz = _tz(t0)
    if pd.isna(t0) or pd.isna(t1) or t1 < t0:
        return _empty_periods(["period"], tz)
    labels = pd.Series(0, index=pd.DatetimeIndex([t0, t1])).resample(window).count()
    period = pd.Series(labels.index)
    start, end = window_bounds(period, window)
    out = pd.DataFrame({"period": period, "start": start, "end": end})
    keep = (out["end"] > t0) & (out["start"] <= t1)
    return out[keep].reset_index(drop=True)


def floor_periods(t0: pd.Timestamp, t1: pd.Timestamp, window: Any) -> pd.DataFrame:
    """Windows labeled by ``floor_local(t, window)`` covering ``[t0, t1]``.

    Args:
        t0: Start of the range.
        t1: End of the range.
        window: Fixed frequency (``"15min"``, ``"1h"``, ``"8h"``, ``"1D"``).

    Returns:
        DataFrame with ``period`` (the floored label), ``start`` (= label)
        and ``end`` (the next label; the last window ends one ``window``
        later, a calendar day for ``"D"``), sorted.
    """
    from ts_shape.events._windows import floor_local

    tz = _tz(t0)
    if pd.isna(t0) or pd.isna(t1) or t1 < t0:
        return _empty_periods(["period"], tz)
    offset = to_offset(window)
    step = pd.Timedelta(hours=1)
    if isinstance(offset, Tick):
        step = min(step, pd.Timedelta(offset))
    probe = pd.Series(pd.date_range(t0, t1, freq=step))
    probe = pd.concat([probe, pd.Series([t1])], ignore_index=True)
    labels = floor_local(probe, window).drop_duplicates().sort_values()
    labels = labels.reset_index(drop=True)
    last_end = labels.iloc[-1] + window_offset(window)
    ends = pd.concat([labels.iloc[1:], pd.Series([last_end])], ignore_index=True)
    return pd.DataFrame({"period": labels, "start": labels, "end": ends})


def _overlaps(
    s: npt.NDArray[np.int64],
    e: npt.NDArray[np.int64],
    ps: npt.NDArray[np.int64],
    pe: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """(interval index, period index, overlap seconds) for positive overlaps."""
    lo = np.searchsorted(pe, s, side="right")
    hi = np.searchsorted(ps, e, side="left")
    counts = np.clip(hi - lo, 0, None)
    total = int(counts.sum())
    idx_int = np.repeat(np.arange(len(s)), counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    idx_per = np.repeat(lo, counts) + offsets
    a = np.maximum(s[idx_int], ps[idx_per])
    b = np.minimum(e[idx_int], pe[idx_per])
    seconds = (b - a).astype("float64") / 1e9
    keep = seconds > 0
    return idx_int[keep], idx_per[keep], seconds[keep]


def _sorted_periods(periods: pd.DataFrame) -> pd.DataFrame:
    return periods.sort_values("start", kind="stable").reset_index(drop=True)


def split_intervals(
    intervals: pd.DataFrame, periods: pd.DataFrame, *, keys: Sequence[str]
) -> pd.DataFrame:
    """Cut each interval at the period boundaries.

    Args:
        intervals: ``start``, ``end`` and any other columns (carried over).
        periods: Non-overlapping periods with ``start``, ``end`` and the
            ``keys`` columns.
        keys: Period label columns copied to each piece.

    Returns:
        One row per (interval, period) with positive overlap: the ``keys``,
        the interval's other columns, ``start`` / ``end`` of the piece,
        ``seconds`` and ``fraction`` (piece / whole interval).
    """
    per = _sorted_periods(periods)
    carried = [c for c in intervals.columns if c not in ("start", "end")]
    if intervals.empty or per.empty:
        cols: dict[str, Any] = {k: pd.Series(dtype=object) for k in keys}
        for c in carried:
            cols[c] = pd.Series(dtype=intervals[c].dtype)
        cols["seconds"] = pd.Series(dtype="float64")
        cols["fraction"] = pd.Series(dtype="float64")
        return pd.DataFrame(cols)
    s, e = to_ns(intervals["start"]), to_ns(intervals["end"])
    ps, pe = to_ns(per["start"]), to_ns(per["end"])
    ii, pi, seconds = _overlaps(s, e, ps, pe)
    out = per.loc[pi, list(keys)].reset_index(drop=True)
    for c in carried:
        out[c] = intervals[c].iloc[ii].to_numpy()
    whole = (e[ii] - s[ii]).astype("float64") / 1e9
    out["seconds"] = seconds
    out["fraction"] = np.divide(
        seconds, whole, out=np.zeros_like(seconds), where=whole > 0
    )
    return out


def time_by_period(
    intervals: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    keys: Sequence[str],
    states: Sequence[str],
) -> pd.DataFrame:
    """Seconds per period and state, intervals split at period boundaries.

    Args:
        intervals: ``start``, ``end``, ``state``.
        periods: Non-overlapping periods with ``start``, ``end``, ``keys``.
        keys: Period label columns.
        states: The state labels to report; each becomes a float column of
            seconds (0 when a period holds none of it). Intervals in other
            states are ignored.

    Returns:
        One row per period that overlaps any interval, in time order, with
        the ``keys``, ``start``, ``end`` and one column per state.
    """
    per = _sorted_periods(periods)
    key_list = list(keys)
    columns = [*key_list, "start", "end", *states]
    ii = pi = np.zeros(0, dtype=np.intp)
    seconds = np.zeros(0, dtype="float64")
    if not intervals.empty and not per.empty:
        s, e = to_ns(intervals["start"]), to_ns(intervals["end"])
        ps, pe = to_ns(per["start"]), to_ns(per["end"])
        ii, pi, seconds = _overlaps(s, e, ps, pe)
    if ii.size == 0:
        cols: dict[str, Any] = {k: pd.Series(dtype=object) for k in key_list}
        cols["start"] = per["start"].iloc[:0].reset_index(drop=True)
        cols["end"] = per["end"].iloc[:0].reset_index(drop=True)
        for st in states:
            cols[st] = pd.Series(dtype="float64")
        return pd.DataFrame(cols)[columns]
    state = intervals["state"].to_numpy(dtype=object)[ii]
    frame = pd.DataFrame({"_p": pi, "state": state, "seconds": seconds})
    table = frame.pivot_table(
        index="_p", columns="state", values="seconds", aggfunc="sum"
    )
    out = per.loc[table.index, [*key_list, "start", "end"]].reset_index(drop=True)
    for st in states:
        col = table[st] if st in table.columns else None
        out[st] = col.fillna(0.0).to_numpy(dtype="float64") if col is not None else 0.0
    return out[columns]


def spread_increments(
    times: pd.Series,
    increments: pd.Series,
    periods: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Counter increments spread over the time they were produced in.

    The increment reported at reading ``t_i`` was produced during
    ``(t_{i-1}, t_i]``. It is shared between the periods that interval
    overlaps, in proportion to the time in each (constant production rate
    between two readings). An increment with no preceding reading, or one
    between two readings at the same time, goes to the period holding its
    reading.

    Args:
        times: Reading times, sorted.
        increments: Increment per reading (aligned with ``times``).
        periods: Non-overlapping periods with ``start``, ``end``, ``keys``.
        keys: Period label columns.

    Returns:
        DataFrame with the ``keys`` and ``quantity`` (float), one row per
        period that received any share (zero shares included).
    """
    per = _sorted_periods(periods)
    key_list = list(keys)
    t = pd.Series(pd.to_datetime(times).to_numpy(), dtype=times.dtype)
    inc = pd.Series(increments.to_numpy(dtype="float64"))
    if t.empty or per.empty:
        cols: dict[str, Any] = {k: pd.Series(dtype=object) for k in key_list}
        cols["quantity"] = pd.Series(dtype="float64")
        return pd.DataFrame(cols)
    tn = to_ns(t)
    prev = np.concatenate([tn[:1], tn[:-1]])
    ps, pe = to_ns(per["start"]), to_ns(per["end"])
    ii, pi, seconds = _overlaps(prev, tn, ps, pe)
    whole = (tn[ii] - prev[ii]).astype("float64") / 1e9
    share = inc.to_numpy()[ii] * seconds / whole
    # Point increments: the first reading and readings at the same time.
    point = np.flatnonzero(tn == prev)
    loc = np.searchsorted(ps, tn[point], side="right") - 1
    hit = (loc >= 0) & (tn[point] < pe[np.clip(loc, 0, None)])
    point_pi = loc[hit]
    point_q = inc.to_numpy()[point[hit]]
    all_pi = np.concatenate([pi, point_pi]).astype(np.intp)
    all_q = np.concatenate([share, point_q])
    if all_pi.size == 0:
        cols = {k: pd.Series(dtype=object) for k in key_list}
        cols["quantity"] = pd.Series(dtype="float64")
        return pd.DataFrame(cols)
    sums = pd.Series(all_q).groupby(all_pi).sum()
    out = per.loc[sums.index, key_list].reset_index(drop=True)
    out["quantity"] = sums.to_numpy(dtype="float64")
    return out


def counter_by_period(
    times: pd.Series,
    values: pd.Series,
    periods: pd.DataFrame,
    reading_keys: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Reset-aware counter quantity and resets per period.

    The reset-aware increments (:func:`~ts_shape.utils.counters.counter_increments`)
    are spread over the time they were produced in
    (:func:`spread_increments`); a reset is a point event and counts in the
    period of the reading that shows it. Missing readings are skipped.

    Args:
        times: Reading times of one counter, sorted.
        values: Counter readings, aligned with ``times``.
        periods: Non-overlapping periods with ``start``, ``end``, ``keys``.
        reading_keys: The ``keys`` of each reading's own period, aligned with
            ``times``; every such period is reported (with 0 when it received
            no production).
        keys: Period label columns.

    Returns:
        DataFrame with the ``keys``, ``quantity`` (float) and ``resets``
        (int), one row per period that holds a reading or received a share,
        sorted by the ``keys``.
    """
    from ts_shape.utils.coerce import to_numeric_checked
    from ts_shape.utils.counters import counter_increments

    key_list = list(keys)
    columns = [*key_list, "quantity", "resets"]
    numeric = to_numeric_checked(values, what="counter readings").astype("float64")
    present = numeric.notna().to_numpy()
    numeric = numeric[present].reset_index(drop=True)
    t = times[present].reset_index(drop=True)
    rk = reading_keys[key_list][present].reset_index(drop=True)
    if numeric.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    increments = counter_increments(numeric)
    shares = spread_increments(t, increments, periods, keys=key_list)
    resets = rk.assign(resets=(numeric.diff() < 0).astype(int).to_numpy())
    resets = resets.groupby(key_list, sort=False, as_index=False)["resets"].sum()
    out = shares.merge(resets, on=key_list, how="outer")
    out["quantity"] = out["quantity"].fillna(0.0).astype(float)
    out["resets"] = out["resets"].fillna(0).astype(int)
    out[key_list] = out[key_list].infer_objects()
    return out.sort_values(key_list, kind="stable").reset_index(drop=True)[columns]
