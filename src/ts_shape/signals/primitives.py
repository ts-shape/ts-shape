"""Spec-driven primitives: the correct way to do the five recurring operations.

Every detector in ts-shape boils down to a few operations on a signal. Done
naively, each has a classic failure on industrial data:

=================  ==========================================================
Operation          Naive failure
=================  ==========================================================
intervals          closes at the last sample: zero-length on change-encoded
                   data, one period short on sampled data, bridges outages
increments         ``last - first`` loses production at a counter reset
per_window         ``resample().mean()`` weights samples, not time
align              ``merge_asof(nearest)`` looks ahead to future setpoints
windows            fixed ``Timedelta`` windows break on DST days / night shifts
=================  ==========================================================

The functions here take a :class:`~ts_shape.signals.spec.SignalSpec` and do
the right thing for that signal's kind and encoding. They are pure functions
on pandas objects (no hidden state), so they can later get alternative
backends without changing their contract.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping

import pandas as pd  # type: ignore
from pandas.tseries.frequencies import to_offset

from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.counters import counter_increments, counter_per_window
from ts_shape.utils.shifts import DEFAULT_SHIFTS
from ts_shape.utils.time_weighted import time_weighted_resample

__all__ = [
    "signal_series",
    "intervals",
    "increments",
    "per_window",
    "align",
    "windows",
    "shift_windows",
]

INTERVAL_COLUMNS = ["start", "end", "state", "duration_seconds", "sample_count"]


def signal_series(
    dataframe: pd.DataFrame,
    spec: SignalSpec,
    *,
    time_column: str = "systime",
    uuid_column: str = "uuid",
) -> pd.Series:
    """The signal's values as a time-indexed Series (sorted, NaN/NaT dropped).

    Rows of other uuids are ignored, so a multi-signal frame can be passed.
    Duplicate timestamps keep the last value.
    """
    rows = dataframe
    if uuid_column in dataframe.columns:
        rows = dataframe[dataframe[uuid_column] == spec.uuid]
    times = pd.to_datetime(rows[time_column])
    series = pd.Series(rows[spec.value_column].to_numpy(), index=times, name=spec.uuid)
    series = series[series.index.notna()].dropna().sort_index(kind="stable")
    return series[~series.index.duplicated(keep="last")]


def intervals(
    dataframe: pd.DataFrame,
    spec: SignalSpec,
    *,
    data_end: pd.Timestamp | None = None,
    time_column: str = "systime",
) -> pd.DataFrame:
    """Constant-value intervals of a held signal, correctly closed.

    * A held value (state, counter, level) lasts until the next sample, for
      both encodings; an ``event`` signal's interval ends at its last sample.
    * A gap longer than ``spec.max_gap`` is "no data": the interval before it
      ends at its last sample and a new interval starts after the gap, even
      if the value is unchanged.
    * The final interval ends at ``data_end`` when given (e.g. "now" or the
      end of the analysis window), otherwise at its last sample.

    Returns:
        DataFrame with ``start``, ``end``, ``state``, ``duration_seconds``,
        ``sample_count`` -- one row per interval, in time order.
    """
    s = signal_series(dataframe, spec, time_column=time_column)
    if s.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)

    times = s.index.to_series()
    next_time = times.shift(-1)
    gap_after = (
        (next_time - times) > spec.max_gap
        if spec.max_gap is not None
        else (pd.Series(False, index=times.index))
    )
    gap_after = gap_after.fillna(False).astype(bool)
    new_segment = (s != s.shift()) | gap_after.shift(fill_value=False)
    segment = new_segment.cumsum()

    if spec.holds_value:
        end = next_time.where(~gap_after, times)
    else:
        end = times.copy()
    last_end = pd.Timestamp(data_end) if data_end is not None else times.iloc[-1]
    end.iloc[-1] = max(last_end, times.iloc[-1]) if spec.holds_value else times.iloc[-1]

    frame = pd.DataFrame(
        {"t": times.to_numpy(), "end": end.to_numpy(), "state": s.to_numpy()},
        index=times.index,
    )
    frame["seg"] = segment.to_numpy()
    out = frame.groupby("seg", sort=True).agg(
        start=("t", "first"),
        end=("end", "last"),
        state=("state", "first"),
        sample_count=("t", "size"),
    )
    out = out.reset_index(drop=True)
    out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
    return out[INTERVAL_COLUMNS]


def increments(
    dataframe: pd.DataFrame,
    spec: SignalSpec,
    *,
    time_column: str = "systime",
) -> pd.Series:
    """Per-reading increase of a counter, reset-aware.

    Each value is the quantity produced since the previous reading, booked
    at the reading that reports it (the first reading contributes 0). With
    ``reset_policy="reset_to_zero"`` a drop is a reset and the post-reset
    value counts as new production; with ``"none"`` decreases are kept
    (negative increments).

    Raises:
        ValueError: If the spec is not a counter.
    """
    if spec.kind != "counter":
        raise ValueError(f"increments() needs a counter; {spec.uuid!r} is {spec.kind}")
    s = pd.to_numeric(signal_series(dataframe, spec, time_column=time_column))
    if spec.reset_policy == "none":
        return s.diff().fillna(0.0).astype(float)
    return counter_increments(s)


def per_window(
    dataframe: pd.DataFrame,
    spec: SignalSpec,
    window: str,
    *,
    data_end: pd.Timestamp | None = None,
    time_column: str = "systime",
) -> pd.Series:
    """The right per-window aggregate for the signal's kind.

    * ``counter`` -- quantity produced in each window (reset-aware increments
      summed per window; no production lost at window boundaries).
    * ``continuous`` / numeric or boolean ``state`` -- time-weighted mean,
      holding each value until the next sample and never across a gap longer
      than ``spec.max_gap``. For a boolean state this is the fraction of time
      it was True.
    * ``event`` -- number of events per window.

    Raises:
        ValueError: For a non-numeric (e.g. string) state; use
            :func:`intervals` to get time per state instead.
    """
    s = signal_series(dataframe, spec, time_column=time_column)
    if spec.kind == "counter":
        return counter_per_window(
            pd.to_numeric(s), window, handle_resets=spec.reset_policy != "none"
        )
    if spec.kind == "event":
        return s.resample(window).count().astype(float)
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        raise ValueError(
            f"per_window() cannot average string state {spec.uuid!r}; "
            "use intervals() for time per state"
        )
    numeric = s.astype(float)
    return time_weighted_resample(
        numeric, window, data_end=data_end, max_gap=spec.max_gap
    )


def align(
    left: pd.DataFrame,
    left_spec: SignalSpec,
    right: pd.DataFrame,
    right_spec: SignalSpec,
    *,
    time_column: str = "systime",
) -> pd.DataFrame:
    """Pair each sample of ``left`` with the ``right`` value in force at that time.

    The right signal is held backward (the last value at or before each left
    sample) and never across more than ``right_spec.max_gap``: no look-ahead
    to a future setpoint, no stale value after an outage.

    Returns:
        DataFrame with ``systime``, ``<left uuid>``, ``<right uuid>``.
    """
    a = signal_series(left, left_spec, time_column=time_column).rename("a")
    b = signal_series(right, right_spec, time_column=time_column).rename("b")
    if a.empty:
        return pd.DataFrame(columns=["systime", left_spec.uuid, right_spec.uuid])
    merged = pd.merge_asof(
        a.rename_axis("systime").reset_index(),
        b.rename_axis("systime").reset_index(),
        on="systime",
        direction="backward",
        tolerance=right_spec.max_gap,
    )
    return merged.rename(columns={"a": left_spec.uuid, "b": right_spec.uuid})


def windows(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    freq: str,
    *,
    tz: str | None = None,
) -> pd.DataFrame:
    """Calendar-correct analysis windows covering ``[start, end)``.

    Window bounds follow the calendar in ``tz``: a ``"1D"`` window on a DST
    day is 23 h or 25 h long, not 24 h.

    Returns:
        DataFrame with ``start``, ``end``, ``duration_seconds``.
    """
    t0, t1 = _localize(pd.Timestamp(start), tz), _localize(pd.Timestamp(end), tz)
    offset = to_offset(freq)
    starts = pd.Series(0, index=pd.DatetimeIndex([t0, t1])).resample(freq).count().index
    starts = starts[starts < t1]
    ends = pd.DatetimeIndex([s + offset for s in starts])
    out = pd.DataFrame({"start": starts, "end": ends})
    out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
    return out


def shift_windows(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    shift_definitions: Mapping[str, tuple[str, str]] | None = None,
    *,
    tz: str | None = None,
) -> pd.DataFrame:
    """One row per (production day, shift) overlapping ``[start, end)``.

    Overnight shifts (e.g. ``22:00 -> 06:00``) are a single window attributed
    to the day they start; wall-clock shift times are localized in ``tz`` so
    a shift spanning a DST change has its real length (7 h or 9 h).

    Returns:
        DataFrame with ``production_day``, ``shift``, ``start``, ``end``,
        ``duration_seconds``.
    """
    shifts = dict(shift_definitions or DEFAULT_SHIFTS)
    t0, t1 = _localize(pd.Timestamp(start), tz), _localize(pd.Timestamp(end), tz)
    rows = []
    day = (t0 - pd.Timedelta(days=1)).date()
    while day <= t1.date():
        for name, (s_str, e_str) in shifts.items():
            s_time = pd.to_datetime(s_str).time()
            e_time = pd.to_datetime(e_str).time()
            e_day = day + _dt.timedelta(days=1) if s_time >= e_time else day
            w_start = _wall(day, s_time, tz)
            w_end = _wall(e_day, e_time, tz)
            if w_end > t0 and w_start < t1:
                rows.append((day, name, w_start, w_end))
        day += _dt.timedelta(days=1)
    out = pd.DataFrame(rows, columns=["production_day", "shift", "start", "end"])
    out = out.sort_values("start").reset_index(drop=True)
    out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
    return out


def _localize(ts: pd.Timestamp, tz: str | None) -> pd.Timestamp:
    if tz is None:
        return ts
    return ts.tz_convert(tz) if ts.tzinfo is not None else ts.tz_localize(tz)


def _wall(day: _dt.date, time: _dt.time, tz: str | None) -> pd.Timestamp:
    """A wall-clock time on ``day`` in ``tz`` (DST gaps shift forward)."""
    ts = pd.Timestamp(_dt.datetime.combine(day, time))
    if tz is None:
        return ts
    return ts.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
