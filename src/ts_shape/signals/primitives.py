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

import numpy as np
import pandas as pd

from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.counters import counter_increments, counter_per_window
from ts_shape.utils.frames import (
    as_datetime,
    localize_like,
    require_columns,
    same_time_axis,
    window_offset,
)
from ts_shape.utils.normalize import normalize_signal_frame, text_mask
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


def _frame_times(
    dataframe: pd.DataFrame, spec: SignalSpec, time_column: str, uuid_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Rows of the signal and their times as ``datetime64``."""
    require_columns(dataframe, (time_column, spec.value_column), where=spec.uuid)
    rows = dataframe
    if uuid_column in dataframe.columns:
        rows = dataframe[text_mask(dataframe[uuid_column], str(spec.uuid))]
    times = as_datetime(rows[time_column], where=f"signal {spec.uuid!r}")
    return rows, times


def signal_series(
    dataframe: pd.DataFrame,
    spec: SignalSpec,
    *,
    time_column: str = "systime",
    uuid_column: str = "uuid",
) -> pd.Series:
    """The signal's values as a time-indexed Series (sorted, NaN/NaT dropped).

    Rows of other uuids are ignored, so a multi-signal frame can be passed
    (uuids are compared as text). Duplicate timestamps keep the last value.
    Nullable and pyarrow dtypes are converted to NumPy dtypes.

    Args:
        dataframe: Long-format frame.
        spec: The signal's spec (``uuid`` and ``value_column``).
        time_column: Timestamp column.
        uuid_column: Signal identifier column; ignored when absent.

    Returns:
        Values indexed by time, named after ``spec.uuid``.

    Raises:
        ColumnNotFoundError: When ``time_column`` or ``spec.value_column`` is
            missing.
    """
    return _series(dataframe, spec, time_column, uuid_column, keep_missing=False)


def _series(
    dataframe: pd.DataFrame,
    spec: SignalSpec,
    time_column: str,
    uuid_column: str,
    *,
    keep_missing: bool,
) -> pd.Series:
    rows, times = _frame_times(dataframe, spec, time_column, uuid_column)
    values = normalize_signal_frame(
        rows[[spec.value_column]].copy(), time_column=time_column
    )[spec.value_column]
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(times), name=spec.uuid)
    series = series[series.index.notna()]
    if not keep_missing:
        series = series.dropna()
    series = series.sort_index(kind="stable")
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

    Args:
        dataframe: Long-format frame.
        spec: The signal's spec.
        data_end: End of the last interval; a naive bound on tz-aware data is
            read in the data's timezone.
        time_column: Timestamp column.

    Returns:
        DataFrame with ``start``, ``end``, ``state``, ``duration_seconds``,
        ``sample_count`` -- one row per interval, in time order.

    Raises:
        ColumnNotFoundError: When ``time_column`` or ``spec.value_column`` is
            missing.
    """
    s = signal_series(dataframe, spec, time_column=time_column)
    if s.empty:
        return pd.DataFrame(
            {
                "start": pd.Series(dtype=s.index.dtype),
                "end": pd.Series(dtype=s.index.dtype),
                "state": pd.Series(dtype=object),
                "duration_seconds": pd.Series(dtype="float64"),
                "sample_count": pd.Series(dtype="int64"),
            }
        )

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
    last_end = (
        localize_like(data_end, times) if data_end is not None else times.iloc[-1]
    )
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

    Args:
        dataframe: Long-format frame.
        spec: The counter's spec.
        time_column: Timestamp column.

    Returns:
        Float increments indexed by reading time.

    Raises:
        ValueError: If the spec is not a counter.
        ColumnNotFoundError: When ``time_column`` or ``spec.value_column`` is
            missing.
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
      than ``spec.max_gap``. For a boolean state this is the fraction of the
      *known* time it was True: a missing reading (``NA``) is "unknown" until
      the next reading and counts neither as True nor as False.
    * ``event`` -- number of events per window.

    Args:
        dataframe: Long-format frame.
        spec: The signal's spec.
        window: Window length as a pandas offset (``"1h"``, ``"1D"``).
        data_end: Hold the last value until this time; a naive bound on
            tz-aware data is read in the data's timezone.
        time_column: Timestamp column.

    Returns:
        A float Series indexed by window start and named after ``spec.uuid``
        (for every kind).

    Raises:
        ValueError: For a non-numeric (e.g. string) state; use
            :func:`intervals` to get time per state instead.
        ColumnNotFoundError: When ``time_column`` or ``spec.value_column`` is
            missing.
    """
    s = _series(dataframe, spec, time_column, "uuid", keep_missing=True)
    end = localize_like(data_end, s.index.to_series()) if data_end is not None else None
    if spec.kind == "counter":
        out = counter_per_window(
            pd.to_numeric(s.dropna()),
            window,
            handle_resets=spec.reset_policy != "none",
        )
    elif spec.kind == "event":
        out = s.dropna().resample(window).count().astype(float)
    elif _is_boolean(s):
        known = s.notna().astype(float)
        true = pd.Series(
            [1.0 if v is True or v is np.True_ else 0.0 for v in s.tolist()],
            index=s.index,
        )
        held_true = time_weighted_resample(
            true, window, data_end=end, max_gap=spec.max_gap
        )
        held_known = time_weighted_resample(
            known, window, data_end=end, max_gap=spec.max_gap
        )
        out = (held_true / held_known).where(held_known > 0)
    else:
        present = s.dropna()
        if present.dtype == object or pd.api.types.is_string_dtype(present):
            raise ValueError(
                f"per_window() cannot average string state {spec.uuid!r}; "
                "use intervals() for time per state"
            )
        out = time_weighted_resample(
            present.astype(float), window, data_end=end, max_gap=spec.max_gap
        )
    return out.astype(float).rename(spec.uuid)


def _is_boolean(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return True
    present = series.dropna()
    return (
        series.dtype == object
        and len(present) > 0
        and all(isinstance(v, bool | np.bool_) for v in present.tolist())
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
    to a future setpoint, no stale value after an outage. The two time
    columns are brought to one unit first (``ns``, ``us``, ``s`` and pyarrow
    timestamps mix freely); tz-aware right times are converted to the left
    timezone.

    Args:
        left: Frame with the left signal.
        left_spec: Spec of the left signal.
        right: Frame with the right signal.
        right_spec: Spec of the right signal.
        time_column: Timestamp column of both frames; also the name of the
            output time column.

    Returns:
        DataFrame with ``<time_column>``, ``<left uuid>``, ``<right uuid>``.
        When both specs have the same uuid the value columns are
        ``<uuid>_left`` and ``<uuid>_right``.

    Raises:
        ValueError: When one side is tz-aware and the other naive.
        ColumnNotFoundError: When a frame lacks ``time_column`` or its value
            column.
    """
    a = signal_series(left, left_spec, time_column=time_column)
    b = signal_series(right, right_spec, time_column=time_column)
    left_name, right_name = str(left_spec.uuid), str(right_spec.uuid)
    if left_name == right_name:
        left_name, right_name = f"{left_name}_left", f"{right_name}_right"
    ta, tb = same_time_axis(a.index.to_series(), b.index.to_series(), where="align")
    if a.empty:
        return pd.DataFrame(
            {
                time_column: pd.Series(dtype=ta.dtype),
                left_name: pd.Series(dtype=a.dtype),
                right_name: pd.Series(dtype=b.dtype),
            }
        )
    lhs = pd.DataFrame(
        {time_column: ta.reset_index(drop=True), "a": a.reset_index(drop=True)}
    )
    rhs = pd.DataFrame(
        {time_column: tb.reset_index(drop=True), "b": b.reset_index(drop=True)}
    )
    merged = pd.merge_asof(
        lhs,
        rhs,
        on=time_column,
        direction="backward",
        tolerance=right_spec.max_gap,
    )
    return merged.rename(columns={"a": left_name, "b": right_name})


def _same_tz(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Read a naive bound in the timezone of the other, aware one."""
    if start.tzinfo is None and end.tzinfo is not None:
        return start.tz_localize(end.tz), end
    if end.tzinfo is None and start.tzinfo is not None:
        return start, end.tz_localize(start.tz)
    return start, end


def windows(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    freq: str,
    *,
    tz: str | None = None,
) -> pd.DataFrame:
    """Analysis windows covering ``[start, end)``, laid out like ``resample``.

    Daily and longer frequencies (``"1D"``, ``"1W"``, ``"MS"`` ...) follow
    the calendar in ``tz``: a ``"1D"`` window on a DST day is 23 h or 25 h
    long, not 24 h, and every window starts at local midnight. Sub-daily
    frequencies (``"12h"``, ``"8h"``, ``"15min"``) are fixed lengths counted
    from midnight of the first day, as ``resample`` counts them: after a DST
    change their wall-clock starts shift by the DST offset (``"12h"`` starts
    at 13:00+02:00 after the spring change, not at 12:00). For wall-clock
    windows such as shifts use :func:`shift_windows`.

    Example:
        >>> from ts_shape.signals.primitives import windows
        >>> w = windows("2024-03-30", "2024-04-01", "12h", tz="Europe/Berlin")
        >>> for row in w.itertuples():
        ...     print(row.start, row.end, row.duration_seconds)
        2024-03-30 00:00:00+01:00 2024-03-30 12:00:00+01:00 43200.0
        2024-03-30 12:00:00+01:00 2024-03-31 00:00:00+01:00 43200.0
        2024-03-31 00:00:00+01:00 2024-03-31 13:00:00+02:00 43200.0
        2024-03-31 13:00:00+02:00 2024-04-01 01:00:00+02:00 43200.0
        >>> list(windows("2024-03-30", "2024-04-01", "1D", tz="Europe/Berlin")
        ...      ["duration_seconds"])
        [86400.0, 82800.0]

    Args:
        start: The first window covers this time.
        end: Exclusive end of the range.
        freq: Window length as a pandas offset (``"1D"``, ``"8h"``).
        tz: Calendar timezone; naive bounds are localized in it, aware bounds
            converted. Without ``tz``, a naive bound next to an aware one is
            read in the aware one's timezone.

    Returns:
        DataFrame with ``start``, ``end``, ``duration_seconds``; empty when
        ``start >= end``.
    """
    t0, t1 = _localize(pd.Timestamp(start), tz), _localize(pd.Timestamp(end), tz)
    t0, t1 = _same_tz(t0, t1)
    offset = window_offset(freq)
    if t0 >= t1:
        starts = pd.DatetimeIndex([], tz=t0.tz)
    else:
        starts = (
            pd.Series(0, index=pd.DatetimeIndex([t0, t1])).resample(freq).count().index
        )
        starts = starts[starts < t1]
    ends = pd.DatetimeIndex([s + offset for s in starts], tz=starts.tz)
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

    Args:
        start: Start of the range.
        end: Exclusive end of the range.
        shift_definitions: ``{name: ("HH:MM", "HH:MM")}``; default
            :data:`ts_shape.utils.shifts.DEFAULT_SHIFTS`.
        tz: Timezone of the wall-clock shift times. Without it, an aware
            bound makes the shifts use that bound's timezone.

    Returns:
        DataFrame with ``production_day``, ``shift``, ``start``, ``end``,
        ``duration_seconds``; empty when ``start >= end``.
    """
    shifts = dict(shift_definitions or DEFAULT_SHIFTS)
    t0, t1 = _localize(pd.Timestamp(start), tz), _localize(pd.Timestamp(end), tz)
    t0, t1 = _same_tz(t0, t1)
    zone = tz if tz is not None else (str(t0.tz) if t0.tzinfo is not None else None)
    rows = []
    if t0 < t1:
        day = (t0 - pd.Timedelta(days=1)).date()
        while day <= t1.date():
            for name, (s_str, e_str) in shifts.items():
                s_time = pd.to_datetime(s_str).time()
                e_time = pd.to_datetime(e_str).time()
                e_day = day + _dt.timedelta(days=1) if s_time >= e_time else day
                w_start = _wall(day, s_time, zone)
                w_end = _wall(e_day, e_time, zone)
                if w_end > t0 and w_start < t1:
                    rows.append((day, name, w_start, w_end))
            day += _dt.timedelta(days=1)
    if not rows:
        time_dtype = f"datetime64[ns, {zone}]" if zone else "datetime64[ns]"
        return pd.DataFrame(
            {
                "production_day": pd.Series(dtype=object),
                "shift": pd.Series(dtype=object),
                "start": pd.Series(dtype=time_dtype),
                "end": pd.Series(dtype=time_dtype),
                "duration_seconds": pd.Series(dtype="float64"),
            }
        )
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
