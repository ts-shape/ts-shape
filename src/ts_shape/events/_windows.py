"""Window bounds for summary and window-flag outputs (shared by all packs).

Summaries carry ``start`` / ``end`` / ``duration_seconds`` (see
:mod:`ts_shape.events._output`). Detectors usually know only a window label:
a ``resample`` / ``pd.Grouper`` label, a calendar ``date``, a ``(date,
shift)`` pair. These helpers derive the real bounds, in the data's timezone
and calendar-correct (a DST day lasts 23 or 25 hours, a month its real
length).

Resample labels need care: for right-labeled frequencies (``'W'``, ``'ME'``,
``'QE'``, ``'YE'`` ...) pandas labels a bin with its *last* day, so the label
is not the window start. :func:`window_bounds` turns such a label into the
true ``[start, end)``.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import pandas as pd
from pandas.tseries.frequencies import to_offset

from ts_shape.events._output import (
    COL_DURATION_S,
    COL_END,
    COL_START,
    empty_event_df,
    finalize_interval_df,
    finalize_summary_df,
)

__all__ = [
    "WINDOW_COLS",
    "window_bounds",
    "with_window_bounds",
    "add_window_bounds",
    "add_duration_seconds",
    "window_summary",
    "finish_events",
    "data_tz",
    "add_shift_window",
    "add_day_window",
    "add_period_window",
]

WINDOW_COLS: list[str] = [COL_START, COL_END, COL_DURATION_S]

_RIGHT_LABELED = {"ME", "YE", "QE", "BME", "BYE", "BQE", "W", "M", "A", "Y", "Q"}

Window: TypeAlias = "str | pd.Timedelta | pd.DateOffset"


def _right_labeled(window: Window) -> bool:
    """Whether ``resample(window)`` labels bins by their right edge."""
    if not isinstance(window, (str, pd.DateOffset)):
        return False
    try:
        code = to_offset(window).rule_code
    except (TypeError, ValueError):
        return False
    return code.split("-")[0] in _RIGHT_LABELED


def _step(window: Window) -> Any:
    return to_offset(window) if isinstance(window, str) else window


def window_bounds(labels: pd.Series, window: Window) -> tuple[pd.Series, pd.Series]:
    """``(start, end)`` of the windows whose resample labels are ``labels``.

    Left-labeled windows (``'h'``, ``'D'``, ``'MS'`` ...): start = label, end =
    label + window. Right-labeled ones (``'W'``, ``'ME'`` ...): the bin is
    ``(label - window, label]`` by day, so start = the day after
    ``label - window`` and end = the day after ``label``. A ``Timedelta``
    window is added as a fixed duration.
    """
    label = pd.to_datetime(labels)
    step = _step(window)
    if _right_labeled(window):
        day = pd.DateOffset(days=1)
        return label.dt.normalize() - step + day, label.dt.normalize() + day
    return label, label + step


def with_window_bounds(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Set ``start`` to the true window start and add ``end`` / ``duration_seconds``.

    ``df["start"]`` holds resample labels. Any existing ``end`` /
    ``duration_seconds`` columns are replaced; the new ones follow ``start``.
    """
    out = df.drop(columns=[c for c in (COL_END, COL_DURATION_S) if c in df.columns])
    if out.empty:
        pos = list(out.columns).index(COL_START) + 1
        out.insert(pos, COL_END, pd.Series(dtype=object, index=out.index))
        out.insert(pos + 1, COL_DURATION_S, pd.Series(dtype=float, index=out.index))
        return out
    start, end = window_bounds(out[COL_START], window)
    out[COL_START] = start
    pos = list(out.columns).index(COL_START) + 1
    out.insert(pos, COL_END, end)
    out.insert(pos + 1, COL_DURATION_S, (end - start).dt.total_seconds())
    return out


def add_window_bounds(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Add ``end`` and ``duration_seconds`` for windows starting at ``start``.

    Like :func:`with_window_bounds`, but existing ``end`` /
    ``duration_seconds`` columns are kept as they are.
    """
    out = df.copy()
    if COL_END not in out.columns:
        if out.empty:
            out[COL_END] = pd.Series(dtype=out[COL_START].dtype, index=out.index)
        else:
            start, end = window_bounds(out[COL_START], window)
            out[COL_START] = start
            out[COL_END] = end
    if COL_DURATION_S not in out.columns:
        out[COL_DURATION_S] = _seconds(out)
    return out


def add_duration_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Add float ``duration_seconds`` (``end`` - ``start``) when missing."""
    out = df.copy()
    if COL_DURATION_S not in out.columns:
        out[COL_DURATION_S] = _seconds(out)
    return out


def _seconds(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    delta = pd.to_datetime(df[COL_END]) - pd.to_datetime(df[COL_START])
    return delta.dt.total_seconds().astype(float)


def window_summary(
    rows: pd.DataFrame | list,
    cols: list[str],
    window: Window,
    *,
    uuid: str | None = None,
    source_uuid: str | None = None,
) -> pd.DataFrame:
    """Canonical summary frame for one row per ``window`` starting at ``start``.

    ``rows`` are records (or a frame) with the columns ``cols`` (``start``
    first). Adds ``end`` / ``duration_seconds`` (and ``uuid`` /
    ``source_uuid`` when given); an empty input yields the full-schema empty
    frame.
    """
    extra = [c for c in cols if c != COL_START]
    if uuid is not None or source_uuid is not None:
        extra = ["uuid", "source_uuid", *extra]
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=cols)
    if df.empty:
        return empty_event_df("summary", extra)
    return finalize_summary_df(
        add_window_bounds(df[cols], window), uuid=uuid, source_uuid=source_uuid
    )


def finish_events(
    events: list[dict[str, Any]],
    cols: Sequence[str],
    shape: Literal["interval", "summary"],
    *,
    uuid: str | None,
    source_uuid: str | None,
    window: Window | None = None,
) -> pd.DataFrame:
    """Build a canonical interval/summary frame from per-row dicts.

    Rows that carry only ``start`` get ``end`` = one ``window`` later
    (calendar-correct). Adds ``duration_seconds``, ``uuid`` and
    ``source_uuid``; empty input yields an empty frame with the same columns.
    """
    full = list(cols)
    if COL_END not in full:
        if window is None:
            raise ValueError("window is required when rows have no 'end' column")
        full.insert(full.index(COL_START) + 1, COL_END)
    if not events:
        return empty_event_df(shape, ["uuid", "source_uuid", *full])
    df = pd.DataFrame(events, columns=list(cols))
    if COL_END not in df.columns:
        start, end = window_bounds(df[COL_START], window)  # type: ignore[arg-type]
        df[COL_START] = start
        df.insert(df.columns.get_loc(COL_START) + 1, COL_END, end)
    finalize = finalize_interval_df if shape == "interval" else finalize_summary_df
    return finalize(df, uuid=uuid, source_uuid=source_uuid)


# ---------------------------------------------------------------------------
# Calendar keys: date, (date, shift), resampled period
# ---------------------------------------------------------------------------


def data_tz(series: pd.Series) -> Any:
    """Return the timezone of a datetime Series (``None`` when naive/not datetime)."""
    try:
        return pd.to_datetime(series).dt.tz
    except (AttributeError, TypeError, ValueError):
        return None


def _localize(ts: pd.Timestamp, tz: Any) -> pd.Timestamp:
    if tz is None:
        return ts
    return ts.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")


def _day(value: Any) -> pd.Timestamp | None:
    if value is None or (not isinstance(value, _dt.date) and pd.isna(value)):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _as_index(values: list[Any], tz: Any) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if tz is not None and idx.tz is None:
        idx = idx.tz_localize(tz)
    return idx


def _insert_window(
    df: pd.DataFrame, starts: list[Any], ends: list[Any], tz: Any
) -> pd.DataFrame:
    out = df.drop(columns=[c for c in WINDOW_COLS if c in df.columns])
    start = pd.Series(_as_index(starts, tz), index=out.index)
    end = pd.Series(_as_index(ends, tz), index=out.index)
    out.insert(0, COL_START, start)
    out.insert(1, COL_END, end)
    out.insert(2, COL_DURATION_S, (end - start).dt.total_seconds())
    return out


def _prepend_empty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i, col in enumerate(WINDOW_COLS):
        if col not in out.columns:
            dtype = float if col == COL_DURATION_S else object
            out.insert(i, col, pd.Series(dtype=dtype, index=out.index))
    return out


def add_shift_window(
    df: pd.DataFrame,
    shift_definitions: dict[str, tuple[str, str]],
    tz: Any,
    *,
    date_col: str = "date",
    shift_col: str = "shift",
) -> pd.DataFrame:
    """Prepend ``start`` / ``end`` / ``duration_seconds`` of each shift window.

    ``start`` is ``date`` at the shift's start time; ``end`` is the shift's
    end time on the same day, or on the next day for an overnight shift
    (start time >= end time, e.g. ``22:00 -> 06:00``). Both are localized to
    ``tz``. Rows whose shift is not in ``shift_definitions`` get ``NaT``.
    """
    if df.empty:
        return _prepend_empty(df)
    bounds = {
        name: (pd.to_datetime(start).time(), pd.to_datetime(end).time())
        for name, (start, end) in shift_definitions.items()
    }
    starts: list[Any] = []
    ends: list[Any] = []
    for date, shift in zip(df[date_col], df[shift_col], strict=False):
        day = _day(date)
        if day is None or shift not in bounds:
            starts.append(pd.NaT)
            ends.append(pd.NaT)
            continue
        t_start, t_end = bounds[shift]
        start = pd.Timestamp.combine(day.date(), t_start)
        end_day = day + pd.Timedelta(days=1) if t_start >= t_end else day
        end = pd.Timestamp.combine(end_day.date(), t_end)
        starts.append(_localize(start, tz))
        ends.append(_localize(end, tz))
    return _insert_window(df, starts, ends, tz)


def add_day_window(
    df: pd.DataFrame,
    tz: Any,
    *,
    date_col: str = "date",
    offset: pd.DateOffset | None = None,
) -> pd.DataFrame:
    """Prepend calendar bounds: ``date`` midnight to ``date + offset`` midnight.

    ``offset`` defaults to one calendar day; pass e.g. ``pd.DateOffset(days=7)``
    for weeks or ``pd.DateOffset(months=1)`` for months. Bounds are wall-clock
    midnights localized to ``tz``, so a DST-change day lasts 23 or 25 hours.
    """
    step = offset if offset is not None else pd.DateOffset(days=1)
    if df.empty:
        return _prepend_empty(df)
    starts: list[Any] = []
    ends: list[Any] = []
    for date in df[date_col]:
        day = _day(date)
        if day is None:
            starts.append(pd.NaT)
            ends.append(pd.NaT)
            continue
        starts.append(_localize(day, tz))
        ends.append(_localize(day + step, tz))
    return _insert_window(df, starts, ends, tz)


def add_period_window(
    df: pd.DataFrame, freq: str, *, period_col: str = "period"
) -> pd.DataFrame:
    """Prepend the bounds of resampled periods (``period`` from ``pd.Grouper``).

    See :func:`window_bounds`; the ``period`` column is kept unchanged.
    """
    out = df.copy()
    if out.empty:
        return _prepend_empty(out)
    start, end = window_bounds(out[period_col], freq)
    out = out.drop(columns=[c for c in WINDOW_COLS if c in out.columns])
    out.insert(0, COL_START, start)
    out.insert(1, COL_END, end)
    out.insert(2, COL_DURATION_S, (end - start).dt.total_seconds())
    return out
