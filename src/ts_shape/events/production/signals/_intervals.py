"""Interval helpers shared by the production signal normalizers.

Two operations show up in every state- or cycle-based production metric, and
both are easy to get subtly wrong:

* **Run collapsing** -- turning a step signal (a state that holds until the next
  update) into intervals. The trap is ending each interval at the *last sample
  of the run* rather than at the *next sample*, which silently discards the time
  between them. On a signal that only reports on change, that is the entire
  duration.
* **Window splitting** -- an interval that straddles a window boundary belongs
  partly to each window. Assigning it to one window by its start timestamp is
  what makes hourly availability add up to more or less than the shift figure.

Both live here so every metric computes them the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

__all__ = [
    "collapse_runs",
    "debounce_intervals",
    "floor_timestamp",
    "split_intervals",
    "window_edges",
]


def floor_timestamp(timestamp: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Round ``timestamp`` down to the start of its ``freq`` window.

    ``Timestamp.floor`` only accepts fixed frequencies, so anchored offsets
    (``"1W"``, ``"MS"``, ``"QS"``) are rolled back to their previous anchor
    instead. That makes weekly and monthly production reports work with the
    same code path as hourly ones.
    """
    offset = to_offset(freq)
    if isinstance(offset, Tick):
        return timestamp.floor(freq)
    normalized = timestamp.normalize()
    if offset.is_on_offset(normalized):
        return normalized
    return offset.rollback(normalized)


def window_edges(lo: pd.Timestamp, hi: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    """Return window boundaries fully covering ``[lo, hi]``.

    The first edge is at or before ``lo`` and the last at or after ``hi``, so
    every point of the range falls inside exactly one window.

    Args:
        lo: Earliest timestamp to cover.
        hi: Latest timestamp to cover.
        freq: Pandas offset alias, fixed (``"15min"``) or anchored (``"1W"``).

    Returns:
        A monotonically increasing :class:`pandas.DatetimeIndex` of at least two
        entries.
    """
    offset = to_offset(freq)
    start = floor_timestamp(lo, freq)
    edges = pd.date_range(start=start, end=max(hi, start) + offset, freq=freq)
    # An anchored offset can land its last edge before ``hi`` (e.g. weekly edges
    # when ``hi`` is mid-week); extend until the range is genuinely covered.
    guard = 0
    while edges[-1] < hi and guard < 1000:
        edges = pd.date_range(start=start, end=edges[-1] + offset, freq=freq)
        guard += 1
    if len(edges) < 2:
        edges = pd.DatetimeIndex([start, start + offset])
    return edges


def collapse_runs(
    frame: pd.DataFrame,
    *,
    value_column: str,
    time_column: str = "systime",
    until: pd.Timestamp | None = None,
    carry_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Collapse a step signal into intervals of constant value.

    Each interval starts at the sample that introduced the value and ends where
    the *next* value begins -- step semantics, the way a PLC signal actually
    behaves. The final interval is closed at ``until``.

    Args:
        frame: Sorted samples with ``time_column`` and ``value_column``.
        value_column: Column whose runs of equal values form the intervals.
        time_column: Timestamp column.
        until: Closing timestamp of the last interval. Defaults to the last
            sample, which gives it zero duration -- pass the end of the
            observation window (or "now") to account for the open interval.
        carry_columns: Extra columns to carry from the first sample of each run.

    Returns:
        A frame with ``start``, ``end``, ``duration_seconds``, ``value_column``
        and any ``carry_columns``.
    """
    columns = ["start", "end", "duration_seconds", value_column, *carry_columns]
    if frame.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    work = frame.reset_index(drop=True)
    run_id = _run_ids(work[value_column])

    first = work.groupby(run_id, sort=True).head(1).reset_index(drop=True)
    starts = first[time_column].reset_index(drop=True)
    horizon = pd.Timestamp(until) if until is not None else work[time_column].iloc[-1]
    ends = starts.shift(-1)
    ends.iloc[-1] = max(horizon, starts.iloc[-1])

    out = pd.DataFrame(
        {
            "start": starts,
            "end": ends,
            "duration_seconds": (ends - starts).dt.total_seconds(),
            value_column: first[value_column].reset_index(drop=True),
        }
    )
    for column in carry_columns:
        out[column] = first[column].reset_index(drop=True)
    return out


def debounce_intervals(
    intervals: pd.DataFrame,
    *,
    min_duration: str,
    value_column: str,
    max_passes: int = 50,
) -> pd.DataFrame:
    """Absorb intervals shorter than ``min_duration`` into their predecessor.

    A bouncing contact or a state bit that flickers for one scan cycle turns a
    day into tens of thousands of intervals, none of them meaningful. Absorbing
    a too-short interval extends the interval before it, then re-merges any
    neighbours that became equal -- repeated until stable, because one merge can
    expose another.

    The first interval has no predecessor, so a too-short leading interval is
    absorbed into its *successor* instead.

    Args:
        intervals: Frame with ``start``, ``end`` and ``value_column``.
        min_duration: Pandas offset string, e.g. ``"2s"``.
        value_column: Column identifying the interval's value/state.
        max_passes: Safety bound on the merge loop.

    Returns:
        A frame with the same columns, re-indexed from zero.
    """
    if intervals.empty or len(intervals) == 1:
        return intervals.reset_index(drop=True)
    threshold = pd.to_timedelta(min_duration)
    if threshold <= pd.Timedelta(0):
        return intervals.reset_index(drop=True)

    outer_start = intervals["start"].iloc[0]
    outer_end = intervals["end"].iloc[-1]
    work = intervals.reset_index(drop=True).copy()
    for _ in range(max_passes):
        work = _merge_equal_neighbours(work, value_column=value_column)
        durations = work["end"] - work["start"]
        short = durations < threshold
        if not short.any() or len(work) == 1:
            break
        # Drop every short interval and hand its time to the neighbour that
        # keeps the timeline gapless: the predecessor extends over it, and a
        # short *leading* interval is absorbed by its successor instead.
        keep = ~short
        if not keep.any():
            # Everything is short -- keep the single longest so the timeline
            # survives rather than returning nothing.
            keep = pd.Series(False, index=work.index)
            keep.iloc[int(durations.to_numpy().argmax())] = True
        work = work[keep].reset_index(drop=True)
        work["end"] = work["start"].shift(-1)
        work.loc[work.index[-1], "end"] = outer_end
        work.loc[work.index[0], "start"] = outer_start
    work = _merge_equal_neighbours(work, value_column=value_column)
    work["duration_seconds"] = (work["end"] - work["start"]).dt.total_seconds()
    return work.reset_index(drop=True)


def _run_ids(values: pd.Series) -> pd.Series:
    """Number consecutive runs of equal values, starting at 1.

    Compared on the string form so nulls group together and an enum/str mix
    counts as one run. The first element must be forced to start a run: on a
    nullable dtype ``values != values.shift()`` yields ``pd.NA`` there, and an
    ``NA`` group key would silently drop the whole first run from any
    subsequent ``groupby``.
    """
    keys = values.astype("string").fillna("\x00<na>")
    changed = keys.ne(keys.shift()).astype("boolean").fillna(True).astype(bool)
    changed.iloc[0] = True
    return changed.cumsum()


def _merge_equal_neighbours(
    intervals: pd.DataFrame, *, value_column: str
) -> pd.DataFrame:
    """Join consecutive intervals carrying the same value."""
    if intervals.empty:
        return intervals
    run_id = _run_ids(intervals[value_column])
    grouped = intervals.groupby(run_id, sort=True)
    merged = grouped.head(1).reset_index(drop=True).copy()
    merged["end"] = grouped["end"].max().reset_index(drop=True)
    merged["duration_seconds"] = (merged["end"] - merged["start"]).dt.total_seconds()
    return merged


def split_intervals(
    intervals: pd.DataFrame,
    freq: str,
    *,
    start_column: str = "start",
    end_column: str = "end",
) -> pd.DataFrame:
    """Cut intervals at window boundaries and tag each piece with its window.

    An interval crossing a boundary contributes to both windows. Splitting it is
    what makes per-hour state times sum exactly to the per-shift figures.

    Args:
        intervals: Frame with datetime ``start_column`` and ``end_column``.
        freq: Window size as a pandas offset alias.
        start_column: Name of the interval start column.
        end_column: Name of the interval end column.

    Returns:
        A copy with one row per (interval, window) piece: ``start_column`` and
        ``end_column`` clipped to the window, ``duration_seconds`` recomputed,
        and a ``window_start`` column. All other columns are carried through.
    """
    if intervals.empty:
        out = intervals.copy()
        out["window_start"] = pd.Series(dtype="datetime64[ns]")
        return out

    work = intervals.reset_index(drop=True).copy()
    work[start_column] = pd.to_datetime(work[start_column])
    work[end_column] = pd.to_datetime(work[end_column])

    edges = window_edges(work[start_column].min(), work[end_column].max(), freq)
    edge_values = edges.to_numpy()
    starts = work[start_column].to_numpy()
    ends = work[end_column].to_numpy()

    # First boundary strictly inside the interval, and the first at/after its
    # end: the count of interior boundaries is the number of extra pieces.
    first_inside = np.searchsorted(edge_values, starts, side="right")
    first_after = np.searchsorted(edge_values, ends, side="left")
    pieces = np.maximum(first_after - first_inside, 0) + 1

    row_index = np.repeat(np.arange(len(work)), pieces)
    within = np.concatenate([np.arange(n) for n in pieces])
    base = first_inside[row_index]

    piece_start = np.where(
        within == 0,
        starts[row_index],
        edge_values[np.clip(base + within - 1, 0, len(edge_values) - 1)],
    )
    is_last = within == (pieces[row_index] - 1)
    piece_end = np.where(
        is_last,
        ends[row_index],
        edge_values[np.clip(base + within, 0, len(edge_values) - 1)],
    )

    out = work.iloc[row_index].reset_index(drop=True)
    out[start_column] = pd.to_datetime(piece_start)
    out[end_column] = pd.to_datetime(piece_end)
    out["duration_seconds"] = (out[end_column] - out[start_column]).dt.total_seconds()
    window_index = np.clip(
        np.searchsorted(edge_values, piece_start, side="right") - 1, 0, None
    )
    out["window_start"] = pd.to_datetime(edge_values[window_index])
    return out
