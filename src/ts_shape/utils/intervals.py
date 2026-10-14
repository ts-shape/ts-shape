"""Helpers for turning a state signal into intervals.

A recurring pattern across the production pack collapses consecutive
equal-state rows into intervals and measures each interval's duration. The
naive implementation closes an interval at the **last sample that still
carries the state**, which never counts the time from that last sample to the
actual transition. Two problems follow:

- On densely, periodically sampled data every interval is undercounted by one
  sample period.
- On **change-encoded** signals (``is_delta=True`` -- one row per transition,
  the library's normal convention) each state is a single row, so
  ``start == end`` and the duration collapses to **zero**. Downtime, micro-stop
  and duty-cycle analyses then silently report nothing.

:func:`state_intervals` closes each interval at the *next* transition instead,
which is correct for both dense and change-encoded data. The behavior is
opt-in via ``interval_end`` so existing callers are unaffected by default.
"""

import pandas as pd  # type: ignore

__all__ = ["state_intervals", "interval_end_times", "INTERVAL_END_MODES"]

INTERVAL_END_MODES = ("last_sample", "next_change")

_COLUMNS = ["start", "end", "state", "duration_seconds", "sample_count"]


def _check_mode(interval_end: str) -> None:
    if interval_end not in INTERVAL_END_MODES:
        raise ValueError(
            f"interval_end must be one of {INTERVAL_END_MODES}, got {interval_end!r}"
        )


def interval_end_times(
    times: pd.Series, interval_end: str = "last_sample"
) -> pd.Series:
    """Per-row closing time for a segment that ends at that row.

    For code that groups consecutive rows into segments itself: a segment
    whose last row has label ``i`` ends at ``interval_end_times(times)[i]``.

    Args:
        times: Timestamps of one signal, ordered in time.
        interval_end: ``"last_sample"`` (default) returns ``times`` unchanged,
            so a segment ends at its own last sample. ``"next_change"``
            returns the *next* row's timestamp, i.e. the instant the following
            segment starts; the final row keeps its own timestamp.

    Returns:
        A Series aligned with ``times``.
    """
    _check_mode(interval_end)
    if interval_end == "last_sample":
        return times
    return times.shift(-1).fillna(times)


def state_intervals(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    state_column: str,
    interval_end: str = "last_sample",
    data_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Collapse consecutive equal-state rows into intervals.

    Args:
        dataframe: Rows for a single signal, ordered in time. Only
            ``time_column`` and ``state_column`` are used.
        time_column: Name of the timestamp column.
        state_column: Name of the (already resolved) state column. Any
            hashable state works -- booleans for run/idle, strings for an
            identifier at a station.
        interval_end: How to close each interval.

            - ``"last_sample"`` (default): ``end`` is the last sample that
              carries the state. Backward compatible; undercounts by one
              sample on dense data and collapses to zero-length intervals on
              change-encoded signals.
            - ``"next_change"``: ``end`` is the timestamp of the next state
              change (the transition instant), so the interval spans the real
              time the state was held. The final, still-open interval is closed
              at ``data_end`` when provided, otherwise at its own last sample.
        data_end: Optional bound for the final open interval when
            ``interval_end="next_change"`` (e.g. the end of the data window or
            "now"). Ignored in ``"last_sample"`` mode.

    Returns:
        DataFrame with columns ``start``, ``end``, ``state``,
        ``duration_seconds`` (float), ``sample_count`` -- one row per interval,
        in time order.

    Raises:
        ValueError: If ``interval_end`` is not one of :data:`INTERVAL_END_MODES`.
    """
    _check_mode(interval_end)

    if dataframe.empty:
        return pd.DataFrame(columns=_COLUMNS)

    d = dataframe[[time_column, state_column]].copy()
    d[time_column] = pd.to_datetime(d[time_column])

    group = (d[state_column] != d[state_column].shift()).cumsum()
    agg = (
        d.groupby(group)
        .agg(
            start=(time_column, "first"),
            _last_sample=(time_column, "last"),
            state=(state_column, "first"),
            sample_count=(time_column, "size"),
        )
        .reset_index(drop=True)
    )

    if interval_end == "next_change":
        # Close each interval at the next interval's start (the transition).
        end = agg["start"].shift(-1)
        fallback = agg["_last_sample"]
        if data_end is not None:
            fallback = fallback.copy()
            fallback.iloc[-1] = pd.Timestamp(data_end)
        end = end.fillna(fallback)
    else:
        end = agg["_last_sample"]

    agg["end"] = end
    agg["duration_seconds"] = (agg["end"] - agg["start"]).dt.total_seconds()
    return agg[_COLUMNS]
