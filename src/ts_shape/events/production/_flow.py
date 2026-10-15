"""Helpers shared by the flow-metrics and line-balancing detectors.

* :func:`to_seconds` -- a duration given as seconds or a pandas offset
  string, in seconds.
* :func:`rising_edge_times` -- the times a boolean signal turns True.
"""

from __future__ import annotations

import pandas as pd

from ts_shape.utils.coerce import bool_or_false

__all__ = ["Number", "to_seconds", "rising_edge_times"]

#: A duration as seconds (``int`` / ``float``) or an offset string (``"55s"``).
Number = int | float | str


def to_seconds(value: Number | None) -> float | None:
    """Coerce a duration (seconds number or pandas offset string) to seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    seconds: float = pd.to_timedelta(value).total_seconds()
    return seconds


def rising_edge_times(
    dataframe: pd.DataFrame, uuid: str, *, time_column: str, value_column: str
) -> pd.Series:
    """Timestamps of the rising edges (False -> True) of one boolean signal.

    A missing value counts as False; a signal True at its first sample has
    a rising edge there.

    Args:
        dataframe: Long-format frame.
        uuid: Signal to read.
        time_column: Timestamp column.
        value_column: Boolean column.

    Returns:
        The edge times, in time order, with a fresh index.
    """
    sig = dataframe[dataframe["uuid"] == uuid].copy().sort_values(time_column)
    if sig.empty:
        return pd.Series(dtype="datetime64[ns]")
    sig[time_column] = pd.to_datetime(sig[time_column])
    state = bool_or_false(sig[value_column])
    prev = state.shift(fill_value=False)
    edges = sig[(~prev) & state]
    return edges[time_column].reset_index(drop=True)
