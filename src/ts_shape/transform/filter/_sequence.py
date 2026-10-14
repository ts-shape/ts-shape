"""Row order and per-signal grouping shared by the change and edge filters.

A change or an edge compares a sample with the previous sample of the same
signal in time. The filters return the caller's rows unchanged and in the
caller's order; only the comparison runs in time order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_shape.utils.frames import as_datetime


def time_order(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    uuid_column: str,
    group_by_uuid: bool,
    where: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions of ``dataframe`` in comparison order, and a group id per row.

    Args:
        dataframe: Input frame.
        time_column: Rows are ordered by this column when the frame has it
            (stable, so rows with the same time keep their order).
        uuid_column: Signal column; used for grouping when present.
        group_by_uuid: Compare within each signal (``True``) or across the
            whole frame (``False``).
        where: Name of the calling function, for error messages.

    Returns:
        ``(order, groups)``: ``order`` holds row positions in comparison
        order; ``groups`` holds, for each position of ``order``, an integer
        group id (all zeros when not grouping).
    """
    n = len(dataframe)
    order = np.arange(n)
    if time_column in dataframe.columns and n:
        times = as_datetime(dataframe[time_column], where=where)
        if times.dt.tz is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        keys = times.to_numpy(dtype="datetime64[ns]").view("int64")
        nat = times.isna().to_numpy()
        # NaT sorts last, like ``sort_values``.
        keys = np.where(nat, np.iinfo(np.int64).max, keys)
        order = np.argsort(keys, kind="stable")
    if group_by_uuid and uuid_column in dataframe.columns and n:
        codes, _ = pd.factorize(dataframe[uuid_column], use_na_sentinel=False)
        codes = np.asarray(codes)[order]
        regroup = np.argsort(codes, kind="stable")
        return order[regroup], codes[regroup]
    return order, np.zeros(n, dtype=np.int64)


def scatter(mask_in_order: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Put a boolean mask computed in comparison order back in row order."""
    out = np.zeros(len(order), dtype=bool)
    out[order] = mask_in_order
    return out


def is_first(groups: np.ndarray) -> np.ndarray:
    """True for the first position of each group (``groups`` is sorted)."""
    first = np.ones(len(groups), dtype=bool)
    first[1:] = groups[1:] != groups[:-1]
    return first
