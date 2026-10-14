"""Per-group counter totals that add up to the signal total.

Computing :func:`~ts_shape.utils.counters.counter_quantity` separately inside
each group (shift, day, operator, part, period ...) loses the production
between the last reading of one group and the first reading of the next.
These helpers compute the reset-aware increments once over the whole signal
(:func:`~ts_shape.utils.counters.counter_by_group`) and book each increment to
the group of the reading that reports it, so the group totals add up to the
signal total. ``handle_resets=False`` keeps the naive per-group
``max(0, last - first)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.counters import counter_by_group, counter_increments

_SELECT = "__counter_select__"


def counter_totals(
    frame: pd.DataFrame,
    value_column: str,
    keys: Sequence[str],
    *,
    handle_resets: bool,
    select: pd.Series | None = None,
) -> pd.DataFrame:
    """Counter quantity and resets per group of one time-sorted counter signal.

    Args:
        frame: Readings of one counter signal, sorted by time, with the
            ``keys`` columns already assigned per reading.
        value_column: Column holding the counter readings.
        keys: Group columns (e.g. ``["date", "shift"]``).
        handle_resets: When True, reset-aware increments are computed over the
            whole signal and booked to the group of the reading that reports
            them. When False each group reports ``max(0, last - first)`` of its
            own readings.
        select: Optional boolean mask (index-aligned with ``frame``) of the
            readings whose groups are reported. The increments are still
            computed on the full signal, so the first selected reading carries
            the production since the reading before it.

    Returns:
        One row per group present among the (selected) readings, sorted by
        ``keys``, with the key columns, ``quantity`` (float) and ``resets``
        (int). A group whose readings are all missing reports ``0``.
    """
    key_list = list(keys)
    columns = [*key_list, "quantity", "resets"]
    base = frame if select is None else frame.loc[select.astype(bool)]
    groups = base[key_list].dropna().drop_duplicates()
    if frame.empty or groups.empty:
        return pd.DataFrame(columns=columns)

    key_series = [frame[k] for k in key_list]
    if select is not None:
        key_series = [select.astype(bool).rename(_SELECT), *key_series]
    totals = counter_by_group(
        frame[value_column], key_series, handle_resets=handle_resets
    ).reset_index()
    if select is not None:
        totals = totals[totals[_SELECT]].drop(columns=_SELECT)

    out = groups.merge(totals, on=key_list, how="left")
    # Key columns as a list-of-dicts frame would type them (text -> str).
    out[key_list] = out[key_list].infer_objects()
    out["quantity"] = out["quantity"].fillna(0.0).astype(float)
    out["resets"] = out["resets"].fillna(0).astype(int)
    return out.sort_values(key_list, kind="stable").reset_index(drop=True)[columns]


def counter_totals_per_period(
    values: pd.Series, window: str, *, handle_resets: bool
) -> pd.DataFrame:
    """Counter quantity and resets per time window, for non-empty windows.

    Args:
        values: Counter readings with a ``DatetimeIndex``, ordered in time.
        window: Resample frequency (e.g. ``'1D'``).
        handle_resets: When True the reset-aware increments of the whole
            signal are summed per window (each booked to the window of the
            reading that reports it). When False each window reports
            ``max(0, last - first)`` of its own readings.

    Returns:
        A DataFrame with ``period`` (window label), ``quantity`` (float) and
        ``resets`` (int), one row per window holding at least one reading.
    """
    numeric = to_numeric_checked(values, what="counter readings").astype("float64")
    present = numeric.dropna()
    size = numeric.resample(window).size()
    if handle_resets:
        quantity = counter_increments(present).resample(window).sum()
    else:
        ends = present.resample(window).agg(["first", "last"])
        quantity = (ends["last"] - ends["first"]).clip(lower=0)
    resets = (present.diff() < 0).astype(int).resample(window).sum()
    out = pd.DataFrame(
        {
            "quantity": quantity.reindex(size.index).fillna(0.0).astype(float),
            "resets": resets.reindex(size.index).fillna(0).astype(int),
        }
    )
    out = out[size > 0]
    out.index.name = "period"
    return out.reset_index()
