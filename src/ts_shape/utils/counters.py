"""Reset-aware helpers for monotonic production counters.

Many industrial counters -- parts produced, scrap, rejects, cumulative
energy -- are expected to increase monotonically. In the field they
frequently reset back to zero (or to a lower value) at a shift change, a
part change, or a controller (PLC) restart. A naive ``last - first``
calculation then silently loses all of the production accumulated in the
window where the reset happened.

These helpers treat any decrease in the counter as a reset and count the
post-reset value as new production, mirroring the approach in
:class:`ts_shape.events.production.part_tracking.PartProductionTracking`.
Use them wherever a quantity is derived from a cumulative counter so the
behaviour is consistent across the library.

Example::

    from ts_shape.utils.counters import counter_quantity

    # A shift where the counter reset once: 100, 300, 500, 0, 150
    values = pd.Series([100, 300, 500, 0, 150])
    counter_quantity(values)                       # 50   (naive last - first)
    counter_quantity(values, handle_resets=True)   # 550  (400 + 150)
"""

import pandas as pd  # type: ignore

__all__ = ["counter_increments", "counter_quantity", "count_resets"]


def counter_increments(values: pd.Series) -> pd.Series:
    """Per-reading production increments from a (possibly resetting) counter.

    A decrease between consecutive readings is interpreted as a reset: the
    counter is assumed to have dropped to (at most) zero and then climbed to
    its current value, so the new value is the production that happened after
    the reset. The first reading has no predecessor and therefore contributes
    ``0`` -- it establishes the baseline rather than counting as production.

    Args:
        values: Counter readings, ordered in time. Non-numeric entries are
            coerced to NaN and treated as gaps.

    Returns:
        A float Series, index-aligned with ``values``, of the parts produced
        since the previous reading.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    prev = numeric.shift(1)
    delta = numeric - prev
    is_reset = delta < 0
    # On a reset the post-reset counter value is the production since the reset.
    increment = delta.where(~is_reset, numeric)
    return increment.fillna(0.0)


def counter_quantity(values: pd.Series, *, handle_resets: bool = False) -> float:
    """Total production represented by a counter series.

    Args:
        values: Counter readings, ordered in time.
        handle_resets: When ``False`` (default) the quantity is the
            backward-compatible ``max(0, last - first)``, so a window
            containing a reset reports too little (often ``0``). When
            ``True`` the quantity is the sum of reset-aware per-reading
            increments (see :func:`counter_increments`).

    Returns:
        The total quantity as a float. Empty input returns ``0.0``.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    if handle_resets:
        return float(counter_increments(numeric).sum())
    return float(max(0.0, numeric.iloc[-1] - numeric.iloc[0]))


def count_resets(values: pd.Series) -> int:
    """Number of resets (strictly decreasing steps) in a counter series.

    Args:
        values: Counter readings, ordered in time.

    Returns:
        The count of readings that are lower than their predecessor.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    return int((numeric.diff() < 0).sum())
