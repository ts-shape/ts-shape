"""Numeric coercion that reports what it could not convert.

``pd.to_numeric(..., errors="coerce")`` turns every value it cannot parse
into ``NaN`` without a trace, so a column of ``"12,5"`` strings or a stray
``"n/a"`` silently drops out of every downstream calculation.
:func:`to_numeric_checked` performs the same coercion and emits a
:class:`~ts_shape.errors.DataQualityWarning` naming how many non-null values
were not numeric.

:func:`bool_or_false` reads a run/state column as plain ``bool`` with
missing values as ``False``, without relying on pandas' deprecated silent
downcasting of ``object`` columns (``.fillna(False).astype(bool)``).
"""

import warnings

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel

__all__ = ["to_numeric_checked", "bool_or_false"]


def to_numeric_checked(values: pd.Series, *, what: str) -> pd.Series:
    """Coerce ``values`` to numbers, warning when non-null values become NaN.

    Args:
        values: Values to coerce.
        what: Short description of the values for the warning message,
            e.g. ``"counter column 'value_integer'"``.

    Returns:
        A numeric Series aligned with ``values``. Values that are not numeric
        are ``NaN``.

    Warns:
        DataQualityWarning: When at least one non-null value is not numeric.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    lost = int((numeric.isna() & values.notna()).sum())
    if lost:
        warnings.warn(
            f"{lost} non-null value(s) in {what} are not numeric and were "
            "treated as missing (NaN).",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    return numeric


def bool_or_false(values: pd.Series) -> pd.Series:
    """``values`` as a NumPy ``bool`` Series; missing values are ``False``.

    Known values are cast with ``bool`` (as ``.astype(bool)`` does), so the
    result equals ``values.fillna(False).astype(bool)`` for every dtype, but
    an ``object`` column is never filled first, which pandas 2.2 reports as
    a deprecated downcast.

    Args:
        values: State values (``bool``, nullable ``boolean``, numbers or
            ``object``).

    Returns:
        A ``bool`` Series with the index and name of ``values``.
    """
    known = values.notna().to_numpy(dtype=bool)
    out = np.zeros(len(values), dtype=bool)
    if known.any():
        out[known] = values[known].astype(bool).to_numpy(dtype=bool)
    return pd.Series(out, index=values.index, name=values.name)
