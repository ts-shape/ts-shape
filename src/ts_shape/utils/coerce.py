"""Numeric coercion that reports what it could not convert.

``pd.to_numeric(..., errors="coerce")`` turns every value it cannot parse
into ``NaN`` without a trace, so a column of ``"12,5"`` strings or a stray
``"n/a"`` silently drops out of every downstream calculation.
:func:`to_numeric_checked` performs the same coercion and emits a
:class:`~ts_shape.errors.DataQualityWarning` naming how many non-null values
were not numeric.
"""

import warnings

import pandas as pd

from ts_shape.errors import DataQualityWarning

__all__ = ["to_numeric_checked"]


def to_numeric_checked(
    values: pd.Series, *, what: str, stacklevel: int = 3
) -> pd.Series:
    """Coerce ``values`` to numbers, warning when non-null values become NaN.

    Args:
        values: Values to coerce.
        what: Short description of the values for the warning message,
            e.g. ``"counter column 'value_integer'"``.
        stacklevel: Passed to :func:`warnings.warn`. The default points at
            the caller of the function that calls this helper.

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
            stacklevel=stacklevel,
        )
    return numeric
