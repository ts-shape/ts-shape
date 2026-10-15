"""Bring a long-format signal frame to the dtypes the detectors compute with.

Data read from parquet, databases or CSV arrives in many dtypes: pandas
nullable types (``Int64``, ``Float64``, ``boolean``, ``string``), pyarrow
types (``int64[pyarrow]``, ``bool[pyarrow]``), or numbers stored as text.
pandas arithmetic on these differs from NumPy in ways that change results
without an error: a comparison with a missing value is ``<NA>`` instead of
``False``, ``cumsum`` is not defined for ``bool[pyarrow]``, and text numbers
concatenate instead of adding up.

:func:`normalize_signal_frame` converts once, up front, so every detector
sees the same NumPy-backed dtypes:

* numeric value columns become ``int64`` (no missing values) or ``float64``;
* numbers stored as text in ``value_double`` / ``value_integer`` are parsed,
  with a :class:`~ts_shape.errors.DataQualityWarning` counting the values that
  are not numbers;
* boolean value columns become ``bool`` (no missing values) or ``object``
  holding ``True`` / ``False`` / ``NaN``, as pandas builds them from NumPy
  data;
* string columns (``value_string``, the uuid column) become ``object``;
* a pyarrow timestamp time column becomes ``datetime64``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel

__all__ = ["normalize_signal_frame", "NUMERIC_VALUE_COLUMNS"]

#: Value columns that hold numbers; text in them is parsed.
NUMERIC_VALUE_COLUMNS = ("value_double", "value_integer")


def _is_extension(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.api.extensions.ExtensionDtype) and not (
        isinstance(series.dtype, pd.DatetimeTZDtype | pd.CategoricalDtype)
    )


def _plain(series: pd.Series) -> pd.Series:
    """One extension-dtype column as the equivalent NumPy-backed column."""
    dtype = series.dtype
    has_missing = bool(series.isna().any())
    if ptypes.is_bool_dtype(dtype):
        if not has_missing:
            return series.astype(bool)
        values = series.astype(object)
        return values.where(series.notna(), np.nan)
    if ptypes.is_integer_dtype(dtype) and not has_missing:
        return series.astype("int64")
    if ptypes.is_numeric_dtype(dtype):
        return series.astype("float64")
    if ptypes.is_datetime64_any_dtype(dtype):
        return pd.to_datetime(series.astype(object))
    values = series.astype(object)
    return values.where(series.notna(), np.nan)


def _arrow_timestamp(series: pd.Series) -> bool:
    dtype = series.dtype
    return isinstance(dtype, pd.ArrowDtype) and ptypes.is_datetime64_any_dtype(dtype)


def normalize_signal_frame(
    frame: pd.DataFrame,
    *,
    time_column: str = "systime",
    uuid_column: str = "uuid",
) -> pd.DataFrame:
    """Return ``frame`` with NumPy-backed dtypes (see the module docstring).

    Only the time column, the uuid column and ``value_*`` columns are
    touched; other columns keep their dtype. ``frame`` is changed in place
    and returned.

    Args:
        frame: Long-format signal data (a copy owned by the caller).
        time_column: Timestamp column.
        uuid_column: Signal identifier column.

    Returns:
        The same frame object, with converted columns.
    """
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        series = frame[column]
        if column == time_column:
            if _arrow_timestamp(series):
                frame[column] = _plain(series)
            continue
        if column != uuid_column and not column.startswith("value_"):
            continue
        if _is_extension(series):
            series = _plain(series)
            frame[column] = series
        if column in NUMERIC_VALUE_COLUMNS and series.dtype == object:
            frame[column] = _parse_numbers(frame, column, uuid_column)
    return frame


def _parse_numbers(frame: pd.DataFrame, column: str, uuid_column: str) -> pd.Series:
    """Text numbers as floats; warn naming the signals with unparseable text."""
    series = frame[column]
    numeric = pd.to_numeric(series, errors="coerce")
    lost = numeric.isna() & series.notna()
    if lost.any():
        where = ""
        if uuid_column in frame.columns:
            uuids = [repr(str(u)) for u in pd.unique(frame.loc[lost, uuid_column])]
            label = "signal" if len(uuids) == 1 else "signals"
            more = f" (+{len(uuids) - 5} more)" if len(uuids) > 5 else ""
            where = f" for {label} {', '.join(uuids[:5])}{more}"
        warnings.warn(
            f"{int(lost.sum())} non-null value(s) in column {column!r}{where} are "
            "not numeric and were treated as missing (NaN).",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    return numeric
