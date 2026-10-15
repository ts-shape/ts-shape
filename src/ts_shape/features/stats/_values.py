"""Column access shared by the statistics classes.

Every statistic reads its column through one of these functions, so a
missing column raises :class:`~ts_shape.errors.ColumnNotFoundError` and
nullable, pyarrow and text columns give the same result as NumPy columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.frames import as_datetime, require_columns
from ts_shape.utils.normalize import _is_extension

__all__ = ["numeric_values", "bool_values", "text_values", "time_values"]


def numeric_values(dataframe: pd.DataFrame, column: str, *, where: str) -> pd.Series:
    """The column as a NumPy numeric Series.

    Nullable and pyarrow numbers become ``int64`` (no missing values) or
    ``float64``; text is parsed, with a
    :class:`~ts_shape.errors.DataQualityWarning` for values that are not
    numbers; an empty untyped column becomes ``float64``.

    Raises:
        ColumnNotFoundError: If the column is missing.
    """
    require_columns(dataframe, [column], where=where)
    values = dataframe[column]
    if _is_extension(values):
        if ptypes.is_bool_dtype(values.dtype):
            values = values.astype("float64")
        elif ptypes.is_integer_dtype(values.dtype) and not values.isna().any():
            return values.astype("int64")
        elif ptypes.is_numeric_dtype(values.dtype):
            return values.astype("float64")
        else:
            values = values.astype(object).where(values.notna(), np.nan)
    if ptypes.is_numeric_dtype(values.dtype):
        return values
    numbers = to_numeric_checked(values, what=f"column {column!r}")
    return numbers.astype("float64")


def bool_values(dataframe: pd.DataFrame, column: str, *, where: str) -> pd.Series:
    """The column with ``True`` / ``False`` / NaN values (object or bool).

    Raises:
        ColumnNotFoundError: If the column is missing.
    """
    require_columns(dataframe, [column], where=where)
    values = dataframe[column]
    if _is_extension(values):
        if not values.isna().any() and ptypes.is_bool_dtype(values.dtype):
            return values.astype(bool)
        return values.astype(object).where(values.notna(), np.nan)
    return values


def text_values(dataframe: pd.DataFrame, column: str, *, where: str) -> pd.Series:
    """The column with a working ``.str`` accessor (missing stays missing).

    Raises:
        ColumnNotFoundError: If the column is missing.
    """
    require_columns(dataframe, [column], where=where)
    values = dataframe[column]
    if values.dtype == object and values.notna().any():
        return values
    return values.astype(object).where(values.notna(), np.nan)


def time_values(dataframe: pd.DataFrame, column: str, *, where: str) -> pd.Series:
    """The column as ``datetime64`` (text parsed, empty columns typed).

    Raises:
        ColumnNotFoundError: If the column is missing.
    """
    require_columns(dataframe, [column], where=where)
    return as_datetime(dataframe[column], where=where)
