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

__all__ = [
    "normalize_signal_frame",
    "NUMERIC_VALUE_COLUMNS",
    "text_keys",
    "text_mask",
    "unique_text",
]

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
    if ptypes.is_string_dtype(dtype):
        return _strings_as_object(series)
    values = series.astype(object)
    return values.where(series.notna(), np.nan)


def _strings_as_object(series: pd.Series) -> pd.Series:
    """A string-dtype column as ``object`` with ``NaN`` for missing values.

    Equal to ``series.astype(object).where(series.notna(), np.nan)``, but
    built from the unique values: every row holding the same text refers to
    one Python ``str``, instead of one new object per row (a uuid or state
    column has few distinct values and many rows).
    """
    codes, uniques = pd.factorize(series)
    labels = np.empty(len(uniques) + 1, dtype=object)
    labels[: len(uniques)] = np.asarray(uniques, dtype=object)
    labels[-1] = np.nan  # code -1: missing
    return pd.Series(labels[codes], index=series.index, name=series.name, dtype=object)


def _non_missing(values: object) -> list[object]:
    return [v for v in values if not _is_missing(v)]  # type: ignore[attr-defined]


def _is_missing(value: object) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _text_is_exact(series: pd.Series, values: list[object]) -> bool:
    """Whether ``str()`` of the unique values matches ``astype(str)`` per row.

    Unique values are found by equality, so an ``object`` column holding
    both ``1`` and ``1.0`` (or ``True``) has one unique value but two texts;
    such columns take the row-wise path.
    """
    return series.dtype != object or all(isinstance(v, str) for v in values)


def unique_text(series: pd.Series) -> set[str]:
    """``set(series.dropna().astype(str).unique())``, from the unique values.

    Never converts the whole column to text: only its distinct values are
    turned into strings (a column with mixed non-text values falls back to
    the row-wise conversion).
    """
    values = _non_missing(pd.unique(series))
    if not _text_is_exact(series, values):
        return set(series.dropna().astype(str).unique())
    return {str(v) for v in values}


def text_mask(series: pd.Series, text: str) -> np.ndarray:
    """Rows whose value, as text, equals ``text`` (missing values never match).

    The same rows as ``series.astype(str) == text``, compared on the unique
    values and mapped back with ``isin``.
    """
    values = _non_missing(pd.unique(series))
    if not _text_is_exact(series, values):
        texts = series.astype(str)
        return np.asarray((texts == text) & series.notna(), dtype=bool)
    hits = [v for v in values if str(v) == text]
    if not hits:
        return np.zeros(len(series), dtype=bool)
    return np.asarray(series.isin(hits), dtype=bool)


def text_keys(series: pd.Series) -> np.ndarray:
    """``series.astype(str).to_numpy()`` for a column without missing values.

    Built from the unique values: rows with the same value share one ``str``.
    """
    codes, uniques = pd.factorize(series)
    values = list(uniques)
    if (codes < 0).any() or not _text_is_exact(series, values):
        return np.asarray(series.astype(str).to_numpy(), dtype=object)
    labels = np.array([str(v) for v in values] + [""], dtype=object)
    keys: np.ndarray = labels[codes]
    return keys


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


def rows_normalize_alike(
    frame: pd.DataFrame,
    *,
    time_column: str = "systime",
    uuid_column: str = "uuid",
) -> bool:
    """Whether :func:`normalize_signal_frame` treats every row on its own.

    True when every column it touches is either left as is or is a string
    column turned into ``object`` value by value. Normalizing a subset of
    the rows then gives exactly those rows of the normalized frame. False
    when a conversion depends on the whole column (nullable or pyarrow
    numbers and booleans pick their dtype from whether any value is
    missing; text numbers are parsed with a warning counting them) or a
    pyarrow time column is converted.
    """
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        series = frame[column]
        if column == time_column:
            if _arrow_timestamp(series):
                return False
            continue
        if column != uuid_column and not column.startswith("value_"):
            continue
        if _is_extension(series):
            if column in NUMERIC_VALUE_COLUMNS or not _is_text_dtype(series.dtype):
                return False
        elif column in NUMERIC_VALUE_COLUMNS and series.dtype == object:
            return False
    return True


def _is_text_dtype(dtype: object) -> bool:
    """A string dtype that :func:`_plain` converts value by value."""
    return (
        ptypes.is_string_dtype(dtype)
        and not ptypes.is_bool_dtype(dtype)
        and not ptypes.is_numeric_dtype(dtype)
        and not ptypes.is_datetime64_any_dtype(dtype)
    )


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
