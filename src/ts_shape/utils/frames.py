"""Input checks shared by every public function outside the detectors.

The detectors in :mod:`ts_shape.events` get their input checks from
:class:`~ts_shape.utils.base.Base`. Transforms, features, signals, the event
log and the context helpers use these functions, so a missing column, an
untyped time column or a time bound in another timezone is handled the same
way everywhere:

* :func:`require_columns` -- a missing column raises
  :class:`~ts_shape.errors.ColumnNotFoundError` naming the function.
* :func:`as_datetime` -- a time column as ``datetime64`` (text parsed,
  pyarrow timestamps converted, an empty untyped column typed).
* :func:`same_time_axis` -- two time columns with one unit and the same
  timezone handling, so they can be merged or compared.
* :func:`localize_like` -- a time bound (``data_end``, ``start``) in the
  timezone of the data.
* :func:`ffill_values` -- forward-fill that keeps the dtype (no silent
  downcasting of ``object`` columns).
* :func:`to_ns` -- times as int64 nanoseconds since the epoch (UTC instants
  for tz-aware values), for NumPy searches and sorts.
* :func:`signal_frame_input` -- decorator: a computing classmethod works on
  a normalized copy of its input frame.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api import types as ptypes
from pandas.tseries.api import guess_datetime_format
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from ts_shape.errors import ColumnNotFoundError
from ts_shape.utils.normalize import normalize_signal_frame

__all__ = [
    "require_columns",
    "as_datetime",
    "same_time_axis",
    "localize_like",
    "localize_in",
    "to_ns",
    "ffill_values",
    "signal_frame_input",
    "window_offset",
]

F = TypeVar("F", bound=Callable[..., Any])


def require_columns(
    dataframe: pd.DataFrame, columns: Iterable[str | None], *, where: str | None
) -> None:
    """Raise when ``dataframe`` lacks any of ``columns``.

    Args:
        dataframe: Input frame.
        columns: Required column names; ``None`` entries are skipped.
        where: Name of the calling function, used as the message prefix;
            ``None`` leaves the prefix out.

    Raises:
        ColumnNotFoundError: Naming the missing columns and the available
            ones. Subclasses ``ValueError``.
    """
    missing = [c for c in columns if c is not None and c not in dataframe.columns]
    if missing:
        label = "column" if len(missing) == 1 else "columns"
        prefix = f"{where}: {label}" if where else label.capitalize()
        raise ColumnNotFoundError(
            f"{prefix} {', '.join(repr(c) for c in missing)} not found. "
            f"Available columns: {list(dataframe.columns)}"
        )


def as_datetime(series: pd.Series, *, where: str) -> pd.Series:
    """Return ``series`` as a NumPy ``datetime64`` Series.

    Text is parsed, a pyarrow timestamp becomes ``datetime64`` with the same
    timezone, and an empty column of any dtype becomes ``datetime64[ns]``.

    Args:
        series: Time values.
        where: Name of the calling function, used in the message.

    Returns:
        The values as ``datetime64`` (tz-aware when the input is).

    Raises:
        ValueError: When non-empty values cannot be read as times.
    """
    if isinstance(series.dtype, pd.ArrowDtype) or (
        isinstance(series.dtype, pd.api.extensions.ExtensionDtype)
        and not isinstance(series.dtype, pd.DatetimeTZDtype)
    ):
        if ptypes.is_datetime64_any_dtype(series.dtype):
            return pd.Series(
                pd.to_datetime(series.astype(object)),
                index=series.index,
                name=series.name,
            )
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return series
    if series.notna().sum() == 0:
        return pd.Series(
            pd.NaT, index=series.index, name=series.name, dtype="datetime64[ns]"
        )
    # Text whose layout pandas cannot infer from the first value (or whose
    # values mix layouts) is parsed value by value, as ``format="mixed"``
    # does; asking for it explicitly avoids pandas' "Could not infer format"
    # warning.
    fmt = "mixed" if _layout_unknown(series) else None
    try:
        return pd.to_datetime(series, format=fmt)
    except (TypeError, ValueError) as exc:
        if fmt is None:
            try:
                return pd.to_datetime(series, format="mixed")
            except (TypeError, ValueError):
                pass
        raise ValueError(
            f"{where}: column {series.name!r} cannot be read as timestamps ({exc})"
        ) from exc


def _layout_unknown(series: pd.Series) -> bool:
    """True when the first non-missing value is text of no inferable layout."""
    valid = series.dropna()
    if valid.empty or not isinstance(valid.iloc[0], str):
        return False
    return guess_datetime_format(valid.iloc[0]) is None


def same_time_axis(
    left: pd.Series, right: pd.Series, *, where: str
) -> tuple[pd.Series, pd.Series]:
    """Bring two time columns to one unit, so they can be merged.

    Args:
        left: Time values of the first frame.
        right: Time values of the second frame.
        where: Name of the calling function, used in the message.

    Returns:
        Both Series as ``datetime64[ns]``, in the timezone of ``left`` when
        both are tz-aware.

    Raises:
        ValueError: When one side is tz-aware and the other is naive.
    """
    a = as_datetime(left, where=where)
    b = as_datetime(right, where=where)
    tz_a, tz_b = a.dt.tz, b.dt.tz
    if (tz_a is None) != (tz_b is None):
        if a.notna().any() and b.notna().any():
            raise ValueError(
                f"{where}: one time column is timezone-aware ({tz_a or tz_b}) and "
                "the other is naive; localize both to the same timezone first"
            )
        tz = tz_a or tz_b
        a = a if a.dt.tz is not None else a.dt.tz_localize(tz)
        b = b if b.dt.tz is not None else b.dt.tz_localize(tz)
    elif tz_a is not None and tz_b is not None and str(tz_a) != str(tz_b):
        b = b.dt.tz_convert(tz_a)
    return a.astype(_ns(a)), b.astype(_ns(b))


def _ns(series: pd.Series) -> Any:
    tz = series.dt.tz
    return f"datetime64[ns, {tz}]" if tz is not None else "datetime64[ns]"


def localize_like(value: Any, reference: pd.Series) -> pd.Timestamp:
    """Return ``value`` as a Timestamp in the timezone of ``reference``.

    A naive bound on tz-aware data is read as local time of the data; an
    aware bound on naive data is converted to UTC and made naive.

    Args:
        value: A time bound (string, datetime, Timestamp).
        reference: The time column the bound is compared with.

    Returns:
        The bound, comparable with ``reference``.
    """
    return localize_in(value, getattr(getattr(reference, "dt", None), "tz", None))


def localize_in(value: Any, tz: Any) -> pd.Timestamp:
    """Return ``value`` as a Timestamp in ``tz`` (naive when ``tz`` is None).

    The rule of :func:`localize_like`, for a timezone instead of a column: a
    naive value is read as wall-clock time in ``tz`` (an ambiguous time is
    read as standard time, a nonexistent one is shifted forward); an aware
    value is converted. With ``tz=None`` an aware value is converted to UTC
    and made naive.

    Args:
        value: A time (string, datetime, Timestamp).
        tz: Target timezone, or None for naive.

    Returns:
        The value as a Timestamp.
    """
    ts = pd.Timestamp(value)
    if tz is not None:
        if ts.tzinfo is None:
            return ts.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
        return ts.tz_convert(tz)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def to_ns(values: pd.Series | pd.Index) -> npt.NDArray[np.int64]:
    """Times as int64 nanoseconds since the epoch.

    Works for naive and tz-aware values of any unit: an aware value gives the
    nanoseconds of its UTC instant, a naive value those of its wall-clock
    reading. ``NaT`` becomes the minimum int64 (``numpy``'s NaT), which sorts
    first.

    Args:
        values: Datetime Series or Index.

    Returns:
        One int64 per value, in input order.
    """
    index = pd.DatetimeIndex(values)
    return np.asarray(index.as_unit("ns").asi8, dtype=np.int64)


def ffill_values(series: pd.Series) -> pd.Series:
    """Forward-fill ``series`` and keep its dtype.

    ``Series.ffill`` on an ``object`` column infers a narrower dtype for the
    result, which pandas 2.2 reports as deprecated silent downcasting. This
    takes every missing value from the last value before it instead and
    leaves the dtype alone; leading missing values stay missing.

    Args:
        series: Values to fill.

    Returns:
        The filled Series, same index, name and dtype.
    """
    valid = series.notna().to_numpy(dtype=bool)
    if valid.all() or not valid.any():
        return series.copy()
    positions = np.where(valid, np.arange(len(series)), -1)
    np.maximum.accumulate(positions, out=positions)
    filled = series.iloc[np.maximum(positions, 0)]
    filled = filled.where(pd.Series(positions >= 0, index=filled.index))
    filled.index = series.index
    return filled


def signal_frame_input(
    *, time_column: str = "time_column", default_time: str = "systime"
) -> Callable[[F], F]:
    """Run a computing classmethod on a normalized copy of its frame.

    The first positional argument after ``cls`` is copied and passed through
    :func:`~ts_shape.utils.normalize.normalize_signal_frame`, so nullable,
    pyarrow and text-number columns compute like NumPy columns. Use it for
    functions that compute on the values; functions that return input rows
    unchanged (filters) keep the caller's dtypes and do not use it.

    Args:
        time_column: Name of the keyword that holds the time column.
        default_time: Time column when the keyword is not passed.

    Returns:
        A decorator for a ``(cls, dataframe, ...)`` method.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(cls: Any, dataframe: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(dataframe, pd.DataFrame):
                dataframe = normalize_signal_frame(
                    dataframe.copy(),
                    time_column=kwargs.get(time_column, default_time),
                )
            return func(cls, dataframe, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


def window_offset(window: Any) -> Any:
    """The pandas offset of ``window``, with ``"D"`` as a calendar day.

    pandas 3 reads ``"1D"`` as a calendar day (23 h or 25 h on a DST
    change); pandas 2 reads it as a fixed 24 h. Window ends are computed the
    pandas-3 way on both, so a daily window on a DST day has its real length.

    Args:
        window: Offset string (``"1h"``, ``"1D"``, ``"MS"``) or offset.

    Returns:
        A pandas offset; for a day frequency one that adds ``n`` calendar
        days. On pandas 3 that is ``to_offset(window)`` itself (pandas' own
        calendar ``Day``, e.g. ``<3 * Days>``); on pandas 2, where ``Day``
        is a fixed 24 h ``Tick``, it is ``pd.DateOffset(days=n)``. Any other
        window gives ``to_offset(window)`` (e.g. ``<24 * Hours>`` for
        ``"24h"``). A ``Timedelta`` (or ``datetime.timedelta`` /
        ``numpy.timedelta64``) window is a fixed duration on both pandas
        versions: ``Timedelta("2D")`` -> ``<48 * Hours>``.
    """
    if isinstance(window, (timedelta, np.timedelta64)):
        offset = to_offset(pd.Timedelta(window))
        if isinstance(offset, Tick) and offset.rule_code == "D":
            # pandas 2 renders whole days as its fixed-length Day tick.
            return pd.offsets.Hour(24 * offset.n)
        return offset
    offset = to_offset(window)
    if isinstance(offset, Tick) and offset.rule_code == "D":
        return pd.DateOffset(days=offset.n)
    return offset
