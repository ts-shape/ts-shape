import functools
import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.frames import as_datetime, require_columns

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _column_first_compat(*bounds: str) -> Callable[[F], F]:
    """Accept the old ``(dataframe, column, bound, ...)`` argument order.

    The bounds used to follow an optional time column, so they had a
    ``None`` default. They now come first and are required. A call that
    still passes the time column first -- recognized because the first
    argument is a column of the frame, or because the bounds are also
    given by keyword, or because there are more positional arguments than
    bounds -- keeps working with a :class:`DeprecationWarning`.
    """

    def decorate(func: F) -> F:
        name = func.__qualname__

        @functools.wraps(func)
        def wrapper(cls: Any, dataframe: Any, *args: Any, **kwargs: Any) -> Any:
            legacy = bool(args) and (
                len(args) > len(bounds)
                or any(b in kwargs for b in bounds)
                or (
                    isinstance(dataframe, pd.DataFrame)
                    and isinstance(args[0], str)
                    and args[0] in dataframe.columns
                    and len(args) > 1
                )
            )
            if legacy:
                if "time_column" in kwargs:
                    raise TypeError(f"{name}() got the time column twice")
                warnings.warn(
                    f"{name}(): passing the time column before the bounds is "
                    f"deprecated; pass {', '.join(bounds)} first and "
                    "time_column=... by keyword; the old order will be removed "
                    "in the next major release",
                    DeprecationWarning,
                    stacklevel=3,
                )
                kwargs["time_column"] = args[0]
                rest = list(args[1:])
                names = [*bounds, "inclusive"]
                for key, value in zip(names, rest):
                    if key in kwargs:
                        raise TypeError(f"{name}() got multiple values for {key!r}")
                    kwargs[key] = value
                args = ()
            return func(cls, dataframe, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


class DateTimeFilter(Base):
    """
    Provides class methods for filtering time columns in a pandas DataFrame.
    Allows specification of which column to operate on.

    Inherits from:
        Base (class): Base class with common initializations for DataFrame handling.

    Bounds given without a timezone are interpreted in the column's timezone,
    so the same ``"2024-01-02"`` works on naive and tz-aware ``systime``
    columns (previously a tz-aware column raised ``TypeError``). Text
    timestamps are parsed, and an empty column of any dtype gives an empty
    result.

    The bounds are the first arguments after the frame; the time column is
    ``time_column=`` (default ``"systime"``). The older order
    ``(dataframe, "systime", "2024-01-02")`` still works with a
    :class:`DeprecationWarning`.
    """

    @staticmethod
    def _times(dataframe: pd.DataFrame, time_column: str, where: str) -> pd.Series:
        """The time column as datetimes (text parsed, empty columns typed)."""
        require_columns(dataframe, [time_column], where=f"DateTimeFilter.{where}")
        return as_datetime(dataframe[time_column], where=f"DateTimeFilter.{where}")

    @staticmethod
    def _bound(column: pd.Series, value: str) -> pd.Timestamp:
        """Parse a bound and align its timezone with ``column``."""
        ts = pd.Timestamp(value)
        tz = getattr(column.dt, "tz", None)
        if tz is not None and ts.tzinfo is None:
            return ts.tz_localize(tz)
        if tz is None and ts.tzinfo is not None:
            raise ValueError(
                f"Bound {value!r} is tz-aware but the column is naive; "
                "pass a naive bound or localize the column first."
            )
        return ts

    @classmethod
    def _after(
        cls, dataframe: pd.DataFrame, time_column: str, bound: str, where: str
    ) -> pd.DataFrame:
        col = cls._times(dataframe, time_column, where)
        return dataframe[(col > cls._bound(col, bound)).to_numpy(dtype=bool)]

    @classmethod
    def _before(
        cls, dataframe: pd.DataFrame, time_column: str, bound: str, where: str
    ) -> pd.DataFrame:
        col = cls._times(dataframe, time_column, where)
        return dataframe[(col < cls._bound(col, bound)).to_numpy(dtype=bool)]

    @classmethod
    def _between(
        cls,
        dataframe: pd.DataFrame,
        time_column: str,
        start: str,
        end: str,
        inclusive: Literal["both", "neither", "left", "right"],
        where: str,
    ) -> pd.DataFrame:
        col = cls._times(dataframe, time_column, where)
        mask = col.between(
            cls._bound(col, start), cls._bound(col, end), inclusive=inclusive
        )
        return dataframe[mask.to_numpy(dtype=bool)]

    @classmethod
    @renamed_kwargs(column_name="time_column")
    @_column_first_compat("date")
    def filter_after_date(
        cls,
        dataframe: pd.DataFrame,
        date: str,
        *,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows after the specified date.

        Args:
            dataframe: Input DataFrame.
            date (str): The cutoff date in 'YYYY-MM-DD' format (exclusive).
            time_column: Name of the column to filter on. Default ``"systime"``.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is after the specified date.

        Raises:
            ColumnNotFoundError: If ``time_column`` is missing.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_after_date(df, "2025-01-02")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-02 12:00']
        """
        return cls._after(dataframe, time_column, date, "filter_after_date")

    @classmethod
    @renamed_kwargs(column_name="time_column")
    @_column_first_compat("date")
    def filter_before_date(
        cls,
        dataframe: pd.DataFrame,
        date: str,
        *,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows before the specified date.

        Args:
            dataframe: Input DataFrame.
            date (str): The cutoff date in 'YYYY-MM-DD' format (exclusive).
            time_column: Name of the column to filter on. Default ``"systime"``.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is before the specified date.

        Raises:
            ColumnNotFoundError: If ``time_column`` is missing.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_before_date(df, "2025-01-02")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 00:00', '2025-01-01 12:00']
        """
        return cls._before(dataframe, time_column, date, "filter_before_date")

    @classmethod
    @renamed_kwargs(column_name="time_column")
    @_column_first_compat("start_date", "end_date")
    def filter_between_dates(
        cls,
        dataframe: pd.DataFrame,
        start_date: str,
        end_date: str,
        *,
        time_column: str = "systime",
        inclusive: Literal["both", "neither", "left", "right"] = "neither",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows between the specified start and end dates.

        Args:
            dataframe: Input DataFrame.
            start_date (str): The start date of the interval in 'YYYY-MM-DD' format.
            end_date (str): The end date of the interval in 'YYYY-MM-DD' format.
            time_column: Name of the column to filter on. Default ``"systime"``.
            inclusive (str): Which bounds to include: 'neither' (default),
                'both', 'left' or 'right'.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is between the specified dates.

        Raises:
            ColumnNotFoundError: If ``time_column`` is missing.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_between_dates(df, "2025-01-01", "2025-01-02")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 12:00']
        """
        return cls._between(
            dataframe,
            time_column,
            start_date,
            end_date,
            inclusive,
            "filter_between_dates",
        )

    @classmethod
    @renamed_kwargs(column_name="time_column")
    @_column_first_compat("datetime")
    def filter_after_datetime(
        cls,
        dataframe: pd.DataFrame,
        datetime: str,
        *,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows after the specified datetime.

        Args:
            dataframe: Input DataFrame.
            datetime (str): The cutoff datetime in 'YYYY-MM-DD HH:MM:SS' format (exclusive).
            time_column: Name of the column to filter on. Default ``"systime"``.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is after the specified datetime.

        Raises:
            ColumnNotFoundError: If ``time_column`` is missing.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_after_datetime(df, "2025-01-01 12:00:00")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-02 00:00', '2025-01-02 12:00']
        """
        return cls._after(dataframe, time_column, datetime, "filter_after_datetime")

    @classmethod
    @renamed_kwargs(column_name="time_column")
    @_column_first_compat("datetime")
    def filter_before_datetime(
        cls,
        dataframe: pd.DataFrame,
        datetime: str,
        *,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows before the specified datetime.

        Args:
            dataframe: Input DataFrame.
            datetime (str): The cutoff datetime in 'YYYY-MM-DD HH:MM:SS' format (exclusive).
            time_column: Name of the column to filter on. Default ``"systime"``.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is before the specified datetime.

        Raises:
            ColumnNotFoundError: If ``time_column`` is missing.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_before_datetime(df, "2025-01-01 12:00:00")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 00:00']
        """
        return cls._before(dataframe, time_column, datetime, "filter_before_datetime")

    @classmethod
    @renamed_kwargs(column_name="time_column")
    @_column_first_compat("start_datetime", "end_datetime")
    def filter_between_datetimes(
        cls,
        dataframe: pd.DataFrame,
        start_datetime: str,
        end_datetime: str,
        *,
        time_column: str = "systime",
        inclusive: Literal["both", "neither", "left", "right"] = "neither",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows between the specified start and end datetimes.

        Args:
            dataframe: Input DataFrame.
            start_datetime (str): The start datetime of the interval in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str): The end datetime of the interval in 'YYYY-MM-DD HH:MM:SS' format.
            time_column: Name of the column to filter on. Default ``"systime"``.
            inclusive (str): Which bounds to include: 'neither' (default),
                'both', 'left' or 'right'.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is between the specified datetimes.

        Raises:
            ColumnNotFoundError: If ``time_column`` is missing.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_between_datetimes(df, "2025-01-01 06:00:00",
            ...                                          "2025-01-02 12:00:00", inclusive="both")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 12:00', '2025-01-02 00:00', '2025-01-02 12:00']
        """
        return cls._between(
            dataframe,
            time_column,
            start_datetime,
            end_datetime,
            inclusive,
            "filter_between_datetimes",
        )
