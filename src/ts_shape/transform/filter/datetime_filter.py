import logging
from typing import Literal

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.frames import as_datetime, require_columns

logger = logging.getLogger(__name__)


class DateTimeFilter(Base):
    """
    Provides class methods for filtering time columns in a pandas DataFrame.
    Allows specification of which column to operate on.

    Inherits from:
        Base (class): Base class with common initializations for DataFrame handling.

    Bounds given without a timezone are interpreted in the column's timezone,
    so the same ``"2024-01-02"`` works on naive and tz-aware ``systime``
    columns (previously a tz-aware column raised ``TypeError``).
    """

    @staticmethod
    def _times(dataframe: pd.DataFrame, column_name: str, where: str) -> pd.Series:
        """The time column as datetimes (text parsed, empty columns typed)."""
        require_columns(dataframe, [column_name], where=f"DateTimeFilter.{where}")
        return as_datetime(dataframe[column_name], where=f"DateTimeFilter.{where}")

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
    def filter_after_date(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        date: str | None = None,
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows after the specified date.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"systime"``.
            date (str): The cutoff date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is after the specified date.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_after_date(df, "systime", "2025-01-02")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-02 12:00']
        """
        if date is None:
            raise ValueError("date parameter is required")
        col = cls._times(dataframe, column_name, "filter_after_date")
        return dataframe[(col > cls._bound(col, date)).to_numpy(dtype=bool)]

    @classmethod
    def filter_before_date(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        date: str | None = None,
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows before the specified date.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"systime"``.
            date (str): The cutoff date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is before the specified date.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_before_date(df, "systime", "2025-01-02")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 00:00', '2025-01-01 12:00']
        """
        if date is None:
            raise ValueError("date parameter is required")
        col = cls._times(dataframe, column_name, "filter_before_date")
        return dataframe[(col < cls._bound(col, date)).to_numpy(dtype=bool)]

    @classmethod
    def filter_between_dates(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        start_date: str | None = None,
        end_date: str | None = None,
        inclusive: Literal["both", "neither", "left", "right"] = "neither",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows between the specified start and end dates.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"systime"``.
            start_date (str): The start date of the interval in 'YYYY-MM-DD' format.
            end_date (str): The end date of the interval in 'YYYY-MM-DD' format.
            inclusive (str): Which bounds to include: 'neither' (default),
                'both', 'left' or 'right'.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is between the specified dates.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_between_dates(df, "systime", "2025-01-01", "2025-01-02")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 12:00']
        """
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date parameters are required")
        col = cls._times(dataframe, column_name, "filter_between_dates")
        mask = col.between(
            cls._bound(col, start_date), cls._bound(col, end_date), inclusive=inclusive
        )
        return dataframe[mask.to_numpy(dtype=bool)]

    @classmethod
    def filter_after_datetime(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        datetime: str | None = None,
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows after the specified datetime.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"systime"``.
            datetime (str): The cutoff datetime in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is after the specified datetime.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_after_datetime(df, "systime", "2025-01-01 12:00:00")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-02 00:00', '2025-01-02 12:00']
        """
        if datetime is None:
            raise ValueError("datetime parameter is required")
        col = cls._times(dataframe, column_name, "filter_after_datetime")
        return dataframe[(col > cls._bound(col, datetime)).to_numpy(dtype=bool)]

    @classmethod
    def filter_before_datetime(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        datetime: str | None = None,
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows before the specified datetime.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"systime"``.
            datetime (str): The cutoff datetime in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is before the specified datetime.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_before_datetime(df, "systime", "2025-01-01 12:00:00")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 00:00']
        """
        if datetime is None:
            raise ValueError("datetime parameter is required")
        col = cls._times(dataframe, column_name, "filter_before_datetime")
        return dataframe[(col < cls._bound(col, datetime)).to_numpy(dtype=bool)]

    @classmethod
    def filter_between_datetimes(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        inclusive: Literal["both", "neither", "left", "right"] = "neither",
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows between the specified start and end datetimes.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"systime"``.
            start_datetime (str): The start datetime of the interval in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str): The end datetime of the interval in 'YYYY-MM-DD HH:MM:SS' format.
            inclusive (str): Which bounds to include: 'neither' (default),
                'both', 'left' or 'right'.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is between the specified datetimes.

        Example:
            >>> import ts_shape
            >>> from ts_shape.transform.filter.datetime_filter import DateTimeFilter
            >>> df = ts_shape.make_timeseries(n_points=4, freq="12h")  # 2025-01-01 00:00 .. 01-02 12:00
            >>> out = DateTimeFilter.filter_between_datetimes(df, "systime", "2025-01-01 06:00:00",
            ...                                          "2025-01-02 12:00:00", inclusive="both")
            >>> out["systime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            ['2025-01-01 12:00', '2025-01-02 00:00', '2025-01-02 12:00']
        """
        if start_datetime is None or end_datetime is None:
            raise ValueError("start_datetime and end_datetime parameters are required")
        col = cls._times(dataframe, column_name, "filter_between_datetimes")
        mask = col.between(
            cls._bound(col, start_datetime),
            cls._bound(col, end_datetime),
            inclusive=inclusive,
        )
        return dataframe[mask.to_numpy(dtype=bool)]
