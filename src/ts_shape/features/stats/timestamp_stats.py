import logging
from typing import cast

import pandas as pd

from ts_shape.features.stats._values import time_values
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)

# pandas ships no type information here (no py.typed, pandas-stubs not
# installed), so every reduction is ``Any``; ``cast`` records the documented
# scalar type without converting it (the numpy scalar is returned as before).


class TimestampStatistics(Base):
    """
    Provides class methods to calculate statistics on timestamp columns in a pandas DataFrame.
    The default column for calculations is 'systime'.
    """

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def count_null(cls, dataframe: pd.DataFrame, time_column: str = "systime") -> int:
        """Returns the number of null (NaN) values in the timestamp column.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Number of missing timestamps.
        """
        return cast(
            int,
            time_values(dataframe, time_column, where="TimestampStatistics.count_null")
            .isna()
            .sum(),
        )

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def count_not_null(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> int:
        """Returns the number of non-null (valid) timestamps in the column.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Number of valid timestamps.
        """
        return cast(
            int,
            time_values(
                dataframe, time_column, where="TimestampStatistics.count_not_null"
            )
            .notna()
            .sum(),
        )

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def earliest_timestamp(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Timestamp:
        """Returns the earliest timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            The earliest timestamp; NaT when there is none.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.earliest_timestamp"
        ).min()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def latest_timestamp(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Timestamp:
        """Returns the latest timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            The latest timestamp; NaT when there is none.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.latest_timestamp"
        ).max()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def timestamp_range(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Timedelta:
        """Returns the time range (difference) between the earliest and latest timestamps.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Latest minus earliest timestamp; NaT when there is none.
        """
        return cls.latest_timestamp(dataframe, time_column) - cls.earliest_timestamp(
            dataframe, time_column
        )

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def most_frequent_timestamp(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Timestamp | None:
        """Returns the most frequent timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            The most frequent timestamp (the earliest on a tie), or ``None``
                when the column holds no timestamp.
        """
        modes = time_values(
            dataframe, time_column, where="TimestampStatistics.most_frequent_timestamp"
        ).mode()
        if modes.empty:
            return None
        return cast(pd.Timestamp, modes.iloc[0])

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def count_most_frequent_timestamp(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> int:
        """Returns the count of the most frequent timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Occurrences of the most frequent timestamp; 0 when there is
                none.
        """
        counts = time_values(
            dataframe,
            time_column,
            where="TimestampStatistics.count_most_frequent_timestamp",
        ).value_counts()
        return int(counts.iloc[0]) if len(counts) else 0

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def year_distribution(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per year.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by year, most frequent first.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.year_distribution"
        ).dt.year.value_counts()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def month_distribution(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per month.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by month number (1-12), most frequent
                first.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.month_distribution"
        ).dt.month.value_counts()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def weekday_distribution(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per weekday.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by weekday (0=Monday, 6=Sunday), most
                frequent first.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.weekday_distribution"
        ).dt.weekday.value_counts()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def hour_distribution(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per hour of the day.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by hour (0-23), most frequent first.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.hour_distribution"
        ).dt.hour.value_counts()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def most_frequent_day(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> int | None:
        """Returns the most frequent day of the week (0=Monday, 6=Sunday).

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            The most frequent weekday (0=Monday; the smallest on a tie), or
                ``None`` when there is no timestamp.
        """
        modes = (
            time_values(
                dataframe, time_column, where="TimestampStatistics.most_frequent_day"
            )
            .dt.weekday.dropna()
            .mode()
        )
        if modes.empty:
            return None
        return int(modes.iloc[0])

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def most_frequent_hour(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> int | None:
        """Returns the most frequent hour of the day (0-23).

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            The most frequent hour (0-23; the smallest on a tie), or
                ``None`` when there is no timestamp.
        """
        modes = (
            time_values(
                dataframe, time_column, where="TimestampStatistics.most_frequent_hour"
            )
            .dt.hour.dropna()
            .mode()
        )
        if modes.empty:
            return None
        return int(modes.iloc[0])

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def average_time_gap(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Timedelta:
        """Returns the average time gap between consecutive timestamps.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Mean gap between consecutive timestamps (in time order); NaT for
                fewer than 2 timestamps.
        """
        sorted_times = (
            time_values(
                dataframe, time_column, where="TimestampStatistics.average_time_gap"
            )
            .dropna()
            .sort_values()
        )
        time_deltas = sorted_times.diff().dropna()
        return time_deltas.mean()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def median_timestamp(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Timestamp:
        """Returns the median timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            The median timestamp; NaT when there is none.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.median_timestamp"
        ).median()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def standard_deviation_timestamps(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Timedelta:
        """Returns the standard deviation of the time differences between consecutive timestamps.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Standard deviation of the gaps between consecutive timestamps;
                NaT for fewer than 3 timestamps.
        """
        sorted_times = (
            time_values(
                dataframe,
                time_column,
                where="TimestampStatistics.standard_deviation_timestamps",
            )
            .dropna()
            .sort_values()
        )
        time_deltas = sorted_times.diff().dropna()
        return time_deltas.std()

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def timestamp_quartiles(
        cls, dataframe: pd.DataFrame, time_column: str = "systime"
    ) -> pd.Series:
        """Returns the 25th, 50th (median), and 75th percentiles of the timestamps.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of the 25th, 50th and 75th percentile timestamps, indexed
                by quantile.
        """
        return time_values(
            dataframe, time_column, where="TimestampStatistics.timestamp_quartiles"
        ).quantile([0.25, 0.5, 0.75])

    @classmethod
    @renamed_kwargs(column_name="time_column")
    def days_with_most_activity(
        cls, dataframe: pd.DataFrame, time_column: str = "systime", n: int = 3
    ) -> pd.Series:
        """Returns the top N days with the most timestamp activity.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.
            n: Number of days to return. Default 3.

        Returns:
            Series of timestamp counts indexed by calendar date, most active first.
        """
        return (
            time_values(
                dataframe,
                time_column,
                where="TimestampStatistics.days_with_most_activity",
            )
            .dt.date.value_counts()
            .head(n)
        )
