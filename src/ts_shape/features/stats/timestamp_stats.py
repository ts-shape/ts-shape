import logging
from typing import cast

import pandas as pd

from ts_shape.features.stats._values import time_values
from ts_shape.utils.base import Base

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
    def count_null(cls, dataframe: pd.DataFrame, column_name: str = "systime") -> int:
        """Returns the number of null (NaN) values in the timestamp column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        return cast(
            int,
            time_values(dataframe, column_name, where="TimestampStatistics.count_null")
            .isna()
            .sum(),
        )

    @classmethod
    def count_not_null(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int:
        """Returns the number of non-null (valid) timestamps in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        return cast(
            int,
            time_values(
                dataframe, column_name, where="TimestampStatistics.count_not_null"
            )
            .notna()
            .sum(),
        )

    @classmethod
    def earliest_timestamp(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timestamp:
        """Returns the earliest timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.earliest_timestamp"
        ).min()

    @classmethod
    def latest_timestamp(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timestamp:
        """Returns the latest timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.latest_timestamp"
        ).max()

    @classmethod
    def timestamp_range(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timedelta:
        """Returns the time range (difference) between the earliest and latest timestamps.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        return cls.latest_timestamp(dataframe, column_name) - cls.earliest_timestamp(
            dataframe, column_name
        )

    @classmethod
    def most_frequent_timestamp(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timestamp | None:
        """Returns the most frequent timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            The most frequent timestamp (the earliest on a tie), or ``None``
                when the column holds no timestamp.
        """
        modes = time_values(
            dataframe, column_name, where="TimestampStatistics.most_frequent_timestamp"
        ).mode()
        if modes.empty:
            return None
        return cast(pd.Timestamp, modes.iloc[0])

    @classmethod
    def count_most_frequent_timestamp(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int:
        """Returns the count of the most frequent timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        counts = time_values(
            dataframe,
            column_name,
            where="TimestampStatistics.count_most_frequent_timestamp",
        ).value_counts()
        return int(counts.iloc[0]) if len(counts) else 0

    @classmethod
    def year_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per year.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by year, most frequent first.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.year_distribution"
        ).dt.year.value_counts()

    @classmethod
    def month_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per month.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by month number (1-12), most frequent
                first.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.month_distribution"
        ).dt.month.value_counts()

    @classmethod
    def weekday_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per weekday.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by weekday (0=Monday, 6=Sunday), most
                frequent first.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.weekday_distribution"
        ).dt.weekday.value_counts()

    @classmethod
    def hour_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per hour of the day.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.

        Returns:
            Series of timestamp counts indexed by hour (0-23), most frequent first.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.hour_distribution"
        ).dt.hour.value_counts()

    @classmethod
    def most_frequent_day(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int | None:
        """Returns the most frequent day of the week (0=Monday, 6=Sunday).

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        modes = (
            time_values(
                dataframe, column_name, where="TimestampStatistics.most_frequent_day"
            )
            .dt.weekday.dropna()
            .mode()
        )
        if modes.empty:
            return None
        return int(modes.iloc[0])

    @classmethod
    def most_frequent_hour(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int | None:
        """Returns the most frequent hour of the day (0-23).

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        modes = (
            time_values(
                dataframe, column_name, where="TimestampStatistics.most_frequent_hour"
            )
            .dt.hour.dropna()
            .mode()
        )
        if modes.empty:
            return None
        return int(modes.iloc[0])

    @classmethod
    def average_time_gap(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timedelta:
        """Returns the average time gap between consecutive timestamps.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        sorted_times = (
            time_values(
                dataframe, column_name, where="TimestampStatistics.average_time_gap"
            )
            .dropna()
            .sort_values()
        )
        time_deltas = sorted_times.diff().dropna()
        return time_deltas.mean()

    @classmethod
    def median_timestamp(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timestamp:
        """Returns the median timestamp in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.median_timestamp"
        ).median()

    @classmethod
    def standard_deviation_timestamps(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timedelta:
        """Returns the standard deviation of the time differences between consecutive timestamps.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        sorted_times = (
            time_values(
                dataframe,
                column_name,
                where="TimestampStatistics.standard_deviation_timestamps",
            )
            .dropna()
            .sort_values()
        )
        time_deltas = sorted_times.diff().dropna()
        return time_deltas.std()

    @classmethod
    def timestamp_quartiles(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the 25th, 50th (median), and 75th percentiles of the timestamps.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
        """
        return time_values(
            dataframe, column_name, where="TimestampStatistics.timestamp_quartiles"
        ).quantile([0.25, 0.5, 0.75])

    @classmethod
    def days_with_most_activity(
        cls, dataframe: pd.DataFrame, column_name: str = "systime", n: int = 3
    ) -> pd.Series:
        """Returns the top N days with the most timestamp activity.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the timestamp column to analyze. Default ``"systime"``.
            n: Number of days to return. Default 3.

        Returns:
            Series of timestamp counts indexed by calendar date, most active first.
        """
        return (
            time_values(
                dataframe,
                column_name,
                where="TimestampStatistics.days_with_most_activity",
            )
            .dt.date.value_counts()
            .head(n)
        )
