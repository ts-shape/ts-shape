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


def _per_signal_gaps(
    dataframe: pd.DataFrame,
    time_column: str,
    uuid_column: str | None,
    *,
    where: str,
) -> pd.Series:
    """Consecutive gaps between timestamps, per-signal when a uuid is present.

    A merged long-format frame mixes rows of different signals; a plain
    ``.diff()`` on the pooled column turns the jump from one signal's last
    sample to another signal's first sample into a sampling gap. Grouping
    by ``uuid_column`` before diffing keeps each gap inside one signal.
    """
    times = time_values(dataframe, time_column, where=where)
    if uuid_column and uuid_column in dataframe.columns:
        uuids = dataframe[uuid_column].reset_index(drop=True)
        ordered = (
            pd.DataFrame({"__t": times.reset_index(drop=True), "__u": uuids})
            .dropna(subset=["__t"])
            .sort_values(["__u", "__t"], kind="stable")
        )
        return ordered.groupby("__u", observed=True, sort=False)["__t"].diff().dropna()
    return times.dropna().sort_values().diff().dropna()


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
        ).dt.day_of_week.value_counts()

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
            .dt.day_of_week.dropna()
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
        cls,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
        uuid_column: str | None = "uuid",
    ) -> pd.Timedelta:
        """Returns the average time gap between consecutive timestamps.

        On a multi-signal long-format frame the gaps are taken per signal
        (grouped by ``uuid_column``) and then averaged, so an inter-signal
        transition in a merged frame never shows up as a sample interval.
        A frame without the uuid column (or with ``uuid_column=None``) is
        treated as a single stream.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.
            uuid_column: Column that identifies each signal; set to ``None``
                to treat the frame as one stream. Default ``"uuid"``.

        Returns:
            Mean gap between consecutive timestamps (in time order), per
                signal and then pooled; NaT for fewer than 2 timestamps on
                every signal.
        """
        time_deltas = _per_signal_gaps(
            dataframe,
            time_column,
            uuid_column,
            where="TimestampStatistics.average_time_gap",
        )
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
        cls,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
        uuid_column: str | None = "uuid",
    ) -> pd.Timedelta:
        """Returns the standard deviation of the time differences between consecutive timestamps.

        Gaps are taken per signal (grouped by ``uuid_column``) and then
        pooled, matching :meth:`average_time_gap`.

        Args:
            dataframe: Input DataFrame.
            time_column: Name of the timestamp column to analyze. Default ``"systime"``.
            uuid_column: Column that identifies each signal; set to ``None``
                to treat the frame as one stream. Default ``"uuid"``.

        Returns:
            Standard deviation of the gaps between consecutive timestamps,
                pooled across signals; NaT for fewer than 3 timestamps in
                total.
        """
        time_deltas = _per_signal_gaps(
            dataframe,
            time_column,
            uuid_column,
            where="TimestampStatistics.standard_deviation_timestamps",
        )
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
