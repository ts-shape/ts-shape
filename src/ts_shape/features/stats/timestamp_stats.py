import logging
import pandas as pd  # type: ignore
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class TimestampStatistics(Base):
    """
    Provides class methods to calculate statistics on timestamp columns in a pandas DataFrame.
    The default column for calculations is 'systime'.
    """

    @classmethod
    def count_null(cls, dataframe: pd.DataFrame, column_name: str = "systime") -> int:
        """Returns the number of null (NaN) values in the timestamp column."""
        return dataframe[column_name].isna().sum()

    @classmethod
    def count_not_null(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int:
        """Returns the number of non-null (valid) timestamps in the column."""
        return dataframe[column_name].notna().sum()

    @classmethod
    def earliest_timestamp(cls, dataframe: pd.DataFrame, column_name: str = "systime"):
        """Returns the earliest timestamp in the column."""
        return dataframe[column_name].min()

    @classmethod
    def latest_timestamp(cls, dataframe: pd.DataFrame, column_name: str = "systime"):
        """Returns the latest timestamp in the column."""
        return dataframe[column_name].max()

    @classmethod
    def timestamp_range(cls, dataframe: pd.DataFrame, column_name: str = "systime"):
        """Returns the time range (difference) between the earliest and latest timestamps."""
        return cls.latest_timestamp(dataframe, column_name) - cls.earliest_timestamp(
            dataframe, column_name
        )

    @classmethod
    def most_frequent_timestamp(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ):
        """Returns the most frequent timestamp in the column."""
        return dataframe[column_name].mode().iloc[0]

    @classmethod
    def count_most_frequent_timestamp(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int:
        """Returns the count of the most frequent timestamp in the column."""
        most_frequent_value = cls.most_frequent_timestamp(dataframe, column_name)
        return dataframe[column_name].value_counts().loc[most_frequent_value]

    @classmethod
    def year_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per year."""
        return dataframe[column_name].dt.year.value_counts()

    @classmethod
    def month_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per month."""
        return dataframe[column_name].dt.month.value_counts()

    @classmethod
    def weekday_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per weekday."""
        return dataframe[column_name].dt.weekday.value_counts()

    @classmethod
    def hour_distribution(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the distribution of timestamps per hour of the day."""
        return dataframe[column_name].dt.hour.value_counts()

    @classmethod
    def most_frequent_day(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int:
        """Returns the most frequent day of the week (0=Monday, 6=Sunday)."""
        return dataframe[column_name].dt.weekday.mode().iloc[0]

    @classmethod
    def most_frequent_hour(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> int:
        """Returns the most frequent hour of the day (0-23)."""
        return dataframe[column_name].dt.hour.mode().iloc[0]

    @classmethod
    def average_time_gap(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timedelta:
        """Returns the average time gap between consecutive timestamps."""
        sorted_times = dataframe[column_name].dropna().sort_values()
        time_deltas = sorted_times.diff().dropna()
        return time_deltas.mean()

    @classmethod
    def median_timestamp(cls, dataframe: pd.DataFrame, column_name: str = "systime"):
        """Returns the median timestamp in the column."""
        return dataframe[column_name].median()

    @classmethod
    def standard_deviation_timestamps(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Timedelta:
        """Returns the standard deviation of the time differences between consecutive timestamps."""
        sorted_times = dataframe[column_name].dropna().sort_values()
        time_deltas = sorted_times.diff().dropna()
        return time_deltas.std()

    @classmethod
    def timestamp_quartiles(
        cls, dataframe: pd.DataFrame, column_name: str = "systime"
    ) -> pd.Series:
        """Returns the 25th, 50th (median), and 75th percentiles of the timestamps."""
        return dataframe[column_name].quantile([0.25, 0.5, 0.75])

    @classmethod
    def days_with_most_activity(
        cls, dataframe: pd.DataFrame, column_name: str = "systime", n: int = 3
    ) -> pd.Series:
        """Returns the top N days with the most timestamp activity."""
        return dataframe[column_name].dt.date.value_counts().head(n)
