import logging
import pandas as pd  # type: ignore
from zoneinfo import available_timezones
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class TimestampConverter(Base):
    """
    A class dedicated to converting high-precision timestamp data (e.g., in seconds, milliseconds, microseconds, or nanoseconds)
    to standard datetime formats with optional timezone adjustment.
    """

    @classmethod
    def convert_to_datetime(
        cls,
        dataframe: pd.DataFrame,
        columns: list,
        unit: str = "ns",
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """
        Converts specified columns from a given timestamp unit to datetime format in a target timezone.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            columns (list): A list of column names with timestamp data to convert.
            unit (str): The unit of the timestamps ('s', 'ms', 'us', or 'ns').
            timezone (str): The target timezone for the converted datetime (default is 'UTC').

        Returns:
            pd.DataFrame: A DataFrame with the converted datetime columns in the specified timezone.
        """
        # Validate unit
        valid_units = ["s", "ms", "us", "ns"]
        if unit not in valid_units:
            raise ValueError(f"Invalid unit '{unit}'. Must be one of {valid_units}.")

        # Validate timezone
        if timezone not in available_timezones():
            raise ValueError(
                f"Invalid timezone '{timezone}'. Use a valid timezone name from available_timezones()."
            )

        df = dataframe.copy()
        for col in columns:
            # Convert timestamps to datetime in UTC first
            df[col] = pd.to_datetime(df[col], unit=unit, utc=True)
            # Adjust to the target timezone
            df[col] = df[col].dt.tz_convert(timezone)

        return df
