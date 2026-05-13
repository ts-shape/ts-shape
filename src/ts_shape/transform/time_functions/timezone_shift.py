import logging
import pandas as pd  # type: ignore
from zoneinfo import available_timezones
from ts_shape.utils.base import Base  # Import Base from the specified path

logger = logging.getLogger(__name__)


class TimezoneShift(Base):
    """
    A class for shifting timestamps in a DataFrame to a different timezone, with methods to handle timezone localization and conversion.
    """

    @classmethod
    def shift_timezone(
        cls,
        dataframe: pd.DataFrame,
        time_column: str,
        input_timezone: str,
        target_timezone: str,
    ) -> pd.DataFrame:
        """
        Shifts timestamps in the specified column of a DataFrame from a given timezone to a target timezone.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to convert.
            input_timezone (str): The timezone of the input timestamps (e.g., 'UTC' or 'America/New_York').
            target_timezone (str): The target timezone to shift to (e.g., 'America/New_York').

        Returns:
            pd.DataFrame: A DataFrame with timestamps converted to the target timezone.
        """
        # Validate timezones
        if input_timezone not in available_timezones():
            raise ValueError(f"Invalid input timezone: {input_timezone}")
        if target_timezone not in available_timezones():
            raise ValueError(f"Invalid target timezone: {target_timezone}")

        # Ensure the time column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(dataframe[time_column]):
            raise ValueError(f"Column '{time_column}' must contain datetime values.")

        # Localize to the specified input timezone if timestamps are naive
        dataframe[time_column] = pd.to_datetime(dataframe[time_column])
        if dataframe[time_column].dt.tz is None:
            dataframe[time_column] = dataframe[time_column].dt.tz_localize(
                input_timezone
            )
        else:
            # Convert from the existing timezone to the specified input timezone, if they differ
            dataframe[time_column] = dataframe[time_column].dt.tz_convert(
                input_timezone
            )

        # Convert to the target timezone
        dataframe[time_column] = dataframe[time_column].dt.tz_convert(target_timezone)

        return dataframe

    @classmethod
    def add_timezone_column(
        cls,
        dataframe: pd.DataFrame,
        time_column: str,
        input_timezone: str,
        target_timezone: str,
    ) -> pd.DataFrame:
        """
        Creates a new column with timestamps converted from an input timezone to a target timezone, without altering the original column.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to convert.
            input_timezone (str): The timezone of the input timestamps.
            target_timezone (str): The target timezone.

        Returns:
            pd.DataFrame: A DataFrame with an additional column for the shifted timezone.
        """
        # Duplicate the DataFrame to prevent modifying the original column
        df_copy = dataframe.copy()

        # Create the new timezone-shifted column
        new_column = f"{time_column}_{target_timezone.replace('/', '_')}"
        df_copy[new_column] = df_copy[time_column]

        # Apply the timezone shift to the new column
        df_copy = cls.shift_timezone(
            df_copy, new_column, input_timezone, target_timezone
        )

        return df_copy

    @classmethod
    def list_available_timezones(cls) -> list:
        """
        Returns a list of all available timezones.

        Returns:
            list: A list of strings representing all available timezones.
        """
        return available_timezones()

    @classmethod
    def detect_timezone_awareness(
        cls, dataframe: pd.DataFrame, time_column: str
    ) -> bool:
        """
        Detects if a time column in a DataFrame is timezone-aware.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to check.

        Returns:
            bool: True if the column is timezone-aware, False otherwise.
        """
        return dataframe[time_column].dt.tz is not None

    @classmethod
    def revert_to_original_timezone(
        cls, dataframe: pd.DataFrame, time_column: str, original_timezone: str
    ) -> pd.DataFrame:
        """
        Reverts a timezone-shifted time column back to the original timezone.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to revert.
            original_timezone (str): The original timezone to revert to.

        Returns:
            pd.DataFrame: A DataFrame with timestamps reverted to the original timezone.
        """
        # Validate the original timezone
        if original_timezone not in available_timezones():
            raise ValueError(f"Invalid original timezone: {original_timezone}")

        # Convert to the original timezone
        dataframe[time_column] = dataframe[time_column].dt.tz_convert(original_timezone)

        return dataframe

    @classmethod
    def calculate_time_difference(
        cls, dataframe: pd.DataFrame, start_column: str, end_column: str
    ) -> pd.Series:
        """
        Calculates the time difference between two timestamp columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            start_column (str): The name of the start time column.
            end_column (str): The name of the end time column.

        Returns:
            pd.Series: A Series with the time differences in seconds.
        """
        # Check if both columns are timezone-aware or both are timezone-naive
        start_is_aware = dataframe[start_column].dt.tz is not None
        end_is_aware = dataframe[end_column].dt.tz is not None

        if start_is_aware != end_is_aware:
            raise ValueError(
                "Both columns must be either timezone-aware or timezone-naive."
            )

        # If timezone-aware, convert both columns to UTC for comparison
        if start_is_aware:
            start_times = dataframe[start_column].dt.tz_convert("UTC")
            end_times = dataframe[end_column].dt.tz_convert("UTC")
        else:
            start_times = dataframe[start_column]
            end_times = dataframe[end_column]

        # Calculate the difference in seconds
        time_difference = (end_times - start_times).dt.total_seconds()

        return time_difference
