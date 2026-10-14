import logging
from zoneinfo import available_timezones

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.frames import as_datetime, require_columns

logger = logging.getLogger(__name__)


def _valid_timezone(name: str, *, what: str = "timezone") -> str:
    """Return the IANA name for ``name``; ``"utc"`` in any case is ``"UTC"``.

    Raises:
        ValueError: If ``name`` is not a known timezone.
    """
    if isinstance(name, str) and name.strip().upper() == "UTC":
        return "UTC"
    if name not in available_timezones():
        raise ValueError(f"Invalid {what}: {name}")
    return name


def _times(dataframe: pd.DataFrame, column: str, where: str) -> pd.Series:
    require_columns(dataframe, [column], where=f"TimezoneShift.{where}")
    return as_datetime(dataframe[column], where=f"TimezoneShift.{where}")


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
        ambiguous: str = "raise",
        nonexistent: str = "raise",
    ) -> pd.DataFrame:
        """
        Shifts timestamps in the specified column of a DataFrame from a given timezone to a target timezone.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to convert.
            input_timezone (str): The timezone of the input timestamps (e.g., 'UTC' or 'America/New_York').
            target_timezone (str): The target timezone to shift to (e.g., 'America/New_York').
            ambiguous (str): How to localize naive wall-clock times that occur
                twice (DST fall-back, e.g. 2024-10-27 02:30 in Europe/Berlin).
                Passed to ``tz_localize``: 'raise' (default), 'infer', 'NaT'.
            nonexistent (str): How to localize naive times that do not exist
                (DST spring-forward, e.g. 2024-03-31 02:30 in Europe/Berlin):
                'raise' (default), 'shift_forward', 'shift_backward', 'NaT'.

        Returns:
            pd.DataFrame: A copy of the DataFrame with timestamps converted to
            the target timezone (the input frame is not modified).
        """
        input_timezone = _valid_timezone(input_timezone, what="input timezone")
        target_timezone = _valid_timezone(target_timezone, what="target timezone")
        require_columns(dataframe, [time_column], where="TimezoneShift.shift_timezone")
        values = dataframe[time_column]
        if values.notna().any() and not (
            pd.api.types.is_datetime64_any_dtype(values.dtype)
        ):
            raise ValueError(f"Column '{time_column}' must contain datetime values.")

        # Work on a copy: the caller's frame must not be mutated.
        dataframe = dataframe.copy()

        # Localize to the specified input timezone if timestamps are naive
        dataframe[time_column] = _times(dataframe, time_column, "shift_timezone")
        if dataframe[time_column].dt.tz is None:
            dataframe[time_column] = dataframe[time_column].dt.tz_localize(
                input_timezone, ambiguous=ambiguous, nonexistent=nonexistent
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
        require_columns(
            dataframe, [time_column], where="TimezoneShift.add_timezone_column"
        )
        target_timezone = _valid_timezone(target_timezone, what="target timezone")
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
    def list_available_timezones(cls) -> list[str]:
        """
        Returns a sorted list of all available timezones.

        Returns:
            list: A sorted list of strings representing all available timezones.
        """
        return sorted(available_timezones())

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
        return (
            _times(dataframe, time_column, "detect_timezone_awareness").dt.tz
            is not None
        )

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
        original_timezone = _valid_timezone(original_timezone, what="original timezone")
        times = _times(dataframe, time_column, "revert_to_original_timezone")
        if times.dt.tz is None and times.isna().all():
            times = times.dt.tz_localize("UTC")
        # Work on a copy: the caller's frame must not be mutated.
        dataframe = dataframe.copy()
        dataframe[time_column] = times.dt.tz_convert(original_timezone)
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
        start = _times(dataframe, start_column, "calculate_time_difference")
        end = _times(dataframe, end_column, "calculate_time_difference")
        # Check if both columns are timezone-aware or both are timezone-naive
        start_is_aware = start.dt.tz is not None
        end_is_aware = end.dt.tz is not None

        if start_is_aware != end_is_aware:
            raise ValueError(
                "Both columns must be either timezone-aware or timezone-naive."
            )

        # If timezone-aware, convert both columns to UTC for comparison
        if start_is_aware:
            start_times = start.dt.tz_convert("UTC")
            end_times = end.dt.tz_convert("UTC")
        else:
            start_times = start
            end_times = end

        # Calculate the difference in seconds
        time_difference = (end_times - start_times).dt.total_seconds().astype("float64")

        return time_difference
