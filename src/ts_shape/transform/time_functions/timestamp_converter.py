import logging

import pandas as pd

from ts_shape.transform.time_functions.timezone_shift import _valid_timezone
from ts_shape.utils.base import Base
from ts_shape.utils.frames import require_columns

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
        columns: str | list[str],
        unit: str = "ns",
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """
        Converts specified columns from a given timestamp unit to datetime format in a target timezone.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            columns (str | list): Column name, or list of column names, with
                timestamp data to convert.
            unit (str): The unit of the timestamps ('s', 'ms', 'us', or 'ns').
            timezone (str): The target timezone for the converted datetime (default is 'UTC').

        Returns:
            pd.DataFrame: A DataFrame with the converted datetime columns in the
                specified timezone, as ``datetime64[ns, timezone]`` whatever
                ``unit`` is.
        """
        # Validate unit
        valid_units = ["s", "ms", "us", "ns"]
        if unit not in valid_units:
            raise ValueError(f"Invalid unit '{unit}'. Must be one of {valid_units}.")

        timezone = _valid_timezone(timezone)
        if isinstance(columns, str):
            columns = [columns]
        require_columns(
            dataframe, columns, where="TimestampConverter.convert_to_datetime"
        )

        df = dataframe.copy()
        for col in columns:
            # Convert timestamps to datetime in UTC first. The resolution pandas
            # picks depends on its version, ``unit`` and the dtype, so pin it.
            df[col] = pd.to_datetime(df[col], unit=unit, utc=True).dt.as_unit("ns")
            # Adjust to the target timezone
            df[col] = df[col].dt.tz_convert(timezone)

        return df
