import logging
import pandas as pd  # type: ignore
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DateTimeFilter(Base):
    """
    Provides class methods for filtering time columns in a pandas DataFrame.
    Allows specification of which column to operate on.

    Inherits from:
        Base (class): Base class with common initializations for DataFrame handling.
    """

    @classmethod
    def filter_after_date(
        cls, dataframe: pd.DataFrame, column_name: str = "systime", date: str = None
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows after the specified date.

        Args:
            date (str): The cutoff date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is after the specified date.

        Example:
        --------
        >>> filtered_data = DateTimeFilter.filter_after_date(df, "systime", "2023-01-01")
        >>> print(filtered_data)
        """
        if date is None:
            raise ValueError("date parameter is required")
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] > pd.to_datetime(date)]

    @classmethod
    def filter_before_date(
        cls, dataframe: pd.DataFrame, column_name: str = "systime", date: str = None
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows before the specified date.

        Args:
            date (str): The cutoff date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is before the specified date.

        Example:
        --------
        >>> filtered_data = DateTimeFilter.filter_before_date(df, "systime", "2023-01-01")
        >>> print(filtered_data)
        """
        if date is None:
            raise ValueError("date parameter is required")
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] < pd.to_datetime(date)]

    @classmethod
    def filter_between_dates(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows between the specified start and end dates.

        Args:
            start_date (str): The start date of the interval in 'YYYY-MM-DD' format.
            end_date (str): The end date of the interval in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is between the specified dates.

        Example:
        --------
        >>> filtered_data = DateTimeFilter.filter_between_dates(df, "systime", "2023-01-01", "2023-02-01")
        >>> print(filtered_data)
        """
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date parameters are required")
        Base._validate_column(dataframe, column_name)
        mask = (dataframe[column_name] > pd.to_datetime(start_date)) & (
            dataframe[column_name] < pd.to_datetime(end_date)
        )
        return dataframe[mask]

    @classmethod
    def filter_after_datetime(
        cls, dataframe: pd.DataFrame, column_name: str = "systime", datetime: str = None
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows after the specified datetime.

        Args:
            datetime (str): The cutoff datetime in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is after the specified datetime.

        Example:
        --------
        >>> filtered_data = DateTimeFilter.filter_after_datetime(df, "systime", "2023-01-01 12:00:00")
        >>> print(filtered_data)
        """
        if datetime is None:
            raise ValueError("datetime parameter is required")
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] > pd.to_datetime(datetime)]

    @classmethod
    def filter_before_datetime(
        cls, dataframe: pd.DataFrame, column_name: str = "systime", datetime: str = None
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows before the specified datetime.

        Args:
            datetime (str): The cutoff datetime in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is before the specified datetime.

        Example:
        --------
        >>> filtered_data = DateTimeFilter.filter_before_datetime(df, "systime", "2023-01-01 12:00:00")
        >>> print(filtered_data)
        """
        if datetime is None:
            raise ValueError("datetime parameter is required")
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] < pd.to_datetime(datetime)]

    @classmethod
    def filter_between_datetimes(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "systime",
        start_datetime: str = None,
        end_datetime: str = None,
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only rows between the specified start and end datetimes.

        Args:
            start_datetime (str): The start datetime of the interval in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str): The end datetime of the interval in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            pd.DataFrame: A DataFrame containing rows where the 'systime' is between the specified datetimes.

        Example:
        --------
        >>> filtered_data = DateTimeFilter.filter_between_datetimes(df, "systime", "2023-01-01 12:00:00", "2023-02-01 12:00:00")
        >>> print(filtered_data)
        """
        if start_datetime is None or end_datetime is None:
            raise ValueError("start_datetime and end_datetime parameters are required")
        Base._validate_column(dataframe, column_name)
        mask = (dataframe[column_name] > pd.to_datetime(start_datetime)) & (
            dataframe[column_name] < pd.to_datetime(end_datetime)
        )
        return dataframe[mask]
