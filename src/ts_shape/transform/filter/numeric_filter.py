import logging
import pandas as pd  # type: ignore
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class IntegerFilter(Base):
    """
    Provides class methods for filtering integer columns in a pandas DataFrame.
    """

    @classmethod
    def filter_value_integer_match(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_integer",
        integer_value: int = 0,
    ) -> pd.DataFrame:
        """Filters rows where 'value_integer' matches the specified integer."""
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] == integer_value]

    @classmethod
    def filter_value_integer_not_match(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_integer",
        integer_value: int = 0,
    ) -> pd.DataFrame:
        """Filters rows where 'value_integer' does not match the specified integer."""
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] != integer_value]

    @classmethod
    def filter_value_integer_between(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_integer",
        min_value: int = 0,
        max_value: int = 100,
    ) -> pd.DataFrame:
        """Filters rows where 'value_integer' is between the specified min and max values (inclusive)."""
        Base._validate_column(dataframe, column_name)
        return dataframe[
            (dataframe[column_name] >= min_value)
            & (dataframe[column_name] <= max_value)
        ]


class DoubleFilter(Base):
    """
    Provides class methods for filtering double (floating-point) columns in a pandas DataFrame,
    particularly focusing on NaN values.
    """

    @classmethod
    def filter_nan_value_double(
        cls, dataframe: pd.DataFrame, column_name: str = "value_double"
    ) -> pd.DataFrame:
        """Filters out rows where 'value_double' is NaN."""
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name].notna()]

    @classmethod
    def filter_value_double_between(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_double",
        min_value: float = 0.0,
        max_value: float = 100.0,
    ) -> pd.DataFrame:
        """Filters rows where 'value_double' is between the specified min and max values (inclusive)."""
        Base._validate_column(dataframe, column_name)
        return dataframe[
            (dataframe[column_name] >= min_value)
            & (dataframe[column_name] <= max_value)
        ]
