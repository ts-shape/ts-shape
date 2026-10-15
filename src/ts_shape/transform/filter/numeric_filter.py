import logging

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)


def _numbers(dataframe: pd.DataFrame, value_column: str) -> pd.Series:
    """The column as numbers; text numbers are parsed (with a warning)."""
    Base._validate_column(dataframe, value_column)
    values = dataframe[value_column]
    if ptypes.is_numeric_dtype(values.dtype) and not ptypes.is_bool_dtype(values.dtype):
        return values
    return to_numeric_checked(values, what=f"column {value_column!r}")


def _rows(dataframe: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Rows where ``mask`` is True; a missing comparison counts as False."""
    return dataframe[np.asarray(mask.fillna(False), dtype=bool)]


class IntegerFilter(Base):
    """
    Provides class methods for filtering integer columns in a pandas DataFrame.
    """

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_value_integer_match(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        integer_value: int = 0,
    ) -> pd.DataFrame:
        """Filters rows where 'value_integer' matches the specified integer.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"value_integer"``.
            integer_value: Value to keep. Default 0.

        Returns:
            The rows whose value equals ``integer_value``, in input order.
        """
        values = _numbers(dataframe, value_column)
        return _rows(dataframe, values == integer_value)

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_value_integer_not_match(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        integer_value: int = 0,
    ) -> pd.DataFrame:
        """Filters rows where 'value_integer' does not match the specified integer.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"value_integer"``.
            integer_value: Value to drop. Default 0.

        Returns:
            The rows whose value differs from ``integer_value`` (missing
                values included), in input order.
        """
        values = _numbers(dataframe, value_column)
        return _rows(dataframe, values != integer_value)

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_value_integer_between(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        min_value: int = 0,
        max_value: int = 100,
    ) -> pd.DataFrame:
        """Filters rows where 'value_integer' is between the specified min and max values (inclusive).

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"value_integer"``.
            min_value: Lower bound, inclusive. Default 0.
            max_value: Upper bound, inclusive. Default 100.

        Returns:
            The rows whose value lies in ``[min_value, max_value]``, in
                input order.
        """
        values = _numbers(dataframe, value_column)
        return _rows(dataframe, (values >= min_value) & (values <= max_value))


class DoubleFilter(Base):
    """
    Provides class methods for filtering double (floating-point) columns in a pandas DataFrame,
    particularly focusing on NaN values.
    """

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_nan_value_double(
        cls, dataframe: pd.DataFrame, value_column: str = "value_double"
    ) -> pd.DataFrame:
        """Filters out rows where 'value_double' is NaN.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"value_double"``.

        Returns:
            The rows that have a value (missing values dropped), in input
                order.
        """
        Base._validate_column(dataframe, value_column)
        return dataframe[dataframe[value_column].notna()]

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_value_double_between(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_double",
        min_value: float = 0.0,
        max_value: float = 100.0,
    ) -> pd.DataFrame:
        """Filters rows where 'value_double' is between the specified min and max values (inclusive).

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"value_double"``.
            min_value: Lower bound, inclusive. Default 0.0.
            max_value: Upper bound, inclusive. Default 100.0.

        Returns:
            The rows whose value lies in ``[min_value, max_value]``, in
                input order.
        """
        values = _numbers(dataframe, value_column)
        return _rows(dataframe, (values >= min_value) & (values <= max_value))
