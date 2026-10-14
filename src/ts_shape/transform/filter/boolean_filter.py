import logging

import pandas as pd

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class IsDeltaFilter(Base):
    """
    Provides class methods for filtering is_delta columns in a pandas DataFrame.
    """

    @classmethod
    def filter_is_delta_true(
        cls, dataframe: pd.DataFrame, column_name: str = "is_delta"
    ) -> pd.DataFrame:
        """Filters rows where 'is_delta' is True.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"is_delta"``.
        """
        Base._validate_column(dataframe, column_name)
        # ``eq`` keeps missing values out of both filters.
        return dataframe[dataframe[column_name].eq(True)]

    @classmethod
    def filter_is_delta_false(
        cls, dataframe: pd.DataFrame, column_name: str = "is_delta"
    ) -> pd.DataFrame:
        """Filters rows where 'is_delta' is False.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"is_delta"``.
        """
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name].eq(False)]


def _previous(
    dataframe: pd.DataFrame, column_name: str, group_by_uuid: bool
) -> pd.Series:
    """Previous value of ``column_name``, optionally within each uuid."""
    if group_by_uuid and "uuid" in dataframe.columns:
        return dataframe.groupby("uuid", sort=False)[column_name].shift(1)
    return dataframe[column_name].shift(1)


class BooleanFilter(Base):
    """
    Provides class methods for filtering boolean columns in a pandas DataFrame,
    particularly focusing on status changes.
    """

    @classmethod
    def filter_falling_value_bool(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_bool",
        *,
        group_by_uuid: bool = True,
    ) -> pd.DataFrame:
        """Filters rows where 'value_bool' changes from True to False.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"value_bool"``.
            group_by_uuid: When ``True`` (default) each row is compared with
                the previous row of the same ``uuid`` (when the frame has a
                ``uuid`` column), so a multi-signal frame produces no edge
                between the last row of one uuid and the first of the next.
                ``False`` compares with the previous row of the frame.
        """
        Base._validate_column(dataframe, column_name)
        previous = _previous(dataframe, column_name, group_by_uuid)
        return dataframe[previous.eq(True) & dataframe[column_name].eq(False)]

    @classmethod
    def filter_raising_value_bool(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_bool",
        *,
        group_by_uuid: bool = True,
    ) -> pd.DataFrame:
        """Filters rows where 'value_bool' changes from False to True.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"value_bool"``.
            group_by_uuid: When ``True`` (default) each row is compared with
                the previous row of the same ``uuid`` (when the frame has a
                ``uuid`` column), so a multi-signal frame produces no edge
                between the last row of one uuid and the first of the next.
                ``False`` compares with the previous row of the frame.
        """
        Base._validate_column(dataframe, column_name)
        previous = _previous(dataframe, column_name, group_by_uuid)
        return dataframe[previous.eq(False) & dataframe[column_name].eq(True)]
