import logging

import pandas as pd  # type: ignore

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
        return dataframe[
            dataframe[column_name] == True  # noqa: E712 pandas-series-bool-comparison
        ]  # noqa: E712 pandas-series-bool-comparison

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
        return dataframe[
            dataframe[column_name] == False  # noqa: E712 pandas-series-bool-comparison
        ]  # noqa: E712 pandas-series-bool-comparison


def _previous(dataframe: pd.DataFrame, column_name: str, group_by_uuid: bool):
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
        group_by_uuid: bool = False,
    ) -> pd.DataFrame:
        """Filters rows where 'value_bool' changes from True to False.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"value_bool"``.
            group_by_uuid: Compare each row with the previous row of the same
                ``uuid``. Default ``False`` compares with the previous row of
                the frame, so on a multi-signal frame the last row of one uuid
                and the first of the next produce a false edge.
        """
        Base._validate_column(dataframe, column_name)
        previous = _previous(dataframe, column_name, group_by_uuid)
        return dataframe[
            (previous == True)  # noqa: E712 pandas-series-bool-comparison
            & (
                dataframe[column_name]  # noqa: E712 pandas-series-bool-comparison
                == False  # noqa: E712 pandas-series-bool-comparison
            )  # noqa: E712 pandas-series-bool-comparison
        ]  # noqa: E712 pandas-series-bool-comparison

    @classmethod
    def filter_raising_value_bool(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_bool",
        *,
        group_by_uuid: bool = False,
    ) -> pd.DataFrame:
        """Filters rows where 'value_bool' changes from False to True.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"value_bool"``.
            group_by_uuid: Compare each row with the previous row of the same
                ``uuid``. Default ``False`` compares with the previous row of
                the frame, so on a multi-signal frame the last row of one uuid
                and the first of the next produce a false edge.
        """
        Base._validate_column(dataframe, column_name)
        previous = _previous(dataframe, column_name, group_by_uuid)
        return dataframe[
            (previous == False)  # noqa: E712 pandas-series-bool-comparison
            & (
                dataframe[column_name]  # noqa: E712 pandas-series-bool-comparison
                == True  # noqa: E712 pandas-series-bool-comparison
            )  # noqa: E712 pandas-series-bool-comparison
        ]  # noqa: E712 pandas-series-bool-comparison
