import logging

import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class StringFilter(Base):
    """
    A class for filtering operations on string columns within a pandas DataFrame.

    Provides class methods for operations on string columns.
    """

    @classmethod
    def filter_na_value_string(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> pd.DataFrame:
        """Filters out rows where the specified string column is NA."""
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name].notna()]

    @classmethod
    def filter_value_string_match(
        cls,
        dataframe: pd.DataFrame,
        string_value: str,
        column_name: str = "value_string",
    ) -> pd.DataFrame:
        """Filters rows where the specified string column matches the provided string."""
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] == string_value]

    @classmethod
    def filter_value_string_not_match(
        cls,
        dataframe: pd.DataFrame,
        string_value: str,
        column_name: str = "value_string",
    ) -> pd.DataFrame:
        """Filters rows where the specified string column does not match the provided string."""
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] != string_value]

    @classmethod
    def filter_string_contains(
        cls, dataframe: pd.DataFrame, substring: str, column_name: str = "value_string"
    ) -> pd.DataFrame:
        """Filters rows where the specified string column contains the provided substring."""
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name].str.contains(substring, na=False)]

    @classmethod
    def regex_clean_value_string(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_string",
        regex_pattern: str = r"(\d+)\s*([a-zA-Z]*)",
        replacement: str = "",
        regex: bool = True,
    ) -> pd.DataFrame:
        """Applies a regex pattern to clean the specified string column."""
        Base._validate_column(dataframe, column_name)
        dataframe = dataframe.copy()
        dataframe[column_name] = dataframe[column_name].str.replace(
            regex_pattern, replacement, regex=regex
        )
        return dataframe

    @classmethod
    def detect_changes_in_string(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_string",
        *,
        group_by_uuid: bool = False,
    ) -> pd.DataFrame:
        """Detects changes from row to row in the specified string column.

        Args:
            group_by_uuid: Compare each row with the previous row of the same
                ``uuid``. Default ``False`` compares with the previous row of
                the frame, so on a multi-signal frame the last row of one uuid
                and the first of the next produce a false edge.
        """
        Base._validate_column(dataframe, column_name)
        if group_by_uuid and "uuid" in dataframe.columns:
            previous = dataframe.groupby("uuid", sort=False)[column_name].shift()
        else:
            previous = dataframe[column_name].shift()
        changes_detected = dataframe[column_name].ne(previous)
        result = dataframe[changes_detected]
        if result.empty:
            logger.info(
                "No changes detected in the '%s' column between consecutive rows.",
                column_name,
            )
        return result
