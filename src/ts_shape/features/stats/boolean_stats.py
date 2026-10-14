import logging

import pandas as pd

from ts_shape.features.stats._values import bool_values
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class BooleanStatistics(Base):
    """
    Provides class methods to calculate statistics on a boolean column in a pandas DataFrame.
    """

    @classmethod
    def count_true(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> int:
        """Returns the count of True values in the boolean column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze. Default
                ``"value_bool"``.
        """
        values = bool_values(
            dataframe, column_name, where="BooleanStatistics.count_true"
        )
        return int(values.eq(True).sum())

    @classmethod
    def count_false(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> int:
        """Returns the count of False values in the boolean column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze. Default
                ``"value_bool"``.
        """
        values = bool_values(
            dataframe, column_name, where="BooleanStatistics.count_false"
        )
        return int(values.eq(False).sum())

    @classmethod
    def count_null(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> int:
        """Returns the count of null (NaN) values in the boolean column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze. Default
                ``"value_bool"``.
        """
        values = bool_values(
            dataframe, column_name, where="BooleanStatistics.count_null"
        )
        return int(values.isna().sum())

    @classmethod
    def count_not_null(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> int:
        """Returns the count of non-null (True or False) values in the boolean column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze. Default
                ``"value_bool"``.
        """
        values = bool_values(
            dataframe, column_name, where="BooleanStatistics.count_not_null"
        )
        return int(values.notna().sum())

    @classmethod
    def true_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> float:
        """Returns the percentage of True values in the boolean column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze. Default
                ``"value_bool"``.


        Returns:
            Percentage (0-100) of the non-null values that are True; NaN
                when the column holds no non-null value.
        """
        true_count = cls.count_true(dataframe, column_name)
        total_count = cls.count_not_null(dataframe, column_name)
        return (true_count / total_count) * 100 if total_count > 0 else float("nan")

    @classmethod
    def false_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> float:
        """Returns the percentage of False values in the boolean column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze. Default
                ``"value_bool"``.


        Returns:
            Percentage (0-100) of the non-null values that are False; NaN
                when the column holds no non-null value.
        """
        false_count = cls.count_false(dataframe, column_name)
        total_count = cls.count_not_null(dataframe, column_name)
        return (false_count / total_count) * 100 if total_count > 0 else float("nan")

    @classmethod
    def mode(cls, dataframe: pd.DataFrame, column_name: str) -> bool | None:
        """Returns the mode (most common value) of the specified boolean column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze.

        Returns:
            The most common non-null value (``False`` on a tie), or ``None``
                when the column holds no non-null value.
        """
        values = bool_values(dataframe, column_name, where="BooleanStatistics.mode")
        modes = values.dropna().mode()
        if modes.empty:
            return None
        return bool(modes.iloc[0])

    @classmethod
    def is_balanced(cls, dataframe: pd.DataFrame, column_name: str) -> bool:
        """Indicates if the distribution is balanced (50% True and False) in the specified boolean column.

        The check is exact on counts: the column is balanced when twice the
        number of True values equals the number of non-null values. Nulls are
        ignored, and an empty (or all-null) column is not balanced.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze.
        """
        n = cls.count_not_null(dataframe, column_name)
        return n > 0 and cls.count_true(dataframe, column_name) * 2 == n

    @classmethod
    def summary_as_dict(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> dict[str, int | float | bool | None]:
        """Returns a summary of boolean statistics for the specified column as a dictionary.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze.

        Returns:
            Dict with keys ``true_count``, ``false_count``, ``true_percentage``,
                ``false_percentage``, ``mode`` and ``is_balanced``.
        """
        return {
            "true_count": cls.count_true(dataframe, column_name),
            "false_count": cls.count_false(dataframe, column_name),
            "true_percentage": cls.true_percentage(dataframe, column_name),
            "false_percentage": cls.false_percentage(dataframe, column_name),
            "mode": cls.mode(dataframe, column_name),
            "is_balanced": cls.is_balanced(dataframe, column_name),
        }

    @classmethod
    def summary_as_dataframe(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """Returns a summary of boolean statistics for the specified column as a DataFrame.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the boolean column to analyze.

        Returns:
            One-row DataFrame with the keys of `summary_as_dict` as columns.
        """
        summary_data = cls.summary_as_dict(dataframe, column_name)
        return pd.DataFrame([summary_data])
