import logging
import pandas as pd  # type: ignore
from typing import Dict, Union
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
        """Returns the count of True values in the boolean column."""
        return dataframe[column_name].sum()

    @classmethod
    def count_false(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> int:
        """Returns the count of False values in the boolean column."""
        return (dataframe[column_name] == False).sum()

    @classmethod
    def count_null(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> int:
        """Returns the count of null (NaN) values in the boolean column."""
        return dataframe[column_name].isna().sum()

    @classmethod
    def count_not_null(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> int:
        """Returns the count of non-null (True or False) values in the boolean column."""
        return dataframe[column_name].notna().sum()

    @classmethod
    def true_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> float:
        """Returns the percentage of True values in the boolean column."""
        true_count = cls.count_true(dataframe, column_name)
        total_count = cls.count_not_null(dataframe, column_name)
        return (true_count / total_count) * 100 if total_count > 0 else 0.0

    @classmethod
    def false_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_bool"
    ) -> float:
        """Returns the percentage of False values in the boolean column."""
        false_count = cls.count_false(dataframe, column_name)
        total_count = cls.count_not_null(dataframe, column_name)
        return (false_count / total_count) * 100 if total_count > 0 else 0.0

    @classmethod
    def mode(cls, dataframe: pd.DataFrame, column_name: str) -> bool:
        """Returns the mode (most common value) of the specified boolean column."""
        return dataframe[column_name].mode()[0]

    @classmethod
    def is_balanced(cls, dataframe: pd.DataFrame, column_name: str) -> bool:
        """Indicates if the distribution is balanced (50% True and False) in the specified boolean column."""
        true_percentage = dataframe[column_name].mean()
        return true_percentage == 0.5

    @classmethod
    def summary_as_dict(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> Dict[str, Union[int, float, bool]]:
        """Returns a summary of boolean statistics for the specified column as a dictionary."""
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
        """Returns a summary of boolean statistics for the specified column as a DataFrame."""
        summary_data = cls.summary_as_dict(dataframe, column_name)
        return pd.DataFrame([summary_data])
