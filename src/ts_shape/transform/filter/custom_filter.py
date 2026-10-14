import logging

import pandas as pd

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class CustomFilter(Base):
    @classmethod
    def filter_custom_conditions(
        cls, dataframe: pd.DataFrame, conditions: str
    ) -> pd.DataFrame:
        """
        Filters the DataFrame based on a set of user-defined conditions passed as a string.

        This method allows for flexible data filtering by evaluating a condition or multiple conditions
        specified in the 'conditions' parameter. The conditions must be provided as a string
        that can be interpreted by pandas' DataFrame.query() method.

        Args:
            dataframe (pd.DataFrame): The DataFrame to apply the filter on.
            conditions (str): A string representing the conditions to filter the DataFrame.
                            The string should be formatted according to pandas query syntax.

        Returns:
            pd.DataFrame: A DataFrame containing only the rows that meet the specified conditions.

        Example:
            >>> import pandas as pd
            >>> from ts_shape.transform.filter.custom_filter import CustomFilter
            >>> df = pd.DataFrame({"age": [25, 35, 45], "score": [90, 85, 70]})
            >>> CustomFilter.filter_custom_conditions(df, "age > 30 and score > 80")
               age  score
            1   35     85

        Note:
            Ensure that the column names and values used in conditions match those in the DataFrame.
            Complex expressions and functions available in pandas query syntax can also be used.
        """
        return dataframe.query(conditions)
