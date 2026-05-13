import logging
import pandas as pd  # type: ignore
from typing import Callable, Any
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class LambdaProcessor(Base):
    """
    Provides class methods for applying lambda or callable functions to columns in a pandas DataFrame.
    This class inherits from Base, ensuring consistency with other processors.
    """

    @classmethod
    def apply_function(
        cls, dataframe: pd.DataFrame, column_name: str, func: Callable[[Any], Any]
    ) -> pd.DataFrame:
        """
        Applies a lambda or callable function to a specified column in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            column_name (str): The name of the column to apply the function to.
            func (Callable): The lambda function or callable to apply to the column.

        Returns:
            pd.DataFrame: The DataFrame with the transformed column.
        """
        if column_name not in dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        dataframe[column_name] = func(dataframe[column_name].values)
        return dataframe
