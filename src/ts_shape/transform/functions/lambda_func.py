import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)


class LambdaProcessor(Base):
    """
    Provides class methods for applying lambda or callable functions to columns in a pandas DataFrame.
    This class inherits from Base, ensuring consistency with other processors.
    """

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def apply_function(
        cls, dataframe: pd.DataFrame, value_column: str, func: Callable[[Any], Any]
    ) -> pd.DataFrame:
        """
        Applies a lambda or callable function to a specified column in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            value_column (str): The name of the column to apply the function to.
            func (Callable): The lambda function or callable to apply to the column.

        Returns:
            pd.DataFrame: A copy of the DataFrame with the transformed column
                (the input frame is not modified).

        Raises:
            ColumnNotFoundError: If the column is missing.
        """
        require_columns(
            dataframe, [value_column], where="LambdaProcessor.apply_function"
        )
        # Work on a copy: the caller's frame must not be mutated.
        dataframe = dataframe.copy()
        dataframe[value_column] = func(dataframe[value_column].values)
        return dataframe
