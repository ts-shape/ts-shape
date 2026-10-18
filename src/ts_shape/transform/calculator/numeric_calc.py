import logging

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)


def _operand(dataframe: pd.DataFrame, value_column: str, where: str) -> pd.Series:
    """The column to calculate on; numbers stored as text are parsed.

    Raises:
        ColumnNotFoundError: If the column is missing.
    """
    require_columns(dataframe, [value_column], where=f"IntegerCalc.{where}")
    values = dataframe[value_column]
    if ptypes.is_object_dtype(values.dtype) or ptypes.is_string_dtype(values.dtype):
        return to_numeric_checked(values, what=f"column {value_column!r}")
    return values


class IntegerCalc(Base):
    """
    Provides class methods for performing calculations on integer columns in a pandas DataFrame.
    """

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def scale_column(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        factor: float = 1,
    ) -> pd.DataFrame:
        """
        Scales the integer column by the given factor.

        Args:
            dataframe (pd.DataFrame): The DataFrame to perform the operation on.
            value_column (str): The column to apply the scaling to.
            factor (float): The scaling factor.

        Returns:
            pd.DataFrame: The DataFrame with the scaled column.
        """
        values = _operand(dataframe, value_column, "scale_column")
        dataframe = dataframe.copy()
        dataframe[value_column] = values * factor
        return dataframe

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def offset_column(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        offset_value: float = 0,
    ) -> pd.DataFrame:
        """
        Offsets the integer column by the given value.

        Args:
            dataframe (pd.DataFrame): The DataFrame to perform the operation on.
            value_column (str): The column to apply the offset to.
            offset_value (float): The value to add (positive) or subtract (negative) from each element in the column.

        Returns:
            pd.DataFrame: The DataFrame with the offset column.
        """
        values = _operand(dataframe, value_column, "offset_column")
        dataframe = dataframe.copy()
        dataframe[value_column] = values + offset_value
        return dataframe

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def divide_column(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        divisor: float = 1,
    ) -> pd.DataFrame:
        """
        Divides each value in the integer column by the given divisor.

        Args:
            dataframe (pd.DataFrame): The DataFrame to perform the operation on.
            value_column (str): The column to apply the division to.
            divisor (float): The value by which to divide each element.

        Returns:
            pd.DataFrame: The DataFrame with the divided column.

        Raises:
            ValueError: If divisor is zero.
        """
        if divisor == 0:
            raise ValueError("divisor cannot be zero")
        values = _operand(dataframe, value_column, "divide_column")
        dataframe = dataframe.copy()
        dataframe[value_column] = values / divisor
        return dataframe

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def subtract_column(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        subtract_value: float = 0,
    ) -> pd.DataFrame:
        """
        Subtracts a given value from each element in the integer column.

        Args:
            dataframe (pd.DataFrame): The DataFrame to perform the operation on.
            value_column (str): The column to apply the subtraction to.
            subtract_value (float): The value to subtract from each element.

        Returns:
            pd.DataFrame: The DataFrame with the subtracted column.
        """
        values = _operand(dataframe, value_column, "subtract_column")
        dataframe = dataframe.copy()
        dataframe[value_column] = values - subtract_value
        return dataframe

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def calculate_with_fixed_factors(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        multiply_factor: float = 1,
        add_factor: float = 0,
    ) -> pd.DataFrame:
        """
        Performs a calculation by multiplying with a factor and then adding an additional factor.

        Args:
            dataframe (pd.DataFrame): The DataFrame to perform the operation on.
            value_column (str): The column to apply the calculations to.
            multiply_factor (float): The factor to multiply each value by. Defaults to 1 (no scaling).
            add_factor (float): The value to add after multiplication. Defaults to 0 (no offset).

        Returns:
            pd.DataFrame: The DataFrame after applying the calculations.
        """
        values = _operand(dataframe, value_column, "calculate_with_fixed_factors")
        dataframe = dataframe.copy()
        dataframe[value_column] = (values * multiply_factor) + add_factor
        return dataframe

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def mod_column(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        mod_value: int = 1,
    ) -> pd.DataFrame:
        """
        Performs a modulus operation on the integer column with a specified value.

        Args:
            dataframe (pd.DataFrame): The DataFrame to perform the operation on.
            value_column (str): The column to apply the modulus operation to.
            mod_value (int): The value to perform the modulus operation with.

        Returns:
            pd.DataFrame: The DataFrame with the modulus operation applied.

        Raises:
            ValueError: If mod_value is zero.
        """
        if mod_value == 0:
            raise ValueError("mod_value cannot be zero")
        values = _operand(dataframe, value_column, "mod_column")
        dataframe = dataframe.copy()
        if isinstance(values.dtype, pd.ArrowDtype):
            # pyarrow has no modulo kernel; compute on the nullable equivalent.
            nullable = "Int64" if ptypes.is_integer_dtype(values.dtype) else "Float64"
            dataframe[value_column] = (values.astype(nullable) % mod_value).astype(
                values.dtype
            )
        else:
            dataframe[value_column] = values % mod_value
        return dataframe

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def power_column(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_integer",
        power_value: float = 1,
    ) -> pd.DataFrame:
        """
        Raises each value in the integer column to the power of a specified value.

        An integer column stays integer only when the result is exact: a
        non-negative whole exponent whose results all fit in int64. A
        negative or fractional exponent, or a result beyond int64 (e.g.
        ``10 ** 64``), gives float64 values (``numpy.float_power``) instead
        of wrapped-around integers or an error: ``[2, 3, 10] ** -1`` is
        ``[0.5, 0.333..., 0.1]``. (A nullable ``Int64`` column becomes
        ``Float64``, a pyarrow integer column ``double[pyarrow]``.) Other
        columns use ``values ** power_value``.

        Args:
            dataframe (pd.DataFrame): The DataFrame to perform the operation on.
            value_column (str): The column to apply the power operation to.
            power_value (float): The exponent to raise each element to.

        Returns:
            pd.DataFrame: The DataFrame with the power operation applied.
        """
        values = _operand(dataframe, value_column, "power_column")
        dataframe = dataframe.copy()
        dataframe[value_column] = _power(values, power_value)
        return dataframe


def _power(values: pd.Series, power_value: float) -> pd.Series:
    """``values ** power_value``, in float64 where integers cannot hold it."""
    if not ptypes.is_integer_dtype(values.dtype) or ptypes.is_bool_dtype(values.dtype):
        return values**power_value
    exponent = float(power_value)
    floats = values.to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(all="ignore"):
        result = np.float_power(floats, exponent)
    fits = not bool((np.abs(result[np.isfinite(floats)]) >= 2.0**63).any())
    if exponent.is_integer() and exponent >= 0 and fits:
        return values ** int(exponent)
    out = pd.Series(result, index=values.index, name=values.name)
    if isinstance(values.dtype, pd.ArrowDtype):
        return out.astype("double[pyarrow]")
    if isinstance(values.dtype, pd.api.extensions.ExtensionDtype):
        return out.astype("Float64")
    return out
