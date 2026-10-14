import logging
from typing import cast

import pandas as pd
from scipy import stats

from ts_shape.features.stats._values import numeric_values
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)

# pandas ships no type information here (no py.typed, pandas-stubs not
# installed), so every reduction is ``Any``; ``cast`` records the documented
# scalar type without converting it (the numpy scalar is returned as before).


class NumericStatistics(Base):
    """
    Provides class methods to calculate statistics on numeric columns in a pandas DataFrame.
    """

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_mean(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the mean of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_mean"
            ).mean(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_median(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the median of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_median"
            ).median(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_std(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the standard deviation of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_std"
            ).std(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_variance(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the variance of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_variance"
            ).var(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_min(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the minimum value of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_min"
            ).min(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_max(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the maximum value of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_max"
            ).max(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_sum(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the sum of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_sum"
            ).sum(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_kurtosis(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the kurtosis of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_kurtosis"
            ).kurt(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_skewness(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the skewness of a specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_skewness"
            ).skew(),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_quantile(
        cls, dataframe: pd.DataFrame, value_column: str, quantile: float
    ) -> float:
        """Calculate a specific quantile of the column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
            quantile: Quantile to compute, between 0 and 1 (0.5 is the median).
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.column_quantile"
            ).quantile(quantile),
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_iqr(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the interquartile range of the column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        values = numeric_values(
            dataframe, value_column, where="NumericStatistics.column_iqr"
        )
        if not values.notna().any():
            return float("nan")
        return float(stats.iqr(values, nan_policy="omit"))

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_range(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the range of the column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        values = numeric_values(
            dataframe, value_column, where="NumericStatistics.column_range"
        )
        return cast(float, values.max() - values.min())

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_mad(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the mean absolute deviation of the column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        series = numeric_values(
            dataframe, value_column, where="NumericStatistics.column_mad"
        )
        return cast(float, (series - series.mean()).abs().mean())

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def column_mode(cls, dataframe: pd.DataFrame, value_column: str) -> float | None:
        """Calculate the mode of a specified column. Returns None if column is empty.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        modes = numeric_values(
            dataframe, value_column, where="NumericStatistics.column_mode"
        ).mode()
        return cast(float, modes.iloc[0]) if not modes.empty else None

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def coefficient_of_variation(
        cls, dataframe: pd.DataFrame, value_column: str
    ) -> float:
        """Calculate the coefficient of variation of the column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.

        Returns:
            Standard deviation divided by the mean, or ``float("nan")`` when the
            mean is 0 (the ratio is undefined).
        """
        mean = cls.column_mean(dataframe, value_column)
        if mean == 0:
            return float("nan")
        return float(cls.column_std(dataframe, value_column) / mean)

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def standard_error_mean(cls, dataframe: pd.DataFrame, value_column: str) -> float:
        """Calculate the standard error of the mean for the column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.
        """
        return cast(
            float,
            numeric_values(
                dataframe, value_column, where="NumericStatistics.standard_error_mean"
            ).sem(),
        )

    @classmethod
    def describe(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Provide a statistical summary for numeric columns in the DataFrame.

        Args:
            dataframe: Input DataFrame; every numeric column is described.

        Returns:
            The output of ``DataFrame.describe()``: one column per numeric column, rows
                ``count``, ``mean``, ``std``, ``min``, ``25%``, ``50%``, ``75%`` and
                ``max``.
        """
        return dataframe.describe()

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def summary_as_dict(
        cls, dataframe: pd.DataFrame, value_column: str
    ) -> dict[str, float | int | None]:
        """Returns a dictionary with comprehensive numeric statistics for the specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.

        Returns:
            Dict with keys ``min``, ``max``, ``mean``, ``median``, ``std``, ``var``,
                ``sum``, ``kurtosis``, ``skewness``, ``q1``, ``q3``, ``iqr``, ``range``,
                ``mad``, ``coeff_var``, ``sem``, ``mode``, ``percentile_90`` and
                ``percentile_10``.
        """
        series = numeric_values(
            dataframe, value_column, where="NumericStatistics.summary_as_dict"
        )
        dataframe = series.to_frame(value_column)
        return {
            "min": cls.column_min(dataframe, value_column),
            "max": cls.column_max(dataframe, value_column),
            "mean": cls.column_mean(dataframe, value_column),
            "median": cls.column_median(dataframe, value_column),
            "std": cls.column_std(dataframe, value_column),
            "var": cls.column_variance(dataframe, value_column),
            "sum": cls.column_sum(dataframe, value_column),
            "kurtosis": cls.column_kurtosis(dataframe, value_column),
            "skewness": cls.column_skewness(dataframe, value_column),
            "q1": cls.column_quantile(dataframe, value_column, 0.25),
            "q3": cls.column_quantile(dataframe, value_column, 0.75),
            "iqr": cls.column_iqr(dataframe, value_column),
            "range": cls.column_range(dataframe, value_column),
            "mad": cls.column_mad(dataframe, value_column),
            "coeff_var": cls.coefficient_of_variation(dataframe, value_column),
            "sem": cls.standard_error_mean(dataframe, value_column),
            "mode": cls.column_mode(dataframe, value_column),
            "percentile_90": cls.column_quantile(dataframe, value_column, 0.90),
            "percentile_10": cls.column_quantile(dataframe, value_column, 0.10),
        }

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def summary_as_dataframe(
        cls, dataframe: pd.DataFrame, value_column: str
    ) -> pd.DataFrame:
        """Returns a DataFrame with comprehensive numeric statistics for the specified column.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the numeric column to analyze.

        Returns:
            One-row DataFrame with the keys of `summary_as_dict` as columns.
        """
        summary_data = cls.summary_as_dict(dataframe, value_column)
        return pd.DataFrame([summary_data])
