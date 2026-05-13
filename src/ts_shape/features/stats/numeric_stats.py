import logging
import pandas as pd  # type: ignore
from scipy import stats
from typing import Dict, Union
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class NumericStatistics(Base):
    """
    Provides class methods to calculate statistics on numeric columns in a pandas DataFrame.
    """

    @classmethod
    def column_mean(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the mean of a specified column."""
        return dataframe[column_name].mean()

    @classmethod
    def column_median(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the median of a specified column."""
        return dataframe[column_name].median()

    @classmethod
    def column_std(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the standard deviation of a specified column."""
        return dataframe[column_name].std()

    @classmethod
    def column_variance(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the variance of a specified column."""
        return dataframe[column_name].var()

    @classmethod
    def column_min(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the minimum value of a specified column."""
        return dataframe[column_name].min()

    @classmethod
    def column_max(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the maximum value of a specified column."""
        return dataframe[column_name].max()

    @classmethod
    def column_sum(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the sum of a specified column."""
        return dataframe[column_name].sum()

    @classmethod
    def column_kurtosis(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the kurtosis of a specified column."""
        return dataframe[column_name].kurt()

    @classmethod
    def column_skewness(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the skewness of a specified column."""
        return dataframe[column_name].skew()

    @classmethod
    def column_quantile(
        cls, dataframe: pd.DataFrame, column_name: str, quantile: float
    ) -> float:
        """Calculate a specific quantile of the column."""
        return dataframe[column_name].quantile(quantile)

    @classmethod
    def column_iqr(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the interquartile range of the column."""
        return stats.iqr(dataframe[column_name])

    @classmethod
    def column_range(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the range of the column."""
        return cls.column_max(dataframe, column_name) - cls.column_min(
            dataframe, column_name
        )

    @classmethod
    def column_mad(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the mean absolute deviation of the column."""
        series = dataframe[column_name]
        return (series - series.mean()).abs().mean()

    @classmethod
    def column_mode(cls, dataframe: pd.DataFrame, column_name: str):
        """Calculate the mode of a specified column. Returns None if column is empty."""
        modes = dataframe[column_name].mode()
        return modes.iloc[0] if not modes.empty else None

    @classmethod
    def coefficient_of_variation(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> float:
        """Calculate the coefficient of variation of the column."""
        mean = cls.column_mean(dataframe, column_name)
        return cls.column_std(dataframe, column_name) / mean if mean != 0 else None

    @classmethod
    def standard_error_mean(cls, dataframe: pd.DataFrame, column_name: str) -> float:
        """Calculate the standard error of the mean for the column."""
        return dataframe[column_name].sem()

    @classmethod
    def describe(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Provide a statistical summary for numeric columns in the DataFrame."""
        return dataframe.describe()

    @classmethod
    def summary_as_dict(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> Dict[str, Union[float, int]]:
        """Returns a dictionary with comprehensive numeric statistics for the specified column."""
        series = dataframe[column_name]
        return {
            "min": cls.column_min(dataframe, column_name),
            "max": cls.column_max(dataframe, column_name),
            "mean": cls.column_mean(dataframe, column_name),
            "median": cls.column_median(dataframe, column_name),
            "std": cls.column_std(dataframe, column_name),
            "var": cls.column_variance(dataframe, column_name),
            "sum": cls.column_sum(dataframe, column_name),
            "kurtosis": cls.column_kurtosis(dataframe, column_name),
            "skewness": cls.column_skewness(dataframe, column_name),
            "q1": cls.column_quantile(dataframe, column_name, 0.25),
            "q3": cls.column_quantile(dataframe, column_name, 0.75),
            "iqr": cls.column_iqr(dataframe, column_name),
            "range": cls.column_range(dataframe, column_name),
            "mad": cls.column_mad(dataframe, column_name),
            "coeff_var": cls.coefficient_of_variation(dataframe, column_name),
            "sem": cls.standard_error_mean(dataframe, column_name),
            "mode": cls.column_mode(dataframe, column_name),
            "percentile_90": cls.column_quantile(dataframe, column_name, 0.90),
            "percentile_10": cls.column_quantile(dataframe, column_name, 0.10),
        }

    @classmethod
    def summary_as_dataframe(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """Returns a DataFrame with comprehensive numeric statistics for the specified column."""
        summary_data = cls.summary_as_dict(dataframe, column_name)
        return pd.DataFrame([summary_data])
