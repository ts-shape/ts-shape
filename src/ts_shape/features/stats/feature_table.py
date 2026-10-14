import logging
from typing import Any

import pandas as pd
from pandas.api.types import (
    infer_dtype,
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from .boolean_stats import BooleanStatistics
from .numeric_stats import NumericStatistics
from .string_stats import StringStatistics

logger = logging.getLogger(__name__)


class DescriptiveFeatures:
    """Compute descriptive statistics for a DataFrame, grouped by UUID.

    Use `compute` to get descriptive statistics for each UUID in the DataFrame.

    Attributes:
        data (pandas.DataFrame): DataFrame containing the data.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with the DataFrame to describe.

        Args:
            dataframe: DataFrame containing the data.
        """
        self.data = dataframe

    def overall_stats(self, group: pd.DataFrame) -> dict[str, int | float]:
        """Compute and return overall statistics for the DataFrame group.

        The returned dictionary contains:

        - **total_rows**: Total number of rows in the group.
        - **total_time**: Total time difference from max and min of 'systime' column.
        - **is_delta_sum**: Sum of the 'is_delta' column.
        - **is_delta_avg**: Mean of the 'is_delta' column.
        - **is_delta_std**: Standard deviation of the 'is_delta' column.

        Args:
            group: DataFrame group (rows of a single UUID) to summarize.

        Returns:
            dict: A dictionary with overall statistics.
        """
        statistics = {
            "total_rows": len(group),
            "total_time": group["systime"].max() - group["systime"].min(),
            "is_delta_sum": group["is_delta"].sum(),
            "is_delta_avg": group["is_delta"].mean(),
            "is_delta_std": group["is_delta"].std(),
        }
        return statistics

    def compute_per_group(self, group: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Compute and return statistics for each column in the DataFrame group.

        Args:
            group: DataFrame group (rows of a single UUID) to summarize.

        Returns:
            dict: A dictionary with overall statistics, and string, numeric, and
                boolean statistics per column.
        """
        results: dict[str, dict[str, Any]] = {"overall": self.overall_stats(group)}
        for col in group.columns:
            if col == "uuid":
                continue
            # Long-format value columns are object dtype as soon as a signal
            # has missing values: judge them by the values they hold.
            kind = (
                infer_dtype(group[col], skipna=True)
                if is_object_dtype(group[col])
                else None
            )
            if kind == "boolean":
                values = group[col].dropna().astype(bool).to_frame(col)
                results[col] = {
                    "boolean_stats": BooleanStatistics.summary_as_dict(values, col)
                }
            elif kind in ("integer", "floating", "mixed-integer-float", "decimal"):
                values = pd.to_numeric(group[col]).to_frame(col)
                results[col] = {
                    "numeric_stats": NumericStatistics.summary_as_dict(values, col)
                }
            elif is_bool_dtype(group[col]):
                # Use BooleanStatistics for boolean columns
                results[col] = {
                    "boolean_stats": BooleanStatistics.summary_as_dict(group, col)
                }
            elif is_numeric_dtype(group[col]):
                # Use NumericStatistics for numeric columns
                results[col] = {
                    "numeric_stats": NumericStatistics.summary_as_dict(group, col)
                }
            elif is_object_dtype(group[col]):
                # Use StringStatistics for string columns
                results[col] = {
                    "string_stats": StringStatistics.summary_as_dict(group, col)
                }

        return results

    def compute(
        self, output_format: str = "dict"
    ) -> pd.DataFrame | dict[str, dict[str, dict[str, Any]]]:
        """Compute and return descriptive statistics for each UUID in the DataFrame.

        Args:
            output_format (str, optional): The desired output format ('dict' or
                'dataframe'). Defaults to 'dict'.

        Returns:
            Union[DataFrame, dict]: A DataFrame or a nested dictionary with the
                UUID as the key and specific statistics related to that UUID's
                data type.
        """
        if output_format == "dataframe":
            rows_list = []

            # Iterate through each group of UUID
            for uuid, group in self.data.groupby("uuid"):
                stats_per_group = self.compute_per_group(group)

                # Iterate through the nested stats and create flat columns
                row_dict = {}
                for section, stats in stats_per_group.items():
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    column_name = f"{uuid}::{section}::{key}::{sub_key}"
                                    row_dict[column_name] = sub_value
                            else:
                                column_name = f"{uuid}::{section}::{key}"
                                row_dict[column_name] = value
                    else:
                        column_name = f"{uuid}::{section}"
                        row_dict[column_name] = stats

                rows_list.append(row_dict)

            return pd.DataFrame(rows_list)

        elif output_format == "dict":
            return self.data.groupby("uuid").apply(self.compute_per_group).to_dict()

        else:
            raise ValueError(
                "Invalid output format. Choose either 'dict' or 'dataframe'."
            )
