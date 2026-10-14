import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ts_shape.utils.frames import as_datetime, require_columns

from ._values import bool_values, numeric_values
from .boolean_stats import BooleanStatistics
from .numeric_stats import NumericStatistics
from .string_stats import StringStatistics

logger = logging.getLogger(__name__)

#: Long-format value columns and the statistics computed for them. They are
#: classified by name, so every signal gets the same keys whatever its values.
_VALUE_KINDS = {
    "value_bool": "boolean",
    "value_double": "numeric",
    "value_integer": "numeric",
    "value_string": "string",
}

#: Columns that are not described per column.
_SKIPPED = {"uuid", "systime", "is_delta"}


def _kind(series: pd.Series) -> str | None:
    """Statistics kind of a column that is not a long-format value column."""
    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_datetime64_any_dtype(series.dtype):
        return None
    if is_numeric_dtype(series.dtype):
        return "numeric"
    if is_object_dtype(series.dtype) or is_string_dtype(series.dtype):
        return "string"
    return None


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

        The ``is_delta_*`` values are NaN when the group has no ``is_delta``
        column (or no value in it).

        Args:
            group: DataFrame group (rows of a single UUID) to summarize.

        Returns:
            dict: A dictionary with overall statistics.
        """
        if "systime" in group.columns and len(group):
            times = as_datetime(group["systime"], where="DescriptiveFeatures")
            total_time = times.max() - times.min()
        else:
            total_time = pd.NaT
        if "is_delta" in group.columns:
            delta = bool_values(group, "is_delta", where="DescriptiveFeatures")
            flags = delta.dropna().astype(float)
        else:
            flags = pd.Series(dtype="float64")
        statistics = {
            "total_rows": len(group),
            "total_time": total_time,
            "is_delta_sum": float(flags.sum()) if len(flags) else np.nan,
            "is_delta_avg": float(flags.mean()) if len(flags) else np.nan,
            "is_delta_std": float(flags.std()) if len(flags) > 1 else np.nan,
        }
        return statistics

    def compute_per_group(self, group: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Compute and return statistics for each column in the DataFrame group.

        Args:
            group: DataFrame group (rows of a single UUID) to summarize.

        Returns:
            dict: A dictionary with overall statistics, and string, numeric, and
                boolean statistics per column. The long-format value columns
                are classified by name (``value_bool`` boolean,
                ``value_double`` / ``value_integer`` numeric, ``value_string``
                string), so every signal has the same keys; other columns by
                dtype. ``systime``, ``uuid`` and ``is_delta`` are not described
                per column.
        """
        results: dict[str, dict[str, Any]] = {"overall": self.overall_stats(group)}
        for col in group.columns:
            if col in _SKIPPED:
                continue
            kind = _VALUE_KINDS.get(str(col)) or _kind(group[col])
            if kind == "boolean":
                values = bool_values(group, col, where="DescriptiveFeatures")
                results[col] = {
                    "boolean_stats": BooleanStatistics.summary_as_dict(
                        values.to_frame(col), col
                    )
                }
            elif kind == "numeric":
                numbers = numeric_values(group, col, where="DescriptiveFeatures")
                results[col] = {
                    "numeric_stats": NumericStatistics.summary_as_dict(
                        numbers.to_frame(col), col
                    )
                }
            elif kind == "string":
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
                data type. An empty frame gives ``{}`` (or an empty DataFrame).

        Raises:
            ValueError: For an unknown ``output_format``.
            ColumnNotFoundError: If the frame has no ``uuid`` column.
        """
        if output_format not in ("dict", "dataframe"):
            raise ValueError(
                "Invalid output format. Choose either 'dict' or 'dataframe'."
            )
        require_columns(self.data, ["uuid"], where="DescriptiveFeatures.compute")
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

        return {
            uuid: self.compute_per_group(group)
            for uuid, group in self.data.groupby("uuid", sort=True)
        }
