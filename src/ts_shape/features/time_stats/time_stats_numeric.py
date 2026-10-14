import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.counters import counter_per_window
from ts_shape.utils.frames import as_datetime, require_columns

logger = logging.getLogger(__name__)


def _prepared(
    dataframe: pd.DataFrame, time_column: str, value_column: str, where: str
) -> pd.DataFrame:
    """A copy with a datetime time column and a float value column."""
    where = f"TimeGroupedStatistics.{where}"
    require_columns(dataframe, [time_column, value_column], where=where)
    df = dataframe.copy()
    df[time_column] = as_datetime(df[time_column], where=where)
    values = df[value_column]
    if not (
        pd.api.types.is_numeric_dtype(values.dtype)
        and not isinstance(values.dtype, pd.api.extensions.ExtensionDtype)
    ):
        values = to_numeric_checked(
            values, what=f"column {value_column!r}", stacklevel=4
        )
        df[value_column] = values.astype("float64")
    return df


class TimeGroupedStatistics(Base):
    """
    A class for calculating time-grouped statistics on numeric data, with class methods to apply various statistical functions.
    """

    @classmethod
    def calculate_statistic(
        cls,
        dataframe: pd.DataFrame,
        time_column: str,
        value_column: str,
        freq: str,
        stat_method: str,
        *,
        group_by_uuid: bool = True,
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate a specified statistic on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
            value_column (str): The name of the numeric column to calculate statistics on.
            freq (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
            stat_method (str): The statistical method to apply ('mean', 'sum', 'min', 'max', 'diff', 'range').
            group_by_uuid (bool): When ``True`` (default) and the frame has a
                ``uuid`` column, compute the statistic per ``uuid`` (result
                indexed by ``uuid`` and time), so signals are never mixed.
                ``False`` pools every row into one series. A frame without a
                ``uuid`` column is always treated as one series.
            handle_resets (bool): For ``'diff'``: when ``True`` (default) the
                column is treated as a cumulative counter and the reset-aware
                increments are summed per interval (see
                :func:`ts_shape.utils.counters.counter_per_window`). When
                ``False`` the result is ``last - first`` within each interval,
                which is the right choice for a signal that is not a counter
                (it goes negative across a counter reset). Ignored for the
                other methods.

        Returns:
            pd.DataFrame: A DataFrame with the time intervals and the calculated statistics.
        """
        require_columns(
            dataframe,
            [time_column, value_column],
            where="TimeGroupedStatistics.calculate_statistic",
        )
        if group_by_uuid and "uuid" in dataframe.columns:
            parts = {
                uuid: cls.calculate_statistic(
                    group,
                    time_column,
                    value_column,
                    freq,
                    stat_method,
                    group_by_uuid=False,
                    handle_resets=handle_resets,
                )
                for uuid, group in dataframe.groupby("uuid", sort=True)
            }
            if not parts:
                return cls.calculate_statistic(
                    dataframe,
                    time_column,
                    value_column,
                    freq,
                    stat_method,
                    group_by_uuid=False,
                    handle_resets=handle_resets,
                )
            return pd.concat(parts, names=["uuid"])

        dataframe = _prepared(
            dataframe, time_column, value_column, "calculate_statistic"
        )
        if handle_resets and stat_method == "diff":
            series = dataframe.sort_values(time_column).set_index(time_column)[
                value_column
            ]
            return counter_per_window(series, freq, handle_resets=True).to_frame(
                "difference"
            )

        # Set the DataFrame index to the time column and resample to the specified frequency
        grouped_df = dataframe.set_index(time_column).resample(freq)

        # Select the calculation method
        if stat_method == "mean":
            result = grouped_df[value_column].mean().to_frame("mean")
        elif stat_method == "sum":
            result = grouped_df[value_column].sum().to_frame("sum")
        elif stat_method == "min":
            result = grouped_df[value_column].min().to_frame("min")
        elif stat_method == "max":
            result = grouped_df[value_column].max().to_frame("max")
        elif stat_method == "diff":
            # Improved diff: last value - first value within each interval
            result = (
                grouped_df[value_column].last() - grouped_df[value_column].first()
            ).to_frame("difference")
        elif stat_method == "range":
            # Range: max value - min value within each interval
            result = (
                grouped_df[value_column].max() - grouped_df[value_column].min()
            ).to_frame("range")
        else:
            raise ValueError(
                "Invalid stat_method. Choose from 'mean', 'sum', 'min', 'max', 'diff', 'range'."
            )

        return result

    @classmethod
    def calculate_statistics(
        cls,
        dataframe: pd.DataFrame,
        time_column: str,
        value_column: str,
        freq: str,
        stat_methods: list,
        *,
        group_by_uuid: bool = True,
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate multiple specified statistics on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
            value_column (str): The name of the numeric column to calculate statistics on.
            freq (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
            stat_methods (list): A list of statistical methods to apply (e.g., ['mean', 'sum', 'diff', 'range']).
            group_by_uuid (bool): See :meth:`calculate_statistic`.
            handle_resets (bool): See :meth:`calculate_statistic`.

        Returns:
            pd.DataFrame: A DataFrame with the time intervals and the calculated statistics for each method.
        """
        # One column per statistic, aligned on the (uuid,) time-window index.
        frames = [
            cls.calculate_statistic(
                dataframe,
                time_column,
                value_column,
                freq,
                method,
                group_by_uuid=group_by_uuid,
                handle_resets=handle_resets,
            )
            for method in stat_methods
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    @classmethod
    def calculate_custom_func(
        cls,
        dataframe: pd.DataFrame,
        time_column: str,
        value_column: str,
        freq: str,
        func: Callable[[pd.Series], Any],
        *,
        group_by_uuid: bool = True,
    ) -> pd.DataFrame:
        """
        Apply a custom aggregation function on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
            value_column (str): The name of the numeric column to calculate statistics on.
            freq (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
            func (callable): Custom function to apply to each group.
            group_by_uuid (bool): See :meth:`calculate_statistic`.

        Returns:
            pd.DataFrame: A DataFrame with the custom calculated statistics in
                the column ``custom``, indexed by time window (and ``uuid``
                when grouping by uuid).
        """
        require_columns(
            dataframe,
            [time_column, value_column],
            where="TimeGroupedStatistics.calculate_custom_func",
        )
        if group_by_uuid and "uuid" in dataframe.columns and not dataframe.empty:
            parts = {
                uuid: cls.calculate_custom_func(
                    group,
                    time_column,
                    value_column,
                    freq,
                    func,
                    group_by_uuid=False,
                )
                for uuid, group in dataframe.groupby("uuid", sort=True)
            }
            if parts:
                return pd.concat(parts, names=["uuid"])
        df = _prepared(dataframe, time_column, value_column, "calculate_custom_func")
        grouped_df = df.set_index(time_column).resample(freq)
        result: pd.DataFrame = grouped_df[value_column].apply(func).to_frame("custom")
        return result
