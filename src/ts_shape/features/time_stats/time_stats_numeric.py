import logging

import pandas as pd  # type: ignore

from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_per_window

logger = logging.getLogger(__name__)


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
        group_by_uuid: bool = False,
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate a specified statistic on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
            value_column (str): The name of the numeric column to calculate statistics on.
            freq (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
            stat_method (str): The statistical method to apply ('mean', 'sum', 'min', 'max', 'diff', 'range').
            group_by_uuid (bool): Compute the statistic per ``uuid`` (result
                indexed by ``uuid`` and time). Default ``False`` pools every
                row, which mixes signals when the frame holds several uuids.
            handle_resets (bool): For ``'diff'`` on a cumulative counter, sum
                reset-aware increments per interval instead of
                ``last - first`` (which goes negative across a reset). See
                :func:`ts_shape.utils.counters.counter_per_window`.

        Returns:
            pd.DataFrame: A DataFrame with the time intervals and the calculated statistics.
        """
        if group_by_uuid:
            parts = {
                uuid: cls.calculate_statistic(
                    group,
                    time_column,
                    value_column,
                    freq,
                    stat_method,
                    handle_resets=handle_resets,
                )
                for uuid, group in dataframe.groupby("uuid", sort=True)
            }
            if not parts:
                return cls.calculate_statistic(
                    dataframe, time_column, value_column, freq, stat_method
                )
            return pd.concat(parts, names=["uuid"])

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
        group_by_uuid: bool = False,
        handle_resets: bool = False,
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
        # Initialize an empty DataFrame for combining results
        result_df = pd.DataFrame()

        # Calculate each requested statistic and join to the result DataFrame
        for method in stat_methods:
            stat_df = cls.calculate_statistic(
                dataframe,
                time_column,
                value_column,
                freq,
                method,
                group_by_uuid=group_by_uuid,
                handle_resets=handle_resets,
            )
            result_df = result_df.join(stat_df, how="outer")

        return result_df

    @classmethod
    def calculate_custom_func(
        cls,
        dataframe: pd.DataFrame,
        time_column: str,
        value_column: str,
        freq: str,
        func,
    ) -> pd.DataFrame:
        """
        Apply a custom aggregation function on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
            value_column (str): The name of the numeric column to calculate statistics on.
            freq (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
            func (callable): Custom function to apply to each group.

        Returns:
            pd.DataFrame: A DataFrame with the custom calculated statistics.
        """
        grouped_df = dataframe.set_index(time_column).resample(freq)
        result = grouped_df[value_column].apply(func).to_frame("custom")
        return result
