import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.counters import counter_per_window
from ts_shape.utils.deprecation import renamed_kwargs
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
        values = to_numeric_checked(values, what=f"column {value_column!r}")
        df[value_column] = values.astype("float64")
    return df


def _flatten(result: pd.DataFrame, *, time_column: str, has_uuid: bool) -> pd.DataFrame:
    """Turn the MultiIndex/Index resample result into a long-format frame.

    ``result`` is indexed either by ``(uuid, time_window)`` (grouped) or by
    ``time_window`` (pooled). The returned frame moves those levels into
    columns, naming the time level ``window_start`` to match the rest of the
    library's canonical time-column vocabulary (see
    ``tests/invariants/test_contract_transform_features.TIME_COLUMNS``) and
    preserves ``uuid`` when the input had it. The stat columns keep their
    names and dtypes.
    """
    if result.empty:
        columns: list[str] = []
        if has_uuid:
            columns.append("uuid")
        columns.append("window_start")
        columns.extend(str(c) for c in result.columns)
        out = pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
        if has_uuid:
            out["uuid"] = out["uuid"].astype(object)
        out["window_start"] = pd.to_datetime(out["window_start"])
        return out
    index = result.index
    if has_uuid and isinstance(index, pd.MultiIndex):
        flat = result.reset_index()
        rename = {}
        if "level_0" in flat.columns and "uuid" not in flat.columns:
            rename["level_0"] = "uuid"
        # The time level is the inner one; pandas names it after the time
        # column (``systime``) or leaves it as ``level_1`` when unnamed.
        time_level_name = index.names[-1]
        time_source = (
            time_level_name
            if time_level_name in flat.columns
            else ("level_1" if "level_1" in flat.columns else time_column)
        )
        if time_source in flat.columns:
            rename[time_source] = "window_start"
        if rename:
            flat = flat.rename(columns=rename)
        cols = [
            "uuid",
            "window_start",
            *[c for c in flat.columns if c not in {"uuid", "window_start"}],
        ]
        return flat[cols]
    flat = result.reset_index()
    time_source = (
        index.name
        if index.name and index.name in flat.columns
        else ("index" if "index" in flat.columns else time_column)
    )
    if time_source in flat.columns:
        flat = flat.rename(columns={time_source: "window_start"})
    cols = ["window_start", *[c for c in flat.columns if c != "window_start"]]
    return flat[cols]


class TimeGroupedStatistics(Base):
    """
    A class for calculating time-grouped statistics on numeric data, with class methods to apply various statistical functions.
    """

    @classmethod
    @renamed_kwargs(freq="window")
    def calculate_statistic(
        cls,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
        value_column: str = "value_double",
        window: str = "1h",
        stat_method: str = "mean",
        *,
        group_by_uuid: bool = True,
        handle_resets: bool = True,
        flat: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate a specified statistic on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
                Default ``"systime"``.
            value_column (str): The name of the numeric column to calculate statistics on.
                Default ``"value_double"``.
            window (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
                Default ``"1h"``. ``freq`` is the deprecated name.
            stat_method (str): The statistical method to apply ('mean', 'sum', 'min', 'max', 'diff', 'range').
                Default ``'mean'``.
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
            flat (bool): When ``True`` return a long-format frame with
                ``window_start`` (and ``uuid`` when grouping by uuid) as
                plain columns and no index of meaning, so the result composes
                with the rest of the library's long-format API. Default
                ``False`` keeps the historical MultiIndex/DatetimeIndex shape.

        Returns:
            pd.DataFrame: A DataFrame with the time intervals and the calculated statistics.
        """
        require_columns(
            dataframe,
            [time_column, value_column],
            where="TimeGroupedStatistics.calculate_statistic",
        )
        has_uuid = group_by_uuid and "uuid" in dataframe.columns
        if has_uuid:
            parts = {
                uuid: cls.calculate_statistic(
                    group,
                    time_column,
                    value_column,
                    window,
                    stat_method,
                    group_by_uuid=False,
                    handle_resets=handle_resets,
                    flat=False,
                )
                for uuid, group in dataframe.groupby("uuid", sort=True)
            }
            if not parts:
                nested = cls.calculate_statistic(
                    dataframe,
                    time_column,
                    value_column,
                    window,
                    stat_method,
                    group_by_uuid=False,
                    handle_resets=handle_resets,
                    flat=False,
                )
                return (
                    _flatten(nested, time_column=time_column, has_uuid=False)
                    if flat
                    else nested
                )
            combined = pd.concat(parts, names=["uuid"])
            return (
                _flatten(combined, time_column=time_column, has_uuid=True)
                if flat
                else combined
            )

        dataframe = _prepared(
            dataframe, time_column, value_column, "calculate_statistic"
        )
        if handle_resets and stat_method == "diff":
            series = dataframe.sort_values(time_column).set_index(time_column)[
                value_column
            ]
            result = counter_per_window(series, window, handle_resets=True).to_frame(
                "difference"
            )
            return (
                _flatten(result, time_column=time_column, has_uuid=False)
                if flat
                else result
            )

        # Set the DataFrame index to the time column and resample to the specified frequency
        grouped_df = dataframe.set_index(time_column).resample(window)

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

        return (
            _flatten(result, time_column=time_column, has_uuid=False)
            if flat
            else result
        )

    @classmethod
    @renamed_kwargs(freq="window")
    def calculate_statistics(
        cls,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
        value_column: str = "value_double",
        window: str = "1h",
        stat_methods: list[str] | None = None,
        *,
        group_by_uuid: bool = True,
        handle_resets: bool = True,
        flat: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate multiple specified statistics on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
                Default ``"systime"``.
            value_column (str): The name of the numeric column to calculate statistics on.
                Default ``"value_double"``.
            window (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
                Default ``"1h"``. ``freq`` is the deprecated name.
            stat_methods (list): A list of statistical methods to apply (e.g., ['mean', 'sum', 'diff', 'range']).
                Default ``['mean', 'min', 'max']``.
            group_by_uuid (bool): See :meth:`calculate_statistic`.
            handle_resets (bool): See :meth:`calculate_statistic`.
            flat (bool): See :meth:`calculate_statistic`.

        Returns:
            pd.DataFrame: A DataFrame with the time intervals and the calculated statistics for each method.
        """
        if stat_methods is None:
            stat_methods = ["mean", "min", "max"]
        # Compute each statistic in the historical indexed shape, concat on
        # the shared (uuid,) time index, and flatten once at the end so the
        # stat columns line up by position regardless of ``flat``.
        frames = [
            cls.calculate_statistic(
                dataframe,
                time_column,
                value_column,
                window,
                method,
                group_by_uuid=group_by_uuid,
                handle_resets=handle_resets,
                flat=False,
            )
            for method in stat_methods
        ]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, axis=1)
        if not flat:
            return combined
        has_uuid = group_by_uuid and "uuid" in dataframe.columns
        return _flatten(combined, time_column=time_column, has_uuid=has_uuid)

    @classmethod
    @renamed_kwargs(freq="window")
    def calculate_custom_func(
        cls,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
        value_column: str = "value_double",
        window: str = "1h",
        func: Callable[[pd.Series], Any] | None = None,
        *,
        group_by_uuid: bool = True,
        flat: bool = False,
    ) -> pd.DataFrame:
        """
        Apply a custom aggregation function on the value column over the grouped time intervals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            time_column (str): The name of the time column to group and sort by.
                Default ``"systime"``.
            value_column (str): The name of the numeric column to calculate statistics on.
                Default ``"value_double"``.
            window (str): Frequency string for time grouping (e.g., 'h' for hourly, 'D' for daily; pandas 3 rejects 'H').
                Default ``"1h"``. ``freq`` is the deprecated name.
            func (callable): Custom function to apply to each group (required).
            group_by_uuid (bool): See :meth:`calculate_statistic`.
            flat (bool): See :meth:`calculate_statistic`.

        Returns:
            pd.DataFrame: A DataFrame with the custom calculated statistics in
                the column ``custom``, indexed by time window (and ``uuid``
                when grouping by uuid).
        """
        if func is None:
            raise TypeError("calculate_custom_func() needs func=<callable>")
        require_columns(
            dataframe,
            [time_column, value_column],
            where="TimeGroupedStatistics.calculate_custom_func",
        )
        has_uuid = group_by_uuid and "uuid" in dataframe.columns
        if has_uuid and not dataframe.empty:
            parts = {
                uuid: cls.calculate_custom_func(
                    group,
                    time_column,
                    value_column,
                    window,
                    func,
                    group_by_uuid=False,
                    flat=False,
                )
                for uuid, group in dataframe.groupby("uuid", sort=True)
            }
            if parts:
                combined = pd.concat(parts, names=["uuid"])
                return (
                    _flatten(combined, time_column=time_column, has_uuid=True)
                    if flat
                    else combined
                )
        df = _prepared(dataframe, time_column, value_column, "calculate_custom_func")
        grouped_df = df.set_index(time_column).resample(window)
        result: pd.DataFrame = grouped_df[value_column].apply(func).to_frame("custom")
        return (
            _flatten(result, time_column=time_column, has_uuid=False)
            if flat
            else result
        )
