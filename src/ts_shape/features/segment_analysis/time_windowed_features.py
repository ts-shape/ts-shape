import logging

import numpy as np
import pandas as pd

from ts_shape.features.segment_analysis._grouped_stats import summarize_groups
from ts_shape.features.segment_analysis.segment_processor import ALL_METRICS
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.frames import as_datetime, window_offset

logger = logging.getLogger(__name__)

#: Canonical window columns added after the metrics by ``compute_long``.
_WINDOW_COLUMNS = ["start", "end", "duration_seconds"]


def _typed_empty(
    columns: list[str], time_dtype: object, keys: list[str]
) -> pd.DataFrame:
    """Empty result with the column dtypes of a real result."""
    data: dict[str, pd.Series] = {}
    for col in columns:
        if col in ("time_window", "start", "end"):
            data[col] = pd.Series(dtype=time_dtype)
        elif col in keys:
            data[col] = pd.Series(dtype=object)
        elif col == "sample_count":
            data[col] = pd.Series(dtype="int64")
        else:
            data[col] = pd.Series(dtype="float64")
    return pd.DataFrame(data)


class TimeWindowedFeatureTable(Base):
    """Build ML-ready feature tables from segmented timeseries data.

    Takes the output of SegmentProcessor.apply_ranges and computes statistical
    metrics per UUID within fixed-size time windows (e.g. every 1 minute).

    Methods:
    - compute_long: One row per (time_window, uuid, segment). Long format.
    - compute: One row per time_window with columns {uuid}__{metric}. Wide format.
    """

    @classmethod
    @renamed_kwargs(freq="window")
    def compute_long(
        cls,
        dataframe: pd.DataFrame,
        window: str = "1min",
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        segment_column: str | None = "segment_value",
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute statistical metrics per UUID per time window (long format).

        Groups the input data into fixed-size time windows of ``window`` and
        computes numeric statistics for each (time_window, uuid, segment) group.

        Args:
            dataframe: Input DataFrame (typically output of SegmentProcessor.apply_ranges).
            window: Pandas frequency string for the time window size (e.g.
                '1min', '30s', '5min'). ``freq`` is the deprecated name.
            time_column: Column containing timestamps.
            uuid_column: Column identifying each timeseries signal.
            value_column: Column containing numeric values.
            segment_column: Column to sub-group by (e.g. 'segment_value').
                Set to None to ignore segments entirely.
            metrics: Subset of metric names to compute. None uses all 19.

        Windows are the bins of ``pd.Grouper(freq=window)`` (for sub-daily
        frequencies they start at midnight of the first day; on tz-aware data
        they are wall-clock windows in the data's zone). A window that spans a
        segment change gives one row per segment. Groups with fewer than 2
        non-missing values are left out.

        Returns:
            DataFrame with columns [time_window, uuid, segment_value?,
            sample_count, metric_1, ..., start, end, duration_seconds], one row
            per (window, uuid, segment) sorted in that order. ``start`` is the
            window start (= ``time_window``), ``end`` the next window start.
            Empty (with these columns) when no group has 2 values.
        """
        cls._validate_column(dataframe, time_column)
        cls._validate_column(dataframe, uuid_column)
        cls._validate_column(dataframe, value_column)
        if segment_column is not None:
            cls._validate_column(dataframe, segment_column)

        if metrics is not None:
            invalid = set(metrics) - set(ALL_METRICS)
            if invalid:
                raise ValueError(
                    f"Unknown metrics: {invalid}. Available: {ALL_METRICS}"
                )

        leading = ["time_window", uuid_column]
        if segment_column is not None:
            leading.append(segment_column)
        wanted = metrics if metrics is not None else ALL_METRICS
        metric_cols = [m for m in ALL_METRICS if m in wanted]
        columns = [*leading, "sample_count", *metric_cols, *_WINDOW_COLUMNS]

        df = dataframe.copy()
        df[time_column] = as_datetime(
            df[time_column], where="TimeWindowedFeatureTable.compute_long"
        )
        keys = leading[1:]
        empty = _typed_empty(columns, df[time_column].dtype, keys)
        if dataframe.empty:
            logger.warning("Empty DataFrame provided.")
            return empty
        # Stable time order: the window grouper then labels rows reliably.
        df = df.iloc[np.argsort(df[time_column].to_numpy(), kind="stable")]

        grouper: list[object] = [pd.Grouper(key=time_column, freq=window), uuid_column]
        if segment_column is not None:
            grouper.append(segment_column)

        result = summarize_groups(df, grouper, value_column)
        if result.empty:
            logger.warning("No windows with sufficient data (>= 2 non-NaN values).")
            return empty
        result.columns = [*leading, *result.columns[len(leading) :]]
        result["start"] = result["time_window"]
        result["end"] = result["time_window"] + window_offset(window)
        result["duration_seconds"] = (
            (result["end"] - result["start"]).dt.total_seconds().astype("float64")
        )
        result = result[columns]

        n_windows = result["time_window"].nunique()
        n_uuids = result[uuid_column].nunique()
        logger.info(
            f"Computed {len(result)} feature rows across "
            f"{n_windows} windows and {n_uuids} UUIDs."
        )
        return result

    @classmethod
    @renamed_kwargs(freq="window")
    def compute(
        cls,
        dataframe: pd.DataFrame,
        window: str = "1min",
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        segment_column: str | None = "segment_value",
        metrics: list[str] | None = None,
        column_separator: str = "__",
    ) -> pd.DataFrame:
        """Compute a wide-format feature table with one row per time window.

        Each column is named ``{uuid}{separator}{metric}`` (e.g.
        ``temperature__mean``).  Windows where a UUID has insufficient data
        are filled with NaN.

        Args:
            dataframe: Input DataFrame (typically output of SegmentProcessor.apply_ranges).
            window: Pandas frequency string for the time window size.
                ``freq`` is the deprecated name.
            time_column: Column containing timestamps.
            uuid_column: Column identifying each timeseries signal.
            value_column: Column containing numeric values.
            segment_column: Column to sub-group by. Set to None to ignore segments.
            metrics: Subset of metric names to compute. None uses all 19.
            column_separator: Separator between uuid and metric in wide column names.

        Returns:
            DataFrame with columns [time_window, segment_value?, {uuid}__{metric}, ...].
        """
        long_df = cls.compute_long(
            dataframe,
            window=window,
            time_column=time_column,
            uuid_column=uuid_column,
            value_column=value_column,
            segment_column=segment_column,
            metrics=metrics,
        )

        index_cols = ["time_window"]
        if segment_column is not None and segment_column in long_df.columns:
            index_cols.append(segment_column)

        if long_df.empty:
            return long_df[index_cols].reset_index(drop=True)

        # Identify metric columns to pivot (window bounds are not metrics)
        value_cols = [
            c
            for c in long_df.columns
            if c not in index_cols + [uuid_column] + _WINDOW_COLUMNS
        ]

        pieces = []
        for col in value_cols:
            pivoted = long_df.pivot_table(
                index=index_cols,
                columns=uuid_column,
                values=col,
                aggfunc="first",
            )
            pivoted.columns = [
                f"{uuid_val}{column_separator}{col}" for uuid_val in pivoted.columns
            ]
            pieces.append(pivoted)

        result = pd.concat(pieces, axis=1).reset_index()

        # Sort columns: index cols first, then alphabetical by uuid, ALL_METRICS order within
        used_metrics = metrics if metrics is not None else ALL_METRICS
        metric_order = list(used_metrics) + ["sample_count"]
        uuids_sorted = sorted(long_df[uuid_column].unique())

        ordered_cols = list(index_cols)
        for uuid_val in uuids_sorted:
            for metric in metric_order:
                col_name = f"{uuid_val}{column_separator}{metric}"
                if col_name in result.columns:
                    ordered_cols.append(col_name)

        result = result[ordered_cols]

        logger.info(
            f"Wide feature table: {len(result)} rows x {len(result.columns)} columns."
        )
        return result
