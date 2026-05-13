import logging
import pandas as pd  # type: ignore
from typing import Optional, List

from ts_shape.utils.base import Base
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.features.segment_analysis.segment_processor import ALL_METRICS

logger = logging.getLogger(__name__)


class TimeWindowedFeatureTable(Base):
    """Build ML-ready feature tables from segmented timeseries data.

    Takes the output of SegmentProcessor.apply_ranges and computes statistical
    metrics per UUID within fixed-size time windows (e.g. every 1 minute).

    Methods:
    - compute_long: One row per (time_window, uuid, segment). Long format.
    - compute: One row per time_window with columns {uuid}__{metric}. Wide format.
    """

    @classmethod
    def compute_long(
        cls,
        dataframe: pd.DataFrame,
        freq: str = "1min",
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        segment_column: Optional[str] = "segment_value",
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute statistical metrics per UUID per time window (long format).

        Groups the input data into fixed-size time windows using ``freq`` and
        computes numeric statistics for each (time_window, uuid, segment) group.

        Args:
            dataframe: Input DataFrame (typically output of SegmentProcessor.apply_ranges).
            freq: Pandas frequency string for the time window size (e.g. '1min', '30s', '5min').
            time_column: Column containing timestamps.
            uuid_column: Column identifying each timeseries signal.
            value_column: Column containing numeric values.
            segment_column: Column to sub-group by (e.g. 'segment_value').
                Set to None to ignore segments entirely.
            metrics: Subset of metric names to compute. None uses all 19.

        Returns:
            DataFrame with columns [time_window, uuid, segment_value?, sample_count, metric_1, ...].
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

        if dataframe.empty:
            logger.warning("Empty DataFrame provided.")
            return pd.DataFrame()

        df = dataframe.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        grouper = [pd.Grouper(key=time_column, freq=freq), uuid_column]
        if segment_column is not None:
            grouper.append(segment_column)

        rows = []
        for group_key, group in df.groupby(grouper):
            numeric_data = group[value_column].dropna()
            if len(numeric_data) < 2:
                continue

            stats = NumericStatistics.summary_as_dict(group, value_column)
            if metrics is not None:
                stats = {k: v for k, v in stats.items() if k in metrics}

            if segment_column is not None:
                window_ts, uuid_val, seg_val = group_key
                stats[segment_column] = seg_val
            else:
                window_ts, uuid_val = group_key

            stats["time_window"] = window_ts
            stats[uuid_column] = uuid_val
            stats["sample_count"] = len(numeric_data)
            rows.append(stats)

        if not rows:
            logger.warning("No windows with sufficient data (>= 2 non-NaN values).")
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        leading = ["time_window", uuid_column]
        if segment_column is not None:
            leading.append(segment_column)
        leading.append("sample_count")
        metric_cols = [c for c in result.columns if c not in leading]
        result = result[leading + metric_cols].reset_index(drop=True)

        n_windows = result["time_window"].nunique()
        n_uuids = result[uuid_column].nunique()
        logger.info(
            f"Computed {len(result)} feature rows across "
            f"{n_windows} windows and {n_uuids} UUIDs."
        )
        return result

    @classmethod
    def compute(
        cls,
        dataframe: pd.DataFrame,
        freq: str = "1min",
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        segment_column: Optional[str] = "segment_value",
        metrics: Optional[List[str]] = None,
        column_separator: str = "__",
    ) -> pd.DataFrame:
        """Compute a wide-format feature table with one row per time window.

        Each column is named ``{uuid}{separator}{metric}`` (e.g.
        ``temperature__mean``).  Windows where a UUID has insufficient data
        are filled with NaN.

        Args:
            dataframe: Input DataFrame (typically output of SegmentProcessor.apply_ranges).
            freq: Pandas frequency string for the time window size.
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
            freq=freq,
            time_column=time_column,
            uuid_column=uuid_column,
            value_column=value_column,
            segment_column=segment_column,
            metrics=metrics,
        )

        if long_df.empty:
            return pd.DataFrame()

        index_cols = ["time_window"]
        if segment_column is not None and segment_column in long_df.columns:
            index_cols.append(segment_column)

        # Identify metric columns to pivot
        value_cols = [c for c in long_df.columns if c not in index_cols + [uuid_column]]

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
