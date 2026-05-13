import logging
import pandas as pd  # type: ignore
from typing import Optional, List

from ts_shape.utils.base import Base
from ts_shape.features.stats.numeric_stats import NumericStatistics

logger = logging.getLogger(__name__)

ALL_METRICS = [
    "min",
    "max",
    "mean",
    "median",
    "std",
    "var",
    "sum",
    "kurtosis",
    "skewness",
    "q1",
    "q3",
    "iqr",
    "range",
    "mad",
    "coeff_var",
    "sem",
    "mode",
    "percentile_90",
    "percentile_10",
]


class SegmentProcessor(Base):
    """Apply extracted time ranges to process data and compute metric profiles.

    Takes the output of SegmentExtractor.extract_time_ranges and uses it to
    filter and annotate process parameter data, then computes statistical
    metrics per UUID per segment.

    Methods:
    - apply_ranges: Filter process data by time ranges, annotate with segment info.
    - compute_metric_profiles: Compute statistical metrics per UUID per segment.
    """

    @classmethod
    def apply_ranges(
        cls,
        dataframe: pd.DataFrame,
        time_ranges: pd.DataFrame,
        uuid_column: str = "uuid",
        time_column: str = "systime",
        target_uuids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Filter process parameter data by extracted time ranges.

        For each time range (segment), selects rows from the main DataFrame
        that fall within [segment_start, segment_end] and annotates them with
        the segment value and index.

        Args:
            dataframe: Input DataFrame with process parameter data (all UUIDs).
            time_ranges: Output from SegmentExtractor.extract_time_ranges.
            uuid_column: Column identifying each timeseries.
            time_column: Column containing timestamps.
            target_uuids: Optional list of UUIDs to include. None keeps all.

        Returns:
            Input DataFrame filtered to the time ranges, with added columns:
            - segment_value: The active order/part number for each row.
            - segment_index: The sequential segment index.
        """
        cls._validate_column(dataframe, time_column)

        if time_ranges.empty:
            logger.warning("No time ranges provided.")
            result = dataframe.iloc[:0].copy()
            result["segment_value"] = pd.Series(dtype="object")
            result["segment_index"] = pd.Series(dtype="int64")
            return result

        df = dataframe.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        if target_uuids is not None:
            df = df[df[uuid_column].isin(target_uuids)]

        # Warn if time ranges overlap
        sorted_ranges = time_ranges.sort_values("segment_start")
        ends = sorted_ranges["segment_end"].values[:-1]
        starts = sorted_ranges["segment_start"].values[1:]
        if len(ends) > 0 and (ends > starts).any():
            overlap_count = int((ends > starts).sum())
            logger.warning(
                f"{overlap_count} overlapping time range(s) detected. "
                f"Overlapping rows will appear multiple times in the output."
            )

        segments = []
        for _, seg in time_ranges.iterrows():
            mask = (df[time_column] >= seg["segment_start"]) & (
                df[time_column] <= seg["segment_end"]
            )
            matched = df[mask].copy()
            matched["segment_value"] = seg["segment_value"]
            matched["segment_index"] = seg["segment_index"]
            segments.append(matched)

        if not segments:
            logger.warning("No data matched any time range.")
            result = dataframe.iloc[:0].copy()
            result["segment_value"] = pd.Series(dtype="object")
            result["segment_index"] = pd.Series(dtype="int64")
            return result

        result = pd.concat(segments, ignore_index=True)
        logger.info(
            f"Applied {len(time_ranges)} ranges: {len(result)} rows across "
            f"{result[uuid_column].nunique()} UUIDs."
        )
        return result

    @classmethod
    def compute_metric_profiles(
        cls,
        dataframe: pd.DataFrame,
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        group_column: str = "segment_value",
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute statistical metrics per UUID per segment.

        Typically called on the output of apply_ranges. Computes metrics per
        (UUID, segment) pair using NumericStatistics.

        Args:
            dataframe: Input DataFrame (output of apply_ranges or similar).
            uuid_column: Column identifying each timeseries.
            value_column: Column containing numeric values.
            group_column: Column to group segments by (e.g. 'segment_value',
                'segment_index'). Use 'segment_value' to aggregate all ranges
                of the same order, or 'segment_index' for individual ranges.
            metrics: Subset of metric names to compute. None uses all 19.

        Returns:
            DataFrame with columns [uuid, <group_column>, sample_count, metric_1, ...].
        """
        cls._validate_column(dataframe, uuid_column)
        cls._validate_column(dataframe, value_column)
        cls._validate_column(dataframe, group_column)

        if metrics is not None:
            invalid = set(metrics) - set(ALL_METRICS)
            if invalid:
                raise ValueError(
                    f"Unknown metrics: {invalid}. Available: {ALL_METRICS}"
                )

        rows = []
        for (uuid_val, group_val), group in dataframe.groupby(
            [uuid_column, group_column]
        ):
            numeric_data = group[value_column].dropna()
            if len(numeric_data) < 2:
                continue

            stats = NumericStatistics.summary_as_dict(group, value_column)
            if metrics is not None:
                stats = {k: v for k, v in stats.items() if k in metrics}
            stats[uuid_column] = uuid_val
            stats[group_column] = group_val
            stats["sample_count"] = len(numeric_data)
            rows.append(stats)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        leading = [uuid_column, group_column, "sample_count"]
        metric_cols = [c for c in result.columns if c not in leading]
        return result[leading + metric_cols].reset_index(drop=True)
