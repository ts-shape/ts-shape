import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.features.segment_analysis._grouped_stats import summarize_groups
from ts_shape.utils.base import Base

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


def _to_tz(times: pd.Series, tz: Any) -> pd.Series:
    """Express range bounds in the data's time zone.

    Naive bounds on tz-aware data are read as local wall-clock times of the
    data's zone. Aware bounds on naive data raise: the data's zone is unknown.
    """
    times_tz = times.dt.tz
    if tz is None and times_tz is not None:
        raise ValueError(
            "time_ranges are tz-aware but the data's time column is naive; "
            "localize the data (or make the ranges naive) first."
        )
    if tz is not None and times_tz is None:
        return times.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    if tz is not None:
        return times.dt.tz_convert(tz)
    return times


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
        target_uuids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Cut process data into the extracted time ranges.

        Every row whose timestamp lies in a range gets that range's
        ``segment_value`` and ``segment_index``. Ranges are half-open,
        ``segment_start <= t < segment_end``, so a sample exactly at a
        boundary belongs to the range that *starts* there and contiguous
        ranges (``interval_end="next_change"``) cut every row exactly once.
        The row exactly at ``segment_end`` belongs to the range only when the
        range's ``end_inclusive`` is True (``extract_time_ranges`` sets it for
        the last segment, closed at the end of the data, and for every
        ``interval_end="last_sample"`` segment). Without an ``end_inclusive``
        column, the end is inclusive when no other range starts at that
        instant. Rows outside every range (before the first range, in gaps
        such as an empty order number) are dropped; rows without a timestamp
        are dropped.

        Overlapping ranges (only possible with a hand-built ``time_ranges``)
        are applied one by one, so a row in two ranges appears twice; a
        warning says how many overlaps there are.

        Args:
            dataframe: Input DataFrame with process parameter data (all UUIDs).
            time_ranges: Output from SegmentExtractor.extract_time_ranges, or
                any frame with ``segment_start``, ``segment_end``,
                ``segment_value`` and ``segment_index`` columns. Naive range
                bounds are read in the data's time zone and aware bounds are
                converted to it; aware bounds on naive data raise
                ``ValueError``.
            uuid_column: Column identifying each timeseries.
            time_column: Column containing timestamps.
            target_uuids: Optional list of UUIDs to include. None keeps all.

        Returns:
            The matching rows of ``dataframe`` (all columns, time column
            parsed to datetime), ordered by range and then by input order,
            with added columns:
            - segment_value: The active order/part number for each row.
            - segment_index: The sequential segment index.
        """
        cls._validate_column(dataframe, time_column)
        for col in ("segment_start", "segment_end", "segment_value", "segment_index"):
            if not time_ranges.empty or col in time_ranges.columns:
                cls._validate_column(time_ranges, col)

        df = dataframe
        if target_uuids is not None:
            cls._validate_column(dataframe, uuid_column)
            df = df[df[uuid_column].isin(target_uuids)]
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        def _empty() -> pd.DataFrame:
            result = df.iloc[:0].copy()
            result["segment_value"] = pd.Series(
                dtype=(
                    time_ranges["segment_value"].dtype
                    if "segment_value" in time_ranges.columns
                    else "object"
                )
            )
            result["segment_index"] = pd.Series(dtype="int64")
            return result

        if time_ranges.empty:
            logger.warning("No time ranges provided.")
            return _empty()

        tz = df[time_column].dt.tz
        starts = _to_tz(pd.to_datetime(time_ranges["segment_start"]), tz)
        ends = _to_tz(pd.to_datetime(time_ranges["segment_end"]), tz)
        usable = (starts.notna() & ends.notna()).to_numpy()
        if not usable.all():
            logger.warning(
                f"{int((~usable).sum())} time range(s) without start or end "
                "were ignored."
            )
        positions = np.flatnonzero(usable)
        if len(positions) == 0 or df.empty:
            return _empty()

        # One time unit for data and ranges: the int64 comparisons below
        # would otherwise compare microseconds with nanoseconds.
        s_idx = pd.DatetimeIndex(starts.iloc[positions]).as_unit("ns")
        e_idx = pd.DatetimeIndex(ends.iloc[positions]).as_unit("ns")
        # Sort by (start, end): a zero-length range sorts before a range
        # starting at the same instant, which then owns that instant.
        order = np.lexsort((e_idx.asi8, s_idx.asi8))
        s_sorted = s_idx[order]
        e_sorted = e_idx[order]
        pos_sorted = positions[order]
        overlap_count = int((e_sorted[:-1] > s_sorted[1:]).sum())
        if "end_inclusive" in time_ranges.columns:
            closed_end = (
                time_ranges["end_inclusive"]
                .astype("boolean")
                .fillna(False)
                .to_numpy(dtype=bool)[pos_sorted]
            )
        else:
            start_set = set(s_idx.asi8.tolist())
            closed_end = np.array(
                [e not in start_set for e in e_sorted.asi8], dtype=bool
            )

        times = pd.DatetimeIndex(df[time_column]).as_unit("ns")
        has_time = ~np.asarray(times.isna())
        if overlap_count == 0:
            # Non-overlapping ranges: one binary search per row.
            k = s_sorted.searchsorted(times, side="right") - 1
            k_safe = np.clip(k, 0, None)
            t_i8 = times.asi8
            e_i8 = e_sorted.asi8[k_safe]
            inside = (
                has_time
                & (k >= 0)
                & ((t_i8 < e_i8) | ((t_i8 == e_i8) & closed_end[k_safe]))
            )
            row_pos = np.flatnonzero(inside)
            range_pos = pos_sorted[k_safe[row_pos]]
        else:
            logger.warning(
                f"{overlap_count} overlapping time range(s) detected. "
                f"Overlapping rows will appear multiple times in the output."
            )
            row_parts: list[np.ndarray] = []
            range_parts: list[np.ndarray] = []
            t_i8 = times.asi8
            for s_val, e_val, closed, pos in zip(
                s_sorted.asi8, e_sorted.asi8, closed_end, pos_sorted, strict=True
            ):
                hit = has_time & (t_i8 >= s_val) & (t_i8 < e_val)
                if closed:
                    hit |= has_time & (t_i8 == e_val)
                rows = np.flatnonzero(hit)
                row_parts.append(rows)
                range_parts.append(np.full(len(rows), pos, dtype=np.int64))
            row_pos = np.concatenate(row_parts)
            range_pos = np.concatenate(range_parts)

        if len(row_pos) == 0:
            logger.warning("No data matched any time range.")
            return _empty()

        # Output order: by range (in time_ranges order), then input order.
        out_order = np.lexsort((row_pos, range_pos))
        row_pos = row_pos[out_order]
        range_pos = range_pos[out_order]

        result = df.iloc[row_pos].reset_index(drop=True)
        result["segment_value"] = (
            time_ranges["segment_value"].iloc[range_pos].reset_index(drop=True)
        )
        result["segment_index"] = (
            time_ranges["segment_index"].iloc[range_pos].reset_index(drop=True)
        )
        if uuid_column in result.columns:
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
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute statistical metrics per UUID per segment.

        Typically called on the output of apply_ranges. Computes the
        statistics of ``NumericStatistics.summary_as_dict`` per (UUID, segment)
        pair, for all pairs at once. Missing values are ignored; pairs with
        fewer than 2 non-missing values are left out.

        Args:
            dataframe: Input DataFrame (output of apply_ranges or similar).
            uuid_column: Column identifying each timeseries.
            value_column: Column containing numeric values.
            group_column: Column to group segments by (e.g. 'segment_value',
                'segment_index'). Use 'segment_value' to aggregate all ranges
                of the same order, or 'segment_index' for individual ranges.
            metrics: Subset of metric names to compute. None uses all 19.

        Returns:
            DataFrame with columns [uuid, <group_column>, sample_count,
            metric_1, ...], one row per pair, sorted by uuid and group.
            ``sample_count`` is the number of non-missing values. Empty (with
            these columns) when no pair has 2 values.
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

        wanted = metrics if metrics is not None else ALL_METRICS
        metric_cols = [m for m in ALL_METRICS if m in wanted]
        columns = [uuid_column, group_column, "sample_count", *metric_cols]
        empty = pd.DataFrame(
            {
                uuid_column: pd.Series(dtype=object),
                group_column: pd.Series(dtype=dataframe[group_column].dtype),
                "sample_count": pd.Series(dtype="int64"),
                **{m: pd.Series(dtype="float64") for m in metric_cols},
            }
        )
        if dataframe.empty:
            return empty
        result = summarize_groups(dataframe, [uuid_column, group_column], value_column)
        if result.empty:
            return empty
        result.columns = [uuid_column, group_column, *result.columns[2:]]
        return result[columns]
