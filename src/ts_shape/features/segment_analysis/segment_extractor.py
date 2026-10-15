import logging
import warnings

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import caller_stacklevel
from ts_shape.utils.intervals import _check_mode

logger = logging.getLogger(__name__)

RANGE_COLUMNS = [
    "segment_value",
    "segment_start",
    "segment_end",
    "segment_duration",
    "segment_index",
    "start",
    "end",
    "duration_seconds",
    "source_uuid",
    "end_inclusive",
]


def _is_blank(values: pd.Series) -> np.ndarray:
    """True for values that are missing or a whitespace-only string."""
    return np.array(
        [pd.isna(v) or (isinstance(v, str) and v.strip() == "") for v in values],
        dtype=bool,
    )


def _add_canonical(result: pd.DataFrame, segment_uuid: str) -> pd.DataFrame:
    """Add the canonical interval columns next to the ``segment_*`` ones."""
    result["start"] = result["segment_start"]
    result["end"] = result["segment_end"]
    result["duration_seconds"] = (
        result["segment_duration"].dt.total_seconds().astype("float64")
    )
    result["source_uuid"] = segment_uuid
    return result[RANGE_COLUMNS]


def _empty_ranges(
    dataframe: pd.DataFrame, time_column: str, value_column: str
) -> pd.DataFrame:
    """Full-schema empty range table, keeping the time and value dtypes."""
    try:
        time_dtype = pd.to_datetime(dataframe[time_column].iloc[:0]).dtype
    except (TypeError, ValueError):
        time_dtype = np.dtype("datetime64[ns]")
    return pd.DataFrame(
        {
            "segment_value": pd.Series(dtype=dataframe[value_column].dtype),
            "segment_start": pd.Series(dtype=time_dtype),
            "segment_end": pd.Series(dtype=time_dtype),
            "segment_duration": pd.Series(dtype="timedelta64[ns]"),
            "segment_index": pd.Series(dtype="int64"),
            "start": pd.Series(dtype=time_dtype),
            "end": pd.Series(dtype=time_dtype),
            "duration_seconds": pd.Series(dtype="float64"),
            "source_uuid": pd.Series(dtype=object),
            "end_inclusive": pd.Series(dtype=bool),
        }
    )


class SegmentExtractor(Base):
    """Extract time ranges from a categorical signal that changes over time.

    Given a signal UUID (e.g. order number or part number) whose value changes
    over time, detects transitions and produces one row per contiguous segment
    with its start time, end time, and active value.

    Methods:
    - extract_time_ranges: Detect value changes and return time range segments.
    """

    @classmethod
    def extract_time_ranges(
        cls,
        dataframe: pd.DataFrame,
        segment_uuid: str,
        uuid_column: str = "uuid",
        value_column: str = "value_string",
        time_column: str = "systime",
        min_duration: str | None = None,
        interval_end: str = "next_change",
        data_end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Detect value transitions and extract time ranges per segment.

        One row per run of consecutive equal values of ``segment_uuid``:
        repeated samples of the same value (a re-sent order number) are one
        segment, and the same value coming back later is a new segment with
        a new ``segment_index``. The signal is ordered by time (rows with the
        same timestamp keep their input order).

        Missing values (NaN / None / ``pd.NA``) mean "no change": they join the
        running segment. Missing values before the first real value, and runs
        whose value is an empty or whitespace-only string ("no order active"),
        are not segments; with ``interval_end="next_change"`` the preceding
        segment ends where such a run starts. Rows without a timestamp are
        skipped with a :class:`~ts_shape.errors.DataQualityWarning`.

        Args:
            dataframe: Input DataFrame in long format. It may hold every
                signal; only the ``segment_uuid`` rows define the segments,
                and (by default) the last timestamp of the whole frame closes
                the last segment.
            segment_uuid: The UUID of the categorical signal to segment by
                (e.g. 'order_number', 'part_number').
            uuid_column: Column identifying each timeseries.
            value_column: Column containing the categorical values.
                Use 'value_string' for string signals, 'value_integer' for
                integer signals, etc. Nullable and pyarrow dtypes work.
            time_column: Column containing timestamps.
            min_duration: Optional minimum segment duration (e.g. '10s', '1min').
                Segments shorter than this are dropped (``segment_index`` is
                renumbered 0..n-1 afterwards).
            interval_end: ``"next_change"`` (default) ends a segment where the
                next value starts (the transition), so change-encoded signals
                -- one row per change -- keep their real duration. The last
                segment is still open at the end of the data and ends at
                ``data_end``. ``"last_sample"`` ends every segment at its own
                last sample, which collapses a single-row segment to zero
                length and leaves the time between segments uncovered.
            data_end: End of the last (open) segment with
                ``interval_end="next_change"``. Default: the latest timestamp
                in ``dataframe`` over all signals, so the process data
                recorded after the last change belongs to the last segment.
                Pass e.g. the end of the query window to close it there. The
                last segment never ends before its own last sample. A naive
                value is read in the data's time zone.

        Returns:
            DataFrame, one row per segment in time order, with columns:
            - segment_value: The active value during this range.
            - segment_start: Start timestamp of the range (first sample).
            - segment_end: End timestamp of the range.
            - segment_duration: Duration as Timedelta.
            - segment_index: Sequential index of the segment (0-based).
            - start, end, duration_seconds: The same range in the canonical
                interval columns (``duration_seconds`` as float).
            - source_uuid: ``segment_uuid``.
            - end_inclusive: Whether the sample exactly at ``segment_end``
                belongs to the segment (used by
                ``SegmentProcessor.apply_ranges``). False when the next run
                starts at that instant; True for the last segment and for
                every ``"last_sample"`` segment.

            Time columns keep the input's time zone. With no segments the
            frame is empty but has all these columns.
        """
        cls._validate_column(dataframe, uuid_column)
        cls._validate_column(dataframe, value_column)
        cls._validate_column(dataframe, time_column)
        _check_mode(interval_end)

        signal = dataframe.loc[
            dataframe[uuid_column] == segment_uuid, [time_column, value_column]
        ]
        if signal.empty:
            logger.warning(f"No data found for UUID '{segment_uuid}'.")
            return _empty_ranges(dataframe, time_column, value_column)

        times = pd.to_datetime(signal[time_column])
        if times.isna().any():
            n_nat = int(times.isna().sum())
            warnings.warn(
                f"{n_nat} row(s) of '{segment_uuid}' have no timestamp and were "
                "skipped.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        keep = times.notna().to_numpy()
        if not keep.any():
            return _empty_ranges(dataframe, time_column, value_column)
        times = times[keep]
        values = signal[value_column][keep]
        # Stable sort: rows with equal timestamps keep their input order.
        order = np.argsort(times.to_numpy(), kind="stable")
        times = times.iloc[order].reset_index(drop=True)
        values = values.iloc[order].reset_index(drop=True)

        # Forward-fill missing values so a NaN row joins the running segment
        # instead of creating spurious one-row groups. Leading NaNs (before
        # any value) stay missing and are skipped.
        filled = values.ffill()
        codes, _ = pd.factorize(filled, use_na_sentinel=True)
        changed = np.empty(len(codes), dtype=bool)
        changed[:1] = True
        changed[1:] = codes[1:] != codes[:-1]
        first = np.flatnonzero(changed)
        last = np.append(first[1:] - 1, len(codes) - 1)

        seg_values = filled.iloc[first].reset_index(drop=True)
        seg_start = times.iloc[first].reset_index(drop=True)
        if interval_end == "next_change":
            seg_end = times.iloc[np.append(first[1:], len(times) - 1)].reset_index(
                drop=True
            )
            # A segment ends where the next run starts, and that instant
            # belongs to the next run; only the last segment (closed at the
            # end of the data) keeps the sample at its end.
            inclusive = np.zeros(len(first), dtype=bool)
            inclusive[-1] = True
            # The last segment is still open at the end of the data: close it
            # at ``data_end`` (default: the last timestamp of the whole frame,
            # so the process data recorded after the last change is covered).
            if data_end is None:
                all_times = pd.to_datetime(dataframe[time_column])
                end_bound = all_times.max()
            else:
                end_bound = pd.Timestamp(data_end)
                if end_bound.tz is None and seg_start.dt.tz is not None:
                    end_bound = end_bound.tz_localize(seg_start.dt.tz)
                elif end_bound.tz is not None and seg_start.dt.tz is not None:
                    end_bound = end_bound.tz_convert(seg_start.dt.tz)
            if pd.notna(end_bound) and end_bound > seg_end.iloc[-1]:
                seg_end.iloc[-1] = end_bound
        else:
            seg_end = times.iloc[last].reset_index(drop=True)
            inclusive = np.ones(len(first), dtype=bool)

        valid = (codes[first] != -1) & ~_is_blank(seg_values)
        result = pd.DataFrame(
            {
                "segment_value": seg_values[valid].reset_index(drop=True),
                "segment_start": seg_start[valid].reset_index(drop=True),
                "segment_end": seg_end[valid].reset_index(drop=True),
                "end_inclusive": inclusive[valid],
            }
        )
        result["segment_duration"] = result["segment_end"] - result["segment_start"]

        if min_duration is not None:
            min_td = pd.Timedelta(min_duration)
            result = result[result["segment_duration"] >= min_td].reset_index(drop=True)

        if result.empty:
            return _empty_ranges(dataframe, time_column, value_column)

        result["segment_index"] = np.arange(len(result), dtype="int64")
        result = _add_canonical(result, segment_uuid)
        logger.info(
            f"Extracted {len(result)} segments from UUID '{segment_uuid}' "
            f"with {result['segment_value'].nunique()} unique values."
        )
        return result
