import logging
import pandas as pd  # type: ignore
from typing import Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


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
        min_duration: Optional[str] = None,
    ) -> pd.DataFrame:
        """Detect value transitions and extract time ranges per segment.

        Args:
            dataframe: Input DataFrame in long format.
            segment_uuid: The UUID of the categorical signal to segment by
                (e.g. 'order_number', 'part_number').
            uuid_column: Column identifying each timeseries.
            value_column: Column containing the categorical values.
                Use 'value_string' for string signals, 'value_integer' for
                integer signals, etc.
            time_column: Column containing timestamps.
            min_duration: Optional minimum segment duration (e.g. '10s', '1min').
                Segments shorter than this are dropped.

        Returns:
            DataFrame with columns:
            - segment_value: The active value during this range.
            - segment_start: Start timestamp of the range.
            - segment_end: End timestamp of the range.
            - segment_duration: Duration as Timedelta.
            - segment_index: Sequential index of the segment.
        """
        cls._validate_column(dataframe, uuid_column)
        cls._validate_column(dataframe, value_column)
        cls._validate_column(dataframe, time_column)

        empty_result = pd.DataFrame(
            columns=[
                "segment_value",
                "segment_start",
                "segment_end",
                "segment_duration",
                "segment_index",
            ]
        )

        # Filter to the segment signal only
        signal = dataframe[dataframe[uuid_column] == segment_uuid].copy()
        if signal.empty:
            logger.warning(f"No data found for UUID '{segment_uuid}'.")
            return empty_result

        signal = signal.sort_values(time_column).reset_index(drop=True)
        signal[time_column] = pd.to_datetime(signal[time_column])

        # Detect value changes — forward-fill NaN so NaN rows join the
        # adjacent segment instead of creating spurious one-row groups.
        values = signal[value_column].ffill()
        changed = values.ne(values.shift())
        group_ids = changed.cumsum()

        rows = []
        for _, group in signal.groupby(group_ids):
            seg_value = group[value_column].iloc[0]
            if pd.isna(seg_value) or (
                isinstance(seg_value, str) and seg_value.strip() == ""
            ):
                continue
            seg_start = group[time_column].iloc[0]
            seg_end = group[time_column].iloc[-1]
            rows.append(
                {
                    "segment_value": seg_value,
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "segment_duration": seg_end - seg_start,
                    "segment_index": len(rows),
                }
            )

        if not rows:
            return empty_result

        result = pd.DataFrame(rows)

        if min_duration is not None:
            min_td = pd.Timedelta(min_duration)
            result = result[result["segment_duration"] >= min_td].reset_index(drop=True)
            result["segment_index"] = range(len(result))

        logger.info(
            f"Extracted {len(result)} segments from UUID '{segment_uuid}' "
            f"with {result['segment_value'].nunique()} unique values."
        )
        return result
