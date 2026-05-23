import logging
import pandas as pd  # type: ignore
from typing import List, Dict, Any, Optional, Tuple

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class LongDowntimeEvents(Base):
    """Production: Long Downtime Detection

    Detect downtime intervals exceeding a minimum duration threshold from a
    boolean or integer state signal, then count production events that occur
    between consecutive long-downtime boundaries.

    Classes:
    - LongDowntimeEvents: Long idle intervals and inter-gap production counts.
      - detect_long_downtime: Find idle/stopped intervals >= min_gap (default 3 h).
      - count_events_between_gaps: Count production events between consecutive
        long-downtime boundaries.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        state_uuid: str,
        *,
        event_uuid: str = "prod:long_downtime",
        value_column: str = "value_bool",
        time_column: str = "systime",
        value_range: tuple[float | None, float | None] | None = None,
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.state_uuid = state_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.value_range = value_range
        if (
            "uuid" in self.dataframe.columns
            and not self.dataframe.empty
            and state_uuid not in self.dataframe["uuid"].values
        ):
            raise ValueError(
                f"UUID '{state_uuid}' not found in dataframe. "
                f"Available UUIDs: {list(self.dataframe['uuid'].unique())}"
            )
        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    def _as_state(self, col: pd.Series) -> pd.Series:
        """Convert a value series to a boolean running/idle state.

        Uses value_range when set (inclusive on both ends); otherwise casts to bool.
        True = running, False = down/idle.
        """
        if self.value_range is not None:
            lower, upper = self.value_range
            mask = pd.Series(True, index=col.index)
            if lower is not None:
                mask &= col >= lower
            if upper is not None:
                mask &= col <= upper
            return mask
        return col.fillna(False).astype(bool)

    def detect_long_downtime(self, min_gap: str = "3h") -> pd.DataFrame:
        """Return idle/stopped intervals whose duration is >= min_gap.

        Columns: start, end, duration_seconds, downtime_index, uuid, source_uuid, is_delta
        """
        empty = pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration_seconds",
                "downtime_index",
                "uuid",
                "source_uuid",
                "is_delta",
            ]
        )
        if self.series.empty:
            return empty

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = self._as_state(s[self.value_column])
        state_change = (s["state"] != s["state"].shift()).cumsum()
        min_td = pd.to_timedelta(min_gap)

        rows: list[dict[str, Any]] = []
        for _, seg in s.groupby(state_change):
            if bool(seg["state"].iloc[0]):
                continue  # running segment — skip
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            duration = end - start
            if duration < min_td:
                continue
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "duration_seconds": duration.total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": self.state_uuid,
                    "is_delta": True,
                }
            )

        if not rows:
            return empty

        result = pd.DataFrame(rows)
        result["downtime_index"] = range(len(result))
        return result[
            [
                "start",
                "end",
                "duration_seconds",
                "downtime_index",
                "uuid",
                "source_uuid",
                "is_delta",
            ]
        ]

    def count_events_between_gaps(
        self,
        production_df: pd.DataFrame,
        production_uuid: str,
        *,
        min_gap: str = "3h",
        value_column: str = "value_integer",
        aggregation: str = "count",
    ) -> pd.DataFrame:
        """Count production events in each window between consecutive long downtimes.

        Args:
            production_df: ts-shape DataFrame with the production signal.
            production_uuid: UUID of the production signal in production_df.
            min_gap: Minimum downtime duration to qualify as a long-downtime boundary.
            value_column: Column used for 'sum' and 'transitions' aggregations.
            aggregation: How to count events in each window:
                - 'count': number of rows in the window
                - 'sum': sum of value_column in the window
                - 'transitions': number of value changes (state changes)

        Returns:
            DataFrame with one row per inter-gap window.
            Columns: start, end, window_duration_seconds, event_count,
                     downtime_index, uuid, source_uuid, is_delta
        """
        empty = pd.DataFrame(
            columns=[
                "start",
                "end",
                "window_duration_seconds",
                "event_count",
                "downtime_index",
                "uuid",
                "source_uuid",
                "is_delta",
            ]
        )

        gaps = self.detect_long_downtime(min_gap)
        if len(gaps) < 2:
            return empty

        prod = (
            production_df[production_df["uuid"] == production_uuid]
            .copy()
            .sort_values("systime")
        )
        prod["systime"] = pd.to_datetime(prod["systime"])

        rows: list[dict[str, Any]] = []
        for i in range(len(gaps) - 1):
            start = gaps.iloc[i]["end"]
            end = gaps.iloc[i + 1]["start"]
            following_index = int(gaps.iloc[i + 1]["downtime_index"])

            subset = prod[(prod["systime"] > start) & (prod["systime"] < end)]

            if aggregation == "count":
                event_count = len(subset)
            elif aggregation == "sum":
                event_count = subset[value_column].sum() if not subset.empty else 0
            elif aggregation == "transitions":
                if subset.empty:
                    event_count = 0
                else:
                    event_count = int(
                        (subset[value_column] != subset[value_column].shift()).sum()
                    )
            else:
                raise ValueError(
                    f"Unknown aggregation '{aggregation}'. "
                    "Use 'count', 'sum', or 'transitions'."
                )

            rows.append(
                {
                    "start": start,
                    "end": end,
                    "window_duration_seconds": (end - start).total_seconds(),
                    "event_count": event_count,
                    "downtime_index": following_index,
                    "uuid": self.event_uuid,
                    "source_uuid": self.state_uuid,
                    "is_delta": True,
                }
            )

        return pd.DataFrame(rows)
