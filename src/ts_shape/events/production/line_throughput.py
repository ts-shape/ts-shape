import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class LineThroughputEvents(Base):
    """Production: Line Throughput

    Methods:
    - count_parts: Part counts per fixed window from a monotonically increasing counter.
    - takt_adherence: Cycle time violations against a takt time from step/boolean triggers.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "prod:throughput",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column

    def count_parts(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1m",
    ) -> pd.DataFrame:
        """Compute parts per window for a counter uuid.

        Returns columns: window_start, uuid, source_uuid, is_delta, count
        """
        c = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if c.empty:
            return pd.DataFrame(
                columns=["window_start", "uuid", "source_uuid", "is_delta", "count"]
            )
        c[self.time_column] = pd.to_datetime(c[self.time_column])
        c = c.set_index(self.time_column)
        # take diff of last values within each window
        grp = c[value_column].resample(window)
        counts = grp.max().ffill().diff().fillna(0).clip(lower=0)
        out = (
            counts.to_frame("count")
            .reset_index()
            .rename(columns={self.time_column: "window_start"})
        )
        out["uuid"] = self.event_uuid
        out["source_uuid"] = counter_uuid
        out["is_delta"] = True
        return out

    def takt_adherence(
        self,
        cycle_uuid: str,
        *,
        value_column: str = "value_bool",
        takt_time: str = "60s",
        min_violation: str = "0s",
    ) -> pd.DataFrame:
        """Flag cycles whose durations exceed the takt_time.

        For boolean triggers: detect True rising edges as cycle boundaries.
        For integer steps: detect increments as cycle boundaries.

        Returns: systime (at boundary), uuid, source_uuid, is_delta, cycle_time_seconds, violation
        """
        s = (
            self.dataframe[self.dataframe["uuid"] == cycle_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if s.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "violation",
                ]
            )
        s[self.time_column] = pd.to_datetime(s[self.time_column])
        if value_column == "value_bool":
            s["prev"] = s[value_column].shift(fill_value=False)
            edges = s[(~s["prev"]) & (s[value_column].fillna(False))]
            times = edges[self.time_column].reset_index(drop=True)
        else:
            s["prev"] = s[value_column].shift(1)
            edges = s[s[value_column].fillna(0) != s["prev"].fillna(0)]
            times = edges[self.time_column].reset_index(drop=True)
        if len(times) < 2:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "violation",
                ]
            )
        cycle_times = (times.diff().dt.total_seconds()).iloc[1:].reset_index(drop=True)
        min_td = pd.to_timedelta(min_violation).total_seconds()
        target = pd.to_timedelta(takt_time).total_seconds()
        viol = (cycle_times - target) >= min_td
        out = pd.DataFrame(
            {
                "systime": times.iloc[1:].reset_index(drop=True),
                "uuid": self.event_uuid,
                "source_uuid": cycle_uuid,
                "is_delta": True,
                "cycle_time_seconds": cycle_times,
                "violation": viol,
            }
        )
        return out

    def throughput_oee(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1h",
        target_rate: Optional[float] = None,
        availability_threshold: float = 0.95,
    ) -> pd.DataFrame:
        """Calculate Overall Equipment Effectiveness (OEE) metrics.

        OEE = Availability × Performance × Quality

        Args:
            counter_uuid: UUID for the part counter signal
            value_column: Column containing counter values
            window: Time window for aggregation
            target_rate: Target production rate (parts per window). If None, uses max observed
            availability_threshold: Threshold for considering equipment available

        Returns:
            DataFrame with columns: window_start, uuid, source_uuid, is_delta,
            actual_count, target_count, availability, performance, oee_score
        """
        parts_df = self.count_parts(
            counter_uuid, value_column=value_column, window=window
        )

        if parts_df.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "actual_count",
                    "target_count",
                    "availability",
                    "performance",
                    "oee_score",
                ]
            )

        # Calculate target rate if not provided
        if target_rate is None:
            target_rate = parts_df["count"].quantile(0.95)

        parts_df["target_count"] = target_rate
        parts_df["actual_count"] = parts_df["count"]

        # Availability: percentage of time equipment was running
        parts_df["availability"] = np.where(parts_df["count"] > 0, 1.0, 0.0)

        # Performance: actual vs target rate
        parts_df["performance"] = np.minimum(parts_df["count"] / target_rate, 1.0)

        # OEE score (simplified - assumes quality = 1.0)
        parts_df["oee_score"] = parts_df["availability"] * parts_df["performance"]

        return parts_df[
            [
                "window_start",
                "uuid",
                "source_uuid",
                "is_delta",
                "actual_count",
                "target_count",
                "availability",
                "performance",
                "oee_score",
            ]
        ]

    def throughput_trends(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1h",
        trend_window: int = 24,
    ) -> pd.DataFrame:
        """Analyze throughput trends with moving averages and degradation detection.

        Args:
            counter_uuid: UUID for the part counter signal
            value_column: Column containing counter values
            window: Time window for counting parts
            trend_window: Number of windows for trend calculation

        Returns:
            DataFrame with throughput, moving average, trend direction, and degradation flag
        """
        parts_df = self.count_parts(
            counter_uuid, value_column=value_column, window=window
        )

        if parts_df.empty or len(parts_df) < trend_window:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "count",
                    "moving_avg",
                    "trend_direction",
                    "degradation_detected",
                ]
            )

        # Calculate moving average
        parts_df["moving_avg"] = (
            parts_df["count"].rolling(window=trend_window, min_periods=1).mean()
        )

        # Calculate trend (positive, negative, stable)
        parts_df["trend_slope"] = parts_df["moving_avg"].diff()
        parts_df["trend_direction"] = pd.cut(
            parts_df["trend_slope"],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["decreasing", "stable", "increasing"],
        )

        # Detect degradation (current significantly below moving average)
        parts_df["degradation_detected"] = parts_df["count"] < (
            parts_df["moving_avg"] * 0.85
        )

        return parts_df[
            [
                "window_start",
                "uuid",
                "source_uuid",
                "is_delta",
                "count",
                "moving_avg",
                "trend_direction",
                "degradation_detected",
            ]
        ]

    def cycle_quality_check(
        self,
        cycle_uuid: str,
        *,
        value_column: str = "value_bool",
        expected_cycle_time: Optional[float] = None,
        tolerance_pct: float = 0.1,
    ) -> pd.DataFrame:
        """Enhanced cycle detection with quality validation.

        Args:
            cycle_uuid: UUID for the cycle trigger signal
            value_column: Column containing cycle trigger (bool/integer)
            expected_cycle_time: Expected cycle time in seconds. If None, uses median
            tolerance_pct: Tolerance percentage for cycle time validation

        Returns:
            DataFrame with cycle times, validation status, and quality flags
        """
        s = (
            self.dataframe[self.dataframe["uuid"] == cycle_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if s.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "expected_time",
                    "deviation_pct",
                    "is_valid",
                    "quality_flag",
                ]
            )

        s[self.time_column] = pd.to_datetime(s[self.time_column])

        # Detect cycle boundaries
        if value_column == "value_bool":
            s["prev"] = s[value_column].shift(fill_value=False)
            edges = s[(~s["prev"]) & (s[value_column].fillna(False))]
            times = edges[self.time_column].reset_index(drop=True)
        else:
            s["prev"] = s[value_column].shift(1)
            edges = s[s[value_column].fillna(0) != s["prev"].fillna(0)]
            times = edges[self.time_column].reset_index(drop=True)

        if len(times) < 2:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "expected_time",
                    "deviation_pct",
                    "is_valid",
                    "quality_flag",
                ]
            )

        cycle_times = (times.diff().dt.total_seconds()).iloc[1:].reset_index(drop=True)

        # Calculate expected cycle time if not provided
        if expected_cycle_time is None:
            expected_cycle_time = cycle_times.median()

        # Calculate deviation
        deviation_pct = (
            (cycle_times - expected_cycle_time) / expected_cycle_time
        ).abs()

        # Validate cycles
        is_valid = deviation_pct <= tolerance_pct

        # Quality flags: good, warning, critical
        quality_flag = pd.cut(
            deviation_pct,
            bins=[-np.inf, 0.1, 0.25, np.inf],
            labels=["good", "warning", "critical"],
        )

        out = pd.DataFrame(
            {
                "systime": times.iloc[1:].reset_index(drop=True),
                "uuid": self.event_uuid,
                "source_uuid": cycle_uuid,
                "is_delta": True,
                "cycle_time_seconds": cycle_times,
                "expected_time": expected_cycle_time,
                "deviation_pct": deviation_pct,
                "is_valid": is_valid,
                "quality_flag": quality_flag,
            }
        )
        return out
