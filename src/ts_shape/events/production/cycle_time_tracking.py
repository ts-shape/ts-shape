"""Cycle time tracking by part number.

Simple, practical module for cycle time analysis:
- Cycle time per part number
- Statistics (min, avg, max, std)
- Slow cycle detection
- Trend analysis
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class CycleTimeTracking(Base):
    """Track cycle times by part number.

    Each UUID represents one signal:
    - part_id_uuid: string signal with current part number
    - cycle_trigger_uuid: boolean/integer signal for cycle completion

    Example usage:
        tracker = CycleTimeTracking(df)

        # Get cycle times by part
        cycles = tracker.cycle_time_by_part(
            part_id_uuid='part_number_signal',
            cycle_trigger_uuid='cycle_complete_signal'
        )

        # Get statistics
        stats = tracker.cycle_time_statistics(
            part_id_uuid='part_number_signal',
            cycle_trigger_uuid='cycle_complete_signal'
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
    ) -> None:
        """Initialize cycle time tracker.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
        """
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column

    def cycle_time_by_part(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
    ) -> pd.DataFrame:
        """Calculate cycle time for each part number.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers

        Returns:
            DataFrame with columns:
            - systime: Cycle completion time
            - part_number: Part number/ID
            - cycle_time_seconds: Cycle time in seconds

        Example:
            >>> cycle_time_by_part('part_id', 'cycle_trigger')
                systime              part_number  cycle_time_seconds
            0   2024-01-01 08:05:30  PART_A       45.2
            1   2024-01-01 08:06:18  PART_A       48.0
            2   2024-01-01 08:07:05  PART_A       47.1
        """
        # Get cycle completion times
        cycles = (
            self.dataframe[self.dataframe["uuid"] == cycle_trigger_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if cycles.empty:
            return pd.DataFrame(
                columns=["systime", "part_number", "cycle_time_seconds"]
            )

        cycles[self.time_column] = pd.to_datetime(cycles[self.time_column])

        # Detect rising edges (cycle completion)
        if value_column_trigger == "value_bool":
            cycles["prev"] = cycles[value_column_trigger].shift(fill_value=False)
            cycle_ends = cycles[
                (~cycles["prev"]) & (cycles[value_column_trigger].fillna(False))
            ]
            times = cycle_ends[self.time_column].reset_index(drop=True)
        else:
            # For integer/counter-based cycles, use value changes
            cycles["prev"] = cycles[value_column_trigger].shift()
            cycle_ends = cycles[cycles[value_column_trigger] != cycles["prev"]]
            times = cycle_ends[self.time_column].reset_index(drop=True)

        if len(times) < 2:
            return pd.DataFrame(
                columns=["systime", "part_number", "cycle_time_seconds"]
            )

        # Calculate cycle times
        cycle_times = times.diff().dt.total_seconds()

        # Get part numbers at each cycle
        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if part_data.empty:
            # Return cycles without part numbers
            return pd.DataFrame(
                {
                    "systime": times.iloc[1:],
                    "part_number": "UNKNOWN",
                    "cycle_time_seconds": cycle_times.iloc[1:],
                }
            )

        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])

        # Match part number to each cycle using merge_asof
        result_df = pd.DataFrame(
            {
                "systime": times.iloc[1:].reset_index(drop=True),
                "cycle_time_seconds": cycle_times.iloc[1:].reset_index(drop=True),
            }
        )

        # Merge with part data
        result_df = pd.merge_asof(
            result_df.sort_values("systime"),
            part_data[[self.time_column, value_column_part]].rename(
                columns={self.time_column: "systime"}
            ),
            on="systime",
            direction="backward",
        )

        result_df = result_df.rename(columns={value_column_part: "part_number"})
        result_df["part_number"] = result_df["part_number"].fillna("UNKNOWN")

        return result_df[["systime", "part_number", "cycle_time_seconds"]]

    def cycle_time_statistics(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
    ) -> pd.DataFrame:
        """Calculate statistics: min, avg, max, std cycle time by part.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers

        Returns:
            DataFrame with statistics per part:
            - part_number: Part number/ID
            - count: Number of cycles
            - min_seconds: Minimum cycle time
            - avg_seconds: Average cycle time
            - max_seconds: Maximum cycle time
            - std_seconds: Standard deviation
            - median_seconds: Median cycle time (robust target)

        Example:
            >>> cycle_time_statistics('part_id', 'cycle_trigger')
                part_number  count  min_seconds  avg_seconds  max_seconds  std_seconds  median_seconds
            0   PART_A       450    42.1         47.5         58.2         3.2          47.1
            1   PART_B       320    55.0         62.8         78.5         5.1          61.9
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=[
                    "part_number",
                    "count",
                    "min_seconds",
                    "avg_seconds",
                    "max_seconds",
                    "std_seconds",
                    "median_seconds",
                ]
            )

        stats = (
            cycle_data.groupby("part_number")["cycle_time_seconds"]
            .agg(
                [
                    ("count", "count"),
                    ("min_seconds", "min"),
                    ("avg_seconds", "mean"),
                    ("max_seconds", "max"),
                    ("std_seconds", "std"),
                    ("median_seconds", "median"),
                ]
            )
            .reset_index()
        )

        return stats

    def detect_slow_cycles(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        threshold_factor: float = 1.5,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
    ) -> pd.DataFrame:
        """Identify cycles that exceed normal time by threshold factor.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            threshold_factor: Cycles slower than median * factor are flagged (default: 1.5)
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers

        Returns:
            DataFrame with slow cycles:
            - systime: Cycle completion time
            - part_number: Part number/ID
            - cycle_time_seconds: Actual cycle time
            - median_seconds: Expected (median) cycle time
            - deviation_factor: How many times slower than expected
            - is_slow: Flag (always True in returned data)

        Example:
            >>> detect_slow_cycles('part_id', 'cycle_trigger', threshold_factor=1.5)
                systime              part_number  cycle_time_seconds  median_seconds  deviation_factor  is_slow
            0   2024-01-01 10:15:30  PART_A       75.2               47.1            1.60              True
            1   2024-01-01 14:22:18  PART_A       82.5               47.1            1.75              True
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "part_number",
                    "cycle_time_seconds",
                    "median_seconds",
                    "deviation_factor",
                    "is_slow",
                ]
            )

        # Calculate median per part
        medians = cycle_data.groupby("part_number")["cycle_time_seconds"].median()
        cycle_data["median_seconds"] = cycle_data["part_number"].map(medians)

        # Calculate deviation factor
        cycle_data["deviation_factor"] = (
            cycle_data["cycle_time_seconds"] / cycle_data["median_seconds"]
        )

        # Flag slow cycles
        cycle_data["is_slow"] = cycle_data["deviation_factor"] >= threshold_factor

        # Return only slow cycles
        return cycle_data[cycle_data["is_slow"]].reset_index(drop=True)

    def cycle_time_trend(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        part_number: str,
        *,
        window_size: int = 20,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
    ) -> pd.DataFrame:
        """Analyze cycle time trends for a specific part.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            part_number: Specific part number to analyze
            window_size: Number of cycles for moving average (default: 20)
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers

        Returns:
            DataFrame with trend data:
            - systime: Cycle completion time
            - cycle_time_seconds: Actual cycle time
            - moving_avg: Moving average cycle time
            - trend: 'improving', 'stable', or 'degrading'

        Example:
            >>> cycle_time_trend('part_id', 'cycle_trigger', 'PART_A')
                systime              cycle_time_seconds  moving_avg  trend
            0   2024-01-01 08:05:30  45.2               47.1        improving
            1   2024-01-01 08:06:18  48.0               47.2        stable
            2   2024-01-01 08:07:05  47.1               47.1        stable
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=["systime", "cycle_time_seconds", "moving_avg", "trend"]
            )

        # Filter for specific part
        part_cycles = cycle_data[cycle_data["part_number"] == part_number].copy()

        if part_cycles.empty or len(part_cycles) < window_size:
            return pd.DataFrame(
                columns=["systime", "cycle_time_seconds", "moving_avg", "trend"]
            )

        # Calculate moving average
        part_cycles["moving_avg"] = (
            part_cycles["cycle_time_seconds"]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )

        # Calculate trend
        part_cycles["trend_slope"] = part_cycles["moving_avg"].diff()

        # Classify trend (improving = getting faster)
        part_cycles["trend"] = pd.cut(
            part_cycles["trend_slope"],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["improving", "stable", "degrading"],
        )

        return part_cycles[["systime", "cycle_time_seconds", "moving_avg", "trend"]]

    def hourly_cycle_time_summary(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
    ) -> pd.DataFrame:
        """Hourly summary of cycle times by part.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers

        Returns:
            DataFrame with hourly statistics:
            - hour: Hour timestamp
            - part_number: Part number/ID
            - cycles_completed: Number of cycles
            - avg_cycle_time: Average cycle time
            - min_cycle_time: Fastest cycle
            - max_cycle_time: Slowest cycle

        Example:
            >>> hourly_cycle_time_summary('part_id', 'cycle_trigger')
                hour                 part_number  cycles_completed  avg_cycle_time  min_cycle_time  max_cycle_time
            0   2024-01-01 08:00:00  PART_A       75               47.2            42.1            55.8
            1   2024-01-01 09:00:00  PART_A       78               46.8            43.0            52.3
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=[
                    "hour",
                    "part_number",
                    "cycles_completed",
                    "avg_cycle_time",
                    "min_cycle_time",
                    "max_cycle_time",
                ]
            )

        # Add hour column
        cycle_data["hour"] = cycle_data["systime"].dt.floor("h")

        # Group by hour and part
        hourly = (
            cycle_data.groupby(["hour", "part_number"])["cycle_time_seconds"]
            .agg(
                [
                    ("cycles_completed", "count"),
                    ("avg_cycle_time", "mean"),
                    ("min_cycle_time", "min"),
                    ("max_cycle_time", "max"),
                ]
            )
            .reset_index()
        )

        return hourly
