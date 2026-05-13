"""Performance/speed loss tracking.

Track when machine runs slower than ideal/target speed:
- Identify slow periods vs target cycle time
- Calculate performance loss in minutes and parts
- Performance trend over time
"""

import logging
import pandas as pd  # type: ignore
from typing import Optional, Dict

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class PerformanceLossTracking(Base):
    """Track performance and speed losses against target cycle times.

    Identifies hidden losses where the machine is running but slower than
    it should be.  Typically 10-20% of production time is lost to speed losses.

    Each UUID represents one signal:
    - cycle_uuid: cycle trigger or production counter
    - part_id_uuid: part number signal (optional, for per-part targets)

    Merge keys: [date, shift] for shift-level, [period] for trend data.

    Pipeline example::

        perf = PerformanceLossTracking(df)
        shift_perf = perf.performance_by_shift('cycle', target_cycle_time=45)
        # → merge with DowntimeTracking.downtime_by_shift() on [date, shift]
        # → merge with QualityTracking.nok_by_shift() on [date, shift]
        # → feed into ShiftHandoverReport.from_shift_data()

    Example usage:
        tracker = PerformanceLossTracking(df)

        # Performance by shift
        perf = tracker.performance_by_shift(
            cycle_uuid='cycle_trigger',
            target_cycle_time=45.0,
        )

        # Identify slow periods
        slow = tracker.slow_periods(
            cycle_uuid='cycle_trigger',
            target_cycle_time=45.0,
            threshold_pct=90,
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: Optional[Dict[str, tuple[str, str]]] = None,
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        time = timestamp.time()
        for shift_name, (start, end) in self.shift_definitions.items():
            start_time = pd.to_datetime(start).time()
            end_time = pd.to_datetime(end).time()
            if start_time < end_time:
                if start_time <= time < end_time:
                    return shift_name
            else:
                if time >= start_time or time < end_time:
                    return shift_name
        return "unknown"

    def _compute_cycle_times(
        self,
        cycle_uuid: str,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Compute per-cycle times from a counter or trigger signal."""
        data = (
            self.dataframe[self.dataframe["uuid"] == cycle_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=[self.time_column, "cycle_time_s"])

        data[self.time_column] = pd.to_datetime(data[self.time_column])

        # If counter: detect increments.  If trigger (bool): each row is a cycle.
        if value_column in data.columns and pd.api.types.is_numeric_dtype(
            data[value_column]
        ):
            data["delta"] = data[value_column].diff()
            data = data[data["delta"] > 0]

        data["cycle_time_s"] = data[self.time_column].diff().dt.total_seconds()
        data = data[data["cycle_time_s"].notna() & (data["cycle_time_s"] > 0)]
        return data

    def performance_by_shift(
        self,
        cycle_uuid: str,
        target_cycle_time: float,
        *,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Calculate performance percentage per shift.

        Performance = (target_cycle_time * actual_parts) / elapsed_run_time.

        Args:
            cycle_uuid: UUID of cycle trigger or counter signal.
            target_cycle_time: Target (ideal) cycle time in seconds.
            value_column: Column with counter values.

        Returns:
            DataFrame with columns:
            - date, shift, actual_parts, avg_cycle_time_s,
              target_cycle_time_s, performance_pct, loss_minutes
        """
        data = self._compute_cycle_times(cycle_uuid, value_column)
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "actual_parts",
                    "avg_cycle_time_s",
                    "target_cycle_time_s",
                    "performance_pct",
                    "loss_minutes",
                ]
            )

        data["shift"] = data[self.time_column].apply(self._assign_shift)
        data["date"] = data[self.time_column].dt.date

        results = []
        for (date, shift), grp in data.groupby(["date", "shift"]):
            actual_parts = len(grp)
            avg_ct = grp["cycle_time_s"].mean()
            elapsed = grp["cycle_time_s"].sum()

            if elapsed > 0 and target_cycle_time > 0:
                ideal_time = actual_parts * target_cycle_time
                performance = (ideal_time / elapsed) * 100.0
            else:
                performance = 0.0

            loss_seconds = max(0, elapsed - actual_parts * target_cycle_time)

            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "actual_parts": actual_parts,
                    "avg_cycle_time_s": round(avg_ct, 2),
                    "target_cycle_time_s": target_cycle_time,
                    "performance_pct": round(min(performance, 100.0), 1),
                    "loss_minutes": round(loss_seconds / 60, 1),
                }
            )

        return pd.DataFrame(results)

    def slow_periods(
        self,
        cycle_uuid: str,
        target_cycle_time: float,
        *,
        threshold_pct: float = 90.0,
        window: str = "1h",
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Identify time windows where performance is below threshold.

        Args:
            cycle_uuid: UUID of cycle trigger or counter signal.
            target_cycle_time: Target cycle time in seconds.
            threshold_pct: Performance must be below this to flag (default 90%).
            window: Rolling window size (default '1h').
            value_column: Column with counter values.

        Returns:
            DataFrame with columns:
            - window_start, window_end, actual_parts, avg_cycle_time_s,
              performance_pct, loss_minutes
        """
        data = self._compute_cycle_times(cycle_uuid, value_column)
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "window_end",
                    "actual_parts",
                    "avg_cycle_time_s",
                    "performance_pct",
                    "loss_minutes",
                ]
            )

        data = data.set_index(self.time_column)
        results = []

        for start, grp in data.groupby(pd.Grouper(freq=window)):
            if grp.empty or len(grp) < 2:
                continue

            actual_parts = len(grp)
            avg_ct = grp["cycle_time_s"].mean()
            elapsed = grp["cycle_time_s"].sum()

            if elapsed > 0 and target_cycle_time > 0:
                ideal_time = actual_parts * target_cycle_time
                performance = (ideal_time / elapsed) * 100.0
            else:
                performance = 0.0

            if performance < threshold_pct:
                loss_seconds = max(0, elapsed - actual_parts * target_cycle_time)
                results.append(
                    {
                        "window_start": start,
                        "window_end": start + pd.Timedelta(window),
                        "actual_parts": actual_parts,
                        "avg_cycle_time_s": round(avg_ct, 2),
                        "performance_pct": round(performance, 1),
                        "loss_minutes": round(loss_seconds / 60, 1),
                    }
                )

        return pd.DataFrame(results)

    def performance_trend(
        self,
        cycle_uuid: str,
        target_cycle_time: float,
        *,
        window: str = "1D",
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Track performance trend over time.

        Args:
            cycle_uuid: UUID of cycle trigger or counter signal.
            target_cycle_time: Target cycle time in seconds.
            window: Time window for aggregation (default '1D').
            value_column: Column with counter values.

        Returns:
            DataFrame with columns:
            - period, actual_parts, avg_cycle_time_s, performance_pct, loss_minutes
        """
        data = self._compute_cycle_times(cycle_uuid, value_column)
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "period",
                    "actual_parts",
                    "avg_cycle_time_s",
                    "performance_pct",
                    "loss_minutes",
                ]
            )

        data = data.set_index(self.time_column)
        results = []

        for period, grp in data.groupby(pd.Grouper(freq=window)):
            if grp.empty:
                continue

            actual_parts = len(grp)
            avg_ct = grp["cycle_time_s"].mean()
            elapsed = grp["cycle_time_s"].sum()

            if elapsed > 0 and target_cycle_time > 0:
                ideal_time = actual_parts * target_cycle_time
                performance = (ideal_time / elapsed) * 100.0
            else:
                performance = 0.0

            loss_seconds = max(0, elapsed - actual_parts * target_cycle_time)

            results.append(
                {
                    "period": period,
                    "actual_parts": actual_parts,
                    "avg_cycle_time_s": round(avg_ct, 2),
                    "performance_pct": round(min(performance, 100.0), 1),
                    "loss_minutes": round(loss_seconds / 60, 1),
                }
            )

        return pd.DataFrame(results)
