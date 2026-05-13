"""Downtime tracking by shift and reason.

Essential module for daily downtime analysis:
- Downtime by shift
- Downtime by reason code
- Availability calculations
- Downtime trends
"""

import logging
import pandas as pd  # type: ignore
from typing import Optional, Dict

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DowntimeTracking(Base):
    """Track machine downtimes by shift and reason.

    Each UUID represents one signal:
    - state_uuid: machine state (running/stopped/idle)
    - reason_uuid: downtime reason code (optional)

    Merge keys: [date, shift] for shift-level, [period] for trend data.

    Pipeline example::

        downtime = DowntimeTracking(df)
        shift_dt = downtime.downtime_by_shift('machine_state')
        # → merge with QualityTracking.nok_by_shift() on [date, shift]
        # → merge with ShiftReporting.shift_production() on [date, shift]
        # → feed into ShiftHandoverReport.from_shift_data()

    Example usage:
        tracker = DowntimeTracking(df)

        # Downtime per shift
        shift_downtime = tracker.downtime_by_shift(
            state_uuid='machine_state',
            running_value='Running'
        )

        # Downtime by reason
        reason_analysis = tracker.downtime_by_reason(
            state_uuid='machine_state',
            reason_uuid='downtime_reason',
            stopped_value='Stopped'
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: Optional[Dict[str, tuple[str, str]]] = None,
    ) -> None:
        """Initialize downtime tracker.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
            shift_definitions: Dictionary mapping shift names to (start, end) times
                              Default: 3-shift operation (06:00-14:00, 14:00-22:00, 22:00-06:00)
        """
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column

        # Default 3-shift operation
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        """Assign shift based on time of day."""
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

    def downtime_by_shift(
        self,
        state_uuid: str,
        *,
        running_value: str = "Running",
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Calculate downtime duration per shift.

        Args:
            state_uuid: UUID for machine state signal
            running_value: Value that indicates machine is running
            value_column: Column containing state values

        Returns:
            DataFrame with downtime by shift:
            - date: Production date
            - shift: Shift name
            - total_minutes: Total shift duration in minutes
            - downtime_minutes: Downtime duration in minutes
            - uptime_minutes: Running time in minutes
            - availability_pct: Uptime percentage

        Example:
            >>> downtime_by_shift('machine_state', running_value='Running')
                date        shift    total_minutes  downtime_minutes  uptime_minutes  availability_pct
            0   2024-01-01  shift_1  480            45.2             434.8           90.6
            1   2024-01-01  shift_2  480            67.5             412.5           85.9
            2   2024-01-01  shift_3  480            92.0             388.0           80.8
        """
        state_data = (
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "total_minutes",
                    "downtime_minutes",
                    "uptime_minutes",
                    "availability_pct",
                ]
            )

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])

        # Assign shifts and dates
        state_data["shift"] = state_data[self.time_column].apply(self._assign_shift)
        state_data["date"] = state_data[self.time_column].dt.date

        # Calculate duration for each state
        state_data["next_time"] = state_data[self.time_column].shift(-1)
        state_data["duration_minutes"] = (
            state_data["next_time"] - state_data[self.time_column]
        ).dt.total_seconds() / 60

        # Mark running vs downtime
        state_data["is_running"] = state_data[value_column] == running_value

        # Group by date and shift
        results = []
        for (date, shift), group in state_data.groupby(["date", "shift"]):
            if group.empty:
                continue

            # Calculate times (exclude last row with NaN duration)
            valid_data = group[group["duration_minutes"].notna()]

            if valid_data.empty:
                continue

            uptime_minutes = valid_data[valid_data["is_running"]][
                "duration_minutes"
            ].sum()
            downtime_minutes = valid_data[~valid_data["is_running"]][
                "duration_minutes"
            ].sum()
            total_minutes = uptime_minutes + downtime_minutes

            availability_pct = (
                (uptime_minutes / total_minutes * 100) if total_minutes > 0 else 0
            )

            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "total_minutes": round(total_minutes, 1),
                    "downtime_minutes": round(downtime_minutes, 1),
                    "uptime_minutes": round(uptime_minutes, 1),
                    "availability_pct": round(availability_pct, 1),
                }
            )

        return pd.DataFrame(results)

    def downtime_by_reason(
        self,
        state_uuid: str,
        reason_uuid: str,
        *,
        stopped_value: str = "Stopped",
        value_column_state: str = "value_string",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Analyze downtime by reason code.

        Args:
            state_uuid: UUID for machine state signal
            reason_uuid: UUID for downtime reason signal
            stopped_value: Value indicating machine is stopped
            value_column_state: Column containing state values
            value_column_reason: Column containing reason codes

        Returns:
            DataFrame with downtime by reason:
            - reason: Reason code
            - occurrences: Number of downtime events
            - total_minutes: Total downtime for this reason
            - avg_minutes: Average downtime per occurrence
            - pct_of_total: Percentage of total downtime

        Example:
            >>> downtime_by_reason('state', 'reason', stopped_value='Stopped')
                reason              occurrences  total_minutes  avg_minutes  pct_of_total
            0   Material_Shortage   12           145.5         12.1         35.2
            1   Tool_Change         8            98.2          12.3         23.8
            2   Quality_Issue       5            76.0          15.2         18.4
        """
        state_data = (
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        reason_data = (
            self.dataframe[self.dataframe["uuid"] == reason_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty or reason_data.empty:
            return pd.DataFrame(
                columns=[
                    "reason",
                    "occurrences",
                    "total_minutes",
                    "avg_minutes",
                    "pct_of_total",
                ]
            )

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        reason_data[self.time_column] = pd.to_datetime(reason_data[self.time_column])

        # Merge state with reason - select only needed columns to avoid suffix issues
        # Note: both dataframes likely have value_string, so we need to be careful
        state_clean = state_data[[self.time_column, value_column_state, "uuid"]].copy()
        state_clean = state_clean.rename(columns={value_column_state: "state"})

        reason_clean = reason_data[[self.time_column, value_column_reason]].copy()
        reason_clean = reason_clean.rename(columns={value_column_reason: "reason"})

        merged = pd.merge_asof(
            state_clean, reason_clean, on=self.time_column, direction="backward"
        )

        # Filter for stopped states
        stopped = merged[merged["state"] == stopped_value].copy()

        if stopped.empty:
            return pd.DataFrame(
                columns=[
                    "reason",
                    "occurrences",
                    "total_minutes",
                    "avg_minutes",
                    "pct_of_total",
                ]
            )

        # Calculate duration
        stopped = stopped.sort_values(self.time_column)
        stopped["next_time"] = stopped[self.time_column].shift(-1)
        stopped["duration_minutes"] = (
            stopped["next_time"] - stopped[self.time_column]
        ).dt.total_seconds() / 60

        # Remove NaN durations
        stopped = stopped[stopped["duration_minutes"].notna()]

        if stopped.empty:
            return pd.DataFrame(
                columns=[
                    "reason",
                    "occurrences",
                    "total_minutes",
                    "avg_minutes",
                    "pct_of_total",
                ]
            )

        # Group by reason
        reason_stats = (
            stopped.groupby("reason")["duration_minutes"]
            .agg(
                [
                    ("occurrences", "count"),
                    ("total_minutes", "sum"),
                    ("avg_minutes", "mean"),
                ]
            )
            .reset_index()
        )

        # Calculate percentage of total
        total_downtime = reason_stats["total_minutes"].sum()
        reason_stats["pct_of_total"] = (
            (reason_stats["total_minutes"] / total_downtime * 100)
            if total_downtime > 0
            else 0
        )

        # Round values
        reason_stats["total_minutes"] = reason_stats["total_minutes"].round(1)
        reason_stats["avg_minutes"] = reason_stats["avg_minutes"].round(1)
        reason_stats["pct_of_total"] = reason_stats["pct_of_total"].round(1)

        # Sort by total downtime descending
        reason_stats = reason_stats.sort_values("total_minutes", ascending=False)

        return reason_stats.reset_index(drop=True)

    def top_downtime_reasons(
        self,
        state_uuid: str,
        reason_uuid: str,
        *,
        top_n: int = 5,
        stopped_value: str = "Stopped",
        value_column_state: str = "value_string",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Get top N downtime reasons (Pareto analysis).

        Args:
            state_uuid: UUID for machine state signal
            reason_uuid: UUID for downtime reason signal
            top_n: Number of top reasons to return
            stopped_value: Value indicating machine is stopped
            value_column_state: Column containing state values
            value_column_reason: Column containing reason codes

        Returns:
            DataFrame with top N reasons and cumulative percentage

        Example:
            >>> top_downtime_reasons('state', 'reason', top_n=5)
                reason              total_minutes  pct_of_total  cumulative_pct
            0   Material_Shortage   145.5         35.2          35.2
            1   Tool_Change         98.2          23.8          59.0
            2   Quality_Issue       76.0          18.4          77.4
        """
        reason_stats = self.downtime_by_reason(
            state_uuid,
            reason_uuid,
            stopped_value=stopped_value,
            value_column_state=value_column_state,
            value_column_reason=value_column_reason,
        )

        if reason_stats.empty:
            return pd.DataFrame(
                columns=["reason", "total_minutes", "pct_of_total", "cumulative_pct"]
            )

        # Get top N
        top_reasons = reason_stats.head(top_n).copy()

        # Calculate cumulative percentage
        top_reasons["cumulative_pct"] = top_reasons["pct_of_total"].cumsum().round(1)

        return top_reasons[
            ["reason", "total_minutes", "pct_of_total", "cumulative_pct"]
        ]

    def availability_trend(
        self,
        state_uuid: str,
        *,
        running_value: str = "Running",
        value_column: str = "value_string",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Calculate availability trend over time.

        Args:
            state_uuid: UUID for machine state signal
            running_value: Value that indicates machine is running
            value_column: Column containing state values
            window: Time window for aggregation (e.g., '1D', '1W')

        Returns:
            DataFrame with availability trend:
            - period: Time period
            - availability_pct: Availability percentage
            - uptime_minutes: Total uptime
            - downtime_minutes: Total downtime

        Example:
            >>> availability_trend('state', window='1D')
                period      availability_pct  uptime_minutes  downtime_minutes
            0   2024-01-01  87.5             1260.0          180.0
            1   2024-01-02  91.2             1313.3          126.7
            2   2024-01-03  85.8             1235.5          204.5
        """
        state_data = (
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return pd.DataFrame(
                columns=[
                    "period",
                    "availability_pct",
                    "uptime_minutes",
                    "downtime_minutes",
                ]
            )

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])

        # Calculate duration for each state
        state_data["next_time"] = state_data[self.time_column].shift(-1)
        state_data["duration_minutes"] = (
            state_data["next_time"] - state_data[self.time_column]
        ).dt.total_seconds() / 60

        # Mark running vs downtime
        state_data["is_running"] = state_data[value_column] == running_value

        # Remove NaN durations
        state_data = state_data[state_data["duration_minutes"].notna()]

        if state_data.empty:
            return pd.DataFrame(
                columns=[
                    "period",
                    "availability_pct",
                    "uptime_minutes",
                    "downtime_minutes",
                ]
            )

        # Group by time window
        state_data = state_data.set_index(self.time_column)

        results = []
        for period, group in state_data.groupby(pd.Grouper(freq=window)):
            if group.empty:
                continue

            uptime = group[group["is_running"]]["duration_minutes"].sum()
            downtime = group[~group["is_running"]]["duration_minutes"].sum()
            total = uptime + downtime

            availability = (uptime / total * 100) if total > 0 else 0

            results.append(
                {
                    "period": period,
                    "availability_pct": round(availability, 1),
                    "uptime_minutes": round(uptime, 1),
                    "downtime_minutes": round(downtime, 1),
                }
            )

        return pd.DataFrame(results)
