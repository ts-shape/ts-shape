"""ISA-18.2 style alarm analysis.

Provides alarm frequency, duration statistics, chattering detection,
and standing alarm identification from boolean alarm signals.
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class AlarmManagementEvents(Base):
    """Analyze alarm signals following ISA-18.2 alarm management principles.

    Works with boolean alarm signals where True = alarm active, False = alarm
    cleared.  Provides metrics for alarm rationalisation and nuisance alarm
    detection.

    Example usage:
        alarms = AlarmManagementEvents(df, alarm_uuid='temp_high_alarm')

        freq = alarms.alarm_frequency(window='1h')
        stats = alarms.alarm_duration_stats()
        chatter = alarms.chattering_detection(min_transitions=5, window='10m')
        standing = alarms.standing_alarms(min_duration='1h')
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        alarm_uuid: str,
        *,
        event_uuid: str = "prod:alarm",
        value_column: str = "value_bool",
        time_column: str = "systime",
    ) -> None:
        """Initialize alarm management analyser.

        Args:
            dataframe: Input DataFrame with timeseries data.
            alarm_uuid: UUID of the alarm signal.
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the boolean alarm state.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.alarm_uuid = alarm_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.alarm_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _intervalize(self) -> pd.DataFrame:
        """Convert point-based alarm signal into ON intervals.

        Each ON interval starts at the timestamp where the alarm becomes True
        and ends at the timestamp of the next False (or the last sample if the
        alarm is still active at the end of data).

        Returns DataFrame with columns: start, end, duration_seconds.
        """
        if self.series.empty:
            return pd.DataFrame(columns=["start", "end", "duration_seconds"])

        s = (
            self.series[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        s["state"] = s[self.value_column].fillna(False).astype(bool)
        s["group"] = (s["state"] != s["state"].shift()).cumsum()

        rows: List[Dict[str, Any]] = []
        groups = list(s.groupby("group"))
        for idx, (_, seg) in enumerate(groups):
            if not seg["state"].iloc[0]:
                continue  # skip OFF intervals
            start = seg[self.time_column].iloc[0]
            # End is the first timestamp of the *next* group (OFF transition),
            # or the last timestamp in this group if no next group exists.
            if idx + 1 < len(groups):
                _, next_seg = groups[idx + 1]
                end = next_seg[self.time_column].iloc[0]
            else:
                end = seg[self.time_column].iloc[-1]
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "duration_seconds": (end - start).total_seconds(),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------

    def alarm_frequency(self, window: str = "1h") -> pd.DataFrame:
        """Count alarm activations per time window.

        An activation is a transition from False to True.

        Args:
            window: Pandas offset alias for the aggregation window.

        Returns:
            DataFrame with columns:
            - window_start
            - alarm_count
            - uuid
            - source_uuid
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=["window_start", "alarm_count", "uuid", "source_uuid"]
            )

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = s[self.value_column].fillna(False).astype(bool)
        s["prev"] = s["state"].shift(fill_value=False)
        # Rising edges = alarm activations
        s["activation"] = s["state"] & ~s["prev"]

        s = s.set_index(self.time_column)
        counts = s["activation"].resample(window).sum().reset_index()
        counts.columns = ["window_start", "alarm_count"]
        counts["alarm_count"] = counts["alarm_count"].astype(int)
        counts["uuid"] = self.event_uuid
        counts["source_uuid"] = self.alarm_uuid
        return counts

    def alarm_duration_stats(self) -> pd.DataFrame:
        """Compute min / avg / max / total duration of alarm-ON states.

        Returns:
            DataFrame with a single row containing:
            - source_uuid
            - alarm_count
            - min_duration_seconds
            - avg_duration_seconds
            - max_duration_seconds
            - total_duration_seconds
        """
        intervals = self._intervalize()

        if intervals.empty:
            return pd.DataFrame(
                columns=[
                    "source_uuid",
                    "alarm_count",
                    "min_duration_seconds",
                    "avg_duration_seconds",
                    "max_duration_seconds",
                    "total_duration_seconds",
                ]
            )

        durations = intervals["duration_seconds"]
        return pd.DataFrame(
            [
                {
                    "source_uuid": self.alarm_uuid,
                    "alarm_count": len(intervals),
                    "min_duration_seconds": round(durations.min(), 2),
                    "avg_duration_seconds": round(durations.mean(), 2),
                    "max_duration_seconds": round(durations.max(), 2),
                    "total_duration_seconds": round(durations.sum(), 2),
                }
            ]
        )

    def chattering_detection(
        self,
        min_transitions: int = 5,
        window: str = "10m",
    ) -> pd.DataFrame:
        """Detect chattering alarms (too many transitions in a short window).

        A chattering alarm is one that toggles on/off rapidly, which is a
        nuisance and masks real alarms.

        Args:
            min_transitions: Minimum state changes within *window* to flag.
            window: Rolling window size (Pandas offset alias).

        Returns:
            DataFrame with columns:
            - window_start
            - window_end
            - transition_count
            - uuid
            - source_uuid
        """
        if self.series.empty or len(self.series) < 2:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "window_end",
                    "transition_count",
                    "uuid",
                    "source_uuid",
                ]
            )

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = s[self.value_column].fillna(False).astype(bool)
        s["transition"] = (s["state"] != s["state"].shift()).astype(int)
        # First row is not a real transition
        s.iloc[0, s.columns.get_loc("transition")] = 0

        s = s.set_index(self.time_column)

        # Rolling count of transitions
        window_td = pd.to_timedelta(window)
        rolling_counts = s["transition"].rolling(window_td, min_periods=1).sum()

        # Find windows that exceed threshold
        flagged = rolling_counts[rolling_counts >= min_transitions]

        if flagged.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "window_end",
                    "transition_count",
                    "uuid",
                    "source_uuid",
                ]
            )

        # Collapse consecutive flagged timestamps into distinct windows
        flagged_times = flagged.index.to_series().reset_index(drop=True)
        transition_counts = flagged.values

        rows: List[Dict[str, Any]] = []
        window_start = flagged_times.iloc[0]
        max_count = int(transition_counts[0])

        for i in range(1, len(flagged_times)):
            gap = flagged_times.iloc[i] - flagged_times.iloc[i - 1]
            if gap > window_td:
                # Close current window
                rows.append(
                    {
                        "window_start": window_start,
                        "window_end": flagged_times.iloc[i - 1],
                        "transition_count": max_count,
                        "uuid": self.event_uuid,
                        "source_uuid": self.alarm_uuid,
                    }
                )
                window_start = flagged_times.iloc[i]
                max_count = int(transition_counts[i])
            else:
                max_count = max(max_count, int(transition_counts[i]))

        # Close last window
        rows.append(
            {
                "window_start": window_start,
                "window_end": flagged_times.iloc[-1],
                "transition_count": max_count,
                "uuid": self.event_uuid,
                "source_uuid": self.alarm_uuid,
            }
        )

        return pd.DataFrame(rows)

    def standing_alarms(self, min_duration: str = "1h") -> pd.DataFrame:
        """Identify alarms that stay active longer than *min_duration*.

        Standing (stale) alarms reduce operator trust and should be
        investigated for shelving or re-engineering.

        Args:
            min_duration: Minimum ON duration to flag (Pandas offset alias).

        Returns:
            DataFrame with columns:
            - start
            - end
            - duration_seconds
            - uuid
            - source_uuid
        """
        intervals = self._intervalize()

        if intervals.empty:
            return pd.DataFrame(
                columns=["start", "end", "duration_seconds", "uuid", "source_uuid"]
            )

        min_td = pd.to_timedelta(min_duration).total_seconds()
        standing = intervals[intervals["duration_seconds"] >= min_td].copy()

        if standing.empty:
            return pd.DataFrame(
                columns=["start", "end", "duration_seconds", "uuid", "source_uuid"]
            )

        standing["uuid"] = self.event_uuid
        standing["source_uuid"] = self.alarm_uuid
        return standing.reset_index(drop=True)
