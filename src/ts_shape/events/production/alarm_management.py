"""ISA-18.2 style alarm analysis.

Provides alarm frequency, duration statistics, chattering detection,
and standing alarm identification from boolean alarm signals.
"""

import logging

import numpy as np
import pandas as pd

from ts_shape.events._output import finalize_interval_df
from ts_shape.events._windows import window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import (
    deprecated_method,
    keyword_only,
    renamed_kwargs,
)
from ts_shape.utils.runs import run_bounds, total_seconds

logger = logging.getLogger(__name__)


class AlarmManagementEvents(Base):
    """Analyze alarm signals following ISA-18.2 alarm management principles.

    Works with boolean alarm signals where True = alarm active, False = alarm
    cleared.  Provides metrics for alarm rationalization and nuisance alarm
    detection.

    Example usage:
        alarms = AlarmManagementEvents(df, alarm_uuid='temp_high_alarm')

        freq = alarms.alarm_frequency(window='1h')
        stats = alarms.alarm_duration_statistics()
        chatter = alarms.chattering_detection(min_transitions=5, window='10min')
        standing = alarms.standing_alarms(min_duration='1h')

    Example:
        >>> import pandas as pd
        >>> from ts_shape import AlarmManagementEvents
        >>> df = pd.DataFrame({
        ...     "systime": pd.to_datetime([
        ...         "2025-01-01 08:00", "2025-01-01 08:05", "2025-01-01 08:20",
        ...         "2025-01-01 09:30", "2025-01-01 09:40", "2025-01-01 09:41"]),
        ...     "uuid": "temp_high",
        ...     "value_bool": [True, False, True, False, True, False],
        ... })
        >>> alarms = AlarmManagementEvents(df, "temp_high")
        >>> alarms.alarm_duration_statistics()[
        ...     ["alarm_count", "max_duration_seconds", "total_duration_seconds"]]
           alarm_count  max_duration_seconds  total_duration_seconds
        0            3                4200.0                  4560.0
        >>> alarms.standing_alarms(min_duration="1h")[["start", "end", "duration_seconds"]]
                        start                 end  duration_seconds
        0 2025-01-01 08:20:00 2025-01-01 09:30:00            4200.0
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
        """Initialize alarm management analyzer.

        Args:
            dataframe: Input DataFrame with timeseries data.
            alarm_uuid: UUID of the alarm signal.
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the boolean alarm state.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.alarm_uuid = alarm_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, alarm_uuid)

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
        state = bool_or_false(s[self.value_column]).to_numpy(dtype=bool)
        runs = run_bounds(state)
        on = state[runs.starts]
        if not on.any():
            return pd.DataFrame([])
        # An ON run ends where the next run (OFF) starts, or at the last
        # sample when the alarm is still active at the end of the data.
        times = s[self.time_column].array
        start = times[runs.starts[on]]
        end = times[np.minimum(runs.ends[on], len(state) - 1)]
        return pd.DataFrame(
            {
                "start": start,
                "end": end,
                "duration_seconds": total_seconds(end - start),
            }
        )

    # ------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------

    @keyword_only(1)
    @renamed_kwargs(count_initial_active="include_initial")
    def alarm_frequency(
        self, window: str = "1h", *, include_initial: bool = True
    ) -> pd.DataFrame:
        """Count alarm activations per time window.

        An activation is a transition from False to True.

        Args:
            window: Pandas offset alias for the aggregation window.
            include_initial: When True (default) an alarm that is already
                active in the first sample is counted as an activation, because
                ``shift(fill_value=False)`` treats the leading True as a rising
                edge. Pass ``False`` to ignore an alarm that was raised before
                the data window began (no real activation occurred inside it).

        Returns:
            DataFrame with columns:
            - start: window start
            - end: window end (start + window)
            - duration_seconds: window length in seconds
            - alarm_count
            - uuid
            - source_uuid
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "alarm_count",
                    "uuid",
                    "source_uuid",
                ]
            )

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = bool_or_false(s[self.value_column])
        s["prev"] = s["state"].shift(fill_value=False)
        # Rising edges = alarm activations
        s["activation"] = s["state"] & ~s["prev"]
        if not include_initial and len(s) > 0 and bool(s["state"].iloc[0]):
            # The alarm was already active at the window start -> not a new
            # activation.
            s.iloc[0, s.columns.get_loc("activation")] = False

        s = s.set_index(self.time_column)
        counts = s["activation"].resample(window).sum().reset_index()
        counts.columns = ["start", "alarm_count"]
        counts["alarm_count"] = counts["alarm_count"].astype(int)
        counts["start"], end = window_bounds(counts["start"], window)
        counts.insert(1, "end", end)
        counts.insert(
            2,
            "duration_seconds",
            (counts["end"] - counts["start"]).dt.total_seconds(),
        )
        counts["uuid"] = self.event_uuid
        counts["source_uuid"] = self.alarm_uuid
        return counts

    alarm_duration_stats = deprecated_method("alarm_duration_statistics")

    def alarm_duration_statistics(self) -> pd.DataFrame:
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
                    "min_duration_seconds": durations.min(),
                    "avg_duration_seconds": durations.mean(),
                    "max_duration_seconds": durations.max(),
                    "total_duration_seconds": durations.sum(),
                }
            ]
        )

    @keyword_only(1)
    def chattering_detection(
        self,
        min_transitions: int = 5,
        window: str = "10min",
    ) -> pd.DataFrame:
        """Detect chattering alarms (too many transitions in a short window).

        A chattering alarm is one that toggles on/off rapidly, which is a
        nuisance and masks real alarms.

        Args:
            min_transitions: Minimum state changes within *window* to flag.
            window: Rolling window size (Pandas offset alias).

        Returns:
            DataFrame with columns:
            - start
            - end
            - duration_seconds: float seconds (end - start)
            - transition_count
            - uuid
            - source_uuid
        """
        if self.series.empty or len(self.series) < 2:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "transition_count",
                    "uuid",
                    "source_uuid",
                ]
            )

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = bool_or_false(s[self.value_column])
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
                    "start",
                    "end",
                    "duration_seconds",
                    "transition_count",
                    "uuid",
                    "source_uuid",
                ]
            )

        # Collapse consecutive flagged timestamps into distinct windows: a new
        # window starts after a gap longer than ``window``.
        flagged_times = flagged.index.array
        counts = flagged.to_numpy(dtype=np.float64).astype(np.int64)
        cut = np.flatnonzero((flagged_times[1:] - flagged_times[:-1]) > window_td) + 1
        first = np.concatenate(([0], cut))
        last = np.concatenate((cut - 1, [len(flagged_times) - 1]))
        rows = pd.DataFrame(
            {
                "start": flagged_times[first],
                "end": flagged_times[last],
                "transition_count": np.maximum.reduceat(counts, first),
                "uuid": self.event_uuid,
                "source_uuid": self.alarm_uuid,
            }
        )

        out = finalize_interval_df(
            rows, uuid=self.event_uuid, source_uuid=self.alarm_uuid
        )
        cols = ["start", "end", "duration_seconds", "transition_count"]
        return out[cols + ["uuid", "source_uuid"]]

    @keyword_only(1)
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
