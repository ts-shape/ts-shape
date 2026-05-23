import logging
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class MachineStateEvents(Base):
    """Production: Machine State

    Detect run/idle transitions and intervals from a boolean state signal.

    Classes:
    - MachineStateEvents: Run/idle state intervals and transitions.
      - detect_run_idle: Intervalize run/idle states with optional min duration filter.
      - transition_events: Point events on state changes (idle->run, run->idle).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        run_state_uuid: str,
        *,
        event_uuid: str = "prod:run_idle",
        value_column: str = "value_bool",
        time_column: str = "systime",
        value_range: tuple[float | None, float | None] | None = None,
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.run_state_uuid = run_state_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.value_range = value_range
        if (
            "uuid" in self.dataframe.columns
            and run_state_uuid not in self.dataframe["uuid"].values
        ):
            raise ValueError(
                f"UUID '{run_state_uuid}' not found in dataframe. "
                f"Available UUIDs: {list(self.dataframe['uuid'].unique())}"
            )
        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])
        self._state_groups: pd.Series | None = None
        self._compute_state_groups()

    def _as_state(self, col: pd.Series) -> pd.Series:
        """Convert a value series to a boolean running/idle state.

        Uses value_range when set (inclusive on both ends); otherwise casts to bool.
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

    def detect_run_idle(self, min_duration: str = "0s") -> pd.DataFrame:
        """Return intervals labeled as 'run' or 'idle'.

        - min_duration: discard intervals shorter than this duration.
        Columns: start, end, uuid, source_uuid, is_delta, state, duration_seconds
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "state",
                    "duration_seconds",
                ]
            )
        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = self._as_state(s[self.value_column])
        state_change = (s["state"] != s["state"].shift()).cumsum()
        min_td = pd.to_timedelta(min_duration)
        rows: list[dict[str, Any]] = []
        for _, seg in s.groupby(state_change):
            state = bool(seg["state"].iloc[0])
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            if (end - start) < min_td:
                continue
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "uuid": self.event_uuid,
                    "source_uuid": self.run_state_uuid,
                    "is_delta": True,
                    "state": "run" if state else "idle",
                    "duration_seconds": (end - start).total_seconds(),
                }
            )
        return pd.DataFrame(rows)

    def transition_events(self) -> pd.DataFrame:
        """Return point events at state transitions.

        Columns: systime, uuid, source_uuid, is_delta, transition ('idle_to_run'|'run_to_idle'), time_since_last_transition_seconds
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "transition",
                    "time_since_last_transition_seconds",
                ]
            )
        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = self._as_state(s[self.value_column])
        s["prev"] = s["state"].shift()
        changes = s[s["state"] != s["prev"]].dropna(subset=["prev"])  # ignore first row
        if changes.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "transition",
                    "time_since_last_transition_seconds",
                ]
            )
        changes = changes.rename(columns={self.time_column: "systime"})
        changes["transition"] = np.where(
            changes["state"] & ~changes["prev"].astype(bool),
            "idle_to_run",
            "run_to_idle",
        )
        changes["time_since_last_transition_seconds"] = (
            changes["systime"].diff().dt.total_seconds()
        )
        return pd.DataFrame(
            {
                "systime": changes["systime"],
                "uuid": self.event_uuid,
                "source_uuid": self.run_state_uuid,
                "is_delta": True,
                "transition": changes["transition"],
                "time_since_last_transition_seconds": changes[
                    "time_since_last_transition_seconds"
                ],
            }
        )

    def _compute_state_groups(self) -> None:
        """Compute and cache state change groups for performance."""
        if self.series.empty:
            self._state_groups = None
            return
        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = self._as_state(s[self.value_column])
        self._state_groups = (s["state"] != s["state"].shift()).cumsum()

    def detect_rapid_transitions(
        self, threshold: str = "5s", min_count: int = 3
    ) -> pd.DataFrame:
        """Identify suspicious rapid state changes.

        - threshold: time window to look for rapid transitions
        - min_count: minimum number of transitions within threshold to be considered rapid
        Returns: DataFrame with start_time, end_time, transition_count, duration_seconds
        """
        transitions = self.transition_events()
        if transitions.empty or len(transitions) < min_count:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "transition_count",
                    "duration_seconds",
                ]
            )

        threshold_td = pd.to_timedelta(threshold)
        rapid_events: list[dict[str, Any]] = []

        for i in range(len(transitions) - min_count + 1):
            start = transitions.iloc[i]["systime"]
            for j in range(i + min_count - 1, len(transitions)):
                end = transitions.iloc[j]["systime"]
                duration = end - start
                if duration <= threshold_td:
                    transition_count = j - i + 1
                    rapid_events.append(
                        {
                            "start": start,
                            "end": end,
                            "transition_count": transition_count,
                            "duration_seconds": duration.total_seconds(),
                        }
                    )
                else:
                    break

        return pd.DataFrame(rapid_events)

    def _detect_data_gaps(self, max_gap: str = "5m") -> int:
        """Detect missing data gaps in the time series.

        - max_gap: maximum acceptable gap between data points
        Returns: count of data gaps detected
        """
        if self.series.empty or len(self.series) < 2:
            return 0

        max_gap_td = pd.to_timedelta(max_gap)
        time_diffs = self.series[self.time_column].diff()
        gaps = time_diffs[time_diffs > max_gap_td]
        return len(gaps)

    def state_quality_metrics(self) -> dict[str, Any]:
        """Return quality metrics for the state data.

        Returns dictionary with:
        - total_transitions: total number of state transitions
        - avg_run_duration: average duration of run states in seconds
        - avg_idle_duration: average duration of idle states in seconds
        - run_idle_ratio: ratio of run time to idle time
        - data_gaps_detected: number of data gaps found
        - rapid_transitions_detected: number of rapid transition events
        """
        transitions = self.transition_events()
        intervals = self.detect_run_idle()

        total_transitions = len(transitions)

        if intervals.empty:
            avg_run_duration = 0.0
            avg_idle_duration = 0.0
            run_idle_ratio = 0.0
        else:
            run_intervals = intervals[intervals["state"] == "run"]
            idle_intervals = intervals[intervals["state"] == "idle"]

            avg_run_duration = (
                run_intervals["duration_seconds"].mean()
                if not run_intervals.empty
                else 0.0
            )
            avg_idle_duration = (
                idle_intervals["duration_seconds"].mean()
                if not idle_intervals.empty
                else 0.0
            )

            total_run_time = (
                run_intervals["duration_seconds"].sum()
                if not run_intervals.empty
                else 0.0
            )
            total_idle_time = (
                idle_intervals["duration_seconds"].sum()
                if not idle_intervals.empty
                else 0.0
            )
            run_idle_ratio = (
                total_run_time / total_idle_time if total_idle_time > 0 else 0.0
            )

        data_gaps_detected = self._detect_data_gaps()
        rapid_transitions_detected = len(self.detect_rapid_transitions())

        return {
            "total_transitions": total_transitions,
            "avg_run_duration": (
                float(avg_run_duration) if not np.isnan(avg_run_duration) else 0.0
            ),
            "avg_idle_duration": (
                float(avg_idle_duration) if not np.isnan(avg_idle_duration) else 0.0
            ),
            "run_idle_ratio": (
                float(run_idle_ratio) if not np.isnan(run_idle_ratio) else 0.0
            ),
            "data_gaps_detected": data_gaps_detected,
            "rapid_transitions_detected": rapid_transitions_detected,
        }
