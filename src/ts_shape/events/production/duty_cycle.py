import logging
from typing import Any

import pandas as pd  # type: ignore

from ts_shape.utils.base import Base
from ts_shape.utils.intervals import state_intervals

logger = logging.getLogger(__name__)


class DutyCycleEvents(Base):
    """Production: Duty Cycle Analysis

    Analyze on/off patterns from a boolean signal: duty cycle percentage,
    interval listing, transition counts, and excessive cycling detection.

    Methods:
    - duty_cycle_per_window: On-time percentage per time window.
    - on_off_intervals: List every on and off interval with duration.
    - cycle_count: Number of on/off transitions per window.
    - excessive_cycling: Flag windows with too many transitions.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "prod:duty_cycle",
        value_column: str = "value_bool",
        time_column: str = "systime",
        interval_end: str = "last_sample",
        data_end: pd.Timestamp | None = None,
    ) -> None:
        """Initialize the duty-cycle analyzer.

        Args:
            interval_end: How on/off intervals are closed. ``"last_sample"``
                (default) ends each interval at its last sample -- backward
                compatible, but a change-encoded (``is_delta``) signal logs one
                row per transition, so every on/off interval collapses to zero
                duration. ``"next_change"`` closes each interval at the next
                transition, correctly measuring change-encoded signals.
            data_end: Optional bound for the final, still-open interval when
                ``interval_end="next_change"``.
        """
        super().__init__(dataframe, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.interval_end = interval_end
        self.data_end = data_end

        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    def on_off_intervals(self) -> pd.DataFrame:
        """List every on and off interval with duration.

        Returns:
            DataFrame with columns: start_time, end_time, state (on/off), duration.
        """
        cols = ["start", "end", "state", "duration"]
        if self.series.empty:
            return pd.DataFrame(columns=cols)

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = s[self.value_column].fillna(False).astype(bool)
        intervals = state_intervals(
            s,
            time_column=self.time_column,
            state_column="state",
            interval_end=self.interval_end,
            data_end=self.data_end,
        )
        if intervals.empty:
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(
            {
                "start": intervals["start"].to_numpy(),
                "end": intervals["end"].to_numpy(),
                "state": ["on" if bool(st) else "off" for st in intervals["state"]],
                "duration": pd.to_timedelta(intervals["duration_seconds"], unit="s"),
            }
        )

    def duty_cycle_per_window(self, window: str = "1h") -> pd.DataFrame:
        """Percentage of time the signal is True per window.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, on_time, off_time, duty_cycle_pct.
        """
        cols = ["start", "on_time", "off_time", "duty_cycle_pct"]
        if self.series.empty:
            return pd.DataFrame(columns=cols)

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = s[self.value_column].fillna(False).astype(float)
        s = s.set_index(self.time_column)

        window_td = pd.to_timedelta(window)
        resampled = s["state"].resample(window).mean()

        events: list[dict[str, Any]] = []
        for ts, pct in resampled.items():
            if pd.isna(pct):
                continue
            on_time = round(pct * window_td.total_seconds(), 2)
            off_time = round((1 - pct) * window_td.total_seconds(), 2)
            events.append(
                {
                    "start": ts,
                    "on_time": on_time,
                    "off_time": off_time,
                    "duty_cycle_pct": round(pct * 100, 2),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def cycle_count(self, window: str = "1h") -> pd.DataFrame:
        """Number of on/off transitions per window.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, on_count, off_count, total_transitions.
        """
        cols = ["start", "on_count", "off_count", "total_transitions"]
        if self.series.empty:
            return pd.DataFrame(columns=cols)

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = s[self.value_column].fillna(False).astype(bool)
        s["transition"] = s["state"] != s["state"].shift()
        s["to_on"] = s["transition"] & s["state"]
        s["to_off"] = s["transition"] & ~s["state"]
        s = s.set_index(self.time_column)

        resampled = s.resample(window).agg(
            on_count=("to_on", "sum"),
            off_count=("to_off", "sum"),
        )
        resampled["total_transitions"] = resampled["on_count"] + resampled["off_count"]

        result = resampled.reset_index().rename(columns={self.time_column: "start"})
        result["on_count"] = result["on_count"].astype(int)
        result["off_count"] = result["off_count"].astype(int)
        result["total_transitions"] = result["total_transitions"].astype(int)

        return result[cols]

    def excessive_cycling(
        self, max_transitions: int = 20, window: str = "1h"
    ) -> pd.DataFrame:
        """Flag windows where transition count exceeds threshold.

        Excessive cycling indicates hunting, instability, or wear risk.

        Args:
            max_transitions: Threshold for flagging.
            window: Resample window.

        Returns:
            DataFrame with columns: start, transition_count,
            avg_on_duration, avg_off_duration.
        """
        cols = [
            "start",
            "transition_count",
            "avg_on_duration",
            "avg_off_duration",
        ]

        counts = self.cycle_count(window)
        if counts.empty:
            return pd.DataFrame(columns=cols)

        excessive = counts[counts["total_transitions"] >= max_transitions].copy()
        if excessive.empty:
            return pd.DataFrame(columns=cols)

        intervals = self.on_off_intervals()
        window_td = pd.to_timedelta(window)

        events: list[dict[str, Any]] = []
        for _, row in excessive.iterrows():
            w_start = row["start"]
            w_end = w_start + window_td

            win_intervals = intervals[
                (intervals["start"] >= w_start) & (intervals["start"] < w_end)
            ]

            on_intervals = win_intervals[win_intervals["state"] == "on"]
            off_intervals = win_intervals[win_intervals["state"] == "off"]

            avg_on = (
                on_intervals["duration"].mean().total_seconds()
                if not on_intervals.empty
                else 0.0
            )
            avg_off = (
                off_intervals["duration"].mean().total_seconds()
                if not off_intervals.empty
                else 0.0
            )

            events.append(
                {
                    "start": w_start,
                    "transition_count": int(row["total_transitions"]),
                    "avg_on_duration": round(avg_on, 2),
                    "avg_off_duration": round(avg_off, 2),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
