import logging
from typing import Any

import pandas as pd  # type: ignore

from ts_shape.events._output import empty_event_df
from ts_shape.events._windows import window_bounds
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
            DataFrame with columns: start, end, duration_seconds (float),
            uuid (event uuid), source_uuid (signal uuid), state (on/off),
            duration (Timedelta).
        """
        cols = ["state", "duration"]
        if self.series.empty:
            return empty_event_df("interval", cols)

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
            return empty_event_df("interval", cols)

        out = pd.DataFrame(
            {
                "start": intervals["start"].to_numpy(),
                "end": intervals["end"].to_numpy(),
                "state": ["on" if bool(st) else "off" for st in intervals["state"]],
                "duration": pd.to_timedelta(intervals["duration_seconds"], unit="s"),
            }
        )
        out.insert(
            2,
            "duration_seconds",
            (
                pd.to_datetime(out["end"]) - pd.to_datetime(out["start"])
            ).dt.total_seconds(),
        )
        out.insert(3, "uuid", self.event_uuid)
        out.insert(4, "source_uuid", self.signal_uuid)
        return out

    def duty_cycle_per_window(
        self, window: str = "1h", *, time_weighted: bool = False
    ) -> pd.DataFrame:
        """Percentage of time the signal is True per window.

        Args:
            window: Resample window.
            time_weighted: When False (default), the duty cycle is the mean of
                the boolean samples in the window -- correct only under uniform
                sampling. On change-encoded / irregular signals the result is
                driven by sample counts (``On@09:00, Off@09:59`` reads 50 %).
                When True, each sample is weighted by the gap to the next sample,
                so the duty cycle reflects wall-clock on-time; ``on_time`` and
                ``off_time`` are then the actual covered seconds in the window.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, on_time, off_time,
            duty_cycle_pct.
        """
        cols = ["start", "on_time", "off_time", "duty_cycle_pct"]
        if self.series.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        s = self.series[[self.time_column, self.value_column]].copy()
        s[self.time_column] = pd.to_datetime(s[self.time_column])
        s = s.sort_values(self.time_column)
        s["state"] = s[self.value_column].fillna(False).astype(float)

        events: list[dict[str, Any]] = []
        if time_weighted:
            s["dur"] = (
                s[self.time_column].shift(-1) - s[self.time_column]
            ).dt.total_seconds()
            s = s[s["dur"].notna()]
            if s.empty:
                return self._with_window(pd.DataFrame(columns=cols), window)
            s["win"] = s[self.time_column].dt.floor(window)
            s["on_dur"] = s["state"] * s["dur"]
            grouped = s.groupby("win").agg(on=("on_dur", "sum"), total=("dur", "sum"))
            for ts, row in grouped.iterrows():
                total = row["total"]
                on_time = row["on"]
                pct = (on_time / total) if total > 0 else 0.0
                events.append(
                    {
                        "start": ts,
                        "on_time": round(on_time, 2),
                        "off_time": round(total - on_time, 2),
                        "duty_cycle_pct": round(pct * 100, 2),
                    }
                )
        else:
            window_td = pd.to_timedelta(window)
            resampled = s.set_index(self.time_column)["state"].resample(window).mean()
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

        return self._with_window(pd.DataFrame(events, columns=cols), window)

    def cycle_count(self, window: str = "1h") -> pd.DataFrame:
        """Number of on/off transitions per window.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, on_count, off_count,
            total_transitions.
        """
        cols = ["start", "on_count", "off_count", "total_transitions"]
        if self.series.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

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

        return self._with_window(result[cols], window)

    def excessive_cycling(
        self, max_transitions: int = 20, window: str = "1h"
    ) -> pd.DataFrame:
        """Flag windows where transition count exceeds threshold.

        Excessive cycling indicates hunting, instability, or wear risk.

        Args:
            max_transitions: Threshold for flagging.
            window: Resample window.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, transition_count,
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
            return self._with_window(pd.DataFrame(columns=cols), window)

        excessive = counts[counts["total_transitions"] >= max_transitions].copy()
        if excessive.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

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

        return self._with_window(pd.DataFrame(events, columns=cols), window)

    def _with_window(self, df: pd.DataFrame, window: str) -> pd.DataFrame:
        """Insert end (start + window), duration_seconds, uuid and source_uuid."""
        out = df.copy()
        if out.empty:
            out.insert(1, "end", pd.Series(dtype="object"))
            out.insert(2, "duration_seconds", pd.Series(dtype="float64"))
        else:
            start, end = window_bounds(out["start"], window)
            out["start"] = start
            out.insert(1, "end", end)
            out.insert(2, "duration_seconds", (out["end"] - start).dt.total_seconds())
        out.insert(3, "uuid", self.event_uuid)
        out.insert(4, "source_uuid", self.signal_uuid)
        return out
