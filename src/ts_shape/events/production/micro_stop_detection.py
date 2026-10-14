import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.events._windows import window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import state_intervals

logger = logging.getLogger(__name__)


class MicroStopEvents(Base):
    """Production: Micro-Stop Detection

    Detect brief idle intervals that individually seem harmless but
    accumulate into significant availability losses.

    Methods:
    - detect_micro_stops: Find idle intervals shorter than max_duration.
    - micro_stop_frequency: Count micro-stops per time window.
    - micro_stop_impact: Time lost to micro-stops per window.
    - micro_stop_patterns: Group micro-stops by hour-of-day.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        run_state_uuid: str,
        *,
        event_uuid: str = "prod:micro_stop",
        value_column: str = "value_bool",
        time_column: str = "systime",
        interval_end: str = "last_sample",
        data_end: pd.Timestamp | None = None,
    ) -> None:
        """Initialize the micro-stop detector.

        Args:
            interval_end: How idle/run intervals are closed. ``"last_sample"``
                (default) ends each interval at its last sample -- backward
                compatible, but a change-encoded (``is_delta``) run-state signal
                logs one row per transition, so every interval collapses to
                zero duration and no lost time is reported. ``"next_change"``
                closes each interval at the next transition, correctly
                measuring change-encoded signals.
            data_end: Optional bound for the final, still-open interval when
                ``interval_end="next_change"``.
        """
        super().__init__(dataframe, time_column=time_column)
        self.run_state_uuid = run_state_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.interval_end = interval_end
        self.data_end = data_end

        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    def _intervalize(self) -> pd.DataFrame:
        """Convert boolean state signal to run/idle intervals.

        Intervals are closed according to the ``interval_end`` set on the
        instance (see :func:`ts_shape.utils.intervals.state_intervals`).
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
                "state": ["run" if bool(st) else "idle" for st in intervals["state"]],
                "duration": pd.to_timedelta(intervals["duration_seconds"], unit="s"),
            }
        )

    @keyword_only(1)
    def detect_micro_stops(
        self,
        max_duration: str = "30s",
        min_duration: str = "0s",
    ) -> pd.DataFrame:
        """Find idle intervals shorter than max_duration.

        Args:
            max_duration: Maximum duration to qualify as a micro-stop.
            min_duration: Minimum duration to include (filter very short glitches).

        Returns:
            DataFrame with columns: start, end, duration_seconds (float),
            uuid (event uuid), source_uuid (run-state uuid), duration
            (Timedelta), preceding_run_duration.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "duration",
            "preceding_run_duration",
        ]
        intervals = self._intervalize()
        if intervals.empty:
            return pd.DataFrame(columns=cols)

        max_td = pd.to_timedelta(max_duration)
        min_td = pd.to_timedelta(min_duration)

        events: list[dict[str, Any]] = []
        for i, row in intervals.iterrows():
            if row["state"] != "idle":
                continue
            dur = row["duration"]
            if dur > max_td or dur < min_td:
                continue

            # Find preceding run duration
            preceding_run = pd.Timedelta(0)
            if i > 0:
                prev = intervals.iloc[i - 1]
                if prev["state"] == "run":
                    preceding_run = prev["duration"]

            events.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "duration_seconds": (row["end"] - row["start"]).total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": self.run_state_uuid,
                    "duration": dur,
                    "preceding_run_duration": preceding_run,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    @keyword_only(1)
    def micro_stop_frequency(
        self, window: str = "1h", max_duration: str = "30s"
    ) -> pd.DataFrame:
        """Count micro-stops per time window.

        Args:
            window: Resample window.
            max_duration: Maximum idle duration to qualify as micro-stop.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, count, total_lost_time,
            pct_of_window.
        """
        cols = ["start", "count", "total_lost_time", "pct_of_window"]
        stops = self.detect_micro_stops(max_duration=max_duration)
        if stops.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        window_td = pd.to_timedelta(window)
        stops["duration_seconds"] = stops["duration"].dt.total_seconds()
        stops_indexed = stops.drop(columns=["end", "uuid", "source_uuid"]).set_index(
            "start"
        )

        resampled = stops_indexed.resample(window).agg(
            count=("duration_seconds", "count"),
            total_lost_seconds=("duration_seconds", "sum"),
        )

        resampled["pct_of_window"] = round(
            resampled["total_lost_seconds"] / window_td.total_seconds() * 100, 2
        )

        result = resampled.reset_index()
        result = result.rename(
            columns={
                "start": "start",
                "total_lost_seconds": "total_lost_time",
            }
        )

        return self._with_window(result[cols], window)

    @keyword_only(1)
    def micro_stop_impact(
        self, window: str = "1h", max_duration: str = "30s"
    ) -> pd.DataFrame:
        """Time lost to micro-stops vs total available time per window.

        Args:
            window: Resample window.
            max_duration: Maximum idle duration to qualify as micro-stop.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, total_run_time,
            total_micro_stop_time, availability_loss_pct.
        """
        cols = [
            "start",
            "total_run_time",
            "total_micro_stop_time",
            "availability_loss_pct",
        ]

        intervals = self._intervalize()
        if intervals.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        max_td = pd.to_timedelta(max_duration)
        _window_td = pd.to_timedelta(window)

        intervals["dur_seconds"] = intervals["duration"].dt.total_seconds()
        intervals["is_micro_stop"] = (intervals["state"] == "idle") & (
            intervals["duration"] <= max_td
        )
        intervals["is_run"] = intervals["state"] == "run"

        intervals_indexed = intervals.set_index("start")

        run_time = (
            intervals_indexed.loc[intervals_indexed["is_run"], "dur_seconds"]
            .resample(window)
            .sum()
        )
        micro_time = (
            intervals_indexed.loc[intervals_indexed["is_micro_stop"], "dur_seconds"]
            .resample(window)
            .sum()
        )

        combined = pd.DataFrame(
            {
                "total_run_time": run_time,
                "total_micro_stop_time": micro_time,
            }
        ).fillna(0)

        total_active = combined["total_run_time"] + combined["total_micro_stop_time"]
        combined["availability_loss_pct"] = np.where(
            total_active > 0,
            round(combined["total_micro_stop_time"] / total_active * 100, 2),
            0.0,
        )

        result = combined.reset_index()
        result = result.rename(columns={"start": "start"})
        return self._with_window(result[cols], window)

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
        out.insert(4, "source_uuid", self.run_state_uuid)
        return out

    @keyword_only(1)
    def micro_stop_patterns(
        self, hour_grouping: bool = True, max_duration: str = "30s"
    ) -> pd.DataFrame:
        """Group micro-stops by hour-of-day to find clustering patterns.

        Args:
            hour_grouping: If True, group by hour. If False, group by shift (8h blocks).
            max_duration: Maximum idle duration to qualify as micro-stop.

        Returns:
            DataFrame with columns: hour (or shift), avg_count, avg_lost_time.
        """
        stops = self.detect_micro_stops(max_duration=max_duration)
        if stops.empty:
            if hour_grouping:
                return pd.DataFrame(columns=["hour", "avg_count", "avg_lost_time"])
            else:
                return pd.DataFrame(columns=["shift", "avg_count", "avg_lost_time"])

        stops["duration_seconds"] = stops["duration"].dt.total_seconds()

        if hour_grouping:
            stops["group"] = stops["start"].dt.hour
            stops["date"] = stops["start"].dt.date

            daily_counts = (
                stops.groupby(["date", "group"])
                .agg(
                    count=("duration_seconds", "count"),
                    lost_time=("duration_seconds", "sum"),
                )
                .reset_index()
            )

            result = (
                daily_counts.groupby("group")
                .agg(
                    avg_count=("count", "mean"),
                    avg_lost_time=("lost_time", "mean"),
                )
                .reset_index()
                .rename(columns={"group": "hour"})
            )

            result["avg_count"] = result["avg_count"].round(2)
            result["avg_lost_time"] = result["avg_lost_time"].round(2)
            return result
        else:

            def assign_shift(hour: int) -> str:
                if 6 <= hour < 14:
                    return "morning"
                elif 14 <= hour < 22:
                    return "afternoon"
                else:
                    return "night"

            stops["group"] = stops["start"].dt.hour.apply(assign_shift)
            stops["date"] = stops["start"].dt.date

            daily_counts = (
                stops.groupby(["date", "group"])
                .agg(
                    count=("duration_seconds", "count"),
                    lost_time=("duration_seconds", "sum"),
                )
                .reset_index()
            )

            result = (
                daily_counts.groupby("group")
                .agg(
                    avg_count=("count", "mean"),
                    avg_lost_time=("lost_time", "mean"),
                )
                .reset_index()
                .rename(columns={"group": "shift"})
            )

            result["avg_count"] = result["avg_count"].round(2)
            result["avg_lost_time"] = result["avg_lost_time"].round(2)
            return result
