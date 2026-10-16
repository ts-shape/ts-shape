import logging

import numpy as np
import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events._windows import window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import state_intervals
from ts_shape.utils.runs import total_seconds
from ts_shape.utils.shifts import assign_shift
from ts_shape.utils.time_accounting import grouper_periods, time_by_period

logger = logging.getLogger(__name__)


# Fixed shift grid of ``pattern_analysis(by="shift")``.
_HOUR_SHIFTS: dict[str, tuple[str, str]] = {
    "morning": ("06:00", "14:00"),
    "afternoon": ("14:00", "22:00"),
    "night": ("22:00", "06:00"),
}


class MicroStopEvents(Base):
    """Production: Micro-Stop Detection

    Detect brief idle intervals that individually seem harmless but
    accumulate into significant availability losses.

    Methods:
    - detect_micro_stops: Find idle intervals shorter than max_duration.
    - micro_stop_frequency: Count micro-stops per time window.
    - micro_stop_impact: Time lost to micro-stops per window.
    - micro_stop_patterns: Group micro-stops by hour-of-day.

    Example:
        >>> import pandas as pd
        >>> from ts_shape import MicroStopEvents
        >>> df = pd.DataFrame({
        ...     "systime": pd.to_datetime([
        ...         "2025-01-01 08:00:00", "2025-01-01 08:10:00", "2025-01-01 08:10:20",
        ...         "2025-01-01 08:30:00", "2025-01-01 08:30:45", "2025-01-01 09:00:00"]),
        ...     "uuid": "run",
        ...     "value_bool": [True, False, True, False, True, True],
        ... })
        >>> stops = MicroStopEvents(df, "run").detect_micro_stops(max_duration="30s")
        >>> stops[["start", "end", "duration_seconds"]]
                        start                 end  duration_seconds
        0 2025-01-01 08:10:00 2025-01-01 08:10:20              20.0

        The 45 s stop at 08:30 is longer than ``max_duration`` and is not listed.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        run_state_uuid: str,
        *,
        event_uuid: str = "prod:micro_stop",
        value_column: str = "value_bool",
        time_column: str = "systime",
        interval_end: str = "next_change",
        data_end: pd.Timestamp | None = None,
    ) -> None:
        """Initialize the micro-stop detector.

        Args:
            dataframe: Long-format input data.
            run_state_uuid: UUID of the boolean run-state signal (True = running,
                False = idle).
            event_uuid: UUID assigned to generated events.
            value_column: Column holding the run state. Missing values count
                as idle.
            time_column: Timestamp column.
            interval_end: How idle/run intervals are closed.
                ``"next_change"`` (default) closes each interval at the next
                transition, which measures both dense and change-encoded
                (``is_delta``) run-state signals. ``"last_sample"`` ends each
                interval at its last sample; on a change-encoded signal (one
                row per transition) every interval then has zero duration and
                no lost time is reported.
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
        s["state"] = bool_or_false(s[self.value_column])
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

        state = intervals["state"].to_numpy()
        duration = intervals["duration"]
        idle = (state == "idle") & (duration <= max_td).to_numpy()
        idle &= (duration >= min_td).to_numpy()
        if not idle.any():
            return pd.DataFrame(columns=cols)

        # Run time just before each stop: the previous interval when it is a
        # run, else zero.
        prev_run = np.r_[False, state[:-1] == "run"]
        preceding = duration.shift(1).where(prev_run, pd.Timedelta(0))[idle]
        if not prev_run[idle].all():
            # A stop without a run before it gets Timedelta(0), whose
            # nanosecond unit the column then takes (as it always has).
            preceding = preceding.astype("timedelta64[ns]")
        stops = intervals.loc[idle].reset_index(drop=True)
        return pd.DataFrame(
            {
                "start": stops["start"],
                "end": stops["end"],
                "duration_seconds": total_seconds(stops["end"] - stops["start"]),
                "uuid": self.event_uuid,
                "source_uuid": self.run_state_uuid,
                "duration": stops["duration"],
                "preceding_run_duration": preceding.reset_index(drop=True),
            },
            columns=cols,
        )

    @keyword_only(1)
    @twin_columns({"pct_of_window": "share_of_window_pct"})
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

            Canonical twin columns: ``share_of_window_pct`` (after ``pct_of_window``); each
            holds the same values as the deprecated column right before it, which is
            removed in the next major release.
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

        resampled["pct_of_window"] = (
            resampled["total_lost_seconds"] / window_td.total_seconds() * 100
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

        A stop is classified as a micro-stop by its full duration; its time
        and the run time are then split at the window boundaries (the ISO
        22400-2 time model), so a stop across the hour counts in both hours.
        Every window from the first interval to the end of the last one is
        reported (0 when it holds no run or micro-stop time).

        Args:
            window: Resample window.
            max_duration: Maximum idle duration to qualify as micro-stop.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, total_run_time,
            total_micro_stop_time, performance_loss_pct,
            availability_loss_pct.

            ``performance_loss_pct`` is the micro-stop time as a percentage of
            run + micro-stop time. Minor stops are a *performance* loss in
            Nakajima's six big losses (idling and minor stoppages), not an
            availability loss. ``availability_loss_pct`` holds the same value
            under its original, misleading name and is kept for
            compatibility.
        """
        cols = [
            "start",
            "total_run_time",
            "total_micro_stop_time",
            "performance_loss_pct",
            "availability_loss_pct",
        ]

        intervals = self._intervalize()
        if intervals.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        max_td = pd.to_timedelta(max_duration)
        _window_td = pd.to_timedelta(window)

        is_micro = (intervals["state"] == "idle") & (intervals["duration"] <= max_td)
        # A stop is classified by its full duration; its time is then split
        # at the window boundaries with the run time (ISO 22400-2 time model).
        labeled = pd.DataFrame(
            {
                "start": pd.to_datetime(intervals["start"]),
                "end": pd.to_datetime(intervals["end"]),
                "state": intervals["state"].where(~is_micro, "micro"),
            }
        )
        periods = grouper_periods(
            labeled["start"].iloc[0], labeled["end"].max(), window
        )
        seconds = time_by_period(
            labeled, periods, keys=["period"], states=["run", "micro"]
        ).set_index("period")
        first, last = labeled["start"].iloc[0], labeled["end"].max()
        span = periods[
            (periods["end"] > first)
            & ((periods["start"] < last) | (periods["start"] <= first))
        ]
        combined = pd.DataFrame(
            {
                "start": span["period"].to_numpy(),
                "total_run_time": span["period"]
                .map(seconds["run"])
                .fillna(0.0)
                .to_numpy(dtype="float64"),
                "total_micro_stop_time": span["period"]
                .map(seconds["micro"])
                .fillna(0.0)
                .to_numpy(dtype="float64"),
            }
        )

        total_active = combined["total_run_time"] + combined["total_micro_stop_time"]
        combined["availability_loss_pct"] = np.where(
            total_active > 0,
            (combined["total_micro_stop_time"] / total_active * 100),
            0.0,
        )

        # Nakajima: minor stops are a performance loss (twin of the old name).
        combined["performance_loss_pct"] = combined["availability_loss_pct"]
        return self._with_window(combined[cols], window)

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

            return result
        else:

            stops["group"] = assign_shift(stops["start"], _HOUR_SHIFTS)
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

            return result
