import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_summary_df,
)
from ts_shape.events._windows import add_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs
from ts_shape.utils.intervals import INTERVAL_END_MODES, interval_end_times

logger = logging.getLogger(__name__)


class ThresholdMonitoringEvents(Base):
    """Engineering: Threshold Monitoring

    Multi-level threshold monitoring with hysteresis for numeric signals.

    Methods:
    - multi_level_threshold: Intervals exceeding configurable warning/alarm/critical levels.
    - threshold_with_hysteresis: Alarm entry/exit with separate high/low thresholds.
    - time_above_threshold: Time and percentage above a threshold per window.
    - threshold_exceedance_trend: Track exceedance frequency over time.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:threshold",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    @keyword_only(2)
    def multi_level_threshold(
        self,
        levels: dict[str, float],
        direction: str = "above",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Detect intervals where signal exceeds configurable threshold levels.

        Args:
            levels: Dict mapping level names to threshold values.
                    e.g. {'warning': 80, 'alarm': 90, 'critical': 95}
            direction: 'above' or 'below'.
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, duration (Timedelta), level, peak_value.
        """
        cols = ["start", "end", "duration", "level", "peak_value"]
        if self.signal.empty:
            return empty_event_df("interval", cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )

        # Sort levels: for 'above', process from highest to lowest
        sorted_levels = sorted(
            levels.items(), key=lambda x: x[1], reverse=(direction == "above")
        )

        all_events: list[dict[str, Any]] = []
        ends = interval_end_times(sig[self.time_column], interval_end)

        for level_name, threshold in sorted_levels:
            if direction == "above":
                exceeded = sig[self.value_column] >= threshold
            else:
                exceeded = sig[self.value_column] <= threshold

            if not exceeded.any():
                continue

            groups = (exceeded != exceeded.shift()).cumsum()
            for _, seg in sig.groupby(groups):
                seg_exc = exceeded.loc[seg.index]
                if not seg_exc.iloc[0]:
                    continue
                start = seg[self.time_column].iloc[0]
                end = ends.loc[seg.index[-1]]
                peak = float(
                    seg[self.value_column].max()
                    if direction == "above"
                    else seg[self.value_column].min()
                )

                all_events.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "level": level_name,
                        "peak_value": peak,
                    }
                )

        if not all_events:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(all_events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @renamed_kwargs(high="upper_limit", low="lower_limit")
    def threshold_with_hysteresis(
        self, upper_limit: float, lower_limit: float
    ) -> pd.DataFrame:
        """Alarm intervals with hysteresis to prevent chattering.

        Enter alarm when signal crosses `upper_limit`, exit when it drops
        below `lower_limit`.

        Args:
            upper_limit: Upper threshold to enter alarm state.
            lower_limit: Lower threshold to exit alarm state.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, duration (Timedelta), peak_value.
        """
        cols = ["start", "end", "duration", "peak_value"]
        if self.signal.empty:
            return empty_event_df("interval", cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        values = sig[self.value_column].values
        times = sig[self.time_column].array

        events: list[dict[str, Any]] = []
        in_alarm = False
        alarm_start = None
        peak = -np.inf

        for i in range(len(values)):
            v = values[i]
            if not in_alarm:
                if v >= upper_limit:
                    in_alarm = True
                    alarm_start = times[i]
                    peak = v
            else:
                peak = max(peak, v)
                if v < lower_limit:
                    events.append(
                        {
                            "start": pd.Timestamp(alarm_start),
                            "end": pd.Timestamp(times[i]),
                            "duration": pd.Timestamp(times[i])
                            - pd.Timestamp(alarm_start),
                            "peak_value": float(peak),
                        }
                    )
                    in_alarm = False
                    peak = -np.inf

        # Close open alarm at end of data
        if in_alarm and alarm_start is not None:
            events.append(
                {
                    "start": pd.Timestamp(alarm_start),
                    "end": pd.Timestamp(times[-1]),
                    "duration": pd.Timestamp(times[-1]) - pd.Timestamp(alarm_start),
                    "peak_value": float(peak),
                }
            )

        if not events:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @keyword_only(2)
    def time_above_threshold(
        self,
        threshold: float,
        window: str = "1h",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Per window: total time and percentage above threshold.

        Args:
            threshold: Value threshold.
            window: Resample window.
            interval_end: ``"next_change"`` (default) holds each value until
                the next sample and measures the real time above the threshold
                inside each window (``pct_above`` is relative to the window
                length). ``"last_sample"`` estimates the time as the fraction of
                *samples* above the threshold times the window length, which is
                only right for uniform sampling.

        Returns:
            DataFrame with columns: start, end, duration_seconds, time_above,
            pct_above, exceedance_count. ``end`` is ``start`` + ``window``.
        """
        cols = ["start", "time_above", "pct_above", "exceedance_count"]
        if self.signal.empty:
            return empty_event_df("summary", cols[1:])

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)
        sig["above"] = (sig[self.value_column] >= threshold).astype(float)

        # Count transitions into above state
        sig["enter_above"] = (sig["above"].diff() == 1).astype(int)

        window_td = pd.to_timedelta(window)
        resampled = sig.resample(window).agg(
            above_frac=("above", "mean"),
            exceedance_count=("enter_above", "sum"),
            sample_count=("above", "count"),
        )
        if interval_end not in INTERVAL_END_MODES:
            raise ValueError(
                f"interval_end must be one of {INTERVAL_END_MODES}, got {interval_end!r}"
            )
        if interval_end == "next_change":
            resampled["above_frac"] = self._time_weighted_above_fraction(
                sig["above"], resampled.index, window_td
            )

        events: list[dict[str, Any]] = []
        for ts, row in resampled.iterrows():
            if row["sample_count"] == 0:
                continue
            pct = float(row["above_frac"] * 100)
            time_above_seconds = row["above_frac"] * window_td.total_seconds()
            events.append(
                {
                    "start": ts,
                    "time_above": float(time_above_seconds),
                    "pct_above": pct,
                    "exceedance_count": int(row["exceedance_count"]),
                }
            )

        if not events:
            return empty_event_df("summary", cols[1:])
        return finalize_summary_df(
            add_window_bounds(pd.DataFrame(events, columns=cols), window)
        )

    @staticmethod
    def _time_weighted_above_fraction(
        above: pd.Series, window_starts: pd.Index, window_td: pd.Timedelta
    ) -> list[float]:
        """Fraction of each window spent above, holding values to the next sample."""
        times = above.index.to_series().reset_index(drop=True)
        seg_end = times.shift(-1).fillna(times)
        flags = above.reset_index(drop=True).astype(bool)
        starts, stops = times[flags], seg_end[flags]
        fractions: list[float] = []
        for w_start in window_starts:
            w_end = w_start + window_td
            overlap = (
                stops.clip(upper=w_end) - starts.clip(lower=w_start)
            ).dt.total_seconds()
            fractions.append(
                float(overlap.clip(lower=0).sum()) / window_td.total_seconds()
            )
        return fractions

    @keyword_only(2)
    def threshold_exceedance_trend(
        self, threshold: float, window: str = "1D"
    ) -> pd.DataFrame:
        """Track exceedance frequency and duration trend over time.

        Args:
            threshold: Value threshold.
            window: Resample window (e.g. '1D' for daily).

        Returns:
            DataFrame with columns: start, end, duration_seconds,
            exceedance_count, total_duration, trend_direction. ``end`` is
            ``start`` + ``window``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "exceedance_count",
            "total_duration",
            "trend_direction",
        ]
        time_above = self.time_above_threshold(threshold, window=window)
        if time_above.empty or len(time_above) < 2:
            return empty_event_df("summary", cols)

        counts = time_above["exceedance_count"].values
        # Simple trend: compare second half to first half
        mid = len(counts) // 2
        first_half_avg = float(np.mean(counts[:mid]))
        second_half_avg = float(np.mean(counts[mid:]))

        if second_half_avg > first_half_avg * 1.1:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        result = time_above[
            ["start", "end", "duration_seconds", "exceedance_count", "time_above"]
        ].copy()
        result = result.rename(columns={"time_above": "total_duration"})
        result["trend_direction"] = trend

        return result[cols]
