import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

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
        super().__init__(dataframe, column_name=time_column)
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

    def multi_level_threshold(
        self, levels: Dict[str, float], direction: str = "above"
    ) -> pd.DataFrame:
        """Detect intervals where signal exceeds configurable threshold levels.

        Args:
            levels: Dict mapping level names to threshold values.
                    e.g. {'warning': 80, 'alarm': 90, 'critical': 95}
            direction: 'above' or 'below'.

        Returns:
            DataFrame with columns: start_time, end_time, duration, level, peak_value.
        """
        cols = ["start_time", "end_time", "duration", "level", "peak_value"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )

        # Sort levels: for 'above', process from highest to lowest
        sorted_levels = sorted(
            levels.items(), key=lambda x: x[1], reverse=(direction == "above")
        )

        all_events: List[Dict[str, Any]] = []

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
                end = seg[self.time_column].iloc[-1]
                peak = float(
                    seg[self.value_column].max()
                    if direction == "above"
                    else seg[self.value_column].min()
                )

                all_events.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "duration": end - start,
                        "level": level_name,
                        "peak_value": peak,
                    }
                )

        return (
            pd.DataFrame(all_events, columns=cols)
            if all_events
            else pd.DataFrame(columns=cols)
        )

    def threshold_with_hysteresis(self, high: float, low: float) -> pd.DataFrame:
        """Alarm intervals with hysteresis to prevent chattering.

        Enter alarm when signal crosses `high`, exit when it drops below `low`.

        Args:
            high: Upper threshold to enter alarm state.
            low: Lower threshold to exit alarm state.

        Returns:
            DataFrame with columns: start_time, end_time, duration, peak_value.
        """
        cols = ["start_time", "end_time", "duration", "peak_value"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        values = sig[self.value_column].values
        times = sig[self.time_column].values

        events: List[Dict[str, Any]] = []
        in_alarm = False
        alarm_start = None
        peak = -np.inf

        for i in range(len(values)):
            v = values[i]
            if not in_alarm:
                if v >= high:
                    in_alarm = True
                    alarm_start = times[i]
                    peak = v
            else:
                peak = max(peak, v)
                if v < low:
                    events.append(
                        {
                            "start_time": pd.Timestamp(alarm_start),
                            "end_time": pd.Timestamp(times[i]),
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
                    "start_time": pd.Timestamp(alarm_start),
                    "end_time": pd.Timestamp(times[-1]),
                    "duration": pd.Timestamp(times[-1]) - pd.Timestamp(alarm_start),
                    "peak_value": float(peak),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def time_above_threshold(
        self, threshold: float, window: str = "1h"
    ) -> pd.DataFrame:
        """Per window: total time and percentage above threshold.

        Args:
            threshold: Value threshold.
            window: Resample window.

        Returns:
            DataFrame with columns: window_start, time_above, pct_above, exceedance_count.
        """
        cols = ["window_start", "time_above", "pct_above", "exceedance_count"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

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

        events: List[Dict[str, Any]] = []
        for ts, row in resampled.iterrows():
            if row["sample_count"] == 0:
                continue
            pct = round(row["above_frac"] * 100, 2)
            time_above_seconds = row["above_frac"] * window_td.total_seconds()
            events.append(
                {
                    "window_start": ts,
                    "time_above": round(time_above_seconds, 2),
                    "pct_above": pct,
                    "exceedance_count": int(row["exceedance_count"]),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def threshold_exceedance_trend(
        self, threshold: float, window: str = "1D"
    ) -> pd.DataFrame:
        """Track exceedance frequency and duration trend over time.

        Args:
            threshold: Value threshold.
            window: Resample window (e.g. '1D' for daily).

        Returns:
            DataFrame with columns: window_start, exceedance_count,
            total_duration, trend_direction.
        """
        cols = ["window_start", "exceedance_count", "total_duration", "trend_direction"]
        time_above = self.time_above_threshold(threshold, window)
        if time_above.empty or len(time_above) < 2:
            return pd.DataFrame(columns=cols)

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

        result = time_above[["window_start", "exceedance_count", "time_above"]].copy()
        result = result.rename(columns={"time_above": "total_duration"})
        result["trend_direction"] = trend

        return result[cols]
