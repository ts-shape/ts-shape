import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._windows import labeled_window_summary
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs

logger = logging.getLogger(__name__)


class TimeToThresholdEvents(Base):
    """Engineering: Time-to-Threshold Prediction

    Answer the question "when will this signal hit X at the current rate?"
    by extrapolating recent trends.  Complements
    :class:`ThresholdMonitoringEvents` (which detects *when you crossed* a
    threshold) with *predictive* analysis.

    Methods:
    - time_to_threshold: Estimate time until the signal reaches a value.
    - time_to_threshold_windows: Per-window time-to-threshold estimates.
    - remaining_useful_range: Time until signal leaves an acceptable band.
    - crossing_forecast: Forecast crossing times for multiple thresholds.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:time_to_threshold",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the time-to-threshold predictor.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            signal_uuid: UUID of the signal to analyze.
            event_uuid: UUID stamped on emitted events.
            value_column: Column holding the signal values.
            time_column: Timestamp column.
        """
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

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @keyword_only(2)
    def time_to_threshold(
        self,
        threshold: float,
        lookback: str = "1h",
        direction: str = "auto",
    ) -> pd.DataFrame:
        """Estimate time until the signal reaches a threshold value.

        Fits a linear trend to the most recent ``lookback`` window and
        extrapolates to find when the threshold would be crossed.

        Args:
            threshold: Target value to predict crossing time for.
            lookback: How far back to look for trend estimation.
            direction: ``'rising'``, ``'falling'``, or ``'auto'``
                (infer from slope sign).

        Returns:
            Single-row DataFrame with columns: current_value, slope_per_second,
            threshold, estimated_seconds, estimated_time, confidence,
            direction.
        """
        cols = [
            "current_value",
            "slope_per_second",
            "threshold",
            "estimated_seconds",
            "estimated_time",
            "confidence",
            "direction",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        sig = self._recent_window(lookback)
        if len(sig) < 2:
            return pd.DataFrame(columns=cols)

        slope, _intercept, r_squared = self._fit_trend(sig)
        current_value = float(sig[self.value_column].iloc[-1])
        current_time = sig[self.time_column].iloc[-1]

        # Determine direction
        if direction == "auto":
            direction = "rising" if slope > 0 else "falling"

        # Check if trend is heading toward or away from threshold
        if slope == 0:
            return pd.DataFrame(
                [
                    {
                        "current_value": float(current_value),
                        "slope_per_second": 0.0,
                        "threshold": threshold,
                        "estimated_seconds": None,
                        "estimated_time": None,
                        "confidence": float(r_squared),
                        "direction": "flat",
                    }
                ],
                columns=cols,
            )

        # Time to reach threshold from current position
        seconds_to_threshold = (threshold - current_value) / slope

        if seconds_to_threshold < 0:
            # Signal is moving away from threshold
            return pd.DataFrame(
                [
                    {
                        "current_value": float(current_value),
                        "slope_per_second": float(slope),
                        "threshold": threshold,
                        "estimated_seconds": None,
                        "estimated_time": None,
                        "confidence": float(r_squared),
                        "direction": direction,
                    }
                ],
                columns=cols,
            )

        estimated_time = current_time + pd.Timedelta(seconds=seconds_to_threshold)

        return pd.DataFrame(
            [
                {
                    "current_value": float(current_value),
                    "slope_per_second": float(slope),
                    "threshold": threshold,
                    "estimated_seconds": float(seconds_to_threshold),
                    "estimated_time": estimated_time,
                    "confidence": float(r_squared),
                    "direction": direction,
                }
            ],
            columns=cols,
        )

    @keyword_only(2)
    def time_to_threshold_windows(
        self,
        threshold: float,
        window: str = "1h",
        lookback: str = "1h",
    ) -> pd.DataFrame:
        """Per-window time-to-threshold estimates.

        For each window, fits a trend to the preceding ``lookback``
        period and estimates when the threshold will be crossed.

        Args:
            threshold: Target value.
            window: Analysis window frequency.
            lookback: Lookback period for trend estimation.

        Returns:
            Summary DataFrame with columns: start, end, duration_seconds
            (the window bounds), window_start (the resample label; a
            deprecated twin of ``start``), current_value,
            slope_per_second, estimated_seconds, confidence, direction.
        """
        cols = [
            "window_start",
            "current_value",
            "slope_per_second",
            "estimated_seconds",
            "confidence",
            "direction",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return labeled_window_summary([], cols, window)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .set_index(self.time_column)
        )
        lookback_td = pd.to_timedelta(lookback)

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            if group.empty:
                continue

            # Use lookback window ending at this window's end
            window_end = ts + pd.to_timedelta(window)
            lookback_start = window_end - lookback_td
            lookback_data = sig.loc[lookback_start:window_end]

            if len(lookback_data) < 2:
                continue

            lookback_df = lookback_data.reset_index()
            slope, _, r_squared = self._fit_trend(lookback_df)
            current_value = float(lookback_data[self.value_column].iloc[-1])

            direction = "rising" if slope > 0 else ("falling" if slope < 0 else "flat")

            est_seconds = None
            if slope != 0:
                secs = (threshold - current_value) / slope
                if secs > 0:
                    est_seconds = float(secs)

            events.append(
                {
                    "window_start": ts,
                    "current_value": float(current_value),
                    "slope_per_second": float(slope),
                    "estimated_seconds": est_seconds,
                    "confidence": float(r_squared),
                    "direction": direction,
                }
            )

        return labeled_window_summary(events, cols, window)

    @keyword_only(3)
    @renamed_kwargs(lower_bound="lower_limit", upper_bound="upper_limit")
    def remaining_useful_range(
        self,
        lower_limit: float,
        upper_limit: float,
        lookback: str = "1h",
    ) -> pd.DataFrame:
        """Estimate time until the signal leaves an acceptable operating band.

        Args:
            lower_limit: Lower acceptable limit.
            upper_limit: Upper acceptable limit.
            lookback: Lookback period for trend estimation.

        Returns:
            Single-row DataFrame with columns: current_value,
            slope_per_second, nearest_bound, nearest_bound_value,
            estimated_seconds, estimated_time, confidence.
        """
        cols = [
            "current_value",
            "slope_per_second",
            "nearest_bound",
            "nearest_bound_value",
            "estimated_seconds",
            "estimated_time",
            "confidence",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        sig = self._recent_window(lookback)
        if len(sig) < 2:
            return pd.DataFrame(columns=cols)

        slope, _, r_squared = self._fit_trend(sig)
        current_value = float(sig[self.value_column].iloc[-1])
        current_time = sig[self.time_column].iloc[-1]

        # Determine which bound the signal is heading toward
        if slope > 0:
            target_bound = upper_limit
            bound_name = "upper"
        elif slope < 0:
            target_bound = lower_limit
            bound_name = "lower"
        else:
            return pd.DataFrame(
                [
                    {
                        "current_value": float(current_value),
                        "slope_per_second": 0.0,
                        "nearest_bound": "none",
                        "nearest_bound_value": None,
                        "estimated_seconds": None,
                        "estimated_time": None,
                        "confidence": float(r_squared),
                    }
                ],
                columns=cols,
            )

        seconds_to_bound = (target_bound - current_value) / slope
        if seconds_to_bound < 0:
            # Already outside the band
            return pd.DataFrame(
                [
                    {
                        "current_value": float(current_value),
                        "slope_per_second": float(slope),
                        "nearest_bound": bound_name,
                        "nearest_bound_value": target_bound,
                        "estimated_seconds": 0.0,
                        "estimated_time": current_time,
                        "confidence": float(r_squared),
                    }
                ],
                columns=cols,
            )

        estimated_time = current_time + pd.Timedelta(seconds=seconds_to_bound)

        return pd.DataFrame(
            [
                {
                    "current_value": float(current_value),
                    "slope_per_second": float(slope),
                    "nearest_bound": bound_name,
                    "nearest_bound_value": target_bound,
                    "estimated_seconds": float(seconds_to_bound),
                    "estimated_time": estimated_time,
                    "confidence": float(r_squared),
                }
            ],
            columns=cols,
        )

    @keyword_only(2)
    def crossing_forecast(
        self,
        thresholds: dict[str, float],
        lookback: str = "1h",
    ) -> pd.DataFrame:
        """Forecast crossing times for multiple named thresholds.

        Args:
            thresholds: Dict mapping names to values, e.g.
                ``{'warning': 80, 'alarm': 90, 'critical': 95}``.
            lookback: Lookback period for trend estimation.

        Returns:
            DataFrame with columns: threshold_name, threshold_value,
            estimated_seconds, estimated_time, reachable.
        """
        cols = [
            "threshold_name",
            "threshold_value",
            "estimated_seconds",
            "estimated_time",
            "reachable",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        sig = self._recent_window(lookback)
        if len(sig) < 2:
            return pd.DataFrame(columns=cols)

        slope, _, _ = self._fit_trend(sig)
        current_value = float(sig[self.value_column].iloc[-1])
        current_time = sig[self.time_column].iloc[-1]

        events: list[dict[str, Any]] = []
        for name, value in thresholds.items():
            if slope == 0:
                events.append(
                    {
                        "threshold_name": name,
                        "threshold_value": value,
                        "estimated_seconds": None,
                        "estimated_time": None,
                        "reachable": False,
                    }
                )
                continue

            secs = (value - current_value) / slope
            reachable = secs > 0
            events.append(
                {
                    "threshold_name": name,
                    "threshold_value": value,
                    "estimated_seconds": float(secs) if reachable else None,
                    "estimated_time": (
                        current_time + pd.Timedelta(seconds=secs) if reachable else None
                    ),
                    "reachable": reachable,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recent_window(self, lookback: str) -> pd.DataFrame:
        """Return the most recent lookback window of data."""
        if self.signal.empty:
            return self.signal
        lookback_td = pd.to_timedelta(lookback)
        end_time = self.signal[self.time_column].iloc[-1]
        start_time = end_time - lookback_td
        return self.signal[self.signal[self.time_column] >= start_time].copy()

    def _fit_trend(self, sig: pd.DataFrame) -> tuple:
        """Fit a linear trend and return (slope_per_second, intercept, r_squared)."""
        times = sig[self.time_column].values
        values = sig[self.value_column].values

        # Convert timestamps to seconds from start
        t0 = times[0]
        t_seconds = (pd.to_datetime(times) - pd.Timestamp(t0)).total_seconds().values

        # Remove NaN
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return 0.0, float(values[mask][0]) if mask.any() else 0.0, 0.0

        t_clean = t_seconds[mask]
        v_clean = values[mask]

        # Linear regression
        t_mean = t_clean.mean()
        v_mean = v_clean.mean()
        ss_tt = np.sum((t_clean - t_mean) ** 2)

        if ss_tt == 0:
            return 0.0, v_mean, 0.0

        slope = np.sum((t_clean - t_mean) * (v_clean - v_mean)) / ss_tt
        intercept = v_mean - slope * t_mean

        # R-squared
        v_pred = slope * t_clean + intercept
        ss_res = np.sum((v_clean - v_pred) ** 2)
        ss_tot = np.sum((v_clean - v_mean) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return float(slope), float(intercept), float(max(0.0, r_squared))
