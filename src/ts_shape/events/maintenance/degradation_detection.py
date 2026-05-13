import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DegradationDetectionEvents(Base):
    """
    Detect degradation patterns in time series signals: trend degradation,
    variance increases, level shifts, and composite health scores.

    Designed for predictive maintenance on manufacturing/industrial IoT data.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "maint:degradation",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # Isolate signal series and ensure proper dtypes/sort
        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    def detect_trend_degradation(
        self,
        window: str = "1h",
        min_slope: float = 0.0,
        direction: str = "decreasing",
    ) -> pd.DataFrame:
        """
        Detect intervals where a rolling linear regression slope exceeds min_slope
        in the given direction, indicating trend-based degradation.

        Args:
            window: Rolling window size (e.g. '1h', '30m').
            min_slope: Minimum absolute slope to consider degradation.
            direction: 'decreasing' (negative slope) or 'increasing' (positive slope).

        Returns:
            DataFrame with columns: start, end, uuid, is_delta, avg_slope,
            total_change, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "avg_slope",
            "total_change",
            "duration_seconds",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        sig["t_seconds"] = (
            sig[self.time_column] - sig[self.time_column].iloc[0]
        ).dt.total_seconds()

        window_td = pd.to_timedelta(window)
        window_seconds = window_td.total_seconds()

        # Compute rolling slope using time-based windows
        slopes = []
        for i in range(len(sig)):
            t_end = sig[self.time_column].iloc[i]
            t_start = t_end - window_td
            mask = (sig[self.time_column] > t_start) & (sig[self.time_column] <= t_end)
            win = sig.loc[mask]
            if len(win) < 2:
                slopes.append(np.nan)
                continue
            x = win["t_seconds"].values
            y = win[self.value_column].values
            try:
                coeffs = np.polyfit(x, y, 1)
                slopes.append(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                logger.debug("Polyfit failed for degradation slope window.")
                slopes.append(np.nan)

        sig["slope"] = slopes

        # Filter by direction and min_slope
        if direction == "decreasing":
            degrading = sig["slope"] <= -abs(min_slope)
        else:
            degrading = sig["slope"] >= abs(min_slope)

        degrading = degrading.fillna(False)

        if not degrading.any():
            return pd.DataFrame(columns=cols)

        # Group contiguous degrading intervals
        group_id = (degrading != degrading.shift()).cumsum()
        events: List[Dict[str, Any]] = []
        for gid, seg in sig.groupby(group_id):
            seg_deg = degrading.loc[seg.index]
            if not seg_deg.iloc[0]:
                continue
            start_time = seg[self.time_column].iloc[0]
            end_time = seg[self.time_column].iloc[-1]
            avg_slope = float(seg["slope"].mean())
            total_change = float(
                seg[self.value_column].iloc[-1] - seg[self.value_column].iloc[0]
            )
            duration = (end_time - start_time).total_seconds()
            events.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "avg_slope": avg_slope,
                    "total_change": total_change,
                    "duration_seconds": duration,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_variance_increase(
        self,
        window: str = "1h",
        threshold_factor: float = 2.0,
    ) -> pd.DataFrame:
        """
        Compare rolling variance against a baseline (first window) and flag
        intervals where the ratio exceeds threshold_factor.

        Args:
            window: Rolling window size.
            threshold_factor: Minimum variance ratio to flag.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            baseline_variance, current_variance, ratio.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "baseline_variance",
            "current_variance",
            "ratio",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        window_td = pd.to_timedelta(window)

        # Compute baseline variance from first window
        first_time = sig[self.time_column].iloc[0]
        baseline_mask = sig[self.time_column] <= first_time + window_td
        baseline_data = sig.loc[baseline_mask, self.value_column]
        if len(baseline_data) < 2:
            return pd.DataFrame(columns=cols)
        baseline_var = float(baseline_data.var())
        if baseline_var == 0:
            baseline_var = np.finfo(float).eps

        # Compute rolling variance using time-based windows
        variances = []
        for i in range(len(sig)):
            t_end = sig[self.time_column].iloc[i]
            t_start = t_end - window_td
            mask = (sig[self.time_column] > t_start) & (sig[self.time_column] <= t_end)
            win = sig.loc[mask, self.value_column]
            if len(win) < 2:
                variances.append(np.nan)
            else:
                variances.append(float(win.var()))

        sig["current_variance"] = variances
        sig["ratio"] = sig["current_variance"] / baseline_var

        exceeded = (sig["ratio"] >= threshold_factor).fillna(False)
        if not exceeded.any():
            return pd.DataFrame(columns=cols)

        # Group contiguous exceeded intervals
        group_id = (exceeded != exceeded.shift()).cumsum()
        events: List[Dict[str, Any]] = []
        for gid, seg in sig.groupby(group_id):
            seg_exc = exceeded.loc[seg.index]
            if not seg_exc.iloc[0]:
                continue
            events.append(
                {
                    "start": seg[self.time_column].iloc[0],
                    "end": seg[self.time_column].iloc[-1],
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "baseline_variance": baseline_var,
                    "current_variance": float(seg["current_variance"].mean()),
                    "ratio": float(seg["ratio"].mean()),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_level_shift(
        self,
        min_shift: float,
        hold: str = "5m",
    ) -> pd.DataFrame:
        """
        CUSUM-like detection for permanent mean shifts in the signal.

        Args:
            min_shift: Minimum absolute shift magnitude to detect.
            hold: Minimum duration the new level must persist.

        Returns:
            DataFrame with columns: systime, uuid, is_delta,
            shift_magnitude, prev_mean, new_mean.
        """
        cols = [
            "systime",
            "uuid",
            "is_delta",
            "shift_magnitude",
            "prev_mean",
            "new_mean",
        ]
        if self.signal.empty or len(self.signal) < 3:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        hold_td = pd.to_timedelta(hold)

        values = sig[self.value_column].values
        times = sig[self.time_column].values

        events: List[Dict[str, Any]] = []

        # CUSUM-based approach: track cumulative deviation from running mean
        running_mean = values[0]
        cusum_pos = 0.0
        cusum_neg = 0.0
        threshold = abs(min_shift) / 2  # allowable slack
        last_shift_idx = 0

        for i in range(1, len(values)):
            deviation = values[i] - running_mean
            cusum_pos = max(0, cusum_pos + deviation - threshold)
            cusum_neg = max(0, cusum_neg - deviation - threshold)

            if cusum_pos > abs(min_shift) or cusum_neg > abs(min_shift):
                # Potential shift detected; compute pre/post means
                prev_mean = float(np.mean(values[last_shift_idx:i]))
                # Check hold: gather data after shift point
                post_mask = (sig[self.time_column] >= pd.Timestamp(times[i])) & (
                    sig[self.time_column] <= pd.Timestamp(times[i]) + hold_td
                )
                post_data = sig.loc[post_mask, self.value_column]
                if len(post_data) >= 1:
                    new_mean = float(post_data.mean())
                    shift_mag = new_mean - prev_mean
                    if abs(shift_mag) >= abs(min_shift):
                        events.append(
                            {
                                "systime": pd.Timestamp(times[i]),
                                "uuid": self.event_uuid,
                                "is_delta": True,
                                "shift_magnitude": shift_mag,
                                "prev_mean": prev_mean,
                                "new_mean": new_mean,
                            }
                        )
                        # Reset CUSUM and update running mean
                        running_mean = new_mean
                        last_shift_idx = i
                        cusum_pos = 0.0
                        cusum_neg = 0.0

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def health_score(
        self,
        window: str = "1h",
        baseline_window: str = "24h",
    ) -> pd.DataFrame:
        """
        Composite 0-100 health score based on mean drift, variance change,
        and trend slope, computed over rolling windows.

        Args:
            window: Rolling window for current metrics.
            baseline_window: Initial period used to establish baseline behaviour.

        Returns:
            DataFrame with columns: systime, uuid, is_delta,
            health_score, mean_drift_pct, variance_ratio, trend_slope.
        """
        cols = [
            "systime",
            "uuid",
            "is_delta",
            "health_score",
            "mean_drift_pct",
            "variance_ratio",
            "trend_slope",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        sig["t_seconds"] = (
            sig[self.time_column] - sig[self.time_column].iloc[0]
        ).dt.total_seconds()

        window_td = pd.to_timedelta(window)
        baseline_td = pd.to_timedelta(baseline_window)

        # Compute baseline statistics
        first_time = sig[self.time_column].iloc[0]
        baseline_mask = sig[self.time_column] <= first_time + baseline_td
        baseline_data = sig.loc[baseline_mask, self.value_column]

        if len(baseline_data) < 2:
            # Not enough baseline data; return perfect health for all points
            result = sig[[self.time_column]].copy()
            result = result.rename(columns={self.time_column: "systime"})
            result["uuid"] = self.event_uuid
            result["is_delta"] = True
            result["health_score"] = 100.0
            result["mean_drift_pct"] = 0.0
            result["variance_ratio"] = 1.0
            result["trend_slope"] = 0.0
            return result[cols]

        baseline_mean = float(baseline_data.mean())
        baseline_var = float(baseline_data.var())
        if baseline_var == 0:
            baseline_var = np.finfo(float).eps
        if baseline_mean == 0:
            baseline_mean_for_pct = np.finfo(float).eps
        else:
            baseline_mean_for_pct = baseline_mean

        # Compute rolling metrics
        rows: List[Dict[str, Any]] = []
        for i in range(len(sig)):
            t_end = sig[self.time_column].iloc[i]
            t_start = t_end - window_td
            mask = (sig[self.time_column] > t_start) & (sig[self.time_column] <= t_end)
            win = sig.loc[mask]

            if len(win) < 2:
                rows.append(
                    {
                        "systime": t_end,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "health_score": 100.0,
                        "mean_drift_pct": 0.0,
                        "variance_ratio": 1.0,
                        "trend_slope": 0.0,
                    }
                )
                continue

            current_mean = float(win[self.value_column].mean())
            current_var = float(win[self.value_column].var())
            mean_drift_pct = (
                abs(current_mean - baseline_mean) / abs(baseline_mean_for_pct) * 100.0
            )
            variance_ratio = current_var / baseline_var

            # Compute slope
            x = win["t_seconds"].values
            y = win[self.value_column].values
            try:
                coeffs = np.polyfit(x, y, 1)
                trend_slope = float(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                trend_slope = 0.0

            # Composite health score: penalize mean drift, variance increase, trend
            # Each component maps to 0-33.3 penalty
            drift_penalty = min(mean_drift_pct / 100.0, 1.0) * 33.3
            var_penalty = min(max(variance_ratio - 1.0, 0.0) / 5.0, 1.0) * 33.3
            slope_penalty = (
                min(
                    abs(trend_slope) * 3600.0 / (abs(baseline_mean_for_pct) + 1e-9), 1.0
                )
                * 33.4
            )

            score = max(
                0.0, min(100.0, 100.0 - drift_penalty - var_penalty - slope_penalty)
            )

            rows.append(
                {
                    "systime": t_end,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "health_score": round(score, 2),
                    "mean_drift_pct": round(mean_drift_pct, 4),
                    "variance_ratio": round(variance_ratio, 4),
                    "trend_slope": round(trend_slope, 8),
                }
            )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
