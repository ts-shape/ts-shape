import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DisturbanceRecoveryEvents(Base):
    """Engineering: Disturbance / Upset Detection and Recovery

    Detect external upsets hitting a process signal and measure how the
    process recovered. Works with a rolling baseline or an optional setpoint.

    Methods:
    - detect_disturbances: Intervals where signal deviates from baseline.
    - recovery_time: How long until signal returns to normal after each upset.
    - disturbance_frequency: Count disturbances per shift/day.
    - before_after_comparison: Did the upset permanently change the process?
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        setpoint_uuid: Optional[str] = None,
        event_uuid: str = "eng:disturbance_recovery",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.setpoint_uuid = setpoint_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

        if setpoint_uuid:
            self._sp = (
                self.dataframe[self.dataframe["uuid"] == setpoint_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            self._sp[self.time_column] = pd.to_datetime(self._sp[self.time_column])
        else:
            self._sp = pd.DataFrame()

    def _compute_baseline_and_deviation(self, baseline_window: str) -> pd.DataFrame:
        """Compute baseline and deviation for the signal."""
        if self.signal.empty:
            return pd.DataFrame(
                columns=[
                    self.time_column,
                    "value",
                    "baseline",
                    "deviation",
                    "rolling_std",
                ]
            )

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.rename(columns={self.value_column: "value"})
        sig = sig.set_index(self.time_column)

        td = pd.Timedelta(baseline_window)

        if not self._sp.empty:
            # Use setpoint as baseline
            sp = (
                self._sp[[self.time_column, self.value_column]]
                .rename(columns={self.value_column: "sp"})
                .set_index(self.time_column)
            )
            merged = pd.merge_asof(
                sig.reset_index().sort_values(self.time_column),
                sp.reset_index().sort_values(self.time_column),
                on=self.time_column,
                direction="nearest",
            ).set_index(self.time_column)
            merged["baseline"] = merged["sp"]
            merged["deviation"] = merged["value"] - merged["baseline"]
            merged["rolling_std"] = (
                merged["value"].rolling(td, min_periods=2).std().fillna(1.0)
            )
            return merged.dropna(subset=["value", "baseline"])
        else:
            # Use rolling mean as baseline
            sig["baseline"] = sig["value"].rolling(td, min_periods=2).mean()
            sig["deviation"] = sig["value"] - sig["baseline"]
            sig["rolling_std"] = (
                sig["value"].rolling(td, min_periods=2).std().fillna(1.0)
            )
            return sig.dropna(subset=["value", "baseline"])

    def detect_disturbances(
        self,
        baseline_window: str = "10m",
        threshold_sigma: float = 3.0,
        min_duration: str = "30s",
    ) -> pd.DataFrame:
        """Detect intervals where signal deviates significantly from baseline.

        Args:
            baseline_window: Window for rolling baseline computation.
            threshold_sigma: Deviation threshold in multiples of rolling std.
            min_duration: Minimum disturbance duration.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            peak_deviation, mean_deviation, direction, duration_seconds,
            disturbance_type.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "peak_deviation",
            "mean_deviation",
            "direction",
            "duration_seconds",
            "disturbance_type",
        ]
        bd = self._compute_baseline_and_deviation(baseline_window)
        if bd.empty:
            return pd.DataFrame(columns=cols)

        exceeded = bd["deviation"].abs() > (threshold_sigma * bd["rolling_std"])
        if not exceeded.any():
            return pd.DataFrame(columns=cols)

        min_td = pd.Timedelta(min_duration)
        groups = (exceeded != exceeded.shift()).cumsum()

        events: List[Dict[str, Any]] = []
        for _, seg_idx in exceeded.groupby(groups):
            if not seg_idx.iloc[0]:
                continue
            seg = bd.loc[seg_idx.index]
            start = seg.index[0]
            end = seg.index[-1]
            dur = end - start
            if dur < min_td:
                continue

            dev = seg["deviation"]
            peak_dev = float(dev.abs().max())
            mean_dev = float(dev.abs().mean())
            direction = "above" if dev.mean() > 0 else "below"
            dur_s = dur.total_seconds()

            # Classify: spike (short), transient (returns), step (doesn't return)
            min_dur_s = pd.Timedelta(min_duration).total_seconds()
            if dur_s < min_dur_s * 3:
                dtype = "spike"
            else:
                # Check if signal returns to baseline after this disturbance
                post = bd.loc[end:]
                if len(post) > 5:
                    post_dev = post["deviation"].iloc[:10].abs().mean()
                    if post_dev < threshold_sigma * bd["rolling_std"].mean() * 0.5:
                        dtype = "transient"
                    else:
                        dtype = "step"
                else:
                    dtype = "transient"

            events.append(
                {
                    "start": start,
                    "end": end,
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "peak_deviation": peak_dev,
                    "mean_deviation": mean_dev,
                    "direction": direction,
                    "duration_seconds": dur_s,
                    "disturbance_type": dtype,
                }
            )

        return pd.DataFrame(events, columns=cols)

    def recovery_time(
        self,
        baseline_window: str = "10m",
        threshold_sigma: float = 3.0,
        recovery_pct: float = 0.95,
        max_recovery: str = "1h",
    ) -> pd.DataFrame:
        """Measure recovery time after each disturbance.

        Args:
            recovery_pct: Signal must return to within (1 - recovery_pct)
                          of peak deviation to be considered recovered.
            max_recovery: Maximum time to look for recovery.

        Returns:
            DataFrame with columns: disturbance_start, disturbance_end,
            recovery_time_seconds, recovered, pre_disturbance_mean,
            post_recovery_mean, residual_offset.
        """
        cols = [
            "disturbance_start",
            "disturbance_end",
            "recovery_time_seconds",
            "recovered",
            "pre_disturbance_mean",
            "post_recovery_mean",
            "residual_offset",
        ]
        disturbances = self.detect_disturbances(baseline_window, threshold_sigma)
        if disturbances.empty:
            return pd.DataFrame(columns=cols)

        bd = self._compute_baseline_and_deviation(baseline_window)
        max_td = pd.Timedelta(max_recovery)
        recovery_threshold = 1.0 - recovery_pct

        events: List[Dict[str, Any]] = []
        for _, dist in disturbances.iterrows():
            d_start = dist["start"]
            d_end = dist["end"]

            # Pre-disturbance mean
            pre_window = pd.Timedelta(baseline_window)
            pre = bd.loc[(bd.index >= d_start - pre_window) & (bd.index < d_start)]
            pre_mean = float(pre["value"].mean()) if not pre.empty else np.nan

            # Scan forward for recovery
            post = bd.loc[(bd.index > d_end) & (bd.index <= d_end + max_td)]
            peak_dev = dist["peak_deviation"]
            thr = peak_dev * recovery_threshold

            recovered = False
            recovery_s = np.nan
            post_mean = np.nan

            if not post.empty and peak_dev > 0:
                recovered_mask = post["deviation"].abs() <= thr
                if recovered_mask.any():
                    recovery_time = post.index[recovered_mask][0]
                    recovery_s = (recovery_time - d_end).total_seconds()
                    recovered = True
                    # Post-recovery mean
                    post_rec = bd.loc[recovery_time : recovery_time + pre_window]
                    post_mean = (
                        float(post_rec["value"].mean())
                        if not post_rec.empty
                        else np.nan
                    )

            residual = (
                post_mean - pre_mean
                if not np.isnan(post_mean) and not np.isnan(pre_mean)
                else np.nan
            )

            events.append(
                {
                    "disturbance_start": d_start,
                    "disturbance_end": d_end,
                    "recovery_time_seconds": recovery_s if recovered else np.nan,
                    "recovered": recovered,
                    "pre_disturbance_mean": pre_mean,
                    "post_recovery_mean": post_mean,
                    "residual_offset": residual,
                }
            )

        return pd.DataFrame(events, columns=cols)

    def disturbance_frequency(
        self,
        window: str = "8h",
        baseline_window: str = "10m",
        threshold_sigma: float = 3.0,
    ) -> pd.DataFrame:
        """Count disturbances per time window.

        Returns:
            DataFrame with columns: window_start, disturbance_count,
            total_disturbance_seconds, pct_time_disturbed,
            avg_recovery_seconds.
        """
        cols = [
            "window_start",
            "disturbance_count",
            "total_disturbance_seconds",
            "pct_time_disturbed",
            "avg_recovery_seconds",
        ]
        disturbances = self.detect_disturbances(baseline_window, threshold_sigma)
        recovery = self.recovery_time(baseline_window, threshold_sigma)

        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        # Create time windows
        t_min = self.signal[self.time_column].min()
        t_max = self.signal[self.time_column].max()
        windows = pd.date_range(start=t_min, end=t_max, freq=window)
        window_td = pd.Timedelta(window)

        events: List[Dict[str, Any]] = []
        for ws in windows:
            we = ws + window_td
            if not disturbances.empty:
                in_window = disturbances[
                    (disturbances["start"] >= ws) & (disturbances["start"] < we)
                ]
                count = len(in_window)
                total_dur = float(in_window["duration_seconds"].sum())
            else:
                count = 0
                total_dur = 0.0

            pct = (total_dur / window_td.total_seconds()) * 100

            if not recovery.empty:
                rec_in = recovery[
                    (recovery["disturbance_start"] >= ws)
                    & (recovery["disturbance_start"] < we)
                    & recovery["recovered"]
                ]
                avg_rec = (
                    float(rec_in["recovery_time_seconds"].mean())
                    if not rec_in.empty
                    else 0.0
                )
            else:
                avg_rec = 0.0

            events.append(
                {
                    "window_start": ws,
                    "disturbance_count": count,
                    "total_disturbance_seconds": total_dur,
                    "pct_time_disturbed": pct,
                    "avg_recovery_seconds": avg_rec,
                }
            )

        return pd.DataFrame(events, columns=cols)

    def before_after_comparison(
        self,
        baseline_window: str = "10m",
        threshold_sigma: float = 3.0,
        comparison_window: str = "5m",
    ) -> pd.DataFrame:
        """Compare process statistics before vs after each disturbance.

        Returns:
            DataFrame with columns: disturbance_start, pre_mean, post_mean,
            pre_std, post_std, mean_shift, variance_ratio.
        """
        cols = [
            "disturbance_start",
            "pre_mean",
            "post_mean",
            "pre_std",
            "post_std",
            "mean_shift",
            "variance_ratio",
        ]
        disturbances = self.detect_disturbances(baseline_window, threshold_sigma)
        if disturbances.empty or self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = self.signal.set_index(self.time_column)[self.value_column]
        comp_td = pd.Timedelta(comparison_window)

        events: List[Dict[str, Any]] = []
        for _, dist in disturbances.iterrows():
            d_start = dist["start"]
            d_end = dist["end"]

            pre = sig.loc[(sig.index >= d_start - comp_td) & (sig.index < d_start)]
            post = sig.loc[(sig.index > d_end) & (sig.index <= d_end + comp_td)]

            if pre.empty or post.empty or len(pre) < 2 or len(post) < 2:
                continue

            pre_mean = float(pre.mean())
            post_mean = float(post.mean())
            pre_std = float(pre.std())
            post_std = float(post.std())

            events.append(
                {
                    "disturbance_start": d_start,
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "pre_std": pre_std,
                    "post_std": post_std,
                    "mean_shift": post_mean - pre_mean,
                    "variance_ratio": (
                        post_std / pre_std if pre_std > 0 else float("inf")
                    ),
                }
            )

        return pd.DataFrame(events, columns=cols)
