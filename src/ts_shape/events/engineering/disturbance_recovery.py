import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_summary_df,
    twin_columns,
)
from ts_shape.events._windows import add_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs

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
        setpoint_uuid: str | None = None,
        event_uuid: str = "eng:disturbance_recovery",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
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
                # The setpoint in force at each sample: ``nearest`` let samples
                # just before a setpoint step take the new value and read as a
                # false deviation.
                direction="backward",
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

    @keyword_only(1)
    @renamed_kwargs(threshold_sigma="n_sigma")
    def detect_disturbances(
        self,
        baseline_window: str = "10min",
        n_sigma: float = 3.0,
        min_duration: str = "30s",
    ) -> pd.DataFrame:
        """Detect intervals where signal deviates significantly from baseline.

        Args:
            baseline_window: Window for rolling baseline computation.
            n_sigma: Deviation threshold in multiples of rolling std.
            min_duration: Minimum disturbance duration.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, peak_deviation, mean_deviation, direction,
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
            return empty_event_df("interval", cols)

        exceeded = bd["deviation"].abs() > (n_sigma * bd["rolling_std"])
        if not exceeded.any():
            return empty_event_df("interval", cols)

        min_td = pd.Timedelta(min_duration)
        groups = (exceeded != exceeded.shift()).cumsum()

        events: list[dict[str, Any]] = []
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
                    if post_dev < n_sigma * bd["rolling_std"].mean() * 0.5:
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

        if not events:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @keyword_only(1)
    @renamed_kwargs(threshold_sigma="n_sigma")
    def recovery_time(
        self,
        baseline_window: str = "10min",
        n_sigma: float = 3.0,
        recovery_pct: float = 0.95,
        max_recovery: str = "1h",
    ) -> pd.DataFrame:
        """Measure recovery time after each disturbance.

        Args:
            baseline_window: Window for the rolling baseline and rolling std
                (pandas offset string). Ignored for the baseline when a
                ``setpoint_uuid`` was given; the setpoint is used instead.
            n_sigma: Deviation threshold in multiples of the rolling std.
            recovery_pct: Signal must return to within (1 - recovery_pct)
                of peak deviation to be considered recovered.
            max_recovery: Maximum time to look for recovery.

        Returns:
            DataFrame with columns: start, end, duration_seconds (of the
            disturbance), uuid, source_uuid, recovery_time_seconds, recovered,
            pre_disturbance_mean, post_recovery_mean, residual_offset.
        """
        cols = [
            "start",
            "end",
            "recovery_time_seconds",
            "recovered",
            "pre_disturbance_mean",
            "post_recovery_mean",
            "residual_offset",
        ]
        disturbances = self.detect_disturbances(
            baseline_window=baseline_window, n_sigma=n_sigma
        )
        if disturbances.empty:
            return empty_event_df("interval", cols)

        bd = self._compute_baseline_and_deviation(baseline_window)
        max_td = pd.Timedelta(max_recovery)
        recovery_threshold = 1.0 - recovery_pct

        events: list[dict[str, Any]] = []
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
                    "start": d_start,
                    "end": d_end,
                    "recovery_time_seconds": recovery_s if recovered else np.nan,
                    "recovered": recovered,
                    "pre_disturbance_mean": pre_mean,
                    "post_recovery_mean": post_mean,
                    "residual_offset": residual,
                }
            )

        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @keyword_only(1)
    @renamed_kwargs(threshold_sigma="n_sigma")
    @twin_columns({"pct_time_disturbed": "time_disturbed_pct"})
    def disturbance_frequency(
        self,
        window: str = "8h",
        baseline_window: str = "10min",
        n_sigma: float = 3.0,
    ) -> pd.DataFrame:
        """Count disturbances per time window.

        Args:
            window: Counting window as a pandas offset string (e.g. '8h' per shift,
                '1D' per day).
            baseline_window: Window for the rolling baseline and rolling std used
                by :meth:`detect_disturbances`.
            n_sigma: Deviation threshold in multiples of the rolling std.

        Returns:
            DataFrame with columns: start, end, duration_seconds,
            disturbance_count, total_disturbance_seconds, pct_time_disturbed,
            avg_recovery_seconds. ``end`` is ``start`` + ``window``.

            Canonical twin columns: ``time_disturbed_pct`` (after
            ``pct_time_disturbed``); each holds the same values as the deprecated column
            right before it, which is removed in the next major release.
        """
        cols = [
            "start",
            "disturbance_count",
            "total_disturbance_seconds",
            "pct_time_disturbed",
            "avg_recovery_seconds",
        ]
        disturbances = self.detect_disturbances(
            baseline_window=baseline_window, n_sigma=n_sigma
        )
        recovery = self.recovery_time(baseline_window=baseline_window, n_sigma=n_sigma)

        if self.signal.empty:
            return empty_event_df("summary", cols[1:])

        # Create time windows
        t_min = self.signal[self.time_column].min()
        t_max = self.signal[self.time_column].max()
        windows = pd.date_range(start=t_min, end=t_max, freq=window)
        window_td = pd.Timedelta(window)

        events: list[dict[str, Any]] = []
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
                    (recovery["start"] >= ws)
                    & (recovery["start"] < we)
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
                    "start": ws,
                    "disturbance_count": count,
                    "total_disturbance_seconds": total_dur,
                    "pct_time_disturbed": pct,
                    "avg_recovery_seconds": avg_rec,
                }
            )

        if not events:
            return empty_event_df("summary", cols[1:])
        return finalize_summary_df(
            add_window_bounds(pd.DataFrame(events, columns=cols), window)
        )

    @keyword_only(1)
    @renamed_kwargs(threshold_sigma="n_sigma")
    def before_after_comparison(
        self,
        baseline_window: str = "10min",
        n_sigma: float = 3.0,
        comparison_window: str = "5min",
    ) -> pd.DataFrame:
        """Compare process statistics before vs after each disturbance.

        Args:
            baseline_window: Window for the rolling baseline and rolling std used
                by :meth:`detect_disturbances`.
            n_sigma: Deviation threshold in multiples of the rolling std.
            comparison_window: Length of the span compared just before the
                disturbance start and just after its end. Disturbances with fewer
                than two samples on either side are skipped.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, pre_mean, post_mean, pre_std, post_std, mean_shift,
            variance_ratio. ``start`` / ``end`` are the disturbance bounds.
        """
        cols = [
            "start",
            "end",
            "pre_mean",
            "post_mean",
            "pre_std",
            "post_std",
            "mean_shift",
            "variance_ratio",
        ]
        disturbances = self.detect_disturbances(
            baseline_window=baseline_window, n_sigma=n_sigma
        )
        if disturbances.empty or self.signal.empty:
            return empty_event_df("summary", ["uuid", "source_uuid"] + cols[2:])

        sig = self.signal.set_index(self.time_column)[self.value_column]
        comp_td = pd.Timedelta(comparison_window)

        events: list[dict[str, Any]] = []
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
                    "start": d_start,
                    "end": d_end,
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

        if not events:
            return empty_event_df("summary", ["uuid", "source_uuid"] + cols[2:])
        return finalize_summary_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )
