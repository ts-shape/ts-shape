import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events.maintenance._rolling import (
    py_max,
    py_min,
    time_window_bounds,
    times_ns,
    window_moments,
)
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import interval_end_times
from ts_shape.utils.runs import run_agg, total_seconds, true_runs

logger = logging.getLogger(__name__)


class DegradationDetectionEvents(Base):
    """
    Detect degradation patterns in time series signals: trend degradation,
    variance increases, level shifts, and composite health scores.

    Designed for predictive maintenance on manufacturing/industrial IoT data.
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "maint:degradation",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the degradation detector.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            signal_uuid: UUID of the signal to analyze; must be present in
                ``dataframe``.
            event_uuid: UUID stamped on emitted events.
            value_column: Column holding the signal values.
            time_column: Timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, signal_uuid)

        # Isolate signal series and ensure proper dtypes/sort
        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    @keyword_only(1)
    def detect_trend_degradation(
        self,
        window: str = "1h",
        min_slope: float = 0.0,
        direction: str = "decreasing",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """
        Detect intervals where a rolling linear regression slope exceeds min_slope
        in the given direction, indicating trend-based degradation.

        Args:
            window: Rolling window size (e.g. '1h', '30min').
            min_slope: Minimum absolute slope to consider degradation.
            direction: 'decreasing' (negative slope) or 'increasing' (positive slope).
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            avg_slope, total_change, duration_seconds. ``source_uuid`` is the
            analyzed signal.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
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

        # Rolling least-squares slope over the time window (t - window, t]. A
        # window with a NaN sample has a NaN slope (as np.polyfit gives); one
        # whose samples all share a timestamp has no slope (NaN).
        lo, hi = time_window_bounds(sig[self.time_column], window_td)
        mom = window_moments(
            sig["t_seconds"].to_numpy(), sig[self.value_column].to_numpy(), lo, hi
        )
        sig["slope"] = np.where(mom["n"] < mom["n_total"], np.nan, mom["slope"])

        # Filter by direction and min_slope
        if direction == "decreasing":
            degrading = sig["slope"] <= -abs(min_slope)
        else:
            degrading = sig["slope"] >= abs(min_slope)

        degrading = degrading.fillna(False)

        if not degrading.any():
            return pd.DataFrame(columns=cols)

        # One interval per contiguous run of degrading samples.
        runs = true_runs(degrading.to_numpy(dtype=bool))
        times = sig[self.time_column].array
        start = times[runs.starts]
        end = interval_end_times(sig[self.time_column], interval_end).array[runs.last]
        values = sig[self.value_column].to_numpy(dtype=np.float64)
        return pd.DataFrame(
            {
                "start": start,
                "end": end,
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "avg_slope": run_agg(sig["slope"], runs, "nanmean"),
                "total_change": values[runs.last] - values[runs.starts],
                "duration_seconds": total_seconds(end - start),
            },
            columns=cols,
        )

    @keyword_only(1)
    def detect_variance_increase(
        self,
        window: str = "1h",
        threshold_factor: float = 2.0,
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """
        Compare rolling variance against a baseline (first window) and flag
        intervals where the ratio exceeds threshold_factor.

        The baseline is ``[first sample, first sample + window)``, the same
        number of samples as each rolling ``(t - window, t]`` window of a
        regular signal, so a stationary signal has ratio 1. Samples inside
        the baseline window are not flagged (their rolling windows are the
        partial warm-up of the baseline).

        Args:
            window: Rolling window size.
            threshold_factor: Minimum variance ratio to flag.
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            baseline_variance, current_variance, ratio, duration_seconds
            (float seconds from start to end).
        """
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "baseline_variance",
            "current_variance",
            "ratio",
            "duration_seconds",
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
        baseline_mask = sig[self.time_column] < first_time + window_td
        baseline_data = sig.loc[baseline_mask, self.value_column]
        if len(baseline_data) < 2:
            return pd.DataFrame(columns=cols)
        baseline_var = float(baseline_data.var())
        if baseline_var == 0:
            baseline_var = np.finfo(float).eps

        # Rolling sample variance over the time window (t - window, t], NaN
        # samples skipped (pandas ``var``).
        lo, hi = time_window_bounds(sig[self.time_column], window_td)
        sig["current_variance"] = window_moments(
            np.zeros(len(sig)), sig[self.value_column].to_numpy(), lo, hi
        )["var"]
        sig["ratio"] = sig["current_variance"] / baseline_var

        # Samples inside the baseline window are the baseline: their rolling
        # windows are partial (a 2-sample warm-up window is no variance
        # increase), so they are not flagged.
        exceeded = (sig["ratio"] >= threshold_factor).fillna(False) & ~baseline_mask
        if not exceeded.any():
            return pd.DataFrame(columns=cols)

        # One interval per contiguous run of exceeded samples.
        runs = true_runs(exceeded.to_numpy(dtype=bool))
        start = sig[self.time_column].array[runs.starts]
        end = interval_end_times(sig[self.time_column], interval_end).array[runs.last]
        return pd.DataFrame(
            {
                "start": start,
                "end": end,
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "baseline_variance": baseline_var,
                "current_variance": run_agg(sig["current_variance"], runs, "nanmean"),
                "ratio": run_agg(sig["ratio"], runs, "nanmean"),
                "duration_seconds": total_seconds(end - start),
            },
            columns=cols,
        )

    @keyword_only(2)
    def detect_level_shift(
        self,
        min_shift: float,
        hold: str = "5min",
    ) -> pd.DataFrame:
        """
        CUSUM-like detection for permanent mean shifts in the signal.

        Args:
            min_shift: Minimum absolute shift magnitude to detect.
            hold: Minimum duration the new level must persist.

        Returns:
            DataFrame with columns: systime, uuid, source_uuid, is_delta,
            shift_magnitude, prev_mean, new_mean.
        """
        cols = [
            "systime",
            "uuid",
            "source_uuid",
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

        values = sig[self.value_column].to_numpy(dtype=float)
        times = sig[self.time_column].array
        # Hold windows [t_i, t_i + hold] as index ranges, and NaN-aware prefix
        # sums (offset by a reference value, for precision) for the window
        # means -- no per-candidate scan of the whole series.
        t_ns = times_ns(sig[self.time_column])
        post_start = np.searchsorted(t_ns, t_ns, side="left")
        post_end = np.searchsorted(t_ns, t_ns + np.int64(hold_td.value), side="right")
        finite = ~np.isnan(values)
        ref = float(values[finite][0]) if finite.any() else 0.0
        csum = np.concatenate(([0.0], np.cumsum(np.where(finite, values - ref, 0.0))))
        ccnt = np.concatenate(([0], np.cumsum(finite)))

        def _mean(a: int, b: int) -> float:
            k = int(ccnt[b] - ccnt[a])
            return ref + float(csum[b] - csum[a]) / k if k else float("nan")

        events: list[dict[str, Any]] = []

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
                # np.mean semantics: NaN if the span holds a NaN sample.
                if ccnt[i] - ccnt[last_shift_idx] < i - last_shift_idx:
                    prev_mean = np.nan
                else:
                    prev_mean = float(_mean(last_shift_idx, i))
                # Check hold: gather data after shift point
                if post_end[i] > post_start[i]:
                    new_mean = float(_mean(post_start[i], post_end[i]))
                    shift_mag = new_mean - prev_mean
                    if abs(shift_mag) >= abs(min_shift):
                        events.append(
                            {
                                "systime": pd.Timestamp(times[i]),
                                "uuid": self.event_uuid,
                                "source_uuid": self.signal_uuid,
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

    @keyword_only(1)
    @twin_columns({"variance_ratio": "variance_factor"})
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
            baseline_window: Initial period used to establish baseline
                behavior: ``[first sample, first sample + baseline_window)``,
                half-open like the rolling windows.

        Returns:
            DataFrame with columns: systime, uuid, source_uuid, is_delta,
            health_score, mean_drift_pct, variance_ratio, variance_factor
            (current / baseline variance; above 1 when the variance grew),
            trend_slope.

            Canonical twin columns: ``variance_factor`` (after
            ``variance_ratio``, which is not a 0-1 fraction); it holds the
            same values as the deprecated column right before it, which is
            removed in the next major release.
        """
        cols = [
            "systime",
            "uuid",
            "source_uuid",
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
        baseline_mask = sig[self.time_column] < first_time + baseline_td
        baseline_data = sig.loc[baseline_mask, self.value_column]

        if len(baseline_data) < 2:
            # Not enough baseline data; return perfect health for all points
            result = sig[[self.time_column]].copy()
            result = result.rename(columns={self.time_column: "systime"})
            result["uuid"] = self.event_uuid
            result["source_uuid"] = self.signal_uuid
            result["is_delta"] = True
            result["health_score"] = 100.0
            result["mean_drift_pct"] = 0.0
            result["variance_ratio"] = 1.0
            result["trend_slope"] = 0.0
            return result[cols]

        baseline_mean = float(baseline_data.mean())
        baseline_var = float(baseline_data.var())
        eps = float(np.finfo(float).eps)
        if baseline_var == 0:
            baseline_var = eps
        baseline_mean_for_pct = eps if baseline_mean == 0 else baseline_mean

        # Rolling metrics over the time window (t - window, t]. Mean and
        # variance skip NaN samples; the trend slope of a window holding a NaN
        # (or whose samples share one timestamp) is NaN. Python ``min``/``max``
        # semantics are kept for the penalties, so NaN metrics behave as before.
        lo, hi = time_window_bounds(sig[self.time_column], window_td)
        mom = window_moments(
            sig["t_seconds"].to_numpy(), sig[self.value_column].to_numpy(), lo, hi
        )
        short = mom["n_total"] < 2
        current_mean = mom["mean"]
        mean_drift_pct = (
            np.abs(current_mean - baseline_mean) / abs(baseline_mean_for_pct) * 100.0
        )
        variance_ratio = mom["var"] / baseline_var
        trend_slope = np.where(mom["n"] < mom["n_total"], np.nan, mom["slope"])
        # No defined slope (all samples at one timestamp): 0, as on a failed fit.
        trend_slope = np.where(
            (mom["n"] == mom["n_total"]) & np.isnan(mom["slope"]), 0.0, trend_slope
        )

        # Composite health score: penalize mean drift, variance increase, trend
        # Each component maps to 0-33.3 penalty
        drift_penalty = py_min(mean_drift_pct / 100.0, 1.0) * 33.3
        var_penalty = py_min(py_max(variance_ratio - 1.0, 0.0) / 5.0, 1.0) * 33.3
        slope_penalty = (
            py_min(
                np.abs(trend_slope) * 3600.0 / (abs(baseline_mean_for_pct) + 1e-9),
                1.0,
            )
            * 33.4
        )
        score = py_max(
            0.0,
            py_min(100.0, 100.0 - drift_penalty - var_penalty - slope_penalty),
        )

        result = pd.DataFrame(
            {
                "systime": sig[self.time_column],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "health_score": np.where(short, 100.0, score).astype(float),
                "mean_drift_pct": np.where(short, 0.0, mean_drift_pct),
                "variance_ratio": np.where(short, 1.0, variance_ratio),
                "trend_slope": np.where(short, 0.0, trend_slope),
            },
            columns=cols,
        )
        return result
