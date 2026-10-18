import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.maintenance._rolling import (
    gather,
    interval_bounds,
    time_window_bounds,
    window_moments,
)
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import interval_end_times
from ts_shape.utils.runs import run_agg, total_seconds, true_runs

logger = logging.getLogger(__name__)


class VibrationAnalysisEvents(Base):
    """
    Analyze vibration signals from industrial equipment: RMS exceedance,
    amplitude growth, and bearing health indicators (kurtosis, crest factor).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "maint:vibration",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the vibration analyzer.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            signal_uuid: UUID of the vibration signal; must be present in ``dataframe``.
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

    @keyword_only(2)
    def detect_rms_exceedance(
        self,
        baseline_rms: float,
        threshold_factor: float = 1.5,
        window: str = "1min",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """
        Compute rolling RMS and flag intervals exceeding baseline_rms * threshold_factor.

        Args:
            baseline_rms: Known baseline RMS value for healthy equipment.
            threshold_factor: Multiplier above baseline to trigger alarm.
            window: Rolling window size.
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            rms_value, baseline_rms, ratio, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "rms_value",
            "baseline_rms",
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
        threshold = baseline_rms * threshold_factor

        # Rolling RMS over the time window (t - window, t]; NaN if the window
        # holds a NaN sample.
        lo, hi = time_window_bounds(sig[self.time_column], window_td)
        values = sig[self.value_column].to_numpy(dtype=float)
        mom = window_moments(np.zeros(len(sig)), values**2, lo, hi)
        sig["rms"] = np.sqrt(np.where(mom["n"] < mom["n_total"], np.nan, mom["mean"]))
        exceeded = (sig["rms"] >= threshold).fillna(False)

        if not exceeded.any():
            return pd.DataFrame(columns=cols)

        # One interval per contiguous run of exceeded samples.
        runs = true_runs(exceeded.to_numpy(dtype=bool))
        start = sig[self.time_column].array[runs.starts]
        end = interval_end_times(sig[self.time_column], interval_end).array[runs.last]
        avg_rms = run_agg(sig["rms"], runs, "nanmean")
        ratio: Any = (
            avg_rms / baseline_rms
            if baseline_rms > 0
            else np.full(runs.n_runs, None, dtype=object)
        )
        return pd.DataFrame(
            {
                "start": start,
                "end": end,
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "rms_value": avg_rms,
                "baseline_rms": baseline_rms,
                "ratio": ratio,
                "duration_seconds": total_seconds(end - start),
            },
            columns=cols,
        )

    @keyword_only(1)
    def detect_amplitude_growth(
        self,
        window: str = "1h",
        growth_threshold: float = 0.1,
    ) -> pd.DataFrame:
        """
        Track peak-to-peak amplitude in non-overlapping windows and flag
        windows where amplitude grows beyond growth_threshold relative to baseline.

        Args:
            window: Window size for amplitude measurement.
            growth_threshold: Minimum fractional growth (e.g. 0.1 = 10%) to flag.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, amplitude, baseline_amplitude, growth_pct.
            ``end`` is ``start`` plus ``window``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "is_delta",
            "amplitude",
            "baseline_amplitude",
            "growth_pct",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        window_td = pd.to_timedelta(window)

        # Generate non-overlapping windows
        t_min = sig[self.time_column].iloc[0]
        t_max = sig[self.time_column].iloc[-1]
        window_starts = pd.date_range(start=t_min, end=t_max, freq=window_td)

        # Peak-to-peak amplitude per window [ws, ws + window) holding at least
        # two samples (NaN samples skipped, as pandas max/min do).
        lo, hi = interval_bounds(sig[self.time_column], window_starts, window_td)
        keep = hi - lo >= 2
        values = sig[self.value_column].to_numpy(dtype=float)
        amplitudes: list[dict[str, Any]] = []
        if keep.any():
            idx, starts = gather(lo[keep], hi[keep])
            with np.errstate(invalid="ignore"):
                amps = np.fmax.reduceat(values[idx], starts) - np.fmin.reduceat(
                    values[idx], starts
                )
            amplitudes = [
                {"start": ws, "end": ws + window_td, "amplitude": float(amp)}
                for ws, amp in zip(window_starts[keep], amps)
            ]

        if not amplitudes:
            return pd.DataFrame(columns=cols)

        # Use first window as baseline
        baseline_amp = amplitudes[0]["amplitude"]
        if baseline_amp == 0:
            baseline_amp = np.finfo(float).eps

        rows: list[dict[str, Any]] = []
        for a in amplitudes:
            growth_pct = (a["amplitude"] - baseline_amp) / baseline_amp
            rows.append(
                {
                    "start": a["start"],
                    "end": a["end"],
                    "duration_seconds": (a["end"] - a["start"]).total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                    "is_delta": True,
                    "amplitude": a["amplitude"],
                    "baseline_amplitude": amplitudes[0]["amplitude"],
                    "growth_pct": float(growth_pct),
                }
            )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    @keyword_only(1)
    def bearing_health_indicators(
        self,
        window: str = "5min",
    ) -> pd.DataFrame:
        """
        Compute bearing health indicators per window: RMS, peak value,
        crest factor (peak/RMS), and kurtosis.

        Args:
            window: Window size for indicator computation.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, rms, peak, crest_factor, kurtosis.
            ``end`` is ``start`` plus ``window``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "is_delta",
            "rms",
            "peak",
            "crest_factor",
            "kurtosis",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        window_td = pd.to_timedelta(window)

        # Generate non-overlapping windows
        t_min = sig[self.time_column].iloc[0]
        t_max = sig[self.time_column].iloc[-1]
        window_starts = pd.date_range(start=t_min, end=t_max, freq=window_td)

        # Indicators per window [ws, ws + window) holding at least two samples;
        # a NaN sample makes the window's indicators NaN (crest factor None).
        lo, hi = interval_bounds(sig[self.time_column], window_starts, window_td)
        keep = hi - lo >= 2
        if not keep.any():
            return pd.DataFrame(columns=cols)
        values = sig[self.value_column].to_numpy(dtype=float)
        idx, starts = gather(lo[keep], hi[keep])
        lens = (hi - lo)[keep]
        v = values[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt(np.add.reduceat(v**2, starts) / lens)
            peak = np.maximum.reduceat(np.abs(v), starts)
            # Excess (Fisher) kurtosis, biased, as scipy.stats.kurtosis computes it.
            mean = np.add.reduceat(v, starts) / lens
            dev = v - np.repeat(mean, lens)
            m2 = np.add.reduceat(dev**2, starts) / lens
            m4 = np.add.reduceat(dev**4, starts) / lens
            zero = m2 <= (np.finfo(float).eps * mean) ** 2
            kurt = np.where(zero, np.nan, m4 / m2**2.0) - 3
        ws = window_starts[keep]
        we = ws + window_td
        return pd.DataFrame(
            {
                "start": ws,
                "end": we,
                "duration_seconds": (we - ws).total_seconds(),
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "rms": rms,
                "peak": peak,
                "crest_factor": [
                    float(p / r) if r > 0 else None for p, r in zip(peak, rms)
                ],
                "kurtosis": kurt,
            },
            columns=cols,
        )
