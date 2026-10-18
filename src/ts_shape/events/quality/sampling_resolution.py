from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import add_twin_columns
from ts_shape.events._windows import labeled_window_summary
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    keyword_only,
    renamed_kwargs,
)

#: Legacy millisecond columns of ``sampling_jitter`` -> their ``_seconds`` twins.
_MS_TWINS = {
    "mean_interval_ms": "mean_interval_seconds",
    "std_interval_ms": "std_interval_seconds",
    "max_gap_ms": "max_gap_seconds",
}
_MS_SCALE = dict.fromkeys(_MS_TWINS, 1e-3)


def _interval_seconds(times: pd.Index) -> np.ndarray:
    """Intervals between consecutive ``times`` in float seconds.

    Computed from integer nanoseconds, so sub-millisecond spacing keeps its
    precision (1.5 ms is 0.0015 s, not 0.001 s; 0.5 ms is not 0).
    """
    ns = np.asarray(pd.DatetimeIndex(times).as_unit("ns").asi8, dtype=np.int64)
    seconds: np.ndarray = np.diff(ns).astype(np.float64) / 1e9
    return seconds


class SamplingResolutionEvents(Base):
    """Quality: Sampling & Resolution Analysis

    Analyze digital measurement system limitations: quantization
    effects, effective resolution, aliasing detection, and sampling
    synchronization/jitter.

    Methods:
    - detect_quantization: Detect A/D quantization in measurements.
    - effective_resolution: Estimate effective vs theoretical resolution.
    - detect_aliasing: Check for Nyquist violations via FFT.
    - sampling_jitter: Measure timing regularity of sampling.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        value_column: str = "value_double",
        event_uuid: str = "quality:sampling_resolution",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_column = time_column

        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    @keyword_only(1)
    def detect_quantization(
        self, window: str = "1h", threshold_ratio: float = 0.3
    ) -> pd.DataFrame:
        """Detect A/D quantization effects in measurements.

        Identifies if measurements are clustered at discrete levels
        rather than continuously distributed.

        Args:
            window: Resample window size.
            threshold_ratio: Ratio of unique values to expected unique
                below which signal is flagged as quantized.

        Returns:
            Summary DataFrame with columns: start, end, duration_seconds
            (the window bounds), window_start (the resample label; a
            deprecated twin of ``start``), n_unique_values,
            expected_unique, quantization_step, quantization_ratio,
            quantized.
        """
        cols = [
            "window_start",
            "n_unique_values",
            "expected_unique",
            "quantization_step",
            "quantization_ratio",
            "quantized",
        ]
        if self.signal.empty:
            return labeled_window_summary([], cols, window)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            n = len(vals)
            if n < 10:
                continue

            unique_vals = np.sort(vals.unique())
            n_unique = len(unique_vals)

            # Quantization step: median of smallest differences
            if n_unique >= 2:
                diffs = np.diff(unique_vals)
                diffs = diffs[diffs > 0]
                if len(diffs) > 0:
                    quant_step = (
                        float(np.median(diffs[diffs <= np.percentile(diffs, 25)]))
                        if len(diffs) >= 4
                        else float(np.min(diffs))
                    )
                else:
                    quant_step = 0.0
            else:
                quant_step = 0.0

            # Expected unique: for continuous data, roughly n (or capped)
            val_range = (
                float(unique_vals[-1] - unique_vals[0]) if n_unique >= 2 else 0.0
            )
            if quant_step > 0:
                expected_unique = int(val_range / quant_step) + 1
            else:
                expected_unique = n

            ratio = n_unique / max(expected_unique, 1)
            quantized = ratio < threshold_ratio and n_unique < n * 0.5

            events.append(
                {
                    "window_start": ts,
                    "n_unique_values": n_unique,
                    "expected_unique": expected_unique,
                    "quantization_step": float(quant_step),
                    "quantization_ratio": float(ratio),
                    "quantized": bool(quantized),
                }
            )

        return labeled_window_summary(events, cols, window)

    @keyword_only(1)
    def effective_resolution(self, window: str = "1h") -> pd.DataFrame:
        """Estimate effective measurement resolution from data.

        Args:
            window: Resample window size.

        Returns:
            Summary DataFrame with columns: start, end, duration_seconds
            (the window bounds), window_start (the resample label; a
            deprecated twin of ``start``), theoretical_resolution,
            effective_resolution, resolution_ratio, assessment.
        """
        cols = [
            "window_start",
            "theoretical_resolution",
            "effective_resolution",
            "resolution_ratio",
            "assessment",
        ]
        if self.signal.empty:
            return labeled_window_summary([], cols, window)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 10:
                continue

            sorted_unique = np.sort(vals.unique())
            if len(sorted_unique) < 2:
                continue

            # Theoretical resolution: smallest observed step
            diffs = np.diff(sorted_unique)
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                continue
            theoretical = float(np.min(diffs))

            # Effective resolution: noise floor from std of consecutive differences
            consecutive_diffs = np.abs(np.diff(vals.values))
            effective = (
                float(np.std(consecutive_diffs))
                if len(consecutive_diffs) > 1
                else theoretical
            )

            ratio = effective / theoretical if theoretical > 0 else 1.0

            if ratio <= 2.0:
                assessment = "adequate"
            elif ratio <= 5.0:
                assessment = "degraded"
            else:
                assessment = "insufficient"

            events.append(
                {
                    "window_start": ts,
                    "theoretical_resolution": float(theoretical),
                    "effective_resolution": float(effective),
                    "resolution_ratio": float(ratio),
                    "assessment": assessment,
                }
            )

        return labeled_window_summary(events, cols, window)

    @keyword_only(1)
    @renamed_kwargs(expected_frequency="expected_frequency_hz")
    def detect_aliasing(
        self,
        window: str = "1h",
        expected_frequency_hz: float | None = None,
    ) -> pd.DataFrame:
        """Check for Nyquist violations using FFT.

        Args:
            window: Resample window size.
            expected_frequency_hz: Known process frequency in Hz. If None,
                dominant frequency is estimated from FFT.

        Returns:
            Summary DataFrame with columns: start, end, duration_seconds
            (the window bounds), window_start (the resample label; a
            deprecated twin of ``start``), sampling_rate_hz,
            nyquist_hz, dominant_frequency, aliasing_risk.
        """
        cols = [
            "window_start",
            "sampling_rate_hz",
            "nyquist_hz",
            "dominant_frequency",
            "aliasing_risk",
        ]
        if self.signal.empty:
            return labeled_window_summary([], cols, window)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            n = len(vals)
            if n < 16:  # Need enough for meaningful FFT
                continue

            # Estimate sampling rate
            times = group.index
            dt = _interval_seconds(times)
            dt = dt[dt > 0]
            if len(dt) == 0:
                continue
            mean_dt = float(np.mean(dt))
            if mean_dt <= 0:
                continue

            sampling_rate = 1.0 / mean_dt
            nyquist = sampling_rate / 2.0

            # FFT to find dominant frequency
            signal_vals = vals.values - vals.values.mean()  # Remove DC
            fft_vals = np.abs(np.fft.rfft(signal_vals))
            freqs = np.fft.rfftfreq(n, d=mean_dt)

            # Skip DC component (index 0)
            if len(fft_vals) > 1:
                dominant_idx = int(np.argmax(fft_vals[1:])) + 1
                dominant_freq = float(freqs[dominant_idx])
            else:
                dominant_freq = 0.0

            check_freq = (
                expected_frequency_hz
                if expected_frequency_hz is not None
                else dominant_freq
            )

            if check_freq > nyquist * 0.9:
                aliasing_risk = "high"
            elif check_freq > nyquist * 0.7:
                aliasing_risk = "medium"
            elif check_freq > nyquist * 0.5:
                aliasing_risk = "low"
            else:
                aliasing_risk = "none"

            events.append(
                {
                    "window_start": ts,
                    "sampling_rate_hz": float(sampling_rate),
                    "nyquist_hz": float(nyquist),
                    "dominant_frequency": float(dominant_freq),
                    "aliasing_risk": aliasing_risk,
                }
            )

        return labeled_window_summary(events, cols, window)

    @keyword_only(1)
    def sampling_jitter(self, window: str = "1h") -> pd.DataFrame:
        """Measure timing jitter (variation in sampling interval).

        Args:
            window: Resample window size.

        Returns:
            Summary DataFrame with columns: start, end, duration_seconds
            (the window bounds), window_start (the resample label; a
            deprecated twin of ``start``), mean_interval_ms,
            mean_interval_seconds, std_interval_ms, std_interval_seconds,
            jitter_pct, max_gap_ms, max_gap_seconds, regularity_score. Each
            ``*_seconds`` column follows its legacy ``*_ms`` twin (the same
            value in seconds); the ``*_ms`` columns are deprecated.
        """
        cols = [
            "window_start",
            "mean_interval_ms",
            "std_interval_ms",
            "jitter_pct",
            "max_gap_ms",
            "regularity_score",
        ]
        if self.signal.empty:
            return add_twin_columns(
                labeled_window_summary([], cols, window), _MS_TWINS, scale=_MS_SCALE
            )

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            if len(group) < 3:
                continue

            times = group.index
            intervals_ms = _interval_seconds(times) * 1000.0
            intervals_ms = intervals_ms[intervals_ms > 0]
            if len(intervals_ms) < 2:
                continue

            mean_ms = float(np.mean(intervals_ms))
            std_ms = float(np.std(intervals_ms))
            jitter_pct = (std_ms / mean_ms * 100) if mean_ms > 0 else 0.0
            max_gap = float(np.max(intervals_ms))

            # Regularity score: 100 for perfect, degrades with jitter
            regularity = max(0.0, 100.0 - jitter_pct * 2)

            events.append(
                {
                    "window_start": ts,
                    "mean_interval_ms": float(mean_ms),
                    "std_interval_ms": float(std_ms),
                    "jitter_pct": float(jitter_pct),
                    "max_gap_ms": float(max_gap),
                    "regularity_score": float(min(100.0, regularity)),
                }
            )

        return add_twin_columns(
            labeled_window_summary(events, cols, window), _MS_TWINS, scale=_MS_SCALE
        )
