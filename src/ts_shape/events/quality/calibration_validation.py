from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only


class CalibrationValidationEvents(Base):
    """Quality: Calibration Validation & Linearity

    Multi-point calibration validation for inline sensors. Assesses
    linearity across measurement range, detects calibration degradation,
    and determines when recalibration is needed.

    Methods:
    - linearity_assessment: Residual analysis at calibration points.
    - calibration_curve: Polynomial fit to calibration data.
    - detect_recalibration_need: Monitor bias/drift for recal triggers.
    - calibration_stability: Bias stability scoring over time.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        reference_uuid: str | None = None,
        reference_value: float | None = None,
        value_column: str = "value_double",
        event_uuid: str = "quality:calibration",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.reference_uuid = reference_uuid
        self.reference_value = reference_value
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_column = time_column

        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

        if self.reference_uuid is not None:
            self._reference = (
                self.dataframe[self.dataframe["uuid"] == self.reference_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            self._reference[self.time_column] = pd.to_datetime(
                self._reference[self.time_column]
            )
        else:
            self._reference = None

    def _get_reference_for_window(self, window_start, window_end) -> float | None:
        """Get reference value for a time window."""
        if self.reference_value is not None:
            return self.reference_value
        if self._reference is not None and not self._reference.empty:
            mask = (self._reference[self.time_column] >= window_start) & (
                self._reference[self.time_column] < window_end
            )
            ref_vals = self._reference.loc[mask, self.value_column].dropna()
            if not ref_vals.empty:
                return float(ref_vals.mean())
        return None

    @staticmethod
    def linearity_assessment(
        calibration_points: dict[float, list[float]],
    ) -> pd.DataFrame:
        """Assess linearity at multiple calibration points.

        Args:
            calibration_points: Dict mapping reference values to lists
                of measured values, e.g. {10.0: [10.1, 9.9, 10.0]}.

        Returns:
            DataFrame with columns: reference, measured_mean,
            measured_std, bias, linearity_error.
        """
        cols = ["reference", "measured_mean", "measured_std", "bias", "linearity_error"]
        if not calibration_points:
            return pd.DataFrame(columns=cols)

        refs = sorted(calibration_points.keys())
        ref_arr = []
        mean_arr = []

        events: list[dict[str, Any]] = []
        for ref in refs:
            measurements = calibration_points[ref]
            if not measurements:
                continue
            measured_mean = float(np.mean(measurements))
            measured_std = (
                float(np.std(measurements, ddof=1)) if len(measurements) > 1 else 0.0
            )
            bias = measured_mean - ref

            ref_arr.append(ref)
            mean_arr.append(measured_mean)

            events.append(
                {
                    "reference": ref,
                    "measured_mean": float(measured_mean),
                    "measured_std": float(measured_std),
                    "bias": float(bias),
                    "linearity_error": 0.0,  # placeholder, computed below
                }
            )

        if len(ref_arr) < 2:
            return (
                pd.DataFrame(events, columns=cols)
                if events
                else pd.DataFrame(columns=cols)
            )

        # Fit best-fit line to reference vs measured
        slope, intercept, _, _, _ = stats.linregress(ref_arr, mean_arr)

        # Linearity error = residual from best-fit line
        for i, e in enumerate(events):
            predicted = slope * ref_arr[i] + intercept
            linearity_error = mean_arr[i] - predicted
            e["linearity_error"] = float(linearity_error)

        return pd.DataFrame(events, columns=cols)

    @staticmethod
    @keyword_only(1)
    def calibration_curve(
        calibration_points: dict[float, list[float]],
        degree: int = 1,
    ) -> pd.DataFrame:
        """Fit polynomial calibration curve.

        Args:
            calibration_points: Dict mapping reference values to
                measured value lists.
            degree: Polynomial degree (1=linear, 2=quadratic, etc).

        Returns:
            DataFrame with columns: coefficient_index, coefficient_value,
            r_squared, max_residual.
        """
        cols = ["coefficient_index", "coefficient_value", "r_squared", "max_residual"]
        if not calibration_points or len(calibration_points) < 2:
            return pd.DataFrame(columns=cols)

        refs = sorted(calibration_points.keys())
        means = [float(np.mean(calibration_points[r])) for r in refs]

        ref_arr = np.array(refs)
        mean_arr = np.array(means)

        # Fit polynomial
        coeffs = np.polyfit(ref_arr, mean_arr, min(degree, len(refs) - 1))
        predicted = np.polyval(coeffs, ref_arr)

        # R² calculation
        ss_res = np.sum((mean_arr - predicted) ** 2)
        ss_tot = np.sum((mean_arr - mean_arr.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        max_residual = float(np.max(np.abs(mean_arr - predicted)))

        events: list[dict[str, Any]] = []
        for i, c in enumerate(coeffs):
            events.append(
                {
                    "coefficient_index": i,
                    "coefficient_value": float(c),
                    "r_squared": float(r_squared),
                    "max_residual": float(max_residual),
                }
            )

        return pd.DataFrame(events, columns=cols)

    @keyword_only(1)
    def detect_recalibration_need(
        self,
        window: str = "1D",
        max_bias: float | None = None,
        max_drift_rate: float | None = None,
    ) -> pd.DataFrame:
        """Monitor bias and drift to detect recalibration need.

        Args:
            window: Resample window size.
            max_bias: Maximum acceptable absolute bias.
            max_drift_rate: Maximum acceptable bias change per window.

        Returns:
            DataFrame with columns: window_start, bias, drift_rate,
            recalibration_needed, reason.
        """
        cols = ["window_start", "bias", "drift_rate", "recalibration_needed", "reason"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)
        if self.reference_uuid is None and self.reference_value is None:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: list[dict[str, Any]] = []
        prev_bias: float | None = None

        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue

            window_end = ts + pd.to_timedelta(window)
            ref = self._get_reference_for_window(ts, window_end)
            if ref is None:
                continue

            bias = float(vals.mean()) - ref
            drift_rate = (bias - prev_bias) if prev_bias is not None else 0.0
            prev_bias = bias

            reasons = []
            if max_bias is not None and abs(bias) > max_bias:
                reasons.append(f"bias={bias:.4f} exceeds max={max_bias}")
            if max_drift_rate is not None and abs(drift_rate) > max_drift_rate:
                reasons.append(
                    f"drift_rate={drift_rate:.4f} exceeds max={max_drift_rate}"
                )

            recal_needed = len(reasons) > 0

            events.append(
                {
                    "window_start": ts,
                    "bias": float(bias),
                    "drift_rate": float(drift_rate),
                    "recalibration_needed": recal_needed,
                    "reason": "; ".join(reasons) if reasons else "",
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    @keyword_only(1)
    def calibration_stability(
        self, window: str = "8h", lookback: int = 10
    ) -> pd.DataFrame:
        """Track calibration bias stability over rolling windows.

        Args:
            window: Resample window size.
            lookback: Number of previous windows for trend assessment.

        Returns:
            DataFrame with columns: window_start, bias_mean, bias_std,
            bias_trend, stability_score.
        """
        cols = [
            "window_start",
            "bias_mean",
            "bias_std",
            "bias_trend",
            "stability_score",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)
        if self.reference_uuid is None and self.reference_value is None:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        # Compute per-window bias
        biases: list[tuple] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue
            window_end = ts + pd.to_timedelta(window)
            ref = self._get_reference_for_window(ts, window_end)
            if ref is None:
                continue
            bias = float(vals.mean()) - ref
            biases.append((ts, bias))

        if not biases:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        bias_values = [b[1] for b in biases]

        for i, (ts, bias) in enumerate(biases):
            start = max(0, i - lookback + 1)
            window_biases = bias_values[start : i + 1]

            bias_mean = float(np.mean(window_biases))
            bias_std = (
                float(np.std(window_biases, ddof=1)) if len(window_biases) > 1 else 0.0
            )

            # Trend from linear regression over lookback
            if len(window_biases) >= 3:
                x = np.arange(len(window_biases), dtype=float)
                slope, _, _, _, _ = stats.linregress(x, window_biases)
                bias_trend = float(slope)
            else:
                bias_trend = 0.0

            # Stability score: 100 = perfect (no bias, no variance, no trend)
            # Deduct for bias magnitude, variance, and trend
            score = 100.0
            score -= min(50.0, abs(bias_mean) * 100)  # bias penalty
            score -= min(30.0, bias_std * 100)  # variance penalty
            score -= min(20.0, abs(bias_trend) * 200)  # trend penalty
            score = max(0.0, score)

            events.append(
                {
                    "window_start": ts,
                    "bias_mean": float(bias_mean),
                    "bias_std": float(bias_std),
                    "bias_trend": float(bias_trend),
                    "stability_score": float(score),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
