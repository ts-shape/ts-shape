import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from scipy import stats  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class SensorDriftEvents(Base):
    """Quality: Sensor Calibration Drift Detection

    Detect gradual calibration drift in inline sensors by tracking
    measurement behavior against reference values or historical baselines.

    Reference can be provided as a UUID (time-aligned signal in the
    DataFrame) or as a fixed float value.

    Methods:
    - detect_zero_drift: Track mean offset from baseline per window.
    - detect_span_drift: Track measurement sensitivity changes over time.
    - drift_trend: Rolling linear trend analysis on signal statistics.
    - calibration_health: Composite health score per window.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        reference_uuid: Optional[str] = None,
        reference_value: Optional[float] = None,
        value_column: str = "value_double",
        event_uuid: str = "quality:sensor_drift",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
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

        # Build reference series if UUID provided
        if self.reference_uuid is not None:
            ref = (
                self.dataframe[self.dataframe["uuid"] == self.reference_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            ref[self.time_column] = pd.to_datetime(ref[self.time_column])
            self._reference_series = ref
        else:
            self._reference_series = None

    def _get_reference_for_window(self, window_start, window_end) -> Optional[float]:
        """Get reference value for a time window."""
        if self.reference_value is not None:
            return self.reference_value
        if self._reference_series is not None and not self._reference_series.empty:
            mask = (self._reference_series[self.time_column] >= window_start) & (
                self._reference_series[self.time_column] < window_end
            )
            ref_vals = self._reference_series.loc[mask, self.value_column].dropna()
            if not ref_vals.empty:
                return float(ref_vals.mean())
        return None

    def detect_zero_drift(
        self, window: str = "8h", threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Track mean offset from baseline per window.

        Args:
            window: Resample window size.
            threshold: Offset threshold for severity. Auto-calculated
                as 3x std of first window if not provided.

        Returns:
            DataFrame with columns: window_start, window_end, mean_offset,
            drift_rate, severity.
        """
        cols = ["window_start", "window_end", "mean_offset", "drift_rate", "severity"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: List[Dict[str, Any]] = []
        prev_offset: Optional[float] = None
        auto_threshold: Optional[float] = None

        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue

            window_end = ts + pd.to_timedelta(window)
            ref = self._get_reference_for_window(ts, window_end)
            if ref is not None:
                mean_offset = float(vals.mean()) - ref
            else:
                # No reference: use first window as baseline
                if not events:
                    mean_offset = 0.0
                    auto_threshold = (
                        float(vals.std()) * 3 if threshold is None else None
                    )
                else:
                    mean_offset = float(vals.mean()) - (
                        events[0]["mean_offset"] + (ref if ref else float(vals.mean()))
                    )
                    # Fallback: offset relative to the first window mean
                    if not events:
                        mean_offset = 0.0
                    else:
                        first_mean = events[0].get("_raw_mean", float(vals.mean()))
                        mean_offset = float(vals.mean()) - first_mean

            # Store raw mean for baseline calculation
            raw_mean = float(vals.mean())

            # Auto-calculate threshold from first window
            if auto_threshold is None and threshold is None and not events:
                auto_threshold = float(vals.std()) * 3

            effective_threshold = (
                threshold
                if threshold is not None
                else (auto_threshold if auto_threshold else 1.0)
            )

            # Drift rate = change in offset between consecutive windows
            drift_rate = (mean_offset - prev_offset) if prev_offset is not None else 0.0
            prev_offset = mean_offset

            # Severity based on threshold
            abs_offset = abs(mean_offset)
            if abs_offset > effective_threshold:
                severity = "critical"
            elif abs_offset > effective_threshold * 0.66:
                severity = "high"
            elif abs_offset > effective_threshold * 0.33:
                severity = "medium"
            else:
                severity = "low"

            event = {
                "window_start": ts,
                "window_end": window_end,
                "mean_offset": round(mean_offset, 6),
                "drift_rate": round(drift_rate, 6),
                "severity": severity,
                "_raw_mean": raw_mean,
            }
            events.append(event)

        # Remove internal field
        for e in events:
            e.pop("_raw_mean", None)

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_span_drift(self, window: str = "8h") -> pd.DataFrame:
        """Track measurement sensitivity changes over time.

        Requires reference_uuid or reference_value. Computes ratio of
        signal mean to reference per window to detect gain/span changes.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, sensitivity,
            sensitivity_change_pct.
        """
        cols = ["window_start", "sensitivity", "sensitivity_change_pct"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)
        if self.reference_value is None and self._reference_series is None:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: List[Dict[str, Any]] = []
        baseline_sensitivity: Optional[float] = None

        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue

            window_end = ts + pd.to_timedelta(window)
            ref = self._get_reference_for_window(ts, window_end)
            if ref is None or ref == 0:
                continue

            sensitivity = float(vals.mean()) / ref

            if baseline_sensitivity is None:
                baseline_sensitivity = sensitivity

            change_pct = (
                (sensitivity - baseline_sensitivity) / abs(baseline_sensitivity)
            ) * 100

            events.append(
                {
                    "window_start": ts,
                    "sensitivity": round(sensitivity, 6),
                    "sensitivity_change_pct": round(change_pct, 4),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def drift_trend(self, window: str = "1D", metric: str = "mean") -> pd.DataFrame:
        """Rolling trend analysis on signal statistics.

        Args:
            window: Resample window size.
            metric: Statistic to trend ('mean' or 'std').

        Returns:
            DataFrame with columns: window_start, value, slope,
            r_squared, direction.
        """
        cols = ["window_start", "value", "slope", "r_squared", "direction"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        # Compute per-window metric
        window_values: List[tuple] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue
            if metric == "std":
                v = float(vals.std())
            else:
                v = float(vals.mean())
            window_values.append((ts, v))

        if len(window_values) < 2:
            return pd.DataFrame(columns=cols)

        timestamps = [wv[0] for wv in window_values]
        values = np.array([wv[1] for wv in window_values])
        x = np.arange(len(values), dtype=float)

        slope, intercept, r_value, _, _ = stats.linregress(x, values)
        r_squared = r_value**2

        # Direction is "stable" unless there is both a meaningful slope
        # AND a strong fit (R² > 0.8). This avoids labelling random noise
        # as trending.
        if r_squared < 0.8 or abs(slope) < 1e-10:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        events: List[Dict[str, Any]] = []
        for i, (ts, v) in enumerate(window_values):
            events.append(
                {
                    "window_start": ts,
                    "value": round(v, 6),
                    "slope": round(slope, 8),
                    "r_squared": round(r_squared, 4),
                    "direction": direction,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def calibration_health(
        self, window: str = "8h", tolerance: Optional[float] = None
    ) -> pd.DataFrame:
        """Composite calibration health score per window.

        Args:
            window: Resample window size.
            tolerance: Acceptable measurement tolerance. Used to normalize
                bias and precision into a 0-100 score.

        Returns:
            DataFrame with columns: window_start, bias, precision,
            health_score.
        """
        cols = ["window_start", "bias", "precision", "health_score"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: List[Dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue

            window_end = ts + pd.to_timedelta(window)
            ref = self._get_reference_for_window(ts, window_end)

            mean_val = float(vals.mean())
            precision = float(vals.std())

            if ref is not None:
                bias = abs(mean_val - ref)
            else:
                bias = 0.0

            # Health score: 100 = perfect, 0 = failing
            if tolerance is not None and tolerance > 0:
                bias_penalty = min(1.0, bias / tolerance) * 50
                precision_penalty = min(1.0, precision / (tolerance / 3)) * 50
                health_score = max(0.0, 100.0 - bias_penalty - precision_penalty)
            else:
                # Without tolerance, use relative scoring
                if precision > 0:
                    health_score = max(0.0, 100.0 - (bias / (precision + 1e-10)) * 25)
                else:
                    health_score = 100.0 if bias == 0 else 50.0

            events.append(
                {
                    "window_start": ts,
                    "bias": round(bias, 6),
                    "precision": round(precision, 6),
                    "health_score": round(health_score, 2),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
