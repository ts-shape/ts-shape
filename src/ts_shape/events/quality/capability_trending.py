import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from scipy import stats  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class CapabilityTrendingEvents(Base):
    """Quality: Process Capability Trending

    Track process capability indices (Cp, Cpk, Pp, Ppk) over rolling time
    windows to detect capability degradation before quality escapes occur.

    Specification limits can be provided as fixed floats or as UUIDs
    referencing signal rows in the DataFrame (matching the
    ToleranceDeviationEvents pattern).

    Methods:
    - capability_over_time: Cp/Cpk/Pp/Ppk per time window.
    - detect_capability_drop: Alert when Cpk falls below threshold.
    - capability_forecast: Extrapolate Cpk trend to predict threshold breach.
    - yield_estimate: Estimated yield, DPMO, and sigma level per window.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        upper_spec: Optional[float] = None,
        lower_spec: Optional[float] = None,
        upper_spec_uuid: Optional[str] = None,
        lower_spec_uuid: Optional[str] = None,
        value_column: str = "value_double",
        event_uuid: str = "quality:capability_trend",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # Resolve spec limits
        if upper_spec is not None:
            self.upper_spec_fixed: Optional[float] = float(upper_spec)
        else:
            self.upper_spec_fixed = None

        if lower_spec is not None:
            self.lower_spec_fixed: Optional[float] = float(lower_spec)
        else:
            self.lower_spec_fixed = None

        self.upper_spec_uuid = upper_spec_uuid
        self.lower_spec_uuid = lower_spec_uuid

        if self.upper_spec_fixed is None and self.upper_spec_uuid is None:
            raise ValueError(
                "Either upper_spec (float) or upper_spec_uuid must be provided"
            )
        if self.lower_spec_fixed is None and self.lower_spec_uuid is None:
            raise ValueError(
                "Either lower_spec (float) or lower_spec_uuid must be provided"
            )

        # Extract signal data
        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    def _resolve_spec_limits(self, window_df: pd.DataFrame) -> tuple:
        """Resolve upper/lower spec limits for a given window."""
        if self.upper_spec_fixed is not None:
            usl = self.upper_spec_fixed
        else:
            spec_rows = self.dataframe[self.dataframe["uuid"] == self.upper_spec_uuid]
            usl = (
                spec_rows[self.value_column].iloc[-1] if not spec_rows.empty else np.nan
            )

        if self.lower_spec_fixed is not None:
            lsl = self.lower_spec_fixed
        else:
            spec_rows = self.dataframe[self.dataframe["uuid"] == self.lower_spec_uuid]
            lsl = (
                spec_rows[self.value_column].iloc[-1] if not spec_rows.empty else np.nan
            )

        return float(usl), float(lsl)

    def capability_over_time(self, window: str = "8h") -> pd.DataFrame:
        """Compute Cp, Cpk, Pp, Ppk per time window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, cp, cpk, pp, ppk,
            mean, std, n_samples.
        """
        cols = ["window_start", "cp", "cpk", "pp", "ppk", "mean", "std", "n_samples"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        # Overall std for Pp/Ppk
        overall_std = sig[self.value_column].std()
        if overall_std == 0 or pd.isna(overall_std):
            overall_std = np.nan

        events: List[Dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            n = len(vals)
            if n < 2:
                continue

            usl, lsl = self._resolve_spec_limits(group)
            if np.isnan(usl) or np.isnan(lsl):
                continue

            mean = float(vals.mean())
            std = float(vals.std())
            tol_range = usl - lsl

            if std > 0:
                cp = tol_range / (6 * std)
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
            else:
                cp = np.nan
                cpk = np.nan

            if overall_std > 0 and not np.isnan(overall_std):
                pp = tol_range / (6 * overall_std)
                ppk = min(
                    (usl - mean) / (3 * overall_std),
                    (mean - lsl) / (3 * overall_std),
                )
            else:
                pp = np.nan
                ppk = np.nan

            events.append(
                {
                    "window_start": ts,
                    "cp": round(cp, 4) if not np.isnan(cp) else np.nan,
                    "cpk": round(cpk, 4) if not np.isnan(cpk) else np.nan,
                    "pp": round(pp, 4) if not np.isnan(pp) else np.nan,
                    "ppk": round(ppk, 4) if not np.isnan(ppk) else np.nan,
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "n_samples": n,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_capability_drop(
        self,
        window: str = "8h",
        min_cpk: float = 1.33,
        lookback: int = 5,
    ) -> pd.DataFrame:
        """Detect windows where Cpk drops below threshold or degrades significantly.

        Args:
            window: Resample window size.
            min_cpk: Minimum acceptable Cpk value.
            lookback: Number of previous windows for rolling average comparison.

        Returns:
            DataFrame with columns: window_start, cpk, prev_avg_cpk, drop_pct, alert.
        """
        cols = ["window_start", "cpk", "prev_avg_cpk", "drop_pct", "alert"]
        cap = self.capability_over_time(window)
        if cap.empty:
            return pd.DataFrame(columns=cols)

        cpk_vals = cap["cpk"].values
        events: List[Dict[str, Any]] = []

        for i in range(len(cap)):
            cpk = cpk_vals[i]
            if np.isnan(cpk):
                continue

            # Rolling average of previous windows
            start = max(0, i - lookback)
            prev = cpk_vals[start:i]
            prev = prev[~np.isnan(prev)]
            prev_avg = float(np.mean(prev)) if len(prev) > 0 else np.nan

            if not np.isnan(prev_avg) and prev_avg > 0:
                drop_pct = round((prev_avg - cpk) / prev_avg * 100, 2)
            else:
                drop_pct = 0.0

            alert = cpk < min_cpk or drop_pct > 20.0

            events.append(
                {
                    "window_start": cap.iloc[i]["window_start"],
                    "cpk": round(cpk, 4),
                    "prev_avg_cpk": (
                        round(prev_avg, 4) if not np.isnan(prev_avg) else np.nan
                    ),
                    "drop_pct": drop_pct,
                    "alert": alert,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def capability_forecast(
        self,
        window: str = "8h",
        horizon: int = 5,
        threshold: float = 1.33,
    ) -> pd.DataFrame:
        """Extrapolate Cpk trend to predict when it will breach threshold.

        Args:
            window: Resample window size.
            horizon: Number of future windows to forecast.
            threshold: Cpk threshold to predict breach for.

        Returns:
            DataFrame with columns: window_start, cpk, trend_slope,
            forecast_cpk, windows_to_threshold.
        """
        cols = [
            "window_start",
            "cpk",
            "trend_slope",
            "forecast_cpk",
            "windows_to_threshold",
        ]
        cap = self.capability_over_time(window)
        if cap.empty or len(cap) < 2:
            return pd.DataFrame(columns=cols)

        cpk_vals = cap["cpk"].values
        valid_mask = ~np.isnan(cpk_vals)
        if valid_mask.sum() < 2:
            return pd.DataFrame(columns=cols)

        x = np.arange(len(cpk_vals))[valid_mask].astype(float)
        y = cpk_vals[valid_mask]
        slope, intercept, _, _, _ = stats.linregress(x, y)

        events: List[Dict[str, Any]] = []
        for i in range(len(cap)):
            cpk = cpk_vals[i]
            if np.isnan(cpk):
                continue

            forecast_cpk = slope * (i + horizon) + intercept

            # Windows until threshold breach
            if slope < 0:
                windows_to = (threshold - intercept - slope * i) / slope
                windows_to = max(0, round(windows_to - i, 1))
            else:
                windows_to = np.nan  # Not degrading

            events.append(
                {
                    "window_start": cap.iloc[i]["window_start"],
                    "cpk": round(cpk, 4),
                    "trend_slope": round(slope, 6),
                    "forecast_cpk": round(forecast_cpk, 4),
                    "windows_to_threshold": windows_to,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def yield_estimate(self, window: str = "8h") -> pd.DataFrame:
        """Estimate yield, DPMO, and sigma level per window.

        Assumes normal distribution within each window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, estimated_yield_pct,
            dpmo, sigma_level.
        """
        cols = ["window_start", "estimated_yield_pct", "dpmo", "sigma_level"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: List[Dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            n = len(vals)
            if n < 2:
                continue

            usl, lsl = self._resolve_spec_limits(group)
            if np.isnan(usl) or np.isnan(lsl):
                continue

            mean = float(vals.mean())
            std = float(vals.std())
            if std <= 0:
                continue

            # P(LSL < X < USL) under normal distribution
            p_upper = stats.norm.cdf(usl, loc=mean, scale=std)
            p_lower = stats.norm.cdf(lsl, loc=mean, scale=std)
            yield_pct = (p_upper - p_lower) * 100
            dpmo = (1 - yield_pct / 100) * 1_000_000
            sigma_level = (
                stats.norm.ppf(1 - dpmo / 1_000_000) + 1.5 if dpmo > 0 else 6.0
            )

            events.append(
                {
                    "window_start": ts,
                    "estimated_yield_pct": round(yield_pct, 4),
                    "dpmo": round(dpmo, 1),
                    "sigma_level": round(sigma_level, 2),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
