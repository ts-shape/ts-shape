import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ts_shape.events._windows import finish_events
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only

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
        upper_spec: float | None = None,
        lower_spec: float | None = None,
        upper_spec_uuid: str | None = None,
        lower_spec_uuid: str | None = None,
        value_column: str = "value_double",
        event_uuid: str = "quality:capability_trend",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # Resolve spec limits
        if upper_spec is not None:
            self.upper_spec_fixed: float | None = float(upper_spec)
        else:
            self.upper_spec_fixed = None

        if lower_spec is not None:
            self.lower_spec_fixed: float | None = float(lower_spec)
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

    @keyword_only(1)
    def capability_over_time(self, window: str = "8h") -> pd.DataFrame:
        """Compute Cp, Cpk, Pp, Ppk per time window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, cp, cpk, pp, ppk, mean, std, n_samples.
            ``end`` is ``start`` plus one ``window``.
        """
        cols = ["start", "cp", "cpk", "pp", "ppk", "mean", "std", "n_samples"]
        if self.signal.empty:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        # Overall std for Pp/Ppk
        overall_std = sig[self.value_column].std()
        if overall_std == 0 or pd.isna(overall_std):
            overall_std = np.nan

        events: list[dict[str, Any]] = []
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
                    "start": ts,
                    "cp": float(cp),
                    "cpk": float(cpk),
                    "pp": float(pp),
                    "ppk": float(ppk),
                    "mean": float(mean),
                    "std": float(std),
                    "n_samples": n,
                }
            )

        return finish_events(
            events,
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )

    @keyword_only(1)
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
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, cpk, prev_avg_cpk, drop_pct, alert.
            ``end`` is ``start`` plus one ``window``.
        """
        cols = ["start", "cpk", "prev_avg_cpk", "drop_pct", "alert"]
        cap = self.capability_over_time(window=window)
        if cap.empty:
            return finish_events(
                [],
                cols,
                "interval",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        cpk_vals = cap["cpk"].values
        events: list[dict[str, Any]] = []

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
                drop_pct = float((prev_avg - cpk) / prev_avg * 100)
            else:
                drop_pct = 0.0

            alert = cpk < min_cpk or drop_pct > 20.0

            events.append(
                {
                    "start": cap.iloc[i]["start"],
                    "cpk": float(cpk),
                    "prev_avg_cpk": (float(prev_avg)),
                    "drop_pct": drop_pct,
                    "alert": alert,
                }
            )

        return finish_events(
            events,
            cols,
            "interval",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )

    @keyword_only(1)
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
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, cpk, trend_slope, forecast_cpk,
            windows_to_threshold. ``end`` is ``start`` plus one ``window``.
        """
        cols = [
            "start",
            "cpk",
            "trend_slope",
            "forecast_cpk",
            "windows_to_threshold",
        ]
        cap = self.capability_over_time(window=window)
        if cap.empty or len(cap) < 2:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        cpk_vals = cap["cpk"].values
        valid_mask = ~np.isnan(cpk_vals)
        if valid_mask.sum() < 2:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        x = np.arange(len(cpk_vals))[valid_mask].astype(float)
        y = cpk_vals[valid_mask]
        slope, intercept, _, _, _ = stats.linregress(x, y)

        events: list[dict[str, Any]] = []
        for i in range(len(cap)):
            cpk = cpk_vals[i]
            if np.isnan(cpk):
                continue

            forecast_cpk = slope * (i + horizon) + intercept

            # Windows until threshold breach
            if slope < 0:
                # The fitted line crosses the threshold at window index
                # (threshold - intercept) / slope; count windows from i.
                breach_at = (threshold - intercept) / slope
                windows_to = max(0, float(breach_at - i))
            else:
                windows_to = np.nan  # Not degrading

            events.append(
                {
                    "start": cap.iloc[i]["start"],
                    "cpk": float(cpk),
                    "trend_slope": float(slope),
                    "forecast_cpk": float(forecast_cpk),
                    "windows_to_threshold": windows_to,
                }
            )

        return finish_events(
            events,
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )

    @keyword_only(1)
    def yield_estimate(self, window: str = "8h") -> pd.DataFrame:
        """Estimate yield, DPMO, and sigma level per window.

        Assumes normal distribution within each window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, estimated_yield_pct, dpmo, sigma_level.
            ``end`` is ``start`` plus one ``window``.
        """
        cols = ["start", "estimated_yield_pct", "dpmo", "sigma_level"]
        if self.signal.empty:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: list[dict[str, Any]] = []
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
                    "start": ts,
                    "estimated_yield_pct": float(yield_pct),
                    "dpmo": float(dpmo),
                    "sigma_level": float(sigma_level),
                }
            )

        return finish_events(
            events,
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )
