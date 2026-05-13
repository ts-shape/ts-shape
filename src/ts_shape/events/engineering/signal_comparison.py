import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class SignalComparisonEvents(Base):
    """Engineering: Signal Comparison

    Compare two related signals (e.g. setpoint vs actual, sensor A vs sensor B)
    and detect divergence, compute deviation statistics, and track correlation.

    Methods:
    - detect_divergence: Intervals where |actual - reference| exceeds tolerance.
    - deviation_statistics: Per-window MAE, max error, RMSE, bias.
    - tracking_error_trend: Whether deviation is growing or shrinking over time.
    - correlation_windows: Per-window Pearson correlation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        reference_uuid: str,
        *,
        event_uuid: str = "eng:signal_comparison",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.reference_uuid = reference_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.reference = (
            self.dataframe[self.dataframe["uuid"] == self.reference_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.reference[self.time_column] = pd.to_datetime(
            self.reference[self.time_column]
        )

    def _align(self, actual_uuid: str) -> pd.DataFrame:
        """Align reference and actual signals by nearest timestamp."""
        actual = (
            self.dataframe[self.dataframe["uuid"] == actual_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        actual[self.time_column] = pd.to_datetime(actual[self.time_column])

        if self.reference.empty or actual.empty:
            return pd.DataFrame(columns=[self.time_column, "ref", "act"])

        ref = self.reference[[self.time_column, self.value_column]].rename(
            columns={self.value_column: "ref"}
        )
        act = actual[[self.time_column, self.value_column]].rename(
            columns={self.value_column: "act"}
        )

        merged = pd.merge_asof(
            ref.sort_values(self.time_column),
            act.sort_values(self.time_column),
            on=self.time_column,
            direction="nearest",
        )
        merged["deviation"] = merged["act"] - merged["ref"]
        merged["abs_deviation"] = merged["deviation"].abs()
        return merged.dropna(subset=["ref", "act"])

    def detect_divergence(
        self,
        actual_uuid: str,
        tolerance: float,
        min_duration: str = "1m",
    ) -> pd.DataFrame:
        """Detect intervals where |actual - reference| > tolerance.

        Args:
            actual_uuid: UUID of the actual/comparison signal.
            tolerance: Maximum acceptable absolute deviation.
            min_duration: Minimum duration of divergence interval.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            max_deviation, mean_deviation, direction.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "max_deviation",
            "mean_deviation",
            "direction",
        ]
        merged = self._align(actual_uuid)
        if merged.empty:
            return pd.DataFrame(columns=cols)

        exceeded = merged["abs_deviation"] > tolerance
        if not exceeded.any():
            return pd.DataFrame(columns=cols)

        min_td = pd.Timedelta(min_duration)
        groups = (exceeded != exceeded.shift()).cumsum()
        events: List[Dict[str, Any]] = []

        for _, seg in merged.groupby(groups):
            seg_exc = exceeded.loc[seg.index]
            if not seg_exc.iloc[0]:
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            if (end - start) < min_td:
                continue
            mean_dev = float(seg["deviation"].mean())
            events.append(
                {
                    "start": start,
                    "end": end,
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "max_deviation": float(seg["abs_deviation"].max()),
                    "mean_deviation": float(seg["abs_deviation"].mean()),
                    "direction": "above" if mean_dev > 0 else "below",
                }
            )

        return pd.DataFrame(events, columns=cols)

    def deviation_statistics(
        self,
        actual_uuid: str,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Per-window deviation statistics.

        Returns:
            DataFrame with columns: window_start, mae, max_error, rmse, bias.
        """
        cols = ["window_start", "mae", "max_error", "rmse", "bias"]
        merged = self._align(actual_uuid)
        if merged.empty:
            return pd.DataFrame(columns=cols)

        indexed = merged.set_index(self.time_column)
        groups = indexed.resample(window)

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
            if group.empty:
                continue
            dev = group["deviation"]
            events.append(
                {
                    "window_start": window_start,
                    "mae": float(dev.abs().mean()),
                    "max_error": float(dev.abs().max()),
                    "rmse": float(np.sqrt((dev**2).mean())),
                    "bias": float(dev.mean()),
                }
            )

        return pd.DataFrame(events, columns=cols)

    def tracking_error_trend(
        self,
        actual_uuid: str,
        window: str = "1D",
    ) -> pd.DataFrame:
        """Track whether deviation is growing or shrinking over time.

        Returns:
            DataFrame with columns: window_start, mae, trend_slope,
            trend_direction.
        """
        cols = ["window_start", "mae", "trend_slope", "trend_direction"]
        stats = self.deviation_statistics(actual_uuid, window)
        if stats.empty or len(stats) < 2:
            return pd.DataFrame(columns=cols)

        result = stats[["window_start", "mae"]].copy()
        # Slope of MAE over consecutive windows
        mae_diff = result["mae"].diff()
        result["trend_slope"] = mae_diff
        result["trend_direction"] = np.where(
            mae_diff > 0.01,
            "worsening",
            np.where(mae_diff < -0.01, "improving", "stable"),
        )
        return result[cols].dropna().reset_index(drop=True)

    def correlation_windows(
        self,
        actual_uuid: str,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Per-window Pearson correlation between reference and actual.

        Returns:
            DataFrame with columns: window_start, correlation, sample_count.
        """
        cols = ["window_start", "correlation", "sample_count"]
        merged = self._align(actual_uuid)
        if merged.empty:
            return pd.DataFrame(columns=cols)

        indexed = merged.set_index(self.time_column)
        groups = indexed.resample(window)

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
            if len(group) < 2:
                continue
            corr = group["ref"].corr(group["act"])
            events.append(
                {
                    "window_start": window_start,
                    "correlation": float(corr) if not np.isnan(corr) else 0.0,
                    "sample_count": len(group),
                }
            )

        return pd.DataFrame(events, columns=cols)
