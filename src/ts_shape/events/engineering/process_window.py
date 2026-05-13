import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ProcessWindowEvents(Base):
    """Engineering: Process Window Analysis

    Analyze time-windowed process statistics for shift reports, SPC context,
    and trend monitoring. Answers 'how is my process doing this hour/shift/day?'

    Methods:
    - windowed_statistics: Per-window count, mean, std, min, max, percentiles.
    - detect_mean_shift: Flag windows where mean shifts significantly.
    - detect_variance_change: Flag windows where variance changes significantly.
    - window_comparison: Compare each window to overall baseline.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:process_window",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    def windowed_statistics(self, window: str = "1h") -> pd.DataFrame:
        """Per-window descriptive statistics.

        Returns:
            DataFrame with columns: window_start, count, mean, std,
            min, max, median, p25, p75, range.
        """
        cols = [
            "window_start",
            "count",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "p25",
            "p75",
            "range",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
            vals = group.dropna()
            if vals.empty:
                continue
            events.append(
                {
                    "window_start": window_start,
                    "count": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "median": float(vals.median()),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "range": float(vals.max() - vals.min()),
                }
            )

        return pd.DataFrame(events, columns=cols)

    def detect_mean_shift(
        self,
        window: str = "1h",
        sensitivity: float = 2.0,
    ) -> pd.DataFrame:
        """Flag windows where mean shifts significantly from the previous window.

        A shift is detected when |current_mean - prev_mean| > sensitivity * prev_std.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            prev_mean, current_mean, shift_sigma.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "prev_mean",
            "current_mean",
            "shift_sigma",
        ]
        stats = self.windowed_statistics(window)
        if len(stats) < 2:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for i in range(1, len(stats)):
            prev = stats.iloc[i - 1]
            curr = stats.iloc[i]
            prev_std = prev["std"] if prev["std"] > 0 else 1e-10
            shift = abs(curr["mean"] - prev["mean"]) / prev_std
            if shift >= sensitivity:
                events.append(
                    {
                        "start": prev["window_start"],
                        "end": curr["window_start"],
                        "uuid": self.event_uuid,
                        "is_delta": False,
                        "prev_mean": prev["mean"],
                        "current_mean": curr["mean"],
                        "shift_sigma": float(shift),
                    }
                )

        return pd.DataFrame(events, columns=cols)

    def detect_variance_change(
        self,
        window: str = "1h",
        ratio_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Flag windows where variance changes significantly.

        A change is detected when (current_std / prev_std) > ratio_threshold
        or < (1 / ratio_threshold).

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            prev_std, current_std, variance_ratio.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "prev_std",
            "current_std",
            "variance_ratio",
        ]
        stats = self.windowed_statistics(window)
        if len(stats) < 2:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for i in range(1, len(stats)):
            prev = stats.iloc[i - 1]
            curr = stats.iloc[i]
            if prev["std"] == 0 and curr["std"] == 0:
                continue
            if prev["std"] == 0:
                ratio = float("inf")
            else:
                ratio = curr["std"] / prev["std"]
            if ratio >= ratio_threshold or (
                ratio > 0 and ratio <= 1.0 / ratio_threshold
            ):
                events.append(
                    {
                        "start": prev["window_start"],
                        "end": curr["window_start"],
                        "uuid": self.event_uuid,
                        "is_delta": False,
                        "prev_std": prev["std"],
                        "current_std": curr["std"],
                        "variance_ratio": float(ratio),
                    }
                )

        return pd.DataFrame(events, columns=cols)

    def window_comparison(self, window: str = "1h") -> pd.DataFrame:
        """Compare each window mean to the overall baseline.

        Returns:
            DataFrame with columns: window_start, mean, z_score_vs_global,
            is_anomalous.
        """
        cols = ["window_start", "mean", "z_score_vs_global", "is_anomalous"]
        stats = self.windowed_statistics(window)
        if stats.empty:
            return pd.DataFrame(columns=cols)

        global_mean = float(stats["mean"].mean())
        global_std = float(stats["mean"].std()) if len(stats) > 1 else 1e-10
        if global_std == 0:
            global_std = 1e-10

        result = stats[["window_start", "mean"]].copy()
        result["z_score_vs_global"] = (result["mean"] - global_mean) / global_std
        result["is_anomalous"] = result["z_score_vs_global"].abs() > 2.0
        return result[cols].reset_index(drop=True)
