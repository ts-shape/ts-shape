import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class OperatingRangeEvents(Base):
    """Engineering: Operating Range Analysis

    Analyze the operating envelope of a signal — what ranges it operates in,
    how often, and when the operating point shifts.

    Methods:
    - operating_envelope: Per-window min/max/mean/percentiles.
    - detect_regime_change: Detect significant shifts in the operating point.
    - time_in_range: Percentage of time within a defined range per window.
    - value_distribution: Histogram of signal values.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:operating_range",
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

    def operating_envelope(self, window: str = "1h") -> pd.DataFrame:
        """Per-window operating envelope statistics.

        Returns:
            DataFrame with columns: window_start, min_value, p5,
            mean_value, p95, max_value, range_width.
        """
        cols = [
            "window_start",
            "min_value",
            "p5",
            "mean_value",
            "p95",
            "max_value",
            "range_width",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
            if group.empty:
                continue
            vals = group.dropna()
            if vals.empty:
                continue
            p5 = float(np.percentile(vals, 5))
            p95 = float(np.percentile(vals, 95))
            events.append(
                {
                    "window_start": window_start,
                    "min_value": float(vals.min()),
                    "p5": p5,
                    "mean_value": float(vals.mean()),
                    "p95": p95,
                    "max_value": float(vals.max()),
                    "range_width": float(vals.max() - vals.min()),
                }
            )

        return pd.DataFrame(events, columns=cols)

    def detect_regime_change(
        self,
        window: str = "1h",
        shift_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Detect significant shifts in the operating point between windows.

        A regime change is flagged when the window mean differs from the
        previous window mean by more than shift_threshold * pooled std.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            prev_mean, new_mean, shift_magnitude.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "prev_mean",
            "new_mean",
            "shift_magnitude",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)

        window_stats: List[Dict[str, Any]] = []
        for window_start, group in groups:
            vals = group.dropna()
            if len(vals) < 2:
                continue
            window_stats.append(
                {
                    "window_start": window_start,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                }
            )

        if len(window_stats) < 2:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for i in range(1, len(window_stats)):
            prev = window_stats[i - 1]
            curr = window_stats[i]
            pooled_std = (prev["std"] + curr["std"]) / 2.0
            if pooled_std == 0:
                pooled_std = 1e-10
            shift_mag = abs(curr["mean"] - prev["mean"]) / pooled_std
            if shift_mag >= shift_threshold:
                events.append(
                    {
                        "start": prev["window_start"],
                        "end": curr["window_start"],
                        "uuid": self.event_uuid,
                        "is_delta": False,
                        "prev_mean": prev["mean"],
                        "new_mean": curr["mean"],
                        "shift_magnitude": float(shift_mag),
                    }
                )

        return pd.DataFrame(events, columns=cols)

    def time_in_range(
        self,
        lower: float,
        upper: float,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Percentage of time within a defined range per window.

        Returns:
            DataFrame with columns: window_start, time_in_range_pct,
            time_below_pct, time_above_pct.
        """
        cols = ["window_start", "time_in_range_pct", "time_below_pct", "time_above_pct"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
            vals = group.dropna()
            if vals.empty:
                continue
            n = len(vals)
            in_range = ((vals >= lower) & (vals <= upper)).sum()
            below = (vals < lower).sum()
            above = (vals > upper).sum()
            events.append(
                {
                    "window_start": window_start,
                    "time_in_range_pct": float(in_range / n * 100),
                    "time_below_pct": float(below / n * 100),
                    "time_above_pct": float(above / n * 100),
                }
            )

        return pd.DataFrame(events, columns=cols)

    def value_distribution(self, n_bins: int = 10) -> pd.DataFrame:
        """Histogram of signal values.

        Returns:
            DataFrame with columns: bin_lower, bin_upper, count, pct,
            cumulative_pct.
        """
        cols = ["bin_lower", "bin_upper", "count", "pct", "cumulative_pct"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        vals = self.signal[self.value_column].dropna()
        if vals.empty:
            return pd.DataFrame(columns=cols)

        counts, bin_edges = np.histogram(vals, bins=n_bins)
        total = int(counts.sum())

        events: List[Dict[str, Any]] = []
        cum = 0
        for i in range(len(counts)):
            c = int(counts[i])
            cum += c
            events.append(
                {
                    "bin_lower": float(bin_edges[i]),
                    "bin_upper": float(bin_edges[i + 1]),
                    "count": c,
                    "pct": float(c / total * 100) if total > 0 else 0.0,
                    "cumulative_pct": float(cum / total * 100) if total > 0 else 0.0,
                }
            )

        return pd.DataFrame(events, columns=cols)
