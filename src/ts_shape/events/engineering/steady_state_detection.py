import logging
import pandas as pd  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class SteadyStateDetectionEvents(Base):
    """Engineering: Steady-State Detection

    Identify when a process signal has settled into a steady operating state
    (low variance, no trend) versus transient/dynamic periods.

    Methods:
    - detect_steady_state: Intervals where rolling std stays below threshold.
    - detect_transient_periods: Inverse — intervals where signal is changing.
    - steady_state_statistics: Summary of steady vs transient time.
    - steady_state_value_bands: Operating band (mean +/- std) per steady interval.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:steady_state",
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

    def _rolling_std(self, window: str) -> pd.Series:
        """Compute rolling std indexed by time."""
        sig = self.signal.set_index(self.time_column)[self.value_column]
        td = pd.Timedelta(window)
        return sig.rolling(td, min_periods=2).std().fillna(0.0)

    def _intervalize(
        self, mask: pd.Series, min_duration: str = "0s"
    ) -> List[Dict[str, Any]]:
        """Convert a boolean mask (indexed by timestamp) into intervals."""
        if mask.empty or not mask.any():
            return []

        min_td = pd.Timedelta(min_duration)
        groups = (mask != mask.shift()).cumsum()
        intervals: List[Dict[str, Any]] = []

        for _, seg in mask.groupby(groups):
            if not seg.iloc[0]:
                continue
            start = seg.index[0]
            end = seg.index[-1]
            dur = end - start
            if dur >= min_td:
                intervals.append({"start": start, "end": end})
        return intervals

    def detect_steady_state(
        self,
        window: str = "5m",
        std_threshold: float = 1.0,
        min_duration: str = "10m",
    ) -> pd.DataFrame:
        """Detect intervals where signal is in steady state.

        Args:
            window: Rolling window for std computation.
            std_threshold: Maximum rolling std to consider steady.
            min_duration: Minimum duration of a steady interval.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            mean_value, std_value, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "mean_value",
            "std_value",
            "duration_seconds",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        rolling_std = self._rolling_std(window)
        steady_mask = rolling_std < std_threshold
        intervals = self._intervalize(steady_mask, min_duration)

        if not intervals:
            return pd.DataFrame(columns=cols)

        sig = self.signal.set_index(self.time_column)[self.value_column]
        events: List[Dict[str, Any]] = []
        for iv in intervals:
            segment = sig.loc[iv["start"] : iv["end"]]
            events.append(
                {
                    "start": iv["start"],
                    "end": iv["end"],
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "mean_value": float(segment.mean()),
                    "std_value": float(segment.std()) if len(segment) > 1 else 0.0,
                    "duration_seconds": (iv["end"] - iv["start"]).total_seconds(),
                }
            )

        return pd.DataFrame(events, columns=cols)

    def detect_transient_periods(
        self,
        window: str = "5m",
        std_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Detect intervals where signal is in transient/dynamic state.

        Args:
            window: Rolling window for std computation.
            std_threshold: Minimum rolling std to consider transient.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            max_std, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "max_std",
            "duration_seconds",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        rolling_std = self._rolling_std(window)
        transient_mask = rolling_std >= std_threshold
        intervals = self._intervalize(transient_mask)

        if not intervals:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for iv in intervals:
            seg_std = rolling_std.loc[iv["start"] : iv["end"]]
            events.append(
                {
                    "start": iv["start"],
                    "end": iv["end"],
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "max_std": float(seg_std.max()),
                    "duration_seconds": (iv["end"] - iv["start"]).total_seconds(),
                }
            )

        return pd.DataFrame(events, columns=cols)

    def steady_state_statistics(
        self,
        window: str = "5m",
        std_threshold: float = 1.0,
        min_duration: str = "10m",
    ) -> Dict[str, Any]:
        """Summary statistics of steady vs transient time.

        Returns:
            Dict with: total_steady_seconds, total_transient_seconds,
            steady_pct, num_steady_periods, avg_steady_duration_seconds.
        """
        if self.signal.empty or len(self.signal) < 2:
            return {
                "total_steady_seconds": 0.0,
                "total_transient_seconds": 0.0,
                "steady_pct": 0.0,
                "num_steady_periods": 0,
                "avg_steady_duration_seconds": 0.0,
            }

        steady = self.detect_steady_state(window, std_threshold, min_duration)
        total_span = (
            self.signal[self.time_column].max() - self.signal[self.time_column].min()
        ).total_seconds()

        total_steady = (
            float(steady["duration_seconds"].sum()) if not steady.empty else 0.0
        )
        total_transient = max(total_span - total_steady, 0.0)
        n = len(steady)

        return {
            "total_steady_seconds": total_steady,
            "total_transient_seconds": total_transient,
            "steady_pct": (total_steady / total_span * 100) if total_span > 0 else 0.0,
            "num_steady_periods": n,
            "avg_steady_duration_seconds": (total_steady / n) if n > 0 else 0.0,
        }

    def steady_state_value_bands(
        self,
        window: str = "5m",
        std_threshold: float = 1.0,
        min_duration: str = "10m",
    ) -> pd.DataFrame:
        """Operating band (mean +/- std) for each steady-state interval.

        Returns:
            DataFrame with columns: start, end, mean_value, lower_band,
            upper_band, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "mean_value",
            "lower_band",
            "upper_band",
            "duration_seconds",
        ]
        steady = self.detect_steady_state(window, std_threshold, min_duration)
        if steady.empty:
            return pd.DataFrame(columns=cols)

        result = steady[["start", "end", "mean_value", "duration_seconds"]].copy()
        result["lower_band"] = steady["mean_value"] - steady["std_value"]
        result["upper_band"] = steady["mean_value"] + steady["std_value"]
        return result[cols].reset_index(drop=True)
