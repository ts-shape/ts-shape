import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class RateOfChangeEvents(Base):
    """Engineering: Rate of Change Events

    Detect rapid changes and step jumps in a numeric signal.

    Methods:
    - detect_rapid_change: Flag intervals where rate of change exceeds threshold.
    - rate_statistics: Per-window rate of change statistics.
    - detect_step_changes: Sudden value jumps within a short duration.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:rate_of_change",
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

    def _compute_rate(self) -> pd.DataFrame:
        """Compute rate of change (value_diff / time_diff_seconds)."""
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=[self.time_column, "rate"])

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        value_diff = sig[self.value_column].diff()
        time_diff = sig[self.time_column].diff().dt.total_seconds()
        sig["rate"] = value_diff / time_diff.replace(0, np.nan)
        return sig.dropna(subset=["rate"])

    def detect_rapid_change(self, threshold: float, window: str = "1m") -> pd.DataFrame:
        """Flag intervals where rate of change exceeds threshold.

        Args:
            threshold: Minimum absolute rate (units/second) to flag.
            window: Minimum duration to group rapid change intervals.

        Returns:
            DataFrame with columns: start_time, end_time, max_rate, direction.
        """
        cols = ["start_time", "end_time", "max_rate", "direction"]
        rate_df = self._compute_rate()
        if rate_df.empty:
            return pd.DataFrame(columns=cols)

        rapid = rate_df["rate"].abs() >= threshold
        if not rapid.any():
            return pd.DataFrame(columns=cols)

        groups = (rapid != rapid.shift()).cumsum()
        events: List[Dict[str, Any]] = []

        for _, seg in rate_df.groupby(groups):
            seg_rapid = rapid.loc[seg.index]
            if not seg_rapid.iloc[0]:
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            max_rate_idx = seg["rate"].abs().idxmax()
            max_rate = float(seg.loc[max_rate_idx, "rate"])

            events.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "max_rate": max_rate,
                    "direction": "increasing" if max_rate > 0 else "decreasing",
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def rate_statistics(self, window: str = "1h") -> pd.DataFrame:
        """Per-window rate of change statistics.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: window_start, mean_rate, std_rate,
            max_rate, min_rate.
        """
        cols = ["window_start", "mean_rate", "std_rate", "max_rate", "min_rate"]
        rate_df = self._compute_rate()
        if rate_df.empty:
            return pd.DataFrame(columns=cols)

        rate_indexed = rate_df.set_index(self.time_column)
        abs_rate = rate_indexed["rate"].abs()

        resampled = abs_rate.resample(window).agg(["mean", "std", "max", "min"])
        resampled = resampled.dropna(subset=["mean"])

        result = resampled.reset_index()
        result.columns = [
            "window_start",
            "mean_rate",
            "std_rate",
            "max_rate",
            "min_rate",
        ]

        for col in ["mean_rate", "std_rate", "max_rate", "min_rate"]:
            result[col] = result[col].round(6)

        return result[cols]

    def detect_step_changes(
        self, min_delta: float, max_duration: str = "5s"
    ) -> pd.DataFrame:
        """Detect sudden value jumps within a short duration.

        Args:
            min_delta: Minimum absolute value change to qualify.
            max_duration: Maximum time window for the step to occur.

        Returns:
            DataFrame with columns: time, value_before, value_after, delta, duration.
        """
        cols = ["time", "value_before", "value_after", "delta", "duration"]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        max_td = pd.to_timedelta(max_duration)

        events: List[Dict[str, Any]] = []
        values = sig[self.value_column].values
        times = sig[self.time_column].values

        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            duration = pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])

            if abs(delta) >= min_delta and duration <= max_td:
                events.append(
                    {
                        "time": pd.Timestamp(times[i]),
                        "value_before": float(values[i - 1]),
                        "value_after": float(values[i]),
                        "delta": float(delta),
                        "duration": duration,
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
