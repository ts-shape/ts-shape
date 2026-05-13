import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class SignalQualityEvents(Base):
    """Quality: Signal Quality Monitoring

    Detect data quality issues in a numeric signal: missing data,
    irregular sampling, out-of-range values, and data completeness.

    Methods:
    - detect_missing_data: Find gaps exceeding expected sampling frequency.
    - sampling_regularity: Inter-sample interval statistics per window.
    - detect_out_of_range: Flag values outside physical/expected bounds.
    - data_completeness: Percentage of expected samples received per window.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "quality:signal",
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

    def detect_missing_data(
        self, expected_freq: str = "1s", tolerance_factor: float = 2.0
    ) -> pd.DataFrame:
        """Find gaps where time between samples exceeds expected frequency.

        Args:
            expected_freq: Expected sampling frequency (e.g. '1s', '100ms').
            tolerance_factor: Multiplier on expected_freq to trigger a gap.

        Returns:
            DataFrame with columns: gap_start, gap_end, gap_duration,
            expected_samples_missing.
        """
        cols = ["gap_start", "gap_end", "gap_duration", "expected_samples_missing"]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        times = self.signal[self.time_column].values
        diffs = np.diff(times)
        expected_td = pd.to_timedelta(expected_freq)
        threshold = expected_td * tolerance_factor

        events: List[Dict[str, Any]] = []
        for i, d in enumerate(diffs):
            gap = pd.Timedelta(d)
            if gap > threshold:
                expected_missing = int(gap / expected_td) - 1
                events.append(
                    {
                        "gap_start": pd.Timestamp(times[i]),
                        "gap_end": pd.Timestamp(times[i + 1]),
                        "gap_duration": gap,
                        "expected_samples_missing": max(expected_missing, 1),
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def sampling_regularity(self, window: str = "1h") -> pd.DataFrame:
        """Inter-sample interval statistics per window.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: window_start, mean_interval, std_interval,
            min_interval, max_interval, regularity_score.
        """
        cols = [
            "window_start",
            "mean_interval",
            "std_interval",
            "min_interval",
            "max_interval",
            "regularity_score",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        sig = self.signal[[self.time_column]].copy().set_index(self.time_column)
        sig["interval"] = sig.index.to_series().diff().dt.total_seconds()
        sig = sig.dropna(subset=["interval"])

        if sig.empty:
            return pd.DataFrame(columns=cols)

        resampled = (
            sig["interval"].resample(window).agg(["mean", "std", "min", "max", "count"])
        )

        events: List[Dict[str, Any]] = []
        for ts, row in resampled.iterrows():
            if row["count"] < 2 or pd.isna(row["mean"]):
                continue
            mean_val = row["mean"]
            std_val = row["std"] if pd.notna(row["std"]) else 0.0
            # Regularity score: 1.0 = perfectly regular, 0.0 = very irregular
            regularity = max(0.0, 1.0 - (std_val / (mean_val + 1e-10)))
            events.append(
                {
                    "window_start": ts,
                    "mean_interval": round(mean_val, 4),
                    "std_interval": round(std_val, 4),
                    "min_interval": round(row["min"], 4),
                    "max_interval": round(row["max"], 4),
                    "regularity_score": round(max(0.0, min(1.0, regularity)), 4),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_out_of_range(self, min_value: float, max_value: float) -> pd.DataFrame:
        """Flag intervals where signal is outside expected range.

        Args:
            min_value: Minimum acceptable value.
            max_value: Maximum acceptable value.

        Returns:
            DataFrame with columns: start_time, end_time, duration,
            min_observed, max_observed, direction.
        """
        cols = [
            "start_time",
            "end_time",
            "duration",
            "min_observed",
            "max_observed",
            "direction",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        out_of_range = (sig[self.value_column] < min_value) | (
            sig[self.value_column] > max_value
        )

        if not out_of_range.any():
            return pd.DataFrame(columns=cols)

        groups = (out_of_range != out_of_range.shift()).cumsum()
        events: List[Dict[str, Any]] = []

        for _, seg in sig.groupby(groups):
            seg_oor = out_of_range.loc[seg.index]
            if not seg_oor.iloc[0]:
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            min_obs = float(seg[self.value_column].min())
            max_obs = float(seg[self.value_column].max())

            if max_obs > max_value:
                direction = "above"
            elif min_obs < min_value:
                direction = "below"
            else:
                direction = "both"

            events.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "min_observed": min_obs,
                    "max_observed": max_obs,
                    "direction": direction,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def data_completeness(
        self, window: str = "1h", expected_freq: str = "1s"
    ) -> pd.DataFrame:
        """Percentage of expected samples actually received per window.

        Args:
            window: Resample window.
            expected_freq: Expected sampling frequency.

        Returns:
            DataFrame with columns: window_start, expected_count,
            actual_count, completeness_pct.
        """
        cols = ["window_start", "expected_count", "actual_count", "completeness_pct"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .set_index(self.time_column)
        )
        window_td = pd.to_timedelta(window)
        expected_td = pd.to_timedelta(expected_freq)
        expected_per_window = int(window_td / expected_td)

        counts = sig[self.value_column].resample(window).count()

        events: List[Dict[str, Any]] = []
        for ts, actual in counts.items():
            completeness = min(100.0, round(actual / expected_per_window * 100, 2))
            events.append(
                {
                    "window_start": ts,
                    "expected_count": expected_per_window,
                    "actual_count": int(actual),
                    "completeness_pct": completeness,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
