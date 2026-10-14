import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
    finalize_summary_df,
)
from ts_shape.events._windows import add_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import REMOVAL, keyword_only

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
        super().__init__(dataframe, time_column=time_column)
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

    @keyword_only(2)
    def detect_rapid_change(
        self, threshold: float, window: str | None = None
    ) -> pd.DataFrame:
        """Flag intervals where rate of change exceeds threshold.

        Each run of consecutive samples whose absolute rate is at or above
        ``threshold`` becomes one interval.

        Args:
            threshold: Minimum absolute rate (units/second) to flag.
            window: Deprecated and not used: the rate is always computed
                between consecutive samples. Passing any value emits a
                ``DeprecationWarning``. Filter the result on
                ``duration_seconds`` to drop short intervals.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, max_rate, direction.
        """
        if window is not None:
            warnings.warn(
                "RateOfChangeEvents.detect_rapid_change: window is not used (the "
                "rate is computed between consecutive samples) and will be "
                f"removed in {REMOVAL}; filter the result on "
                "duration_seconds instead",
                DeprecationWarning,
                stacklevel=4,
            )
        cols = ["start", "end", "max_rate", "direction"]
        empty = empty_event_df("interval", ["max_rate", "direction"])
        rate_df = self._compute_rate()
        if rate_df.empty:
            return empty

        rapid = rate_df["rate"].abs() >= threshold
        if not rapid.any():
            return empty

        groups = (rapid != rapid.shift()).cumsum()
        events: list[dict[str, Any]] = []

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
                    "start": start,
                    "end": end,
                    "max_rate": max_rate,
                    "direction": "increasing" if max_rate > 0 else "decreasing",
                }
            )

        if not events:
            return empty
        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @keyword_only(1)
    def rate_statistics(self, window: str = "1h") -> pd.DataFrame:
        """Per-window rate of change statistics.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, end, duration_seconds, mean_rate,
            std_rate, max_rate, min_rate. ``end`` is ``start`` + ``window``.
        """
        cols = ["start", "mean_rate", "std_rate", "max_rate", "min_rate"]
        rate_df = self._compute_rate()
        if rate_df.empty:
            return empty_event_df("summary", cols[1:])

        rate_indexed = rate_df.set_index(self.time_column)
        abs_rate = rate_indexed["rate"].abs()

        resampled = abs_rate.resample(window).agg(["mean", "std", "max", "min"])
        resampled = resampled.dropna(subset=["mean"])

        result = resampled.reset_index()
        result.columns = [
            "start",
            "mean_rate",
            "std_rate",
            "max_rate",
            "min_rate",
        ]

        return finalize_summary_df(add_window_bounds(result[cols], window))

    @keyword_only(2)
    def detect_step_changes(
        self, min_delta: float, max_duration: str = "5s"
    ) -> pd.DataFrame:
        """Detect sudden value jumps within a short duration.

        Args:
            min_delta: Minimum absolute value change to qualify.
            max_duration: Maximum time window for the step to occur.

        Returns:
            DataFrame with columns: systime, uuid, source_uuid, time,
            value_before, value_after, delta, duration. ``systime`` equals
            ``time`` (the instant after the jump).
        """
        cols = ["time", "value_before", "value_after", "delta", "duration"]
        empty = empty_event_df("point", cols)
        if self.signal.empty or len(self.signal) < 2:
            return empty

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        max_td = pd.to_timedelta(max_duration)

        events: list[dict[str, Any]] = []
        values = sig[self.value_column].values
        times = sig[self.time_column].array

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

        if not events:
            return empty
        out = pd.DataFrame(events, columns=cols)
        out["systime"] = out["time"]
        return finalize_point_df(
            out, uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
