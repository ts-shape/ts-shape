"""Signal quality: is the data itself trustworthy?

Finds missing data (gaps longer than the expected period), measures
sampling regularity, flags out-of-range values, and reports data
completeness per calendar window (real window lengths on DST days).
"""

import logging
from typing import Any

import pandas as pd
from pandas.tseries.frequencies import to_offset

from ts_shape.events._windows import finish_events
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import interval_end_times

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

    @keyword_only(1)
    def detect_missing_data(
        self, expected_freq: str = "1s", tolerance_factor: float = 2.0
    ) -> pd.DataFrame:
        """Find gaps where time between samples exceeds expected frequency.

        Args:
            expected_freq: Expected sampling frequency (e.g. '1s', '100ms').
            tolerance_factor: Multiplier on expected_freq to trigger a gap.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, gap_duration, expected_samples_missing.
            ``gap_duration`` is a Timedelta; ``duration_seconds`` is the
            same span in float seconds.
        """
        cols = ["start", "end", "gap_duration", "expected_samples_missing"]
        if self.signal.empty or len(self.signal) < 2:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

        times = self.signal[self.time_column].array
        diffs = times[1:] - times[:-1]
        expected_td = pd.to_timedelta(expected_freq)
        threshold = expected_td * tolerance_factor

        events: list[dict[str, Any]] = []
        for i, d in enumerate(diffs):
            gap = pd.Timedelta(d)
            if gap > threshold:
                expected_missing = int(gap / expected_td) - 1
                events.append(
                    {
                        "start": pd.Timestamp(times[i]),
                        "end": pd.Timestamp(times[i + 1]),
                        "gap_duration": gap,
                        "expected_samples_missing": max(expected_missing, 1),
                    }
                )

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    def sampling_regularity(self, window: str = "1h") -> pd.DataFrame:
        """Inter-sample interval statistics per window.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, mean_interval, std_interval, min_interval,
            max_interval, regularity_score. ``end`` is ``start`` plus one ``window``.
        """
        cols = [
            "start",
            "mean_interval",
            "std_interval",
            "min_interval",
            "max_interval",
            "regularity_score",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        sig = self.signal[[self.time_column]].copy().set_index(self.time_column)
        sig["interval"] = sig.index.to_series().diff().dt.total_seconds()
        sig = sig.dropna(subset=["interval"])

        if sig.empty:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        resampled = (
            sig["interval"].resample(window).agg(["mean", "std", "min", "max", "count"])
        )

        events: list[dict[str, Any]] = []
        for ts, row in resampled.iterrows():
            if row["count"] < 2 or pd.isna(row["mean"]):
                continue
            mean_val = row["mean"]
            std_val = row["std"] if pd.notna(row["std"]) else 0.0
            # Regularity score: 1.0 = perfectly regular, 0.0 = very irregular
            regularity = max(0.0, 1.0 - (std_val / (mean_val + 1e-10)))
            events.append(
                {
                    "start": ts,
                    "mean_interval": round(mean_val, 4),
                    "std_interval": round(std_val, 4),
                    "min_interval": round(row["min"], 4),
                    "max_interval": round(row["max"], 4),
                    "regularity_score": round(max(0.0, min(1.0, regularity)), 4),
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

    def detect_out_of_range(
        self,
        min_value: float,
        max_value: float,
        *,
        interval_end: str = "last_sample",
    ) -> pd.DataFrame:
        """Flag intervals where signal is outside expected range.

        Args:
            min_value: Minimum acceptable value.
            max_value: Maximum acceptable value.
            interval_end: ``"last_sample"`` (default) ends an interval at its
                last flagged sample. ``"next_change"`` ends it at the next sample
                (the transition), so change-encoded signals -- one row per change --
                no longer collapse to zero-length intervals. See
                :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, duration, min_observed, max_observed, direction.
            ``duration`` is a Timedelta; ``duration_seconds`` is the same
            span in float seconds.
        """
        cols = [
            "start",
            "end",
            "duration",
            "min_observed",
            "max_observed",
            "direction",
        ]
        if self.signal.empty:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        out_of_range = (sig[self.value_column] < min_value) | (
            sig[self.value_column] > max_value
        )

        if not out_of_range.any():
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

        groups = (out_of_range != out_of_range.shift()).cumsum()
        ends = interval_end_times(sig[self.time_column], interval_end)
        events: list[dict[str, Any]] = []

        for _, seg in sig.groupby(groups):
            seg_oor = out_of_range.loc[seg.index]
            if not seg_oor.iloc[0]:
                continue
            start = seg[self.time_column].iloc[0]
            end = ends.loc[seg.index[-1]]
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
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "min_observed": min_obs,
                    "max_observed": max_obs,
                    "direction": direction,
                }
            )

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    def data_completeness(
        self, window: str = "1h", expected_freq: str = "1s"
    ) -> pd.DataFrame:
        """Percentage of expected samples actually received per window.

        Args:
            window: Resample window.
            expected_freq: Expected sampling frequency.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, expected_count, actual_count, completeness_pct.
            ``end`` is ``start`` plus one ``window``.
        """
        cols = ["start", "expected_count", "actual_count", "completeness_pct"]
        if self.signal.empty:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .set_index(self.time_column)
        )
        expected_td = pd.to_timedelta(expected_freq)
        offset = to_offset(window)

        counts = sig[self.value_column].resample(window).count()

        events: list[dict[str, Any]] = []
        for ts, actual in counts.items():
            # The window's real length: a '1D' window on the DST days of a
            # tz-aware signal is 23 h or 25 h, not 24 h.
            expected_per_window = int(((ts + offset) - ts) / expected_td)
            completeness = (
                min(100.0, round(actual / expected_per_window * 100, 2))
                if expected_per_window > 0
                else 100.0
            )
            events.append(
                {
                    "start": ts,
                    "expected_count": expected_per_window,
                    "actual_count": int(actual),
                    "completeness_pct": completeness,
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
