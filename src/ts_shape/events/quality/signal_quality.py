"""Signal quality: is the data itself trustworthy?

Finds missing data (gaps longer than the expected period), measures
sampling regularity, flags out-of-range values, and reports data
completeness per calendar window (real window lengths on DST days).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events._windows import finish_events
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    keyword_only,
    renamed_kwargs,
)
from ts_shape.utils.frames import window_offset
from ts_shape.utils.intervals import interval_end_times
from ts_shape.utils.runs import run_agg, true_runs

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

    Example:
        A 1 s signal with an 8 s gap and one implausible reading:

        >>> import pandas as pd
        >>> from ts_shape import SignalQualityEvents
        >>> df = pd.DataFrame({
        ...     "systime": pd.to_datetime([
        ...         "2025-01-01 00:00:00", "2025-01-01 00:00:01", "2025-01-01 00:00:02",
        ...         "2025-01-01 00:00:10", "2025-01-01 00:00:11"]),
        ...     "uuid": "temp",
        ...     "value_double": [20.0, 20.5, 21.0, 95.0, 21.0],
        ... })
        >>> quality = SignalQualityEvents(df, "temp")
        >>> quality.detect_missing_data(expected_interval="1s")[["start", "end", "duration_seconds"]]
                        start                 end  duration_seconds
        0 2025-01-01 00:00:02 2025-01-01 00:00:10               8.0
        >>> quality.detect_out_of_range(0.0, 50.0)[["start", "end", "duration_seconds"]]
                        start                 end  duration_seconds
        0 2025-01-01 00:00:10 2025-01-01 00:00:11               1.0
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "quality:signal",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the signal quality monitor.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            signal_uuid: UUID of the signal to analyze.
            event_uuid: UUID stamped on emitted events.
            value_column: Column holding the signal values.
            time_column: Timestamp column.
        """
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
    @renamed_kwargs(expected_freq="expected_interval")
    def detect_missing_data(
        self, expected_interval: str = "1s", tolerance_factor: float = 2.0
    ) -> pd.DataFrame:
        """Find gaps where time between samples exceeds expected frequency.

        Args:
            expected_interval: Expected sampling interval (e.g. '1s', '100ms').
            tolerance_factor: Multiplier on expected_interval to trigger a gap.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, gap_duration, gap_duration_seconds,
            expected_samples_missing. ``gap_duration`` is a Timedelta;
            ``gap_duration_seconds`` (and ``duration_seconds``) is the same
            span in float seconds.
        """
        cols = [
            "start",
            "end",
            "gap_duration",
            "gap_duration_seconds",
            "expected_samples_missing",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

        times = self.signal[self.time_column].array
        diffs = times[1:] - times[:-1]
        expected_td = pd.to_timedelta(expected_interval)
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
                        "gap_duration_seconds": gap.total_seconds(),
                        "expected_samples_missing": max(expected_missing, 1),
                    }
                )

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    @twin_columns(
        {
            "mean_interval": "mean_interval_seconds",
            "std_interval": "std_interval_seconds",
            "min_interval": "min_interval_seconds",
            "max_interval": "max_interval_seconds",
        }
    )
    def sampling_regularity(self, window: str = "1h") -> pd.DataFrame:
        """Inter-sample interval statistics per window.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, mean_interval, mean_interval_seconds, std_interval,
            std_interval_seconds, min_interval, min_interval_seconds,
            max_interval, max_interval_seconds, regularity_score. ``end`` is
            ``start`` plus one ``window``. Intervals are in seconds.

            Canonical twin columns: ``mean_interval_seconds``,
            ``std_interval_seconds``, ``min_interval_seconds``,
            ``max_interval_seconds``, each right after its deprecated
            unitless column (same values), which is removed in the next major
            release.
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

        keep = (resampled["count"] >= 2).to_numpy() & resampled[
            "mean"
        ].notna().to_numpy()
        kept = resampled[keep]
        mean_val = kept["mean"].to_numpy(dtype=np.float64)
        std_val = kept["std"].fillna(0.0).to_numpy(dtype=np.float64)
        # Regularity score: 1.0 = perfectly regular, 0.0 = very irregular
        regularity = np.maximum(0.0, 1.0 - (std_val / (mean_val + 1e-10)))
        events = pd.DataFrame(
            {
                "start": kept.index,
                "mean_interval": mean_val,
                "std_interval": std_val,
                "min_interval": kept["min"].to_numpy(dtype=np.float64),
                "max_interval": kept["max"].to_numpy(dtype=np.float64),
                "regularity_score": np.minimum(1.0, regularity),
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
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Flag intervals where signal is outside expected range.

        Args:
            min_value: Minimum acceptable value.
            max_value: Maximum acceptable value.
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so change-encoded signals -- one
                row per change -- report the real time out of range.
                ``"last_sample"`` ends it at its last flagged sample. See
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

        runs = true_runs(out_of_range.to_numpy(dtype=bool))
        start = sig[self.time_column].array[runs.starts]
        end = interval_end_times(sig[self.time_column], interval_end).array[runs.last]
        min_obs = run_agg(sig[self.value_column], runs, "nanmin")
        max_obs = run_agg(sig[self.value_column], runs, "nanmax")
        direction = np.where(
            max_obs > max_value,
            "above",
            np.where(min_obs < min_value, "below", "both"),
        ).astype(object)
        events = pd.DataFrame(
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
    @renamed_kwargs(expected_freq="expected_interval")
    def data_completeness(
        self, window: str = "1h", expected_interval: str = "1s"
    ) -> pd.DataFrame:
        """Percentage of expected samples actually received per window.

        The expected count is the full window's length over
        ``expected_interval``, so a window the data covers only partly (the
        first and last one) reads below 100 %: 1 s data from 10:30:00 to
        11:29:59 reads 50 % in both hourly windows, as
        ``DataGapAnalysisEvents.coverage_by_period`` does.

        Args:
            window: Resample window.
            expected_interval: Expected sampling interval.

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
        expected_td = pd.to_timedelta(expected_interval)
        offset = window_offset(window)

        counts = sig[self.value_column].resample(window).count()

        events: list[dict[str, Any]] = []
        for ts, actual in counts.items():
            # The window's real length: a '1D' window on the DST days of a
            # tz-aware signal is 23 h or 25 h, not 24 h.
            expected_per_window = int(((ts + offset) - ts) / expected_td)
            completeness = (
                min(100.0, actual / expected_per_window * 100)
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
