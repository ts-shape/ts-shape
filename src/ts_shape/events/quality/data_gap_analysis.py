import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DataGapAnalysisEvents(Base):
    """Quality: Data Gap Analysis

    Answer the question "where are the holes in my data?" by analysing
    gaps in a numeric signal's timestamps.  Complements
    :class:`SignalQualityEvents` (which detects individual missing-data
    events) with pattern-level analysis: gap summaries, per-period
    coverage, and interpolation-candidate identification.

    Methods:
    - find_gaps: Locate all gaps longer than a threshold.
    - gap_summary: Aggregate statistics across all gaps.
    - coverage_by_period: Data coverage percentage per time window.
    - interpolation_candidates: Gaps small enough to interpolate safely.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "quality:data_gap",
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

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def find_gaps(self, min_gap: str = "5s") -> pd.DataFrame:
        """Locate all gaps longer than *min_gap*.

        Args:
            min_gap: Minimum gap duration to report (e.g. ``'5s'``,
                ``'1min'``, ``'1h'``).

        Returns:
            DataFrame with columns: gap_start, gap_end, gap_duration_seconds,
            samples_before_gap, samples_after_gap.
        """
        cols = [
            "gap_start",
            "gap_end",
            "gap_duration_seconds",
            "samples_before_gap",
            "samples_after_gap",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        threshold = pd.to_timedelta(min_gap)
        times = self.signal[self.time_column].values
        diffs = np.diff(times)

        events: List[Dict[str, Any]] = []
        for i, d in enumerate(diffs):
            gap = pd.Timedelta(d)
            if gap >= threshold:
                events.append(
                    {
                        "gap_start": pd.Timestamp(times[i]),
                        "gap_end": pd.Timestamp(times[i + 1]),
                        "gap_duration_seconds": round(gap.total_seconds(), 3),
                        "samples_before_gap": i + 1,
                        "samples_after_gap": len(times) - (i + 1),
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def gap_summary(self, min_gap: str = "5s") -> pd.DataFrame:
        """Aggregate statistics across all gaps.

        Args:
            min_gap: Minimum gap duration to include (e.g. ``'5s'``).

        Returns:
            Single-row DataFrame with columns: total_gaps, total_missing_seconds,
            longest_gap_seconds, shortest_gap_seconds, mean_gap_seconds,
            data_span_seconds, gap_fraction.
        """
        cols = [
            "total_gaps",
            "total_missing_seconds",
            "longest_gap_seconds",
            "shortest_gap_seconds",
            "mean_gap_seconds",
            "data_span_seconds",
            "gap_fraction",
        ]
        gaps = self.find_gaps(min_gap=min_gap)
        if gaps.empty:
            return pd.DataFrame(
                [{c: 0 if c != "gap_fraction" else 0.0 for c in cols}],
                columns=cols,
            )

        durations = gaps["gap_duration_seconds"]
        total_missing = float(durations.sum())

        # Total span from first to last sample
        span = 0.0
        if not self.signal.empty and len(self.signal) >= 2:
            first = self.signal[self.time_column].iloc[0]
            last = self.signal[self.time_column].iloc[-1]
            span = (last - first).total_seconds()

        return pd.DataFrame(
            [
                {
                    "total_gaps": len(gaps),
                    "total_missing_seconds": round(total_missing, 3),
                    "longest_gap_seconds": round(float(durations.max()), 3),
                    "shortest_gap_seconds": round(float(durations.min()), 3),
                    "mean_gap_seconds": round(float(durations.mean()), 3),
                    "data_span_seconds": round(span, 3),
                    "gap_fraction": round(total_missing / span, 4) if span > 0 else 0.0,
                }
            ],
            columns=cols,
        )

    def coverage_by_period(
        self, freq: str = "1h", min_gap: str | None = None
    ) -> pd.DataFrame:
        """Data coverage percentage per time window.

        For each window, reports how much of the window actually contains
        data (based on first and last sample timestamps minus any internal
        gaps).

        Args:
            freq: Resample frequency (e.g. ``'1h'``, ``'1D'``, ``'15min'``).
            min_gap: Minimum inter-sample interval to count as a gap.
                Defaults to 2x the median sampling interval (auto-detected).

        Returns:
            DataFrame with columns: period_start, sample_count,
            coverage_pct, gap_count, gap_seconds.
        """
        cols = [
            "period_start",
            "sample_count",
            "coverage_pct",
            "gap_count",
            "gap_seconds",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .set_index(self.time_column)
        )
        window_td = pd.to_timedelta(freq)
        window_secs = window_td.total_seconds()

        # Auto-detect a sensible min_gap if not provided
        if min_gap is None:
            times = self.signal[self.time_column].values
            if len(times) >= 2:
                diffs = pd.to_timedelta(np.diff(times))
                median_interval = diffs.median()
                min_gap_td = median_interval * 2
                min_gap = str(min_gap_td)
            else:
                min_gap = "1s"

        # Get gaps for context
        all_gaps = self.find_gaps(min_gap=min_gap)

        counts = sig[self.value_column].resample(freq).count()

        events: List[Dict[str, Any]] = []
        for ts, count in counts.items():
            window_end = ts + window_td
            # Find gaps overlapping this window
            gap_secs = 0.0
            gap_count = 0
            if not all_gaps.empty:
                overlapping = all_gaps[
                    (all_gaps["gap_start"] < window_end) & (all_gaps["gap_end"] > ts)
                ]
                gap_count = len(overlapping)
                for _, g in overlapping.iterrows():
                    # Clip gap to window boundaries
                    clip_start = max(g["gap_start"], ts)
                    clip_end = min(g["gap_end"], window_end)
                    gap_secs += (clip_end - clip_start).total_seconds()

            coverage = max(0.0, min(100.0, (1.0 - gap_secs / window_secs) * 100))
            events.append(
                {
                    "period_start": ts,
                    "sample_count": int(count),
                    "coverage_pct": round(coverage, 2),
                    "gap_count": gap_count,
                    "gap_seconds": round(gap_secs, 3),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def interpolation_candidates(
        self, max_gap: str = "10s", min_gap: str = "0s"
    ) -> pd.DataFrame:
        """Identify gaps small enough to safely interpolate.

        Args:
            max_gap: Maximum gap duration to consider for interpolation.
            min_gap: Minimum gap duration (ignore trivially small gaps).

        Returns:
            DataFrame with columns: gap_start, gap_end, gap_duration_seconds,
            value_before, value_after, value_jump, safe_to_interpolate.
            ``safe_to_interpolate`` is True when the value jump across the
            gap is within 2 standard deviations of the signal.
        """
        cols = [
            "gap_start",
            "gap_end",
            "gap_duration_seconds",
            "value_before",
            "value_after",
            "value_jump",
            "safe_to_interpolate",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        max_td = pd.to_timedelta(max_gap)
        min_td = pd.to_timedelta(min_gap)
        times = self.signal[self.time_column].values
        values = self.signal[self.value_column].values
        diffs = np.diff(times)

        # Use 2 * global std as a "safe jump" threshold
        signal_std = float(np.nanstd(values))
        safe_threshold = 2.0 * signal_std if signal_std > 0 else float("inf")

        events: List[Dict[str, Any]] = []
        for i, d in enumerate(diffs):
            gap = pd.Timedelta(d)
            if gap < min_td or gap > max_td:
                continue

            val_before = float(values[i]) if not np.isnan(values[i]) else None
            val_after = float(values[i + 1]) if not np.isnan(values[i + 1]) else None
            if val_before is not None and val_after is not None:
                jump = abs(val_after - val_before)
            else:
                jump = float("nan")

            safe = not np.isnan(jump) and jump <= safe_threshold

            events.append(
                {
                    "gap_start": pd.Timestamp(times[i]),
                    "gap_end": pd.Timestamp(times[i + 1]),
                    "gap_duration_seconds": round(gap.total_seconds(), 3),
                    "value_before": val_before,
                    "value_after": val_after,
                    "value_jump": round(jump, 6) if not np.isnan(jump) else None,
                    "safe_to_interpolate": safe,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
