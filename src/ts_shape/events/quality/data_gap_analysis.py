import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ts_shape.events._windows import finish_events
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_method, keyword_only, renamed_kwargs
from ts_shape.utils.frames import window_offset
from ts_shape.utils.runs import total_seconds

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


def _floats(values: np.ndarray) -> np.ndarray:
    """``values`` as float64, NaN-checked element by element for non-numbers."""
    if values.dtype.kind in "biuf":
        return values.astype(np.float64)
    return np.array(
        [np.nan if np.isnan(v) else float(v) for v in values], dtype=np.float64
    )


def _float_or_none(values: np.ndarray) -> Any:
    """A column as a list of ``float``-or-``None`` would build it: float64
    with NaN for the missing values, or all ``None`` (object) when none is
    known."""
    if values.size and np.isnan(values).all():
        return [None] * values.size
    return values


class DataGapAnalysisEvents(Base):
    """Quality: Data Gap Analysis

    Answer the question "where are the holes in my data?" by analyzing
    gaps in a numeric signal's timestamps.  Complements
    :class:`SignalQualityEvents` (which detects individual missing-data
    events) with pattern-level analysis: gap summaries, per-period
    coverage, and interpolation-candidate identification.

    Methods:
    - detect_gaps: Locate all gaps longer than a threshold.
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
        """Initialize the data gap analyzer.

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

    def _min_gap(self, min_gap: str | pd.Timedelta | None) -> pd.Timedelta:
        """``min_gap`` as a Timedelta; ``None`` is 2x the median sampling interval.

        The median is taken over the positive steps between consecutive
        samples (duplicate timestamps are no sampling interval). Without any
        positive step the default is 1 s.
        """
        if min_gap is not None:
            return pd.to_timedelta(min_gap)
        times = self.signal[self.time_column].array
        if len(times) >= 2:
            diffs = pd.Series(pd.to_timedelta(times[1:] - times[:-1]))
            positive = diffs[diffs > pd.Timedelta(0)]
            if len(positive):
                return pd.Timedelta(positive.median() * 2)
        return pd.Timedelta(seconds=1)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    if TYPE_CHECKING:

        @deprecated(
            "DataGapAnalysisEvents.find_gaps() is deprecated, use .detect_gaps()"
        )
        def find_gaps(self, min_gap: str | None = None) -> pd.DataFrame: ...

    else:
        find_gaps = deprecated_method("detect_gaps")

    @keyword_only(1)
    def detect_gaps(self, min_gap: str | pd.Timedelta | None = None) -> pd.DataFrame:
        """Locate all gaps longer than *min_gap*.

        A gap is a step between two consecutive samples strictly longer than
        ``min_gap``: regular 10-minute sampling with ``min_gap="10min"`` has
        no gaps.

        Args:
            min_gap: Gap threshold (e.g. ``'5s'``, ``'1min'``, ``'1h'``).
                ``None`` (default) is twice the median positive sampling
                interval, the rule of :meth:`coverage_by_period`, so a
                regularly sampled signal has no gaps. A zero-length step
                (duplicate timestamp) is never a gap, even with
                ``min_gap="0s"``.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, gap_duration_seconds, samples_before_gap,
            samples_after_gap. ``gap_duration_seconds`` equals
            ``duration_seconds``.
        """
        cols = [
            "start",
            "end",
            "gap_duration_seconds",
            "samples_before_gap",
            "samples_after_gap",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

        threshold = self._min_gap(min_gap)
        times = self.signal[self.time_column].array
        diffs = times[1:] - times[:-1]

        # Strictly longer than min_gap; a zero step (duplicate timestamp) is
        # never a gap.
        i = np.flatnonzero((diffs > threshold) & (diffs > pd.Timedelta(0)))
        events = pd.DataFrame(
            {
                "start": times[i],
                "end": times[i + 1],
                "gap_duration_seconds": total_seconds(diffs[i]),
                "samples_before_gap": i + 1,
                "samples_after_gap": len(times) - (i + 1),
            }
        )

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    def gap_summary(self, min_gap: str | pd.Timedelta | None = None) -> pd.DataFrame:
        """Aggregate statistics across all gaps.

        Args:
            min_gap: Gap threshold, as in :meth:`detect_gaps` (``None``:
                twice the median sampling interval).

        Returns:
            Single-row DataFrame with columns: total_gaps, total_missing_seconds,
            longest_gap_seconds, shortest_gap_seconds, mean_gap_seconds,
            data_span_seconds, gap_fraction. ``data_span_seconds`` is the
            time from the first to the last sample, with or without gaps.
            Without gaps ``total_missing_seconds``, ``longest_gap_seconds``
            and ``gap_fraction`` are 0 and ``shortest_gap_seconds`` /
            ``mean_gap_seconds`` are NaN (there is no gap to measure).
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
        gaps = self.detect_gaps(min_gap=min_gap)

        # Total span from first to last sample (also when there is no gap)
        span = 0.0
        if not self.signal.empty and len(self.signal) >= 2:
            first = self.signal[self.time_column].iloc[0]
            last = self.signal[self.time_column].iloc[-1]
            span = float((last - first).total_seconds())

        if gaps.empty:
            return pd.DataFrame(
                [
                    {
                        "total_gaps": 0,
                        "total_missing_seconds": 0.0,
                        "longest_gap_seconds": 0.0,
                        "shortest_gap_seconds": float("nan"),
                        "mean_gap_seconds": float("nan"),
                        "data_span_seconds": span,
                        "gap_fraction": 0.0,
                    }
                ],
                columns=cols,
            )

        durations = gaps["gap_duration_seconds"]
        total_missing = float(durations.sum())

        return pd.DataFrame(
            [
                {
                    "total_gaps": len(gaps),
                    "total_missing_seconds": float(total_missing),
                    "longest_gap_seconds": float(durations.max()),
                    "shortest_gap_seconds": float(durations.min()),
                    "mean_gap_seconds": float(durations.mean()),
                    "data_span_seconds": float(span),
                    "gap_fraction": float(total_missing / span) if span > 0 else 0.0,
                }
            ],
            columns=cols,
        )

    @keyword_only(1)
    @renamed_kwargs(freq="window")
    def coverage_by_period(
        self, window: str = "1h", min_gap: str | pd.Timedelta | None = None
    ) -> pd.DataFrame:
        """Data coverage percentage per time window.

        For each window, reports how much of the window actually contains
        data (based on first and last sample timestamps minus any internal
        gaps).

        Args:
            window: Resample window (e.g. ``'1h'``, ``'1D'``, ``'15min'``).
            min_gap: Gap threshold, as in :meth:`detect_gaps`: a step
                strictly longer than it is a gap. Defaults to 2x the median
                sampling interval (auto-detected).

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, sample_count, coverage_pct, gap_count,
            gap_seconds. ``end`` is the next period start.
        """
        cols = [
            "start",
            "sample_count",
            "coverage_pct",
            "gap_count",
            "gap_seconds",
        ]
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
        offset = window_offset(window)

        # Gaps for context; without min_gap, 2x the median positive sampling
        # interval (duplicate timestamps -- zero intervals -- are ignored: a
        # zero median made every interval a gap).
        all_gaps = self.detect_gaps(min_gap=min_gap)

        counts = sig[self.value_column].resample(window).count()

        events: list[dict[str, Any]] = []
        for ts, count in counts.items():
            # The window's real length (23 h / 25 h on DST days for '1D').
            end = ts + offset
            window_secs = (end - ts).total_seconds()
            # Find gaps overlapping this window
            gap_secs = 0.0
            gap_count = 0
            if not all_gaps.empty:
                overlapping = all_gaps[
                    (all_gaps["start"] < end) & (all_gaps["end"] > ts)
                ]
                gap_count = len(overlapping)
                for _, g in overlapping.iterrows():
                    # Clip gap to window boundaries
                    clip_start = max(g["start"], ts)
                    clip_end = min(g["end"], end)
                    gap_secs += (clip_end - clip_start).total_seconds()

            coverage = max(0.0, min(100.0, (1.0 - gap_secs / window_secs) * 100))
            events.append(
                {
                    "start": ts,
                    "sample_count": int(count),
                    "coverage_pct": float(coverage),
                    "gap_count": gap_count,
                    "gap_seconds": float(gap_secs),
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

    @keyword_only(1)
    def interpolation_candidates(
        self, max_gap: str = "10s", min_gap: str = "0s"
    ) -> pd.DataFrame:
        """Identify gaps small enough to safely interpolate.

        Args:
            max_gap: Maximum gap duration to consider for interpolation.
            min_gap: Minimum gap duration (ignore trivially small gaps).
                A zero-length step (duplicate timestamp) is never a gap.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, gap_duration_seconds, value_before, value_after,
            value_jump, safe_to_interpolate.
            ``safe_to_interpolate`` is True when the value jump across the
            gap is within 2 standard deviations of the signal.
        """
        cols = [
            "start",
            "end",
            "gap_duration_seconds",
            "value_before",
            "value_after",
            "value_jump",
            "safe_to_interpolate",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

        max_td = pd.to_timedelta(max_gap)
        min_td = pd.to_timedelta(min_gap)
        times = self.signal[self.time_column].array
        values = self.signal[self.value_column].values
        diffs = times[1:] - times[:-1]

        # Use 2 * global std as a "safe jump" threshold
        signal_std = float(np.nanstd(values))
        safe_threshold = 2.0 * signal_std if signal_std > 0 else float("inf")

        # (A NaT step fails every comparison and is kept, as before.)
        i = np.flatnonzero(
            ~((diffs <= pd.Timedelta(0)) | (diffs < min_td) | (diffs > max_td))
        )
        before = _floats(values[i])
        after = _floats(values[i + 1])
        known = ~np.isnan(before) & ~np.isnan(after)
        with np.errstate(invalid="ignore"):
            jump = np.where(known, np.abs(after - before), np.nan)
            safe = ~np.isnan(jump) & (jump <= safe_threshold)
        events = pd.DataFrame(
            {
                "start": times[i],
                "end": times[i + 1],
                "gap_duration_seconds": total_seconds(diffs[i]),
                "value_before": _float_or_none(before),
                "value_after": _float_or_none(after),
                "value_jump": _float_or_none(jump),
                "safe_to_interpolate": safe,
            }
        )

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
