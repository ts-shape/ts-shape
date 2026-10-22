import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    twin_columns,
)
from ts_shape.events._windows import window_summary
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs
from ts_shape.utils.time_weighted import time_weighted_resample

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

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:operating_range",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the operating range analyzer.

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
    def operating_envelope(self, window: str = "1h") -> pd.DataFrame:
        """Per-window operating envelope statistics.

        Args:
            window: Resample window (pandas offset string, e.g. '1h').

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, min_value, p5, mean_value, p95, max_value,
            range_width. ``end`` is ``start`` + ``window``.
        """
        cols = [
            "start",
            "min_value",
            "p5",
            "mean_value",
            "p95",
            "max_value",
            "range_width",
        ]
        if self.signal.empty:
            return self._summary([], cols, window)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)

        events: list[dict[str, Any]] = []
        for start, group in groups:
            if group.empty:
                continue
            vals = group.dropna()
            if vals.empty:
                continue
            p5 = float(np.percentile(vals, 5))
            p95 = float(np.percentile(vals, 95))
            events.append(
                {
                    "start": start,
                    "min_value": float(vals.min()),
                    "p5": p5,
                    "mean_value": float(vals.mean()),
                    "p95": p95,
                    "max_value": float(vals.max()),
                    "range_width": float(vals.max() - vals.min()),
                }
            )

        return self._summary(events, cols, window)

    def _summary(
        self, events: list[dict[str, Any]], cols: list[str], window: str
    ) -> pd.DataFrame:
        return window_summary(
            events,
            cols,
            window,
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @keyword_only(1)
    def detect_regime_change(
        self,
        window: str = "1h",
        shift_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Detect significant shifts in the operating point between windows.

        A regime change is flagged when the window mean differs from the
        previous window mean by more than shift_threshold * pooled std.

        Args:
            window: Resample window (pandas offset string). Windows with fewer than
                two samples are skipped.
            shift_threshold: Minimum shift between consecutive window means, in
                multiples of the two windows' average std, to flag a regime change.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, prev_mean, new_mean, shift_magnitude.
            ``start`` is the previous window's start and ``end`` the new
            window's start.
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
            return empty_event_df("interval", cols)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)

        window_stats: list[dict[str, Any]] = []
        for start, group in groups:
            vals = group.dropna()
            if len(vals) < 2:
                continue
            window_stats.append(
                {
                    "start": start,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                }
            )

        if len(window_stats) < 2:
            return empty_event_df("interval", cols)

        events: list[dict[str, Any]] = []
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
                        "start": prev["start"],
                        "end": curr["start"],
                        "uuid": self.event_uuid,
                        "is_delta": False,
                        "prev_mean": prev["mean"],
                        "new_mean": curr["mean"],
                        "shift_magnitude": float(shift_mag),
                    }
                )

        if not events:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @keyword_only(3)
    @renamed_kwargs(lower="lower_limit", upper="upper_limit")
    def time_in_range(
        self,
        lower_limit: float,
        upper_limit: float,
        window: str = "1h",
        time_weighted: bool = True,
    ) -> pd.DataFrame:
        """Share of each window spent within, below and above a range.

        By default the ``*_pct`` columns are percentages of *time*: each value
        is held until the next sample (zero-order hold), so a window without
        samples that holds an earlier value is reported too. With
        ``time_weighted=False`` they are percentages of the *samples* in each
        window (equal to the share of time only for evenly spaced data).

        Args:
            lower_limit: Lower bound of the range (inclusive), in signal units.
            upper_limit: Upper bound of the range (inclusive), in signal units.
            window: Resample window (pandas offset string).
            time_weighted: ``True`` (default) weights each value by how long it
                was held; ``False`` counts samples.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, time_in_range_pct, time_below_pct, time_above_pct.
            ``end`` is ``start`` + ``window``.
        """
        cols = ["start", "time_in_range_pct", "time_below_pct", "time_above_pct"]
        if self.signal.empty:
            return self._summary([], cols, window)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        events: list[dict[str, Any]] = []
        if time_weighted:
            vals = to_numeric_checked(
                indexed, what=f"signal {self.signal_uuid!r}"
            ).dropna()
            shares = {
                name: time_weighted_resample(mask.astype(float), window)
                for name, mask in (
                    (
                        "time_in_range_pct",
                        (vals >= lower_limit) & (vals <= upper_limit),
                    ),
                    ("time_below_pct", vals < lower_limit),
                    ("time_above_pct", vals > upper_limit),
                )
            }
            frame = pd.DataFrame(shares).dropna() * 100
            for start, row in frame.iterrows():
                events.append({"start": start, **row.astype(float).to_dict()})
            return self._summary(events, cols, window)

        groups = indexed.resample(window)
        for start, group in groups:
            vals = group.dropna()
            if vals.empty:
                continue
            n = len(vals)
            in_range = ((vals >= lower_limit) & (vals <= upper_limit)).sum()
            below = (vals < lower_limit).sum()
            above = (vals > upper_limit).sum()
            events.append(
                {
                    "start": start,
                    "time_in_range_pct": float(in_range / n * 100),
                    "time_below_pct": float(below / n * 100),
                    "time_above_pct": float(above / n * 100),
                }
            )

        return self._summary(events, cols, window)

    @keyword_only(1)
    @twin_columns({"count": "sample_count"})
    def value_distribution(self, n_bins: int = 10) -> pd.DataFrame:
        """Histogram of signal values.

        Args:
            n_bins: Number of equal-width histogram bins between the signal's
                minimum and maximum.

        Returns:
            DataFrame with columns: bin_lower, bin_upper, count,
            sample_count (samples in the bin), pct, cumulative_pct.

            Canonical twin columns: ``sample_count`` (after ``count``); it holds the same
            values as the deprecated column right before it, which is removed
            in the next major release.
        """
        cols = ["bin_lower", "bin_upper", "count", "pct", "cumulative_pct"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        vals = self.signal[self.value_column].dropna()
        if vals.empty:
            return pd.DataFrame(columns=cols)

        counts, bin_edges = np.histogram(vals, bins=n_bins)
        total = int(counts.sum())

        events: list[dict[str, Any]] = []
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
