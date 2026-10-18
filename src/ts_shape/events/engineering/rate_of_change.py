import logging
import warnings

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
    finalize_summary_df,
    twin_columns,
)
from ts_shape.events._windows import add_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import REMOVAL, caller_stacklevel, keyword_only
from ts_shape.utils.runs import run_argmax, true_runs

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
        """Initialize the rate-of-change detector.

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

    def _compute_rate(self) -> pd.DataFrame:
        """Compute rate of change (value_diff / time_diff_seconds)."""
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=[self.time_column, "rate", "prev_time"])

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        value_diff = sig[self.value_column].diff()
        time_diff = sig[self.time_column].diff().dt.total_seconds()
        sig["rate"] = value_diff / time_diff.replace(0, np.nan)
        # The rate at sample i describes the step from sample i-1 to i.
        sig["prev_time"] = sig[self.time_column].shift()
        return sig.dropna(subset=["rate"])

    @keyword_only(2)
    def detect_rapid_change(
        self, threshold: float, window: str | None = None
    ) -> pd.DataFrame:
        """Flag intervals where rate of change exceeds threshold.

        Each run of consecutive samples whose absolute rate is at or above
        ``threshold`` becomes one interval. The rate at a sample describes the
        step from the previous sample to it, so an interval starts at the
        sample *before* the first fast step and ends at the last sample of
        the run: a single fast step from ``t[i-1]`` to ``t[i]`` lasts
        ``t[i] - t[i-1]``, not zero.

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
                stacklevel=caller_stacklevel(),
            )
        cols = ["start", "end", "max_rate", "direction"]
        empty = empty_event_df("interval", ["max_rate", "direction"])
        rate_df = self._compute_rate()
        if rate_df.empty:
            return empty

        rate = rate_df["rate"].to_numpy(dtype=np.float64)
        rapid = np.abs(rate) >= threshold
        if not rapid.any():
            return empty

        # One interval per run of rapid samples; max_rate is the rate with the
        # largest magnitude (first one on ties).
        runs = true_runs(rapid)
        times = rate_df[self.time_column].array
        prev_times = rate_df["prev_time"].array
        max_rate = rate[run_argmax(np.abs(rate), runs)]
        events = pd.DataFrame(
            {
                "start": prev_times[runs.starts],
                "end": times[runs.last],
                "max_rate": max_rate,
                "direction": np.where(max_rate > 0, "increasing", "decreasing").astype(
                    object
                ),
            },
            columns=cols,
        )

        return finalize_interval_df(
            events, uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    @twin_columns(
        {
            "mean_rate": "mean_abs_rate",
            "std_rate": "std_abs_rate",
            "max_rate": "max_abs_rate",
            "min_rate": "min_abs_rate",
        }
    )
    def rate_statistics(self, window: str = "1h") -> pd.DataFrame:
        """Per-window statistics of the absolute rate of change.

        The statistics are over ``|rate|`` (signal units per second between
        consecutive samples), so a falling signal gives positive values.

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, end, duration_seconds, mean_rate,
            mean_abs_rate, std_rate, std_abs_rate, max_rate, max_abs_rate,
            min_rate, min_abs_rate. ``end`` is ``start`` + ``window``.

            Canonical twin columns: ``mean_abs_rate``, ``std_abs_rate``,
            ``max_abs_rate`` and ``min_abs_rate``, each right after the
            column of the same values. The ``*_rate`` columns hold absolute
            values today; in the next major release they become statistics
            of the signed rate, so read the ``*_abs_rate`` columns for the
            absolute rate.
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
            DataFrame with columns: systime, uuid, source_uuid, time, start,
            end, value_before, value_after, delta, duration,
            duration_seconds. ``systime`` and ``end`` equal ``time`` (the
            instant after the jump; ``time`` is a deprecated twin), ``start``
            is the sample before the jump, and ``duration_seconds`` is the
            float-seconds twin of the Timedelta ``duration``.
        """
        cols = [
            "time",
            "start",
            "end",
            "value_before",
            "value_after",
            "delta",
            "duration",
            "duration_seconds",
        ]
        empty = empty_event_df("point", cols)
        if self.signal.empty or len(self.signal) < 2:
            return empty

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        max_td = pd.to_timedelta(max_duration)

        values = sig[self.value_column].values
        times = sig[self.time_column].array
        delta = values[1:] - values[:-1]
        duration = times[1:] - times[:-1]
        hit = np.flatnonzero((np.abs(delta) >= min_delta) & (duration <= max_td))
        if not hit.size:
            return empty
        out = pd.DataFrame(
            {
                "time": times[hit + 1],
                "start": times[hit],
                "end": times[hit + 1],
                "value_before": values[hit].astype(np.float64),
                "value_after": values[hit + 1].astype(np.float64),
                "delta": delta[hit].astype(np.float64),
                "duration": duration[hit],
                "duration_seconds": (
                    pd.to_timedelta(duration[hit]).total_seconds().astype(np.float64)
                ),
            },
            columns=cols,
        )
        out["systime"] = out["time"]
        return finalize_point_df(
            out, uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
