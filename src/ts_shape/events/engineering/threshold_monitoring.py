import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_summary_df,
    twin_columns,
)
from ts_shape.events._windows import add_window_bounds, window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs
from ts_shape.utils.intervals import INTERVAL_END_MODES, interval_end_times

logger = logging.getLogger(__name__)


def _pairwise_sum_slice(values: np.ndarray, offset: int, length: int) -> float:
    """numpy's pairwise sum of a length-``length`` array that is zero except
    for ``values`` at ``[offset, offset + len(values))``.

    Equals ``full.sum()`` bit for bit (``full`` the zero-padded array) but
    only touches the blocks of numpy's summation tree that hold ``values``;
    an all-zero block sums to exactly 0.0, which leaves any partial sum
    unchanged.
    """
    end = offset + values.size

    def block(lo: int, n: int) -> float:
        if lo + n <= offset or lo >= end:
            return 0.0
        if n > 128:
            n2 = n // 2
            n2 -= n2 % 8
            return block(lo, n2) + block(lo + n2, n - n2)
        seg = np.zeros(n, dtype=np.float64)
        a, b = max(lo, offset), min(lo + n, end)
        seg[a - lo : b - lo] = values[a - offset : b - offset]
        if n < 8:
            res = 0.0
            for v in seg.tolist():
                res += v
            return res
        full = n - n % 8
        r = seg[:full].reshape(-1, 8).sum(axis=0).tolist()
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for v in seg[full:].tolist():
            res += v
        return float(res)

    return block(0, length)


def _time_weighted_above_fraction_loop(
    above: pd.Series, window_starts: pd.Index, window_ends: pd.Index
) -> list[float]:
    """Reference per-window loop (used when a timestamp is missing)."""
    times = above.index.to_series().reset_index(drop=True)
    seg_end = times.shift(-1).fillna(times)
    flags = above.reset_index(drop=True).astype(bool)
    starts, stops = times[flags], seg_end[flags]
    fractions: list[float] = []
    for w_start, w_end in zip(window_starts, window_ends, strict=True):
        overlap = (
            stops.clip(upper=w_end) - starts.clip(lower=w_start)
        ).dt.total_seconds()
        window_s = (w_end - w_start).total_seconds()
        fractions.append(float(overlap.clip(lower=0).sum()) / window_s)
    return fractions


class ThresholdMonitoringEvents(Base):
    """Engineering: Threshold Monitoring

    Multi-level threshold monitoring with hysteresis for numeric signals.

    Methods:
    - multi_level_threshold: Intervals exceeding configurable warning/alarm/critical levels.
    - threshold_with_hysteresis: Alarm entry/exit with separate high/low thresholds.
    - time_above_threshold: Time and percentage above a threshold per window.
    - threshold_exceedance_trend: Track exceedance frequency over time.

    Example:
        >>> import pandas as pd
        >>> from ts_shape import ThresholdMonitoringEvents
        >>> df = pd.DataFrame({
        ...     "systime": pd.date_range("2025-01-01 08:00", periods=6, freq="1min"),
        ...     "uuid": "temp",
        ...     "value_double": [70.0, 82.0, 91.0, 85.0, 75.0, 70.0],
        ... })
        >>> monitor = ThresholdMonitoringEvents(df, "temp")
        >>> monitor.multi_level_threshold({"warning": 80.0, "alarm": 90.0})[
        ...     ["start", "end", "level"]]
                        start                 end    level
        0 2025-01-01 08:02:00 2025-01-01 08:03:00    alarm
        1 2025-01-01 08:01:00 2025-01-01 08:04:00  warning
        >>> monitor.threshold_with_hysteresis(upper_limit=80.0, lower_limit=76.0)[
        ...     ["start", "end", "duration_seconds"]]
                        start                 end  duration_seconds
        0 2025-01-01 08:01:00 2025-01-01 08:04:00             180.0
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:threshold",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the threshold monitor.

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

    @keyword_only(2)
    def multi_level_threshold(
        self,
        levels: dict[str, float],
        direction: str = "above",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Detect intervals where signal exceeds configurable threshold levels.

        Args:
            levels: Dict mapping level names to threshold values.
                    e.g. {'warning': 80, 'alarm': 90, 'critical': 95}
            direction: 'above' or 'below'.
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, duration (Timedelta), level, peak_value.
        """
        cols = ["start", "end", "duration", "level", "peak_value"]
        if self.signal.empty:
            return empty_event_df("interval", cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )

        # Sort levels: for 'above', process from highest to lowest
        sorted_levels = sorted(
            levels.items(), key=lambda x: x[1], reverse=(direction == "above")
        )

        all_events: list[dict[str, Any]] = []
        ends = interval_end_times(sig[self.time_column], interval_end)

        for level_name, threshold in sorted_levels:
            if direction == "above":
                exceeded = sig[self.value_column] >= threshold
            else:
                exceeded = sig[self.value_column] <= threshold

            if not exceeded.any():
                continue

            # Runs of exceeding samples as index ranges [first, last].
            exc = exceeded.to_numpy(dtype=bool)
            edges = np.diff(np.r_[False, exc, False].astype(np.int8))
            first = np.flatnonzero(edges == 1)
            last = np.flatnonzero(edges == -1) - 1
            vals = sig[self.value_column].to_numpy(dtype=float)
            starts = sig[self.time_column].iloc[first]
            end_times = ends.iloc[last]
            for i, (start, end) in enumerate(zip(starts, end_times)):
                seg_vals = vals[first[i] : last[i] + 1]
                peak = float(seg_vals.max() if direction == "above" else seg_vals.min())
                all_events.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "level": level_name,
                        "peak_value": peak,
                    }
                )

        if not all_events:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(all_events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @renamed_kwargs(high="upper_limit", low="lower_limit")
    def threshold_with_hysteresis(
        self,
        upper_limit: float,
        lower_limit: float,
        *,
        max_gap: str | pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Alarm intervals with hysteresis to prevent chattering.

        Enter alarm when signal crosses `upper_limit`, exit when it drops
        below `lower_limit`.

        Args:
            upper_limit: Upper threshold to enter alarm state.
            lower_limit: Lower threshold to exit alarm state.
            max_gap: Longest time between two samples an alarm may span (e.g.
                ``"15min"``). A longer gap is "no data": an open alarm ends
                at its last sample before the gap, and after the gap the
                signal must reach ``upper_limit`` again to start a new one.
                ``None`` (default): no limit.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, duration (Timedelta), peak_value.
        """
        cols = ["start", "end", "duration", "peak_value"]
        if self.signal.empty:
            return empty_event_df("interval", cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        values = sig[self.value_column].values
        times = sig[self.time_column].array

        events: list[dict[str, Any]] = []
        in_alarm = False
        alarm_start = None
        peak = -np.inf
        gap_after: np.ndarray = np.zeros(len(values), dtype=bool)
        if max_gap is not None:
            steps = sig[self.time_column].diff().shift(-1)
            gap_after = (steps > pd.Timedelta(max_gap)).to_numpy(dtype=bool)

        for i in range(len(values)):
            v = values[i]
            if not in_alarm:
                if v >= upper_limit:
                    in_alarm = True
                    alarm_start = times[i]
                    peak = v
            else:
                peak = max(peak, v)
                if v < lower_limit:
                    events.append(
                        {
                            "start": pd.Timestamp(alarm_start),
                            "end": pd.Timestamp(times[i]),
                            "duration": pd.Timestamp(times[i])
                            - pd.Timestamp(alarm_start),
                            "peak_value": float(peak),
                        }
                    )
                    in_alarm = False
                    peak = -np.inf
            if in_alarm and gap_after[i]:
                # No data across the gap: close the alarm at its last sample.
                events.append(
                    {
                        "start": pd.Timestamp(alarm_start),
                        "end": pd.Timestamp(times[i]),
                        "duration": pd.Timestamp(times[i]) - pd.Timestamp(alarm_start),
                        "peak_value": float(peak),
                    }
                )
                in_alarm = False
                peak = -np.inf

        # Close open alarm at end of data
        if in_alarm and alarm_start is not None:
            events.append(
                {
                    "start": pd.Timestamp(alarm_start),
                    "end": pd.Timestamp(times[-1]),
                    "duration": pd.Timestamp(times[-1]) - pd.Timestamp(alarm_start),
                    "peak_value": float(peak),
                }
            )

        if not events:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
        )

    @keyword_only(2)
    @twin_columns({"time_above": "time_above_seconds", "pct_above": "time_above_pct"})
    def time_above_threshold(
        self,
        threshold: float,
        window: str = "1h",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Per window: total time and percentage above threshold.

        Args:
            threshold: Value threshold.
            window: Resample window.
            interval_end: ``"next_change"`` (default) holds each value until
                the next sample and measures the real time above the threshold
                inside each window (``pct_above`` is relative to the window
                length). A window without a sample of its own that a held
                value covers is reported too. The time after the last sample
                is not counted. ``"last_sample"`` estimates the time as the
                fraction of *samples* above the threshold times the window
                length, which is only right for uniform sampling; it reports
                only windows that contain samples.

        Returns:
            DataFrame with columns: start, end, duration_seconds, time_above,
            pct_above, exceedance_count. ``end`` is ``start`` + ``window``,
            calendar-correct: on tz-aware data a daily window on a DST change
            lasts 23 h or 25 h, and ``time_above`` / ``pct_above`` are
            measured against that real ``duration_seconds``.
            ``exceedance_count`` counts the samples that start an exceedance
            (at or above ``threshold`` while the previous sample was below);
            a signal that is already above at its first sample starts one
            there.

            ``time_above`` is in seconds.

            Canonical twin columns: ``time_above_seconds`` (after
            ``time_above``), ``time_above_pct`` (after ``pct_above``); each holds the
            same values as the deprecated column right before it, which is removed in
            the next major release.
        """
        cols = ["start", "time_above", "pct_above", "exceedance_count"]
        if self.signal.empty:
            return empty_event_df("summary", cols[1:])

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)
        sig["above"] = (sig[self.value_column] >= threshold).astype(float)

        # Count transitions into above state; the first sample starts an
        # exceedance when it is already above.
        prev_above = sig["above"].shift(1, fill_value=0.0)
        sig["enter_above"] = ((sig["above"] == 1) & (prev_above == 0)).astype(int)

        resampled = sig.resample(window).agg(
            above_frac=("above", "mean"),
            exceedance_count=("enter_above", "sum"),
            sample_count=("above", "count"),
        )
        if interval_end not in INTERVAL_END_MODES:
            raise ValueError(
                f"interval_end must be one of {INTERVAL_END_MODES}, got {interval_end!r}"
            )
        # The real [start, end) of each window: a calendar day lasts 23 h or
        # 25 h on a DST change, so time shares use that, not a fixed length.
        w_start, w_end = window_bounds(resampled.index.to_series(), window)
        resampled["window_seconds"] = (w_end - w_start).dt.total_seconds().to_numpy()
        if interval_end == "next_change":
            resampled["above_frac"] = self._time_weighted_above_fraction(
                sig["above"], pd.Index(w_start), pd.Index(w_end)
            )

        if interval_end == "next_change":
            # Every window between the first and the last sample holds a
            # value, with or without a sample of its own.
            kept = resampled
        else:
            kept = resampled[resampled["sample_count"] != 0]
        if kept.empty:
            return empty_event_df("summary", cols[1:])
        above_frac = kept["above_frac"].to_numpy(dtype=np.float64)
        events = pd.DataFrame(
            {
                "start": kept.index,
                "time_above": above_frac
                * kept["window_seconds"].to_numpy(dtype=np.float64),
                "pct_above": above_frac * 100,
                "exceedance_count": kept["exceedance_count"]
                .to_numpy(dtype=np.float64)
                .astype(np.int64),
            },
            columns=cols,
        )
        return finalize_summary_df(add_window_bounds(events, window))

    @staticmethod
    def _time_weighted_above_fraction(
        above: pd.Series, window_starts: pd.Index, window_ends: pd.Index
    ) -> list[float]:
        """Fraction of each window spent above, holding values to the next sample.

        Per window this is the sum, over all flagged samples, of the overlap
        of ``[sample, next sample)`` with the window (clipped at 0), divided
        by the window's real length (``window_ends - window_starts``: 23 h
        or 25 h for a calendar day on a DST change). Only the samples near
        the window overlap it, so the sum is computed over that slice -- with
        the pairwise summation order of the full-length sum, so the result is
        bit-identical.
        """
        times = above.index.to_series().reset_index(drop=True)
        if bool(times.isna().any()):
            return _time_weighted_above_fraction_loop(above, window_starts, window_ends)
        seg_end = times.shift(-1).fillna(times)
        flags = above.reset_index(drop=True).astype(bool).to_numpy()
        starts = np.asarray(times.array[flags].asi8, dtype=np.int64)
        stops = np.asarray(seg_end.array[flags].asi8, dtype=np.int64)
        per_second = 10.0 ** {"s": 0, "ms": 3, "us": 6, "ns": 9}[times.dt.unit]
        unit = times.dt.unit
        w_lo = np.asarray(pd.DatetimeIndex(window_starts).as_unit(unit).asi8, np.int64)
        w_hi = np.asarray(pd.DatetimeIndex(window_ends).as_unit(unit).asi8, np.int64)
        first = np.searchsorted(stops, w_lo, side="right")
        last = np.searchsorted(starts, w_hi, side="left")
        fractions: list[float] = []
        for lo, hi, a, b in zip(w_lo, w_hi, first, last, strict=True):
            if b > a:
                overlap = (
                    np.minimum(stops[a:b], hi) - np.maximum(starts[a:b], lo)
                ) / per_second
                total = _pairwise_sum_slice(
                    np.maximum(overlap, 0.0), int(a), starts.size
                )
            else:
                total = 0.0
            fractions.append(float(total) / ((hi - lo) / per_second))
        return fractions

    @keyword_only(2)
    def threshold_exceedance_trend(
        self, threshold: float, window: str = "1D"
    ) -> pd.DataFrame:
        """Track exceedance frequency and duration trend over time.

        Args:
            threshold: Value threshold.
            window: Resample window (e.g. '1D' for daily).

        Returns:
            DataFrame with columns: start, end, duration_seconds,
            exceedance_count, total_duration, total_duration_seconds,
            trend_direction. ``end`` is ``start`` + ``window``;
            ``total_duration_seconds`` is the time above ``threshold`` in the
            window, in seconds.

            Canonical twin columns: ``total_duration_seconds`` (after
            ``total_duration``); it holds the same values as the deprecated
            column right before it, which is removed in the next major
            release.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "exceedance_count",
            "total_duration",
            "total_duration_seconds",
            "trend_direction",
        ]
        time_above = self.time_above_threshold(threshold, window=window)
        if time_above.empty or len(time_above) < 2:
            return empty_event_df("summary", cols)

        counts = time_above["exceedance_count"].values
        # Simple trend: compare second half to first half
        mid = len(counts) // 2
        first_half_avg = float(np.mean(counts[:mid]))
        second_half_avg = float(np.mean(counts[mid:]))

        if second_half_avg > first_half_avg * 1.1:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        result = time_above[
            ["start", "end", "duration_seconds", "exceedance_count"]
        ].copy()
        result["total_duration"] = time_above["time_above_seconds"]
        result["total_duration_seconds"] = time_above["time_above_seconds"]
        result["trend_direction"] = trend

        return result[cols]
