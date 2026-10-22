import logging
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_summary_df,
    twin_columns,
)
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.runs import total_seconds, true_runs

logger = logging.getLogger(__name__)


def _bridge_pauses(moving: np.ndarray, max_pause: int) -> np.ndarray:
    """Set short False runs between two True runs to True."""
    out = moving.copy()
    true_idx = np.flatnonzero(moving)
    if len(true_idx) < 2:
        return out
    for left, right in pairwise(true_idx):
        if 1 < right - left <= max_pause + 1:
            out[left + 1 : right] = True
    return out


class WarmUpCoolDownEvents(Base):
    """Engineering: Warm-Up / Cool-Down Analysis

    Detect and characterize warm-up and cool-down curves — common for ovens,
    extruders, molds, hydraulic systems. Analyzes the shape, consistency,
    and timing of monotonic temperature/pressure ramps.

    Methods:
    - detect_warmup: Intervals where signal rises by at least min_rise.
    - detect_cooldown: Intervals where signal falls by at least min_fall.
    - warmup_consistency: Compare warm-up durations and rates for consistency.
    - time_to_target: Time from warmup start until target value is reached.
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:warmup",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the warm-up / cool-down analyzer.

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

    def _detect_ramps(
        self,
        direction: str,
        min_change: float,
        min_duration: str = "1min",
        max_pause: int = 0,
    ) -> list[dict[str, Any]]:
        """Monotonic ramp intervals as dicts (see :meth:`_ramp_frame`)."""
        frame = self._ramp_frame(direction, min_change, min_duration, max_pause)
        return [] if frame is None else frame.to_dict("records")

    _RAMP_COLUMNS = [
        "start",
        "end",
        "start_value",
        "end_value",
        "change",
        "duration_seconds",
        "avg_rate",
    ]

    def _ramp_frame(
        self,
        direction: str,
        min_change: float,
        min_duration: str = "1min",
        max_pause: int = 0,
    ) -> pd.DataFrame | None:
        """Detect monotonic ramp intervals (rising or falling).

        A ramp is a run of sample-to-sample steps in ``direction``. With
        ``max_pause > 0``, up to that many consecutive flat or reversed steps
        between two moving stretches are bridged, so one noisy or repeated
        sample does not split a ramp in two. Returns None when there is none.
        """
        if self.signal.empty or len(self.signal) < 3:
            return None

        min_td = pd.Timedelta(min_duration)
        # A repeated sample (same time and value) is not a pause in a ramp.
        sig = (
            self.signal[[self.time_column, self.value_column]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if len(sig) < 3:
            return None
        vals = sig[self.value_column].values
        times = sig[self.time_column].array

        # Smoothed diff direction
        diff = np.diff(vals)
        if direction == "rising":
            moving = diff > 0
        else:
            moving = diff < 0
        if max_pause > 0:
            moving = _bridge_pauses(moving, max_pause)

        # A run of moving steps [s, e) over the diff indices spans the samples
        # s..e (a ramp reaching the end of the data ends at the last sample).
        runs = true_runs(moving)
        s_idx, e_idx = runs.starts, runs.ends
        s_val = vals[s_idx].astype(np.float64)
        e_val = vals[e_idx].astype(np.float64)
        change = np.abs(e_val - s_val)
        s_time, e_time = times[s_idx], times[e_idx]
        dur = e_time - s_time
        keep = (change >= min_change) & (dur >= min_td)
        if not keep.any():
            return None
        dur_s = total_seconds(dur[keep])
        change = change[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_rate = np.where(dur_s > 0, change / dur_s, 0.0)
        return pd.DataFrame(
            {
                "start": s_time[keep],
                "end": e_time[keep],
                "start_value": s_val[keep],
                "end_value": e_val[keep],
                "change": change,
                "duration_seconds": dur_s,
                "avg_rate": avg_rate,
            },
            columns=self._RAMP_COLUMNS,
        )

    @keyword_only(2)
    def detect_warmup(
        self,
        min_rise: float,
        min_duration: str = "1min",
        *,
        max_pause: int = 0,
    ) -> pd.DataFrame:
        """Detect intervals where signal rises by at least min_rise.

        Args:
            min_rise: Minimum total value increase to qualify.
            min_duration: Minimum duration of the ramp.
            max_pause: Number of consecutive flat or reversed sample steps
                tolerated inside a ramp (default 0: any pause ends it).

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            start_value, end_value, rise, duration_seconds, avg_rate.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "start_value",
            "end_value",
            "rise",
            "duration_seconds",
            "avg_rate",
        ]
        ramps = self._ramp_frame("rising", min_rise, min_duration, max_pause)
        if ramps is None:
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(
            {
                "start": ramps["start"],
                "end": ramps["end"],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": False,
                "start_value": ramps["start_value"],
                "end_value": ramps["end_value"],
                "rise": ramps["change"],
                "duration_seconds": ramps["duration_seconds"],
                "avg_rate": ramps["avg_rate"],
            },
            columns=cols,
        )

    @keyword_only(2)
    def detect_cooldown(
        self,
        min_fall: float,
        min_duration: str = "1min",
        *,
        max_pause: int = 0,
    ) -> pd.DataFrame:
        """Detect intervals where signal falls by at least min_fall.

        Args:
            min_fall: Minimum total value decrease to qualify (positive number).
            min_duration: Minimum duration of the ramp.
            max_pause: Number of consecutive flat or reversed sample steps
                tolerated inside a ramp (default 0: any pause ends it).

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            start_value, end_value, fall, duration_seconds, avg_rate.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "start_value",
            "end_value",
            "fall",
            "duration_seconds",
            "avg_rate",
        ]
        ramps = self._ramp_frame("falling", min_fall, min_duration, max_pause)
        if ramps is None:
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(
            {
                "start": ramps["start"],
                "end": ramps["end"],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": False,
                "start_value": ramps["start_value"],
                "end_value": ramps["end_value"],
                "fall": ramps["change"],
                "duration_seconds": ramps["duration_seconds"],
                "avg_rate": ramps["avg_rate"],
            },
            columns=cols,
        )

    @keyword_only(2)
    @twin_columns(
        {"deviation_from_median_duration": "deviation_from_median_duration_seconds"}
    )
    def warmup_consistency(
        self,
        min_rise: float,
        min_duration: str = "1min",
    ) -> pd.DataFrame:
        """Compare warm-up curves for consistency in duration and rate.

        Args:
            min_rise: Minimum total value increase for a warm-up, passed to
                :meth:`detect_warmup`.
            min_duration: Minimum warm-up duration (pandas offset string), passed to
                :meth:`detect_warmup`.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, warmup_index, avg_rate,
            deviation_from_median_duration,
            deviation_from_median_duration_seconds. ``start`` / ``end`` are
            the warm-up's bounds; the deviation from the median warm-up
            duration is in seconds.

            Canonical twin columns: ``deviation_from_median_duration_seconds``
            (after ``deviation_from_median_duration``); it holds the same
            values as the deprecated column right before it, which is removed
            in the next major release.
        """
        extra = [
            "uuid",
            "source_uuid",
            "warmup_index",
            "avg_rate",
            "deviation_from_median_duration",
        ]
        warmups = self.detect_warmup(min_rise, min_duration=min_duration)
        if warmups.empty:
            return empty_event_df("summary", extra)

        median_dur = warmups["duration_seconds"].median()
        result = pd.DataFrame(
            {
                "warmup_index": range(len(warmups)),
                "start": warmups["start"].array,
                "end": warmups["end"].array,
                "duration_seconds": warmups["duration_seconds"].values,
                "avg_rate": warmups["avg_rate"].values,
                "deviation_from_median_duration": (
                    warmups["duration_seconds"].values - median_dur
                ),
            }
        )
        return finalize_summary_df(
            result, uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(2)
    def time_to_target(
        self,
        target_value: float,
        direction: str = "rising",
    ) -> pd.DataFrame:
        """Time from each ramp start until target value is reached.

        Args:
            target_value: The target value to reach.
            direction: 'rising' or 'falling'.

        Returns:
            DataFrame with columns: start, end (= target_reached_at),
            duration_seconds, uuid, source_uuid, target_reached_at,
            time_to_target_seconds, overshoot. ``overshoot`` is how far the
            signal goes beyond the target (0 when it does not) from the
            time the target is reached until the next ramp starts, or the
            end of the data for the last ramp.
        """
        cols = ["start", "target_reached_at", "time_to_target_seconds", "overshoot"]
        empty = empty_event_df("summary", ["uuid", "source_uuid", *cols[1:]])
        if self.signal.empty or len(self.signal) < 3:
            return empty

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )

        # Detect ramp starts using a minimal change threshold
        ramps = self._detect_ramps(direction, min_change=0.01, min_duration="0s")
        if not ramps:
            return empty

        # Index arithmetic on the sorted series instead of a full-series mask
        # per ramp: the first sample at/after each ramp start and the first
        # sample from there that meets the target.
        times = sig[self.time_column]
        values = sig[self.value_column].to_numpy(dtype=float)
        n = len(values)
        with np.errstate(invalid="ignore"):
            hit = (
                values >= target_value
                if direction == "rising"
                else values <= target_value
            )
        next_hit = np.minimum.accumulate(np.where(hit, np.arange(n), n)[::-1])[::-1]
        next_hit = np.append(next_hit, n)
        firsts = times.searchsorted(pd.Series([r["start"] for r in ramps]), side="left")
        ramp_firsts = np.asarray(firsts, dtype=np.int64)

        events: list[dict[str, Any]] = []
        for r, first in zip(ramps, firsts):
            j = next_hit[first]
            if j >= n:
                continue
            reached_time = times.iloc[j]
            time_to = (reached_time - r["start"]).total_seconds()

            # Overshoot: the extreme beyond the target from the time it is
            # reached until the next ramp starts (or the data ends), so a
            # later, separate ramp's peak is not booked to this one. pandas
            # max/min skip NaN, hence fmax/fmin.
            k = int(times.searchsorted(reached_time, side="left"))
            nxt = np.searchsorted(ramp_firsts, k, side="right")
            stop = int(ramp_firsts[nxt]) if nxt < ramp_firsts.size else n
            span = values[k : max(stop, k + 1)]
            if direction == "rising":
                overshoot = max(float(np.fmax.reduce(span)) - target_value, 0.0)
            else:
                overshoot = max(target_value - float(np.fmin.reduce(span)), 0.0)

            events.append(
                {
                    "start": r["start"],
                    "target_reached_at": reached_time,
                    "time_to_target_seconds": time_to,
                    "overshoot": overshoot,
                }
            )

        if not events:
            return empty
        result = pd.DataFrame(events, columns=cols)
        result["end"] = result["target_reached_at"]
        return finalize_summary_df(
            result, uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
