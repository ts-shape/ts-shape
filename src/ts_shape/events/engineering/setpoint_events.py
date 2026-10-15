import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_summary_df,
)
from ts_shape.events._windows import add_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import caller_stacklevel, keyword_only

logger = logging.getLogger(__name__)

# ---- helpers for the per-window KPIs ----

_SETTLE_COLS = [
    "start",
    "uuid",
    "is_delta",
    "t_settle_seconds",
    "settled",
    "settled_within_window",
]
_OVERSHOOT_COLS = [
    "start",
    "uuid",
    "is_delta",
    "overshoot_abs",
    "overshoot_pct",
    "overshoot_ratio",
    "t_peak_seconds",
    "undershoot_abs",
    "undershoot_pct",
    "undershoot_ratio",
    "t_undershoot_seconds",
    "oscillation_count",
    "oscillation_amplitude",
]
_SETTLE_DERIVATIVE_COLS = [
    "start",
    "uuid",
    "is_delta",
    "t_settle_seconds",
    "settled",
    "final_rate",
]
_RISE_COLS = ["start", "uuid", "is_delta", "rise_time_seconds", "reached_end"]
_DECAY_COLS = ["start", "uuid", "is_delta", "decay_rate_lambda", "fit_quality_r2"]
_FREQUENCY_COLS = [
    "start",
    "uuid",
    "is_delta",
    "oscillation_freq_hz",
    "period_seconds",
]


@dataclass(frozen=True)
class _Changes:
    """The setpoint changes: time, new value, step and previous value."""

    t0: list[pd.Timestamp]
    t0_ns: np.ndarray
    s_new: list[float]
    delta: list[float]
    prev: list[float | None]
    times: pd.Series


@dataclass(frozen=True)
class _Responses:
    """The actual signal (time order) and each change's window ``lo:hi``."""

    times: pd.Series  # Timestamps, positional index
    ticks: np.ndarray  # int64 nanoseconds
    raw: np.ndarray  # int64 in the time column's own unit
    per_second: int  # units of ``raw`` per second
    values: np.ndarray  # float64
    lo: np.ndarray
    hi: np.ndarray


def _nanmax(values: np.ndarray, valid: np.ndarray) -> float:
    """``Series.max()``: the largest non-NaN value, NaN when there is none."""
    return float(values[valid].max()) if valid.any() else np.nan


def _nanmin(values: np.ndarray, valid: np.ndarray) -> float:
    """``Series.min()``: the smallest non-NaN value, NaN when there is none."""
    return float(values[valid].min()) if valid.any() else np.nan


def _nanargmax(values: np.ndarray, valid: np.ndarray) -> int:
    """Position of ``Series.idxmax()``: the first largest non-NaN value."""
    return int(np.argmax(np.where(valid, values, -np.inf)))


def _nanargmin(values: np.ndarray, valid: np.ndarray) -> int:
    """Position of ``Series.idxmin()``: the first smallest non-NaN value."""
    return int(np.argmin(np.where(valid, values, np.inf)))


def _nanmean(values: np.ndarray, valid: np.ndarray) -> float:
    """``Series.mean()``: pairwise sum of the non-NaN values over their count
    (NaN filled with 0 in place, as pandas does), NaN when there is none."""
    count = int(valid.sum())
    if not count:
        return float("nan")
    return float(np.where(valid, values, 0.0).sum() / count)


def _held_signs(err: np.ndarray) -> np.ndarray:
    """``np.sign(err)`` with 0 (and NaN) replaced by the previous sign, the
    leading ones by the first sign (``replace(0, nan).ffill().bfill()``);
    all NaN when there is no sign."""
    signs = np.sign(err)
    known = ~(np.isnan(signs) | (signs == 0))
    if not known.any():
        return np.full(len(err), np.nan)
    last = np.where(known, np.arange(len(err)), -1)
    np.maximum.accumulate(last, out=last)
    last[last < 0] = int(np.argmax(known))
    filled: np.ndarray = signs[last]
    return filled


class SetpointChangeEvents(Base):
    """
    Detect step/ramp changes on a setpoint signal and compute follow-up KPIs
    like time-to-settle and overshoot based on an actual (process) value.

    Schema assumptions (columns):
    - uuid, sequence_number, systime, plctime, is_delta
    - value_integer, value_string, value_double, value_bool, value_bytes

    Example:
        >>> import pandas as pd
        >>> from ts_shape import SetpointChangeEvents
        >>> df = pd.DataFrame({
        ...     "systime": pd.date_range("2025-01-01 08:00", periods=4, freq="10min"),
        ...     "uuid": "oven:setpoint",
        ...     "value_double": [50.0, 50.0, 60.0, 60.0],
        ... })
        >>> steps = SetpointChangeEvents(df, "oven:setpoint").detect_setpoint_steps(min_delta=5.0)
        >>> steps[["start", "prev_level", "new_level", "magnitude"]]
                        start  prev_level  new_level  magnitude
        0 2025-01-01 08:20:00        50.0       60.0       10.0
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        setpoint_uuid: str,
        *,
        event_uuid: str = "eng:setpoint_change",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.setpoint_uuid = setpoint_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # isolate setpoint series and ensure proper dtypes/sort
        self.sp = (
            self.dataframe[self.dataframe["uuid"] == self.setpoint_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.sp[self.time_column] = pd.to_datetime(self.sp[self.time_column])

        # Cache for performance optimization
        self._actual_cache: dict[str, pd.DataFrame] = {}
        # The change table and, per (actual uuid, lookahead), the response
        # windows of every change: gathered once, shared by all KPIs.
        self._changes_cache: _Changes | None = None
        self._responses_cache: dict[tuple[str, int], _Responses] = {}

    def _get_actual(self, actual_uuid: str) -> pd.DataFrame:
        """
        Get and cache actual signal data for performance optimization.
        """
        if actual_uuid not in self._actual_cache:
            actual = (
                self.dataframe[self.dataframe["uuid"] == actual_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            actual[self.time_column] = pd.to_datetime(actual[self.time_column])
            self._actual_cache[actual_uuid] = actual
        return self._actual_cache[actual_uuid]

    def _change_intervals(
        self, rows: list[dict[str, Any]], cols: list[str]
    ) -> pd.DataFrame:
        """Canonical interval frame for setpoint changes (full schema if empty)."""
        if not rows:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(rows, columns=cols),
            uuid=self.event_uuid,
            source_uuid=self.setpoint_uuid,
        )

    def _kpi_frame(
        self,
        rows: list[dict[str, Any]],
        cols: list[str],
        actual_uuid: str,
        lookahead: str,
    ) -> pd.DataFrame:
        """Canonical summary frame for per-change KPIs.

        One row per setpoint change: ``start`` is the change instant, ``end``
        is ``start`` + ``lookahead`` (the analyzed response window) and
        ``source_uuid`` is ``"<setpoint_uuid>,<actual_uuid>"``.
        """
        source = f"{self.setpoint_uuid},{actual_uuid}"
        extra = ["uuid", "source_uuid", *[c for c in cols if c != "start"]]
        if not rows:
            return empty_event_df("summary", extra)
        df = add_window_bounds(
            pd.DataFrame(rows, columns=cols), pd.to_timedelta(lookahead)
        )
        return finalize_summary_df(df, uuid=self.event_uuid, source_uuid=source)

    # ---- Change detection ----
    @keyword_only(2)
    def detect_setpoint_steps(
        self,
        min_delta: float,
        min_hold: str = "0s",
        filter_noise: bool = False,
        noise_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Point events at times where the setpoint changes by >= min_delta and the
        new level holds for at least `min_hold` (no subsequent change within that time).

        Args:
            min_delta: Minimum magnitude of change to detect
            min_hold: Minimum duration the new level must hold
            filter_noise: If True, filter out changes smaller than noise_threshold
            noise_threshold: Threshold for noise filtering (absolute value)

        Returns:
            DataFrame with columns: start, end (== start), duration_seconds
            (0.0), uuid, source_uuid, is_delta, change_type='step', magnitude,
            prev_level, new_level.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "change_type",
            "magnitude",
            "prev_level",
            "new_level",
        ]
        if self.sp.empty:
            return self._change_intervals([], cols)

        sp = self.sp[[self.time_column, self.value_column]].copy()

        # Apply noise filtering if requested
        if filter_noise:
            # Group consecutive values within noise_threshold and use mean
            vals = sp[self.value_column].values.copy()
            current_group = vals[0]
            for i in range(1, len(vals)):
                if abs(vals[i] - current_group) <= noise_threshold:
                    vals[i] = current_group
                else:
                    current_group = vals[i]
            sp["filtered_value"] = vals
            sp[self.value_column] = sp["filtered_value"]
            sp = sp.drop(columns=["filtered_value"])

        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        change_mask = sp["delta"].abs() >= float(min_delta)

        # hold condition: next change must be after min_hold
        change_times = sp.loc[change_mask, self.time_column]
        min_hold_td = pd.to_timedelta(min_hold)
        next_change_times = change_times.shift(-1)
        hold_ok = (
            next_change_times - change_times >= min_hold_td
        ) | next_change_times.isna()
        valid_change_times = change_times[hold_ok]

        rows: list[dict[str, Any]] = []
        # The first setpoint sample at each change time (binary search on the
        # sorted series instead of a full-series comparison per change).
        first_at = sp[self.time_column].searchsorted(valid_change_times, side="left")
        for t, pos in zip(valid_change_times, first_at):
            row = sp.iloc[pos]
            rows.append(
                {
                    "start": t,
                    "end": t,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "change_type": "step",
                    "magnitude": float(row["delta"]),
                    "prev_level": float(row["prev"]) if pd.notna(row["prev"]) else None,
                    "new_level": float(row[self.value_column]),
                }
            )

        return self._change_intervals(rows, cols)

    @keyword_only(2)
    def detect_setpoint_ramps(
        self, min_rate: float, min_duration: str = "0s"
    ) -> pd.DataFrame:
        """
        Interval events where |dS/dt| >= min_rate for at least `min_duration`.

        Args:
            min_rate: Minimum absolute setpoint rate of change, in setpoint units
                per second, for a sample-to-sample step to count as ramping.
            min_duration: Minimum time from the first to the last ramping sample of
                a contiguous run (pandas offset string). Default ``"0s"``.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, change_type='ramp', avg_rate, delta.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "change_type",
            "avg_rate",
            "delta",
        ]
        if self.sp.empty:
            return self._change_intervals([], cols)

        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["dt_s"] = sp[self.time_column].diff().dt.total_seconds()
        sp["dv"] = sp[self.value_column].diff()
        # A sample sharing its timestamp with the previous one (dt = 0) has no
        # rate (it was +-inf, so always "ramping"): its rate is NaN, it does not
        # count as ramping and its jump is not part of a ramp's delta.
        dup = sp["dt_s"] <= 0
        if dup.any():
            warnings.warn(
                f"detect_setpoint_ramps(): {int(dup.sum())} setpoint sample(s) "
                "share a timestamp with the previous sample; their rate is "
                "undefined and they are not counted as ramping",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        sp["rate"] = sp["dv"] / sp["dt_s"].where(~dup)
        rate_mask = sp["rate"].abs() >= float(min_rate)

        # group contiguous True segments
        group_id = (rate_mask != rate_mask.shift()).cumsum()
        events: list[dict[str, Any]] = []
        min_d = pd.to_timedelta(min_duration)
        for _, seg in sp.groupby(group_id):
            seg_mask_true = rate_mask.loc[seg.index]
            if not seg_mask_true.any():
                continue
            # boundaries
            start_time = seg.loc[seg_mask_true, self.time_column].iloc[0]
            end_time = seg.loc[seg_mask_true, self.time_column].iloc[-1]
            if (end_time - start_time) < min_d:
                continue
            avg_rate = seg.loc[seg_mask_true, "rate"].mean()
            delta = seg.loc[seg_mask_true, "dv"].sum()
            events.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "change_type": "ramp",
                    "avg_rate": float(avg_rate) if pd.notna(avg_rate) else None,
                    "delta": float(delta) if pd.notna(delta) else None,
                }
            )

        return self._change_intervals(events, cols)

    def detect_setpoint_changes(
        self,
        *,
        min_delta: float = 0.0,
        min_rate: float | None = None,
        min_hold: str = "0s",
        min_duration: str = "0s",
    ) -> pd.DataFrame:
        """
        Unified setpoint change table (steps + ramps) with standardized columns.

        Args:
            min_delta: Minimum step magnitude, passed to
                :meth:`detect_setpoint_steps`.
            min_rate: Minimum ramp rate in setpoint units per second, passed to
                :meth:`detect_setpoint_ramps`. None skips ramp detection.
            min_hold: Minimum time a new step level must hold, passed to
                :meth:`detect_setpoint_steps`.
            min_duration: Minimum ramp duration, passed to
                :meth:`detect_setpoint_ramps`.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, change_type, magnitude, prev_level,
            new_level, avg_rate, delta.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "is_delta",
            "change_type",
            "magnitude",
            "prev_level",
            "new_level",
            "avg_rate",
            "delta",
        ]
        steps = self.detect_setpoint_steps(min_delta=min_delta, min_hold=min_hold)
        ramps = (
            self.detect_setpoint_ramps(min_rate=min_rate, min_duration=min_duration)
            if min_rate is not None
            else empty_event_df("interval")
        )
        # ensure uniform columns
        if not steps.empty:
            steps = steps.assign(avg_rate=None, delta=None)[cols]
        if not ramps.empty:
            ramps = ramps.assign(magnitude=None, prev_level=None, new_level=None)[cols]
        frames = [df for df in (steps, ramps) if not df.empty]
        if not frames:
            return empty_event_df("interval", cols)
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["start", "end"])

    # ---- Follow-up KPIs ----
    #
    # Every KPI looks at the actual signal in one window per setpoint change,
    # ``[t0, t0 + lookahead]``. The change table and the window bounds of all
    # changes (one vectorized binary search) are computed once per (actual
    # signal, lookahead) and shared by every KPI -- control_quality_metrics
    # evaluates all of them from one gather. Each window is then read as a
    # slice of the actual signal's value and time arrays. Sums stay numpy's
    # pairwise sums over each window (as pandas computes them), so every KPI
    # is bit-identical to evaluating it on the window's rows with pandas.

    def _change_table(self) -> _Changes:
        """Setpoint changes (rows where the setpoint differs from the previous
        sample), computed once."""
        if self._changes_cache is None:
            sp = self.sp[[self.time_column, self.value_column]].copy()
            sp["prev"] = sp[self.value_column].shift(1)
            sp["delta"] = sp[self.value_column] - sp["prev"]
            changes = sp.loc[
                sp["delta"].abs() > 0,
                [self.time_column, self.value_column, "delta", "prev"],
            ].reset_index(drop=True)
            records = changes.to_dict("records")
            t0 = [r[self.time_column] for r in records]
            self._changes_cache = _Changes(
                t0=t0,
                t0_ns=pd.DatetimeIndex(changes[self.time_column]).as_unit("ns").asi8,
                s_new=[float(r[self.value_column]) for r in records],
                delta=[
                    float(r["delta"]) if pd.notna(r["delta"]) else 0.0 for r in records
                ],
                prev=[
                    float(r["prev"]) if pd.notna(r["prev"]) else None for r in records
                ],
                times=changes[self.time_column],
            )
        return self._changes_cache

    def _responses(self, actual_uuid: str, look_td: pd.Timedelta) -> _Responses:
        """The actual signal as arrays and every change's window
        ``[t0, t0 + look_td]`` as row bounds ``lo:hi`` (cached)."""
        key = (actual_uuid, pd.Timedelta(look_td).value)
        if key not in self._responses_cache:
            actual = self._get_actual(actual_uuid)
            times = actual[self.time_column].reset_index(drop=True)
            index = pd.DatetimeIndex(times)
            changes = self._change_table()
            ticks = index.as_unit("ns").asi8
            starts = pd.DatetimeIndex(changes.times).as_unit("ns")
            ends = (pd.DatetimeIndex(changes.times) + look_td).as_unit("ns")
            self._responses_cache[key] = _Responses(
                times=times,
                ticks=ticks,
                raw=index.asi8,
                per_second=int(pd.Timedelta(1, "s") / pd.Timedelta(1, index.unit)),
                values=actual[self.value_column].to_numpy(
                    dtype="float64", na_value=np.nan
                ),
                lo=np.searchsorted(ticks, starts.asi8, side="left"),
                hi=np.searchsorted(ticks, ends.asi8, side="right"),
            )
        return self._responses_cache[key]

    def _settle_rows(
        self,
        actual_uuid: str,
        tol: float,
        settle_pct: float | None,
        hold: str,
        lookahead: str,
    ) -> list[dict[str, Any]]:
        hold_ns = pd.to_timedelta(hold).value
        ch = self._change_table()
        resp = self._responses(actual_uuid, pd.to_timedelta(lookahead))
        # The response to a step is evaluated until the lookahead ends or the
        # next (strictly later) setpoint change, whichever is first: after
        # that the process follows a different setpoint.
        nxt = np.searchsorted(ch.t0_ns, ch.t0_ns, side="right")
        next_ns = np.where(
            nxt < len(ch.t0_ns), ch.t0_ns[np.minimum(nxt, len(ch.t0_ns) - 1)], 0
        )
        before_next = np.searchsorted(resp.ticks, next_ns, side="left")
        rows: list[dict[str, Any]] = []
        for i, t0 in enumerate(ch.t0):
            lo = int(resp.lo[i])
            hi = int(resp.hi[i])
            if nxt[i] < len(ch.t0_ns):
                hi = min(hi, int(before_next[i]))
            delta = ch.delta[i]
            # Calculate tolerance based on settle_pct or use absolute tol
            if settle_pct is not None and delta != 0:
                effective_tol = abs(delta) * settle_pct
            else:
                effective_tol = tol
            row: dict[str, Any] = {
                "start": t0,
                "uuid": self.event_uuid,
                "is_delta": True,
                "t_settle_seconds": None,
                "settled": False,
                "settled_within_window": False,
            }
            if hi > lo:
                err = np.abs(resp.values[lo:hi] - ch.s_new[i])
                present = np.flatnonzero(~np.isnan(err))  # a missing reading
                inside = err[present] <= effective_tol  # is no evidence
                # Settling time (Ogata): the instant after which the response
                # enters the band and stays inside it until the end of the
                # evaluation window -- the start of the final inside run, not
                # the first entry (an underdamped response passes through the
                # band long before it settles).
                if inside.size and inside[-1]:
                    outside = np.flatnonzero(~inside)
                    first_in = int(outside[-1]) + 1 if outside.size else 0
                    enter = lo + int(present[first_in])
                    t_enter = resp.times.iloc[enter]
                    row["settled_within_window"] = True
                    row["t_settle_seconds"] = (t_enter - t0).total_seconds()
                    # ``hold`` confirms the settle: the final inside run must
                    # last at least that long before the window ends.
                    last = lo + int(present[-1])
                    row["settled"] = bool(
                        resp.ticks[last] - resp.ticks[enter] >= hold_ns
                    )
            rows.append(row)
        return rows

    def time_to_settle(
        self,
        actual_uuid: str,
        *,
        tol: float = 0.0,
        settle_pct: float | None = None,
        hold: str = "0s",
        lookahead: str = "10min",
    ) -> pd.DataFrame:
        """
        For each setpoint change (any change), compute the settling time: the
        time after which the actual signal enters the ±tolerance band around
        the new setpoint and *stays* inside it until the end of the evaluation
        window (Ogata, *Modern Control Engineering*, settling time ``t_s``).
        The first entry into the band is not the settling time: an
        underdamped response crosses the band on every swing long before it
        settles.

        Args:
            actual_uuid: UUID of the actual/process value signal
            tol: Absolute tolerance, used when ``settle_pct`` is None.
            settle_pct: Tolerance as a fraction of the step magnitude (e.g.
                0.02 for 2%). When given it wins over ``tol``; passing both
                emits a ``UserWarning``.
            hold: Minimum duration the final inside-band run must last
                before the window ends for the settle to count as confirmed
                (``settled``).
            lookahead: Evaluation window after each change. It also ends at
                the next (later) setpoint change, after which the process
                follows a different setpoint. The response must be inside the
                band at the window's last sample to have settled.

        Returns:
            DataFrame with columns: start, end (start + the lookahead window),
            duration_seconds, uuid, source_uuid ("<setpoint_uuid>,<actual_uuid>"),
            is_delta, t_settle_seconds (seconds from the change to the start
            of the final inside-band run; ``None`` when the response is
            outside the band at the end of the window), settled_within_window
            (True when the response is inside the band at the end of the
            window, so a settling time exists) and settled
            (settled_within_window and the final inside-band run lasts at
            least ``hold``).
        """
        if settle_pct is not None and tol != 0.0:
            warnings.warn(
                "time_to_settle(): both tol and settle_pct were passed; "
                "settle_pct wins and tol is ignored",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        rows = (
            []
            if self.sp.empty
            else self._settle_rows(actual_uuid, tol, settle_pct, hold, lookahead)
        )
        return self._kpi_frame(rows, _SETTLE_COLS, actual_uuid, lookahead)

    def _overshoot_rows(self, actual_uuid: str, window: str) -> list[dict[str, Any]]:
        ch = self._change_table()
        resp = self._responses(actual_uuid, pd.to_timedelta(window))
        out_rows: list[dict[str, Any]] = []
        for i, t0 in enumerate(ch.t0):
            lo, hi = int(resp.lo[i]), int(resp.hi[i])
            if hi <= lo:
                out_rows.append(
                    {"start": t0, "uuid": self.event_uuid, "is_delta": True}
                    | dict.fromkeys(_OVERSHOOT_COLS[3:])
                )
                continue
            delta = ch.delta[i]
            err = resp.values[lo:hi] - ch.s_new[i]
            valid = ~np.isnan(err)

            # Undershoot is the swing back past the new setpoint *after* the
            # process first reaches it. Measured over the whole window it was
            # always the initial step deviation (100 % on every step).
            reached = err >= 0 if delta >= 0 else err <= 0
            after = err[int(np.argmax(reached)) :] if reached.any() else err[:0]
            after_valid = ~np.isnan(after)

            def _time(pos: int) -> pd.Timestamp:
                return resp.times.iloc[lo + pos]

            # Overshoot (in direction of change); the extremes skip NaN.
            t_undershoot = None
            if delta >= 0:
                peak = _nanmax(err, valid)
                t_peak = _time(_nanargmax(err, valid)) if peak > 0 else None
                # Undershoot (opposite direction)
                undershoot_val = _nanmin(after, after_valid) if after.size else 0.0
                if undershoot_val < 0:
                    t_undershoot = _time(
                        hi - lo - after.size + _nanargmin(after, after_valid)
                    )
            else:
                low = _nanmin(err, valid)
                peak = -low  # magnitude for downward step
                t_peak = _time(_nanargmin(err, valid)) if low < 0 else None
                # Undershoot (opposite direction)
                high = _nanmax(after, after_valid) if after.size else np.nan
                undershoot_val = -high if after.size else 0.0
                if after.size and high > 0:
                    t_undershoot = _time(
                        hi - lo - after.size + _nanargmax(after, after_valid)
                    )

            overshoot_abs = float(peak) if pd.notna(peak) and peak > 0 else 0.0
            overshoot_ratio = overshoot_abs / abs(delta) if delta != 0 else None
            undershoot_abs = (
                float(abs(undershoot_val))
                if pd.notna(undershoot_val) and abs(undershoot_val) > 0
                else 0.0
            )
            undershoot_ratio = undershoot_abs / abs(delta) if delta != 0 else None

            # Oscillation detection: count zero crossings. A sample exactly
            # on the setpoint holds the previous sign so a crossing through
            # an exact zero is counted once, not twice.
            err_sign = _held_signs(err)
            changes = err_sign[1:] != err_sign[:-1]
            oscillation_count = max(0, int(changes[~np.isnan(err_sign[1:])].sum()))

            # Oscillation amplitude: average of peak deviations
            oscillation_amplitude = float(_nanmean(np.abs(err), valid))

            out_rows.append(
                {
                    "start": t0,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "overshoot_abs": overshoot_abs if overshoot_abs > 0 else None,
                    "overshoot_pct": (
                        100.0 * float(overshoot_ratio)
                        if overshoot_ratio is not None and overshoot_abs > 0
                        else None
                    ),
                    "overshoot_ratio": (
                        float(overshoot_ratio)
                        if overshoot_ratio is not None and overshoot_abs > 0
                        else None
                    ),
                    "t_peak_seconds": (
                        (t_peak - t0).total_seconds() if t_peak is not None else None
                    ),
                    "undershoot_abs": undershoot_abs if undershoot_abs > 0 else None,
                    "undershoot_pct": (
                        100.0 * float(undershoot_ratio)
                        if undershoot_ratio is not None and undershoot_abs > 0
                        else None
                    ),
                    "undershoot_ratio": (
                        float(undershoot_ratio)
                        if undershoot_ratio is not None and undershoot_abs > 0
                        else None
                    ),
                    "t_undershoot_seconds": (
                        (t_undershoot - t0).total_seconds()
                        if t_undershoot is not None
                        else None
                    ),
                    "oscillation_count": oscillation_count,
                    "oscillation_amplitude": oscillation_amplitude,
                }
            )
        return out_rows

    def overshoot_metrics(
        self,
        actual_uuid: str,
        *,
        window: str = "10min",
    ) -> pd.DataFrame:
        """
        For each change, compute peak overshoot, undershoot, and oscillation metrics
        relative to the new setpoint within a lookahead window.

        Args:
            actual_uuid: UUID of the actual (process value) signal.
            window: Lookahead window after each setpoint change in which the peak,
                undershoot and oscillations are measured (pandas offset string).

        Returns:
            DataFrame with columns: start, end (start + the lookahead window),
            duration_seconds, uuid, source_uuid ("<setpoint_uuid>,<actual_uuid>"),
            is_delta, overshoot_abs, overshoot_pct, overshoot_ratio,
            t_peak_seconds, undershoot_abs, undershoot_pct, undershoot_ratio,
            t_undershoot_seconds, oscillation_count, oscillation_amplitude.

            ``overshoot_pct`` is the peak overshoot in percent of the step
            magnitude (0-100+, Ogata's maximum percent overshoot ``M_p``);
            ``overshoot_ratio`` is the same quantity as a fraction (0-1+).
            ``undershoot_pct`` / ``undershoot_ratio`` are the swing back past
            the new setpoint after it is first reached, in the same two units.
        """
        rows = [] if self.sp.empty else self._overshoot_rows(actual_uuid, window)
        return self._kpi_frame(rows, _OVERSHOOT_COLS, actual_uuid, window)

    def _settle_derivative_rows(
        self, actual_uuid: str, rate_threshold: float, lookahead: str, hold: str
    ) -> list[dict[str, Any]]:
        hold_td = pd.to_timedelta(hold)
        hold_ns = hold_td.value
        ch = self._change_table()
        resp = self._responses(actual_uuid, pd.to_timedelta(lookahead))
        rows: list[dict[str, Any]] = []
        for i, t0 in enumerate(ch.t0):
            lo, hi = int(resp.lo[i]), int(resp.hi[i])
            row: dict[str, Any] = {
                "start": t0,
                "uuid": self.event_uuid,
                "is_delta": True,
                "t_settle_seconds": None,
                "settled": False,
                "final_rate": None,
            }
            if hi - lo < 2:
                rows.append(row)
                continue

            # Rate of change between consecutive samples (NaN for the first).
            with np.errstate(divide="ignore", invalid="ignore"):
                dt_s = np.diff(resp.raw[lo:hi]) / resp.per_second
                rate = np.abs(np.diff(resp.values[lo:hi]) / dt_s)
            rate = np.concatenate(([np.nan], rate))

            # Find when rate drops below threshold
            below = rate <= rate_threshold
            if below.any():
                if hold_td.total_seconds() > 0:
                    # The first run below the threshold lasting ``hold``.
                    edges = np.flatnonzero(np.diff(below.astype(np.int8)))
                    run_starts = np.concatenate(([0], edges + 1))
                    run_ends = np.concatenate((edges + 1, [hi - lo]))
                    for a, b in zip(run_starts, run_ends, strict=True):
                        if not below[a]:
                            continue
                        if resp.ticks[lo + b - 1] - resp.ticks[lo + a] >= hold_ns:
                            row["t_settle_seconds"] = (
                                resp.times.iloc[lo + a] - t0
                            ).total_seconds()
                            row["settled"] = True
                            row["final_rate"] = float(np.sum(rate[a:b]) / (b - a))
                            break
                else:
                    # First entry below threshold
                    first = int(np.argmax(below))
                    row["t_settle_seconds"] = (
                        resp.times.iloc[lo + first] - t0
                    ).total_seconds()
                    row["settled"] = True
                    row["final_rate"] = float(rate[first])
            rows.append(row)
        return rows

    def time_to_settle_derivative(
        self,
        actual_uuid: str,
        *,
        rate_threshold: float = 0.01,
        lookahead: str = "10min",
        hold: str = "0s",
    ) -> pd.DataFrame:
        """
        Detect settling based on rate of change (derivative) falling below threshold.
        More sensitive to when the process has truly stopped moving.

        Args:
            actual_uuid: UUID of the actual/process value signal
            rate_threshold: Maximum absolute rate of change to consider settled
            lookahead: Maximum time window to search for settling
            hold: Minimum duration the rate must stay below threshold

        Returns:
            DataFrame with columns: start, end (start + the lookahead window),
            duration_seconds, uuid, source_uuid ("<setpoint_uuid>,<actual_uuid>"),
            is_delta, t_settle_seconds, settled, final_rate.
        """
        rows = (
            []
            if self.sp.empty
            else self._settle_derivative_rows(
                actual_uuid, rate_threshold, lookahead, hold
            )
        )
        return self._kpi_frame(rows, _SETTLE_DERIVATIVE_COLS, actual_uuid, lookahead)

    def _rise_rows(
        self, actual_uuid: str, start_pct: float, end_pct: float, lookahead: str
    ) -> list[dict[str, Any]]:
        ch = self._change_table()
        resp = self._responses(actual_uuid, pd.to_timedelta(lookahead))
        rows: list[dict[str, Any]] = []
        for i, t0 in enumerate(ch.t0):
            lo, hi = int(resp.lo[i]), int(resp.hi[i])
            s_new = ch.s_new[i]
            prev = ch.prev[i]
            s_prev = prev if prev is not None else s_new
            delta = ch.delta[i]
            row: dict[str, Any] = {
                "start": t0,
                "uuid": self.event_uuid,
                "is_delta": True,
                "rise_time_seconds": None,
                "reached_end": False,
            }
            if hi > lo and delta != 0:
                # Target levels and the first sample crossing each.
                start_level = s_prev + delta * start_pct
                end_level = s_prev + delta * end_pct
                values = resp.values[lo:hi]
                if delta > 0:
                    start_crossed = values >= start_level
                    end_crossed = values >= end_level
                else:
                    start_crossed = values <= start_level
                    end_crossed = values <= end_level
                if start_crossed.any() and end_crossed.any():
                    t_start = resp.times.iloc[lo + int(np.argmax(start_crossed))]
                    t_end = resp.times.iloc[lo + int(np.argmax(end_crossed))]
                    row["rise_time_seconds"] = (t_end - t_start).total_seconds()
                    row["reached_end"] = True
            rows.append(row)
        return rows

    def rise_time(
        self,
        actual_uuid: str,
        *,
        start_pct: float = 0.1,
        end_pct: float = 0.9,
        lookahead: str = "10min",
    ) -> pd.DataFrame:
        """
        Compute rise time: time for actual to go from start_pct to end_pct of the setpoint change.
        Typically measured from 10% to 90% of the final value.

        Args:
            actual_uuid: UUID of the actual/process value signal
            start_pct: Starting percentage of change (e.g., 0.1 for 10%)
            end_pct: Ending percentage of change (e.g., 0.9 for 90%)
            lookahead: Maximum time window to search

        Returns:
            DataFrame with columns: start, end (start + the lookahead window),
            duration_seconds, uuid, source_uuid ("<setpoint_uuid>,<actual_uuid>"),
            is_delta, rise_time_seconds, reached_end.
        """
        rows = (
            []
            if self.sp.empty
            else self._rise_rows(actual_uuid, start_pct, end_pct, lookahead)
        )
        return self._kpi_frame(rows, _RISE_COLS, actual_uuid, lookahead)

    def _decay_rows(
        self, actual_uuid: str, lookahead: str, min_points: int
    ) -> list[dict[str, Any]]:
        ch = self._change_table()
        resp = self._responses(actual_uuid, pd.to_timedelta(lookahead))
        rows: list[dict[str, Any]] = []
        failed_fits: list[pd.Timestamp] = []
        for i, t0 in enumerate(ch.t0):
            lo, hi = int(resp.lo[i]), int(resp.hi[i])
            row: dict[str, Any] = {
                "start": t0,
                "uuid": self.event_uuid,
                "is_delta": True,
                "decay_rate_lambda": None,
                "fit_quality_r2": None,
            }
            if hi <= lo or hi - lo < min_points:
                rows.append(row)
                continue

            # Error and time since the change; near-zero errors are left out
            # of the log fit.
            err = np.abs(resp.values[lo:hi] - ch.s_new[i])
            valid = err > 1e-6
            if valid.sum() < min_points:
                rows.append(row)
                continue
            err_valid = err[valid]
            t_valid = (resp.ticks[lo:hi][valid] - ch.t0_ns[i]) / 1e9

            try:
                # Linear fit to log(error) vs time: log(err) = log(A) - lambda*t
                log_err = np.log(err_valid)
                coeffs = np.polyfit(t_valid, log_err, 1)
                decay_lambda = -coeffs[0]  # negative slope

                # Calculate R^2
                log_err_pred = np.polyval(coeffs, t_valid)
                ss_res = np.sum((log_err - log_err_pred) ** 2)
                ss_tot = np.sum((log_err - log_err.sum() / len(log_err)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                row["decay_rate_lambda"] = (
                    float(decay_lambda) if decay_lambda > 0 else None
                )
                row["fit_quality_r2"] = float(r2)
            except (np.linalg.LinAlgError, ValueError):
                failed_fits.append(t0)
            rows.append(row)

        if failed_fits:
            warnings.warn(
                f"Exponential decay fit failed for {len(failed_fits)} setpoint "
                f"change(s) of {actual_uuid!r} (first at {failed_fits[0]}); "
                "their decay_rate_lambda and fit_quality_r2 are None.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        return rows

    def decay_rate(
        self,
        actual_uuid: str,
        *,
        lookahead: str = "10min",
        min_points: int = 5,
    ) -> pd.DataFrame:
        """
        Estimate exponential decay rate of the settling behavior.
        Fits error(t) = A * exp(-lambda * t) and returns lambda.

        Args:
            actual_uuid: UUID of the actual/process value signal
            lookahead: Time window for analysis
            min_points: Minimum number of points required for fitting

        Returns:
            DataFrame with columns: start, end (start + the lookahead window),
            duration_seconds, uuid, source_uuid ("<setpoint_uuid>,<actual_uuid>"),
            is_delta, decay_rate_lambda, fit_quality_r2.
        """
        rows = (
            []
            if self.sp.empty
            else self._decay_rows(actual_uuid, lookahead, min_points)
        )
        return self._kpi_frame(rows, _DECAY_COLS, actual_uuid, lookahead)

    def _frequency_rows(
        self, actual_uuid: str, window: str, min_oscillations: int
    ) -> list[dict[str, Any]]:
        ch = self._change_table()
        resp = self._responses(actual_uuid, pd.to_timedelta(window))
        rows: list[dict[str, Any]] = []
        for i, t0 in enumerate(ch.t0):
            lo, hi = int(resp.lo[i]), int(resp.hi[i])
            row: dict[str, Any] = {
                "start": t0,
                "uuid": self.event_uuid,
                "is_delta": True,
                "oscillation_freq_hz": None,
                "period_seconds": None,
            }
            if hi > lo:
                # Detect zero crossings. A sample exactly on the setpoint
                # (sign 0) holds the previous sign; otherwise one crossing
                # through an exact zero counted twice, 1 sample apart, halving
                # the period.
                err_sign = _held_signs(resp.values[lo:hi] - ch.s_new[i])
                sign_changes = np.where(np.diff(err_sign) != 0)[0]
                # each oscillation has 2 crossings
                if len(sign_changes) >= min_oscillations * 2:
                    # Average period (2 crossings = 1 period)
                    gaps = np.diff(resp.raw[lo + sign_changes]) / resp.per_second
                    if len(gaps) > 0:
                        period = gaps.sum() / len(gaps) * 2
                        freq = 1.0 / period if period > 0 else None
                        row["oscillation_freq_hz"] = (
                            float(freq) if freq is not None else None
                        )
                        row["period_seconds"] = float(period)
            rows.append(row)
        return rows

    def oscillation_frequency(
        self,
        actual_uuid: str,
        *,
        window: str = "10min",
        min_oscillations: int = 2,
    ) -> pd.DataFrame:
        """
        Estimate the frequency of oscillations during settling.
        Counts zero crossings and estimates period.

        Args:
            actual_uuid: UUID of the actual/process value signal
            window: Time window for analysis
            min_oscillations: Minimum number of oscillations to compute frequency

        Returns:
            DataFrame with columns: start, end (start + the lookahead window),
            duration_seconds, uuid, source_uuid ("<setpoint_uuid>,<actual_uuid>"),
            is_delta, oscillation_freq_hz, period_seconds.
        """
        rows = (
            []
            if self.sp.empty
            else self._frequency_rows(actual_uuid, window, min_oscillations)
        )
        return self._kpi_frame(rows, _FREQUENCY_COLS, actual_uuid, window)

    def _steady_state_rows(
        self, actual_uuid: str, lookahead: str
    ) -> list[dict[str, Any]]:
        ch = self._change_table()
        resp = self._responses(actual_uuid, pd.to_timedelta(lookahead))
        rows: list[dict[str, Any]] = []
        for i, t0 in enumerate(ch.t0):
            lo, hi = int(resp.lo[i]), int(resp.hi[i])
            ss_error = None
            if hi > lo:
                # Use last 10% of window for steady-state
                tail = resp.values[lo + int((hi - lo) * 0.9) : hi]
                if len(tail) > 0:
                    dev = np.abs(tail - ch.s_new[i])
                    ss_error = float(_nanmean(dev, ~np.isnan(dev)))
            rows.append({"start": t0, "steady_state_error": ss_error})
        return rows

    def control_quality_metrics(
        self,
        actual_uuid: str,
        *,
        tol: float = 0.0,
        settle_pct: float | None = None,
        hold: str = "0s",
        lookahead: str = "10min",
        rate_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Comprehensive control quality metrics combining multiple performance indicators.

        Computes all available metrics for each setpoint change and returns them in a single DataFrame.
        This includes: settling time, rise time, overshoot, undershoot, oscillations, and decay characteristics.
        The change table and the response windows are gathered once and
        shared by all of them.

        Args:
            actual_uuid: UUID of the actual/process value signal
            tol: Absolute tolerance for settling, used when ``settle_pct``
                is None.
            settle_pct: Settling tolerance as a fraction of the step
                magnitude. When given it wins over ``tol``; passing both
                emits a ``UserWarning``.
            hold: Minimum duration to confirm settling
            lookahead: Time window for all analyses
            rate_threshold: Rate threshold for derivative-based settling

        Returns:
            DataFrame with comprehensive metrics including:
            - start, end (start + lookahead), duration_seconds, uuid,
                source_uuid ("<setpoint_uuid>,<actual_uuid>"), is_delta
            - t_settle_seconds, settled, settled_within_window (from
                time_to_settle)
            - t_settle_derivative_seconds (from time_to_settle_derivative)
            - rise_time_seconds (from rise_time)
            - overshoot_abs, overshoot_pct (percent), overshoot_ratio
                (fraction) (from overshoot_metrics)
            - undershoot_abs, undershoot_pct (percent), undershoot_ratio
            - oscillation_count, oscillation_amplitude, oscillation_freq_hz
            - decay_rate_lambda, fit_quality_r2
            - steady_state_error (final error in window)
        """
        if self.sp.empty:
            return empty_event_df(
                "summary",
                [
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "t_settle_seconds",
                    "settled",
                    "settled_within_window",
                    "t_settle_derivative_seconds",
                    "rise_time_seconds",
                    "overshoot_abs",
                    "overshoot_pct",
                    "overshoot_ratio",
                    "undershoot_abs",
                    "undershoot_pct",
                    "undershoot_ratio",
                    "oscillation_count",
                    "oscillation_amplitude",
                    "oscillation_freq_hz",
                    "decay_rate_lambda",
                    "fit_quality_r2",
                    "steady_state_error",
                ],
            )

        if settle_pct is not None and tol != 0.0:
            warnings.warn(
                "control_quality_metrics(): both tol and settle_pct were "
                "passed; settle_pct wins and tol is ignored",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
            tol = 0.0  # unused once settle_pct is set

        # Every metric from the one shared gather (change table + windows).
        def kpi(rows: list[dict[str, Any]], cols: list[str]) -> pd.DataFrame:
            return self._kpi_frame(rows, cols, actual_uuid, lookahead)

        settle_df = kpi(
            self._settle_rows(actual_uuid, tol, settle_pct, hold, lookahead),
            _SETTLE_COLS,
        )
        settle_deriv_df = kpi(
            self._settle_derivative_rows(actual_uuid, rate_threshold, lookahead, hold),
            _SETTLE_DERIVATIVE_COLS,
        )
        rise_df = kpi(self._rise_rows(actual_uuid, 0.1, 0.9, lookahead), _RISE_COLS)
        overshoot_df = kpi(
            self._overshoot_rows(actual_uuid, lookahead), _OVERSHOOT_COLS
        )
        decay_df = kpi(self._decay_rows(actual_uuid, lookahead, 5), _DECAY_COLS)
        freq_df = kpi(self._frequency_rows(actual_uuid, lookahead, 2), _FREQUENCY_COLS)
        ss_error_df = pd.DataFrame(
            self._steady_state_rows(actual_uuid, lookahead),
            columns=["start", "steady_state_error"],
        )

        # Merge all metrics on 'start'. Every KPI method emits one row per
        # setpoint change in the same order, so two changes at the same
        # instant are matched by their occurrence number instead of being
        # cross-joined.
        def _keyed(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
            out = frame[cols].copy()
            out["_occ"] = out.groupby("start").cumcount()
            return out

        result = settle_df.copy()
        result["_occ"] = result.groupby("start").cumcount()
        parts = [
            _keyed(settle_deriv_df, ["start", "t_settle_seconds"]).rename(
                columns={"t_settle_seconds": "t_settle_derivative_seconds"}
            ),
            _keyed(rise_df, ["start", "rise_time_seconds"]),
            _keyed(
                overshoot_df,
                [
                    "start",
                    "overshoot_abs",
                    "overshoot_pct",
                    "overshoot_ratio",
                    "undershoot_abs",
                    "undershoot_pct",
                    "undershoot_ratio",
                    "oscillation_count",
                    "oscillation_amplitude",
                ],
            ),
            _keyed(freq_df, ["start", "oscillation_freq_hz"]),
            _keyed(decay_df, ["start", "decay_rate_lambda", "fit_quality_r2"]),
            _keyed(ss_error_df, ["start", "steady_state_error"]),
        ]
        for part in parts:
            result = result.merge(part, on=["start", "_occ"], how="left")
        result = result.drop(columns="_occ")

        return result
