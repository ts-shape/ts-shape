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
from ts_shape.events._windows import add_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs
from ts_shape.utils.frames import to_ns
from ts_shape.utils.runs import true_runs
from ts_shape.utils.time_accounting import grouper_periods, split_intervals

logger = logging.getLogger(__name__)


def _split_steps(
    times: pd.Series,
    s_ns: np.ndarray,
    e_ns: np.ndarray,
    window: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Cut the steps ``[s_ns[k], e_ns[k])`` at the ``resample(window)`` bounds.

    Returns ``(periods, p_idx, i_idx, a_rel, seconds)``: the windows covering
    ``times`` (``period`` label, true ``start`` / ``end``) and, per piece with
    positive length, its window position, its step position, its start in
    seconds after the window start and its length in seconds. ``None`` when
    no piece has positive length.
    """
    if s_ns.size == 0 or times.empty:
        return None
    periods = grouper_periods(times.iloc[0], times.iloc[-1], window)
    if periods.empty:
        return None
    periods = periods.reset_index(drop=True)
    ps_all = to_ns(periods["start"])
    # Only instants matter for the split: work on UTC nanoseconds.
    steps = pd.DataFrame(
        {
            "start": s_ns.view("datetime64[ns]"),
            "end": e_ns.view("datetime64[ns]"),
            "_i": np.arange(s_ns.size, dtype=np.int64),
        }
    )
    grid = pd.DataFrame(
        {
            "start": ps_all.view("datetime64[ns]"),
            "end": to_ns(periods["end"]).view("datetime64[ns]"),
            "_p": np.arange(len(periods), dtype=np.int64),
        }
    )
    pieces = split_intervals(steps, grid, keys=["_p"])
    if pieces.empty:
        return None
    p_idx = pieces["_p"].to_numpy(dtype=np.int64)
    i_idx = pieces["_i"].to_numpy(dtype=np.int64)
    ps = ps_all[p_idx]
    a_rel = (np.maximum(s_ns[i_idx], ps) - ps).astype(np.float64) / 1e9
    seconds = pieces["seconds"].to_numpy(dtype=np.float64)
    return periods, p_idx, i_idx, a_rel, seconds


class ControlLoopHealthEvents(Base):
    """Engineering: Control Loop Health

    Continuously assess PID/control loop health from setpoint + actual pairs,
    independent of setpoint changes. Computes error integrals, detects
    oscillation in the error signal, and checks for valve saturation.

    Methods:
    - error_integrals: Per-window IAE, ISE, ITAE, bias.
    - detect_oscillation: Sustained oscillation in the error signal.
    - output_saturation: Valve pegged at limits.
    - loop_health_summary: Shift-level report card.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        setpoint_uuid: str,
        actual_uuid: str,
        *,
        output_uuid: str | None = None,
        event_uuid: str = "eng:control_loop_health",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the control loop health monitor.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            setpoint_uuid: UUID of the setpoint signal.
            actual_uuid: UUID of the actual (process value) signal.
            output_uuid: UUID of the controller output signal, used by
                :meth:`output_saturation`. ``None`` (default) when there is none.
            event_uuid: UUID stamped on emitted events.
            value_column: Numeric column holding the signal values.
            time_column: Timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.setpoint_uuid = setpoint_uuid
        self.actual_uuid = actual_uuid
        self.output_uuid = output_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self._sp = self._load_signal(setpoint_uuid)
        self._pv = self._load_signal(actual_uuid)
        self._out = self._load_signal(output_uuid) if output_uuid else pd.DataFrame()
        self._aligned = self._align()

    def _load_signal(self, uuid: str) -> pd.DataFrame:
        sig = (
            self.dataframe[self.dataframe["uuid"] == uuid]
            .copy()
            .sort_values(self.time_column)
        )
        sig[self.time_column] = pd.to_datetime(sig[self.time_column])
        return sig

    def _align(self) -> pd.DataFrame:
        """Align each PV sample with the setpoint in force at that time.

        PV drives the timeline and the setpoint is held until it changes
        (backward as-of match). Using SP as the left side dropped dense PV
        data on a change-encoded setpoint, and ``nearest`` let a sample pick
        up a setpoint that had not been set yet.
        """
        if self._sp.empty or self._pv.empty:
            return pd.DataFrame(columns=[self.time_column, "sp", "pv", "error"])

        sp = self._sp[[self.time_column, self.value_column]].rename(
            columns={self.value_column: "sp"}
        )
        pv = self._pv[[self.time_column, self.value_column]].rename(
            columns={self.value_column: "pv"}
        )
        merged = pd.merge_asof(
            pv.sort_values(self.time_column),
            sp.sort_values(self.time_column),
            on=self.time_column,
            direction="backward",
        )[[self.time_column, "sp", "pv"]]
        merged["error"] = merged["pv"] - merged["sp"]
        return merged.dropna(subset=["sp", "pv"])

    @keyword_only(1)
    def error_integrals(
        self, window: str = "1h", *, max_gap: str | None = None
    ) -> pd.DataFrame:
        """Per-window error integrals for control loop performance.

        The error ``e = PV - SP`` is piecewise constant between samples and
        integrated over time; a step that crosses a window boundary is split
        at the boundary, so every second between the first and the last
        sample is booked to the window it falls in (none is lost at window
        edges, and a window with no sample of its own still receives the
        time a held error covers in it).

        - ``IAE = integral |e| dt``, ``ISE = integral e**2 dt``.
        - ``ITAE = integral (t - window_start) * |e| dt``: time is measured
          from the start of the window, and the integral is exact for the
          piecewise-constant error.

        Args:
            window: Resample window (e.g. '1h', '8h').
            max_gap: Default ``None`` gives each error the time since the
                previous sample (the step ``(t[i-1], t[i]]`` carries
                ``e[i]``), with no gap cap. When set, each error is held
                until the next sample (zero-order hold, the step
                ``[t[i], t[i+1])`` carries ``e[i]``) and any step longer than
                ``max_gap`` contributes nothing, so a communication gap is
                not integrated as one huge error. In both modes no time is
                integrated before the first or after the last sample.

        Returns:
            DataFrame with columns: start, end, duration_seconds, iae, ise,
            itae, bias, sample_count. ``end`` is ``start`` + ``window``.
            ``bias`` is the mean error of the samples in the window (NaN for
            a window without samples) and ``sample_count`` their number.
            Windows that receive no integrated time are omitted.
        """
        cols = ["start", "iae", "ise", "itae", "bias", "sample_count"]
        if self._aligned.empty:
            return empty_event_df("summary", cols[1:])

        times = self._aligned[self.time_column].reset_index(drop=True)
        error = self._aligned["error"].to_numpy(dtype=np.float64)
        tn = to_ns(times)
        if max_gap is None:
            # Step (t[i-1], t[i]] carries e[i].
            s_ns, e_ns, err = tn[:-1], tn[1:], error[1:]
        else:
            s_ns, e_ns, err = tn[:-1], tn[1:], error[:-1]
            ok = (e_ns - s_ns) <= pd.Timedelta(max_gap).value
            s_ns, e_ns, err = s_ns[ok], e_ns[ok], err[ok]
        pieces = _split_steps(times, s_ns, e_ns, window)
        if pieces is None:
            return empty_event_df("summary", cols[1:])
        periods, p_idx, i_idx, a_rel, seconds = pieces
        e_abs = np.abs(err[i_idx])
        b_rel = a_rel + seconds
        n_per = len(periods)
        iae = np.bincount(p_idx, weights=e_abs * seconds, minlength=n_per)
        ise = np.bincount(p_idx, weights=err[i_idx] ** 2 * seconds, minlength=n_per)
        itae = np.bincount(
            p_idx, weights=e_abs * (b_rel**2 - a_rel**2) / 2.0, minlength=n_per
        )
        covered = np.bincount(p_idx, weights=seconds, minlength=n_per)
        labels = pd.DatetimeIndex(periods["period"])

        per_sample = (
            pd.Series(error, index=pd.DatetimeIndex(times))
            .resample(window)
            .agg(["mean", "count"])
            .reindex(labels)
        )
        keep = covered > 0
        events = pd.DataFrame(
            {
                "start": labels[keep],
                "iae": iae[keep],
                "ise": ise[keep],
                "itae": itae[keep],
                "bias": per_sample["mean"].to_numpy(dtype=np.float64)[keep],
                "sample_count": per_sample["count"]
                .fillna(0)
                .to_numpy(dtype=np.int64)[keep],
            },
            columns=cols,
        )
        if events.empty:
            return empty_event_df("summary", cols[1:])
        return finalize_summary_df(add_window_bounds(events, window))

    @keyword_only(1)
    def detect_oscillation(
        self,
        window: str = "5min",
        min_crossings: int = 4,
    ) -> pd.DataFrame:
        """Detect sustained oscillation in the error signal.

        Args:
            window: Analysis window size.
            min_crossings: Minimum zero-crossings to flag oscillation.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid (setpoint and actual uuids, comma-joined), is_delta,
            crossing_count, estimated_period_seconds, amplitude,
            damping_direction.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "crossing_count",
            "estimated_period_seconds",
            "amplitude",
            "damping_direction",
        ]
        if self._aligned.empty or len(self._aligned) < 4:
            return empty_event_df("interval", cols)

        df = self._aligned.set_index(self.time_column)
        td = pd.Timedelta(window)
        groups = df.resample(window)

        osc_windows: list[dict[str, Any]] = []
        for start, group in groups:
            if len(group) < 4:
                continue

            error = group["error"].values
            signs = np.sign(error)
            crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

            if crossings < min_crossings:
                continue

            # Estimate period from crossing intervals
            cross_idx = np.where(np.abs(np.diff(signs)) > 0)[0]
            times = group.index
            if len(cross_idx) >= 2:
                cross_times = times[cross_idx]
                intervals = np.diff(to_ns(cross_times)).astype(np.float64) / 1e9
                # Full period = 2 × half-period (crossing to crossing)
                est_period = float(np.mean(intervals)) * 2
            else:
                est_period = 0.0

            amplitude = float(np.max(np.abs(error)))

            # Damping: compare amplitude in first half vs second half
            mid = len(error) // 2
            amp_first = float(np.max(np.abs(error[:mid]))) if mid > 0 else 0.0
            amp_second = float(np.max(np.abs(error[mid:]))) if mid > 0 else 0.0
            if amp_second > amp_first * 1.1:
                damping = "growing"
            elif amp_second < amp_first * 0.9:
                damping = "decaying"
            else:
                damping = "sustained"

            osc_windows.append(
                {
                    "start": start,
                    "end": start + td,
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "crossing_count": crossings,
                    "estimated_period_seconds": est_period,
                    "amplitude": amplitude,
                    "damping_direction": damping,
                }
            )

        if not osc_windows:
            return empty_event_df("interval", cols)

        # Merge contiguous oscillating windows
        events: list[dict[str, Any]] = []
        current = osc_windows[0].copy()
        for i in range(1, len(osc_windows)):
            w = osc_windows[i]
            if w["start"] <= current["end"]:
                current["end"] = w["end"]
                current["crossing_count"] += w["crossing_count"]
                current["amplitude"] = max(current["amplitude"], w["amplitude"])
                current["estimated_period_seconds"] = (
                    current["estimated_period_seconds"] + w["estimated_period_seconds"]
                ) / 2
                current["damping_direction"] = w["damping_direction"]
            else:
                events.append(current)
                current = w.copy()
        events.append(current)

        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=f"{self.setpoint_uuid},{self.actual_uuid}",
        )

    @keyword_only(1)
    @renamed_kwargs(high_limit="upper_limit", low_limit="lower_limit")
    @twin_columns(
        {
            "pct_time_at_high": "time_at_high_pct",
            "pct_time_at_low": "time_at_low_pct",
            "pct_time_saturated": "time_saturated_pct",
        }
    )
    def output_saturation(
        self,
        upper_limit: float = 100.0,
        lower_limit: float = 0.0,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Detect when controller output is pegged at limits.

        Requires output_uuid in constructor.

        Args:
            upper_limit: Upper output limit, in the output signal's units. Samples
                within 1% of the ``upper_limit - lower_limit`` range of it count as
                saturated high. Default 100.0.
            lower_limit: Lower output limit, in the output signal's units. Samples
                within 1% of the range of it count as saturated low. Default 0.0.
            window: Resample window as a pandas offset string (e.g. '1h').

        Shares and run lengths are measured in time, not samples: each
        output sample is held until the next one (zero-order hold, the
        module's ``interval_end="next_change"`` convention), steps are split
        at the window boundaries, and the time after the last sample is not
        observed (the last sample opens no step). A single saturated sample
        therefore counts for the time until the next sample.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid (the output uuid), pct_time_at_high, pct_time_at_low,
            pct_time_saturated, longest_saturation_seconds. One row per
            window that holds observed time; ``end`` is ``start`` +
            ``window``. The percentages are shares of the observed (held)
            time in the window; ``longest_saturation_seconds`` is the
            longest stretch of consecutive saturated time inside the window
            (a run that crosses a window boundary is cut there).

            Canonical twin columns: ``time_at_high_pct`` (after ``pct_time_at_high``),
            ``time_at_low_pct`` (after ``pct_time_at_low``), ``time_saturated_pct``
            (after ``pct_time_saturated``); each holds the same values as the deprecated
            column right before it, which is removed in the next major release.
        """
        cols = [
            "start",
            "pct_time_at_high",
            "pct_time_at_low",
            "pct_time_saturated",
            "longest_saturation_seconds",
        ]
        if self._out.empty:
            return empty_event_df("interval", cols)

        times = self._out[self.time_column].reset_index(drop=True)
        values = self._out[self.value_column].to_numpy(dtype=np.float64)
        tol = (upper_limit - lower_limit) * 0.01  # 1% of range
        tn = to_ns(times)
        with np.errstate(invalid="ignore"):
            high = (values >= upper_limit - tol)[:-1]
            low = (values <= lower_limit + tol)[:-1]
        pieces = _split_steps(times, tn[:-1], tn[1:], window)
        if pieces is None:
            return empty_event_df("interval", cols)
        periods, p_idx, i_idx, _, seconds = pieces
        n_per = len(periods)
        covered = np.bincount(p_idx, weights=seconds, minlength=n_per)
        t_high = np.bincount(p_idx, weights=seconds * high[i_idx], minlength=n_per)
        t_low = np.bincount(p_idx, weights=seconds * low[i_idx], minlength=n_per)

        # Runs of consecutive saturated steps, as time spans, cut per window.
        longest = np.zeros(n_per, dtype=np.float64)
        runs = true_runs(high | low)
        if runs.n_runs:
            run_pieces = _split_steps(
                times, tn[:-1][runs.starts], tn[1:][runs.last], window
            )
            if run_pieces is not None:
                _, rp, _, _, rsec = run_pieces
                np.maximum.at(longest, rp, rsec)

        keep = covered > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_high = np.where(keep, t_high / covered * 100, np.nan)
            pct_low = np.where(keep, t_low / covered * 100, np.nan)
        events = pd.DataFrame(
            {
                "start": pd.DatetimeIndex(periods["period"])[keep],
                "pct_time_at_high": pct_high[keep],
                "pct_time_at_low": pct_low[keep],
                "pct_time_saturated": (pct_high + pct_low)[keep],
                "longest_saturation_seconds": longest[keep],
            },
            columns=cols,
        )
        if events.empty:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            add_window_bounds(events, window),
            uuid=self.event_uuid,
            source_uuid=self.output_uuid,
        )

    @keyword_only(1)
    @twin_columns({"pct_saturated": "saturated_pct"})
    def loop_health_summary(
        self,
        window: str = "8h",
        oscillation_window: str = "5min",
        min_crossings: int = 4,
    ) -> pd.DataFrame:
        """Shift-level report card combining all loop health metrics.

        Oscillation is detected with :meth:`detect_oscillation` on short
        analysis windows of ``oscillation_window`` (not on the report
        ``window``: a whole shift would almost always contain enough
        zero-crossings). ``oscillation_count`` is the number of merged
        oscillation episodes that start inside each report window.

        Args:
            window: Report window as a pandas offset string (e.g. '8h' for a
                shift). Used for the error integrals, the saturation share
                and to bucket the oscillation episodes.
            oscillation_window: Analysis window passed to
                :meth:`detect_oscillation` (default '5min').
            min_crossings: Minimum zero-crossings per ``oscillation_window``
                to flag oscillation, passed to :meth:`detect_oscillation`.

        Returns:
            DataFrame with columns: start, end, duration_seconds, iae, bias,
            oscillation_count, pct_saturated, health_grade. ``end`` is
            ``start`` + ``window``.

            Canonical twin columns: ``saturated_pct`` (after ``pct_saturated``); each
            holds the same values as the deprecated column right before it, which is
            removed in the next major release.
        """
        cols = [
            "start",
            "iae",
            "bias",
            "oscillation_count",
            "pct_saturated",
            "health_grade",
        ]
        integrals = self.error_integrals(window=window)
        if integrals.empty:
            return empty_event_df("summary", cols[1:])

        osc = self.detect_oscillation(
            window=oscillation_window, min_crossings=min_crossings
        )
        sat = self.output_saturation(window=window)

        events: list[dict[str, Any]] = []
        for _, row in integrals.iterrows():
            ws = row["start"]
            we = ws + pd.Timedelta(window)

            # Count oscillation events in this window
            if not osc.empty:
                osc_in = osc[(osc["start"] >= ws) & (osc["start"] < we)]
                osc_count = len(osc_in)
            else:
                osc_count = 0

            # Saturation in this window
            if not sat.empty:
                sat_row = sat[sat["start"] == ws]
                pct_sat = (
                    float(sat_row["pct_time_saturated"].iloc[0])
                    if not sat_row.empty
                    else 0.0
                )
            else:
                pct_sat = 0.0

            # Grade: count issues
            iae_median = (
                float(integrals["iae"].median()) if len(integrals) > 1 else row["iae"]
            )
            issues = 0
            if row["iae"] > iae_median * 1.5:
                issues += 1
            if osc_count > 0:
                issues += 1
            if pct_sat > 5.0:
                issues += 1

            grade = "good" if issues == 0 else ("fair" if issues == 1 else "poor")

            events.append(
                {
                    "start": ws,
                    "iae": row["iae"],
                    "bias": row["bias"],
                    "oscillation_count": osc_count,
                    "pct_saturated": pct_sat,
                    "health_grade": grade,
                }
            )

        return finalize_summary_df(
            add_window_bounds(pd.DataFrame(events, columns=cols), window)
        )
