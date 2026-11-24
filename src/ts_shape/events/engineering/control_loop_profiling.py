"""Mapping-free control-loop profiler: find the loops, rank their health.

:class:`ControlLoopProfileEvents` needs no machine cycle and no names. It
reads the numeric signals of one controller and:

* **finds setpoints**: signals that hold their value for minutes between
  steps (judged on their own readings, so polled and on-change logging
  both work);
* **pairs** each setpoint with the continuous signal that follows its steps
  (moves in the step's direction by a good part of it) and tracks it the
  rest of the time; the pairing is one-to-one, best score first;
* **finds the output** (optional): the continuous signal that moves at once
  (within ``KICK_SAMPLES`` grid samples) and consistently with the setpoint
  steps -- the controller's reaction; a signal that reacts at once is never
  taken for the process value, which reacts only after its dead time;
* **measures** every loop with
  :class:`~ts_shape.events.engineering.setpoint_events.SetpointChangeEvents`
  (settling, rise, overshoot, oscillation per step) and
  :class:`~ts_shape.events.engineering.control_loop_health.ControlLoopHealthEvents`
  (IAE, output saturation);
* **diagnoses** ``oscillating`` (overshoot and repeated crossings, or a
  settling loop that overshoots by 30 % and more), ``sluggish`` (most steps
  do not settle in time) or ``ok``, and ranks the
  loops by a health score (mean percentile rank of the badness metrics).

A loop with fewer than ``min_steps`` steps gets steady-state metrics only
(``too_few_units``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents
from ts_shape.events.production._unit_profile import (
    MAD_SCALE,
    REASON_BELOW_RESOLUTION,
    REASON_OK,
    REASON_TOO_FEW,
    _ProfilerBase,
    evaluation_frame,
    memo,
)

__all__ = ["ControlLoopProfileEvents"]

#: A setpoint holds its value at least this long between changes (median);
#: a continuous signal changes sooner.
SP_MIN_HOLD_SECONDS = 60.0
#: Most grid points of the common time grid.
GRID_MAX_POINTS = 50_000
#: A signal sets the grid period only with at least this many readings.
GRID_MIN_READINGS = 50
#: A followed step moves the process value by at least this share of it.
FOLLOW_MIN_SHARE = 0.2
#: Pairing thresholds: step-follow ratio and tracking ratio.
PAIR_MIN_FOLLOW = 0.6
PAIR_MIN_TRACKING = 0.5
#: Tracking band: |PV - SP| within this share of the setpoint's range.
TRACKING_BAND = 0.25
#: Output: consistent reaction share and minimum move (x the output noise).
OUTPUT_MIN_CONSISTENCY = 0.8
OUTPUT_MIN_MOVE = 3.0
#: Grid samples from a step in which the output reacts at once ...
KICK_SAMPLES = 2
#: ... by at least this share of the step.
KICK_MIN_RATIO = 0.05
#: Diagnosis thresholds.
OSCILLATION_MIN_OVERSHOOT_PCT = 25.0
OSCILLATION_MIN_COUNT = 2.0
SLUGGISH_MAX_SETTLED_SHARE = 0.5
#: A loop that settles but overshoots this much is too aggressive.
UNDERDAMPED_MIN_OVERSHOOT_PCT = 30.0
#: The grid period must be below this share of the response window.
RESOLUTION_SHARE = 0.1


class ControlLoopProfileEvents(_ProfilerBase):
    """Find the control loops among the signals and rank their health.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_control_loops(seed=1)
        >>> loops = ts_shape.ControlLoopProfileEvents(frame).loop_register()
        >>> dict(zip(loops["setpoint_uuid"], loops["diagnosis"]))["plc:tic102_sp"]
        'oscillating'
    """

    PROFILE_KIND = "loops"
    PROFILE_TABLES = ("loop_pairs", "loop_steps", "loop_register")
    PROFILE_ORDER = 90
    NEEDS_CYCLES = False

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        signal_uuids: list[str] | None = None,
        min_steps: int = 5,
        settle_pct: float = 0.1,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        event_uuid: str = "eng:control_loop_profile",
    ) -> None:
        """Initialize the control-loop profiler.

        Args:
            dataframe: Long-format signals of one controller or PLC.
            signal_uuids: Signals to use (default: all numeric ones).
            min_steps: Setpoint steps a loop needs for step metrics.
            settle_pct: Settling band as a share of each step.
            time_column: Timestamp column.
            uuid_column: Signal column.
            value_column: Numeric value column.
            event_uuid: ``uuid`` of the step records.
        """
        super().__init__(
            dataframe,
            signal_uuids=signal_uuids,
            time_column=time_column,
            uuid_column=uuid_column,
            event_uuid=event_uuid,
        )
        self.value_column = value_column
        self.min_steps = int(min_steps)
        self.settle_pct = float(settle_pct)

    # ------------------------------------------------------------------
    # Common grid
    # ------------------------------------------------------------------

    @memo
    def _grid(self) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
        """Grid times (ns), every numeric signal on it (held), grid period (s)."""
        empty: tuple[np.ndarray, dict[str, np.ndarray], float] = (
            np.array([], dtype=np.int64),
            {},
            0.0,
        )
        if self.value_column not in self.dataframe.columns:
            return empty
        signals = self._signals()
        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        steps = []
        for uuid in sorted(signals.uuids):
            v = signals.numeric(uuid, self.value_column)
            ok = np.isfinite(v)
            t, v = signals.ns(uuid)[ok], v[ok]
            if len(t) < 3:
                continue
            series[uuid] = (t, v)
            d = np.diff(t)
            d = d[d > 0]
            if len(d) >= GRID_MIN_READINGS:
                steps.append(float(np.median(d)) / 1e9)
        if not series:
            return empty
        t_min = min(int(t[0]) for t, _ in series.values())
        t_max = max(int(t[-1]) for t, _ in series.values())
        # The grid follows the most often read signal: with on-change logging
        # the setpoints are read rarely, the process values and outputs often.
        period = float(np.min(steps)) if steps else 1.0
        period = max(period, (t_max - t_min) / 1e9 / GRID_MAX_POINTS, 0.1)
        grid = np.arange(t_min, t_max + 1, int(period * 1e9), dtype=np.int64)
        held = {}
        for uuid, (t, v) in series.items():
            pos = np.searchsorted(t, grid, side="right") - 1
            held[uuid] = np.where(pos >= 0, v[np.clip(pos, 0, None)], np.nan)
        return grid, held, period

    def _resolution(self) -> float:
        return self._grid()[2]

    @memo
    def _roles(self) -> tuple[list[str], list[str]]:
        """Setpoint candidates and continuous signals.

        Judged on each signal's own readings, not on the common grid: a
        setpoint holds its value for minutes between changes, a process value
        changes within seconds -- whether it is polled (every reading a new
        noisy value) or logged on change with a deadband (every reading a
        change).
        """
        _, held, _ = self._grid()
        signals = self._signals()
        setpoints, continuous = [], []
        for uuid in held:
            v = signals.numeric(uuid, self.value_column)
            ok = np.isfinite(v)
            t, v = signals.ns(uuid)[ok], v[ok]
            if len(v) < 3:
                continue
            change = np.flatnonzero(v[1:] != v[:-1]) + 1
            if len(change) < 2:
                continue
            holds = np.diff(np.r_[t[0], t[change]]) / 1e9
            hold = float(np.median(holds))
            if hold >= SP_MIN_HOLD_SECONDS and len(np.unique(v)) >= 2:
                setpoints.append(uuid)
            elif hold < SP_MIN_HOLD_SECONDS and len(change) >= 10:
                continuous.append(uuid)
        return setpoints, continuous

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    @staticmethod
    def _step_index(sp: np.ndarray) -> np.ndarray:
        """Grid positions where the setpoint steps (from one value to another;
        its first reading after a gap is not a step)."""
        prev = np.r_[np.nan, sp[:-1]]
        return np.flatnonzero(np.isfinite(sp) & np.isfinite(prev) & (sp != prev))

    @staticmethod
    def _late_median(x: np.ndarray, lo: int, hi: int) -> float:
        if hi <= lo:
            return float("nan")
        a = lo + int(0.8 * (hi - lo))
        part = x[a:hi]
        part = part[np.isfinite(part)]
        return float(np.median(part)) if len(part) else float("nan")

    def _follow(self, sp: np.ndarray, pv: np.ndarray) -> tuple[float, float, int]:
        """Step-follow ratio, tracking ratio and step count."""
        idx = self._step_index(sp)
        if len(idx) == 0:
            return 0.0, 0.0, 0
        bounds = np.r_[0, idx, len(sp)]
        follow = []
        for j, k in enumerate(idx):
            prev, nxt = bounds[j], bounds[j + 2]
            delta = sp[k] - sp[k - 1]
            before = self._late_median(pv, prev, k)
            after = self._late_median(pv, k, nxt)
            if not (np.isfinite(before) and np.isfinite(after)) or delta == 0:
                continue
            move = (after - before) / delta
            follow.append(move >= FOLLOW_MIN_SHARE)
        ok = np.isfinite(sp) & np.isfinite(pv)
        span = float(np.nanpercentile(sp, 95) - np.nanpercentile(sp, 5))
        span = span if span > 0 else float(np.nanmax(np.abs(np.diff(sp[ok]))) or 1.0)
        tracking = float(np.mean(np.abs(pv[ok] - sp[ok]) <= TRACKING_BAND * span))
        return (float(np.mean(follow)) if follow else 0.0), tracking, len(idx)

    def _kick(self, sp: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        """Reaction at the step itself: consistency and median size.

        The controller output reacts within the scan of a setpoint step; the
        process value only after its dead time and lag. Returns the share of
        steps whose immediate move has the step's sign (signed, -1 .. 1) and
        the median immediate move per unit of step, in units of the step,
        or ``(0, 0)`` when the moves stay within the signal's noise.
        """
        idx = self._step_index(sp)
        idx = idx[(idx >= 1) & (idx + KICK_SAMPLES <= len(x))]
        if len(idx) < 2:
            return 0.0, 0.0
        noise = float(np.nanmedian(np.abs(np.diff(x)))) * MAD_SCALE or 1e-9
        delta = sp[idx] - sp[idx - 1]
        post = np.array([np.nanmean(x[k : k + KICK_SAMPLES]) for k in idx])
        move = post - x[idx - 1]
        ok = np.isfinite(move) & (delta != 0)
        if ok.sum() < 2 or float(np.median(np.abs(move[ok]))) < OUTPUT_MIN_MOVE * noise:
            return 0.0, 0.0
        signs = np.sign(move[ok]) * np.sign(delta[ok])
        return float(np.mean(signs)), float(np.median(np.abs(move[ok] / delta[ok])))

    def _reacts_at_once(self, sp: np.ndarray, x: np.ndarray) -> tuple[bool, float]:
        consistency, size = self._kick(sp, x)
        return (
            abs(consistency) >= OUTPUT_MIN_CONSISTENCY and size >= KICK_MIN_RATIO,
            abs(consistency),
        )

    @memo
    def _pairs(self) -> pd.DataFrame:
        _, held, _ = self._grid()
        setpoints, continuous = self._roles()
        scores = []
        for sp in setpoints:
            for pv in continuous:
                if self._reacts_at_once(held[sp], held[pv])[0]:
                    continue  # an output, not a process value
                follow, tracking, steps = self._follow(held[sp], held[pv])
                if follow >= PAIR_MIN_FOLLOW and tracking >= PAIR_MIN_TRACKING:
                    scores.append((follow * tracking, sp, pv, follow, tracking, steps))
        scores.sort(key=lambda r: -r[0])
        used: set[str] = set()
        rows = []
        for score, sp, pv, follow, tracking, steps in scores:
            if sp in used or pv in used:
                continue
            used.update((sp, pv))
            rows.append(
                {
                    "setpoint_uuid": sp,
                    "actual_uuid": pv,
                    "step_count": steps,
                    "step_follow_ratio": follow,
                    "tracking_ratio": tracking,
                    "pair_score": score,
                }
            )
        # Any signal not used yet may be an output: one logged with a deadband
        # changes slowly between steps yet reacts at once to each of them.
        free = [c for c in held if c not in used]
        for row in rows:
            op, consistency = None, 0.0
            for uuid in free:
                reacts, score = self._reacts_at_once(
                    held[str(row["setpoint_uuid"])], held[uuid]
                )
                if reacts and score > consistency:
                    op, consistency = uuid, score
            row["output_uuid"] = op
            row["output_consistency_ratio"] = consistency
            if op is not None:
                free.remove(op)
        out = pd.DataFrame(
            rows,
            columns=[
                "setpoint_uuid",
                "actual_uuid",
                "output_uuid",
                "step_count",
                "step_follow_ratio",
                "tracking_ratio",
                "pair_score",
                "output_consistency_ratio",
            ],
        )
        return out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _lookahead(self, sp_uuid: str) -> float:
        grid, held, period = self._grid()
        idx = self._step_index(held[sp_uuid])
        gaps = np.diff(np.r_[idx, len(grid)]) * period
        median_gap = float(np.median(gaps)) if len(gaps) else 600.0
        return float(np.clip(0.8 * median_gap, 60.0, 1800.0))

    @memo
    def _steps(self) -> pd.DataFrame:
        frames = []
        for row in self._pairs().itertuples(index=False):
            if row.step_count < self.min_steps:
                continue
            look = self._lookahead(row.setpoint_uuid)
            q = SetpointChangeEvents(
                self.dataframe,
                row.setpoint_uuid,
                event_uuid=self.event_uuid,
                value_column=self.value_column,
                time_column=self.time_column,
            ).control_quality_metrics(
                row.actual_uuid,
                settle_pct=self.settle_pct,
                lookahead=f"{int(look)}s",
            )
            if q.empty:
                continue
            q = q.copy()
            q.insert(0, "loop", row.setpoint_uuid)
            frames.append(q)
        out = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=[
                    "loop",
                    "start",
                    "end",
                    "duration_seconds",
                    "uuid",
                    "source_uuid",
                ]
            )
        )
        return out

    @memo
    def _register(self) -> pd.DataFrame:
        pairs = self._pairs()
        steps = self._steps()
        grid, held, period = self._grid()
        hours = max((grid[-1] - grid[0]) / 3.6e12, 1e-9) if len(grid) else 1.0
        rows = []
        for row in pairs.itertuples(index=False):
            sp, pv = held[row.setpoint_uuid], held[row.actual_uuid]
            ok = np.isfinite(sp) & np.isfinite(pv)
            span = float(np.nanpercentile(sp, 95) - np.nanpercentile(sp, 5)) or 1.0
            # IAE per hour from the health detector, in setpoint spans.
            health = ControlLoopHealthEvents(
                self.dataframe,
                row.setpoint_uuid,
                row.actual_uuid,
                output_uuid=row.output_uuid,
                value_column=self.value_column,
                time_column=self.time_column,
            )
            iae = float(health.error_integrals(window="1h")["iae"].sum())
            saturated = np.nan
            if row.output_uuid is not None:
                sat = health.output_saturation(window="1h")
                cols = [c for c in ("pct_time_at_high", "pct_time_at_low") if c in sat]
                if len(sat) and cols:
                    saturated = float(sat[cols].sum(axis=1).mean()) / 100.0
            part = steps[steps["loop"] == row.setpoint_uuid] if len(steps) else steps
            has_steps = len(part) > 0
            settled = (
                float(part["settled_within_window"].astype(float).mean())
                if has_steps
                else np.nan
            )
            overshoot = float(part["overshoot_pct"].median()) if has_steps else np.nan
            oscillations = (
                float(part["oscillation_count"].median()) if has_steps else np.nan
            )
            settle = float(part["t_settle_seconds"].median()) if has_steps else np.nan
            if not has_steps:
                diagnosis = "unknown"
            elif (
                overshoot >= OSCILLATION_MIN_OVERSHOOT_PCT
                and oscillations >= OSCILLATION_MIN_COUNT
            ):
                diagnosis = "oscillating"
            elif settled < SLUGGISH_MAX_SETTLED_SHARE:
                diagnosis = "sluggish"
            elif overshoot >= UNDERDAMPED_MIN_OVERSHOOT_PCT:
                diagnosis = "oscillating"  # settles, but far too aggressive
            else:
                diagnosis = "ok"
            rows.append(
                {
                    "loop": row.setpoint_uuid,
                    "setpoint_uuid": row.setpoint_uuid,
                    "actual_uuid": row.actual_uuid,
                    "output_uuid": row.output_uuid,
                    "step_count": int(row.step_count),
                    "settled_share_ratio": settled,
                    "median_settle_seconds": settle,
                    "median_overshoot_pct": overshoot,
                    "median_oscillation_count": oscillations,
                    "iae_per_hour_spans": iae / hours / span,
                    "tracking_ratio": float(row.tracking_ratio),
                    "saturated_share_ratio": saturated,
                    "mean_abs_error": float(np.mean(np.abs(pv[ok] - sp[ok]))),
                    "diagnosis": diagnosis,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            out = pd.DataFrame(
                columns=[
                    "loop",
                    "setpoint_uuid",
                    "actual_uuid",
                    "output_uuid",
                    "step_count",
                    "settled_share_ratio",
                    "median_settle_seconds",
                    "median_overshoot_pct",
                    "median_oscillation_count",
                    "iae_per_hour_spans",
                    "tracking_ratio",
                    "saturated_share_ratio",
                    "mean_abs_error",
                    "diagnosis",
                    "health_score",
                    "health_rank",
                ]
            )
            return out
        bad = pd.DataFrame(
            {
                "unsettled": 1.0 - out["settled_share_ratio"],
                "overshoot": out["median_overshoot_pct"],
                "oscillation": out["median_oscillation_count"],
                "iae": out["iae_per_hour_spans"],
                "saturated": out["saturated_share_ratio"],
            }
        )
        ranks = bad.rank(pct=True)
        out["health_score"] = 1.0 - ranks.mean(axis=1, skipna=True).fillna(0.5)
        out["health_rank"] = (
            out["health_score"].rank(method="min", ascending=True).astype(int)
        )
        out = out.sort_values("health_rank", kind="stable").reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def loop_pairs(self) -> pd.DataFrame:
        """The control loops found among the signals.

        Returns:
            One row per loop: ``setpoint_uuid``, ``actual_uuid``,
            ``output_uuid`` (None when no output reacts), ``step_count``,
            ``step_follow_ratio``, ``tracking_ratio``, ``pair_score`` and
            ``output_consistency_ratio``.
        """
        return self._pairs().copy()

    def loop_steps(self) -> pd.DataFrame:
        """Response metrics of every setpoint step of every loop.

        Returns:
            One row per step (loops with at least ``min_steps`` steps):
            ``loop`` plus the columns of
            :meth:`SetpointChangeEvents.control_quality_metrics
            <ts_shape.events.engineering.setpoint_events.SetpointChangeEvents.control_quality_metrics>`.
        """
        return self._steps().copy()

    def loop_register(self) -> pd.DataFrame:
        """Health of every loop, worst first.

        Returns:
            One row per loop: ``loop``, ``setpoint_uuid``, ``actual_uuid``,
            ``output_uuid``, ``step_count``, ``settled_share_ratio``,
            ``median_settle_seconds``, ``median_overshoot_pct``,
            ``median_oscillation_count``, ``iae_per_hour_spans`` (IAE per
            hour in setpoint spans), ``tracking_ratio``,
            ``saturated_share_ratio``, ``mean_abs_error``, ``diagnosis``
            (``oscillating`` / ``sluggish`` / ``ok`` / ``unknown``),
            ``health_score`` (0 worst .. 1 best) and ``health_rank``
            (1 = worst).
        """
        return self._register().copy()

    def evaluation_window(self) -> pd.DataFrame:
        """Whether each loop's metrics can be trusted, and why not.

        Returns:
            One row per loop (or one ``all`` row without any): ``scope``,
            ``unit_count`` (setpoint steps), ``evaluable``, ``reason``,
            ``time_resolution_seconds`` (the common grid period) and
            ``lookahead_seconds`` (the response window).
        """
        _, _, period = self._grid()
        pairs = self._pairs()
        if pairs.empty:
            return self._evaluation_row(
                0,
                self.min_steps,
                missing="setpoint with a follower",
                lookahead_seconds=np.nan,
            )
        rows = []
        for row in pairs.itertuples(index=False):
            look = self._lookahead(row.setpoint_uuid)
            if row.step_count < self.min_steps:
                reason = REASON_TOO_FEW
            elif period > RESOLUTION_SHARE * look:
                reason = REASON_BELOW_RESOLUTION
            else:
                reason = REASON_OK
            rows.append(
                {
                    "scope": row.setpoint_uuid,
                    "unit_count": int(row.step_count),
                    "evaluable": reason == REASON_OK,
                    "reason": reason,
                    "time_resolution_seconds": period,
                    "lookahead_seconds": look,
                }
            )
        return evaluation_frame(rows, ["lookahead_seconds"])

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable`` (at least one loop is), ``reason``, ``loop_count``,
            ``diagnoses`` (count per diagnosis) and ``worst_loop``.
        """
        register = self._register()
        return {
            **self._summary_head(),
            "loop_count": int(len(register)),
            "diagnoses": {
                str(k): int(v) for k, v in register["diagnosis"].value_counts().items()
            },
            "worst_loop": str(register["loop"].iloc[0]) if len(register) else None,
        }
