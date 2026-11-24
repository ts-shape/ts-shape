"""Mapping-free curve signatures: recurring fault shapes of analog curves.

:class:`CurveSignatureEvents` reads every analog signal of one PLC over each
cycle of its cycle profile and learns, per product variant:

1. the **golden curve**: per point of a fixed grid over the variant's best
   demonstrated cycle (P10), the median of the normal cycles and their
   P10..P90 band;
2. per cycle, its **deviation**: the residual against the golden curve in
   robust units (median absolute deviation per point), its mean as the
   ``deviation_score``, the **phase** of the cycle where it deviates most
   (the segment of the standard view, ``from → to``) and the direction
   (``high`` / ``low``);
3. **fault patterns**: the residuals of the anomalous cycles are clustered
   (hierarchical, average linkage); a cluster of at least
   :data:`PATTERN_MIN_CYCLES` cycles is a recurring fault shape, named after
   its signal, direction and phase, and linked to the cycles' classes and
   the time they lost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

from ts_shape.events.production._unit_profile import (
    MAD_SCALE,
    _UnitProfiler,
    evaluation_frame,
    evaluation_reason,
    memo,
)
from ts_shape.events.production.cycle_profiling import CycleProfileEvents

__all__ = ["CurveSignatureEvents"]

#: Grid points of a curve over the variant's P10 cycle.
CURVE_POINTS = 60
#: Points of a residual used for clustering.
PATTERN_POINTS = 16
#: A cluster with at least this many cycles is a pattern.
PATTERN_MIN_CYCLES = 3
#: A warm-up run of off-band cycles begins within this many cycles after a
#: pause or variant change.
RESTART_CYCLES = 3
#: Cut height of the clustering, in median nearest-neighbour distances.
PATTERN_CUT = 1.5
#: At most this many anomalous cycles per signal are clustered directly.
CLUSTER_MAX_CYCLES = 500
#: A cycle needs this many readings of a signal to have a curve.
CURVE_MIN_SAMPLES = 8
#: The score adds this percentile of a cycle's |residual| (divided by
#: :data:`PEAK_WEIGHT`) to its mean, so a short spike counts.
PEAK_PERCENTILE = 97.5
PEAK_WEIGHT = 4.0
#: Floor of the per-point scale, relative to the signal's overall spread.
SCALE_FLOOR = 0.05


def _curve(ns: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The held value at every grid time (NaN before the first reading)."""
    idx = np.searchsorted(ns, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def _shift_tolerant(
    matrix: np.ndarray, median: np.ndarray, shift: int = 1
) -> np.ndarray:
    """``matrix - median`` with the smallest |difference| within ``shift`` points.

    The median curve is moved by up to ``shift`` grid points either way; at
    the ends it is held (its first / last value), not wrapped around.
    """
    best = matrix - median
    n = len(median)
    for k in range(1, min(shift, n - 1) + 1):
        later = np.r_[np.full(k, median[0]), median[:-k]]
        earlier = np.r_[median[k:], np.full(k, median[-1])]
        for moved in (later, earlier):
            diff = matrix - moved
            better = np.abs(diff) < np.abs(best)
            best = np.where(better, diff, best)
    return np.asarray(best, dtype=float)


def _cluster_shapes(small: np.ndarray) -> np.ndarray:
    """Cluster labels of downsampled residual shapes (average linkage).

    At most :data:`CLUSTER_MAX_CYCLES` shapes are clustered (an evenly
    spaced sample); the others join the cluster of their nearest clustered
    shape when it is within the cut distance, else stay alone. Memory stays
    bounded on long recordings with many anomalous cycles.
    """
    n = len(small)
    if n > CLUSTER_MAX_CYCLES:
        core = np.linspace(0, n - 1, CLUSTER_MAX_CYCLES).round().astype(int)
    else:
        core = np.arange(n)
    ref = small[core]
    dist = np.sqrt(((ref[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = float(np.median(dist.min(axis=1))) if len(ref) > 1 else 0.0
    np.fill_diagonal(dist, 0.0)
    if not np.isfinite(nn) or nn <= 0:
        core_labels = np.ones(len(ref), dtype=int)
    else:
        condensed = dist[np.triu_indices(len(ref), k=1)]
        tree = hierarchy.linkage(condensed, method="average")
        core_labels = hierarchy.fcluster(tree, PATTERN_CUT * nn, criterion="distance")
    labels = np.zeros(n, dtype=int)
    labels[core] = core_labels
    rest = np.setdiff1d(np.arange(n), core)
    if len(rest):
        next_label = int(core_labels.max()) + 1
        for i in rest:
            d = np.sqrt(((ref - small[i]) ** 2).sum(axis=1))
            j = int(np.argmin(d))
            if nn > 0 and d[j] <= PATTERN_CUT * nn:
                labels[i] = core_labels[j]
            else:
                labels[i] = next_label
                next_label += 1
    return labels


class CurveSignatureEvents(_UnitProfiler):
    """Golden curves, per-cycle deviations and recurring fault shapes.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=400, seed=5, curve_faults={"spike_in_hold": 0.05})
        >>> curves = ts_shape.CurveSignatureEvents(frame)
        >>> "plc:pressure" in set(curves.curve_golden()["signal"])
        True
    """

    PROFILE_KIND = "curves"
    PROFILE_TABLES = ("curve_golden", "curve_cycle_scores", "curve_patterns")
    PROFILE_ORDER = 70

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        n_sigma: float = 3.0,
        min_cycles: int = 30,
        max_signals: int = 20,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:curve_signature",
        **cycle_options: Any,
    ) -> None:
        """Initialize the curve-signature profiler.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            n_sigma: A cycle whose deviation score is this many robust sigmas
                above the median score (and at least twice it) is anomalous.
            min_cycles: A variant and signal with fewer normal cycles is not
                evaluable.
            max_signals: At most this many analog signals (the most sampled).
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the cycle score records.
            **cycle_options: Options of the cycle profile built when
                ``cycle_profile`` is None.
        """
        super().__init__(
            dataframe,
            cycle_profile=cycle_profile,
            signal_uuids=signal_uuids,
            time_column=time_column,
            uuid_column=uuid_column,
            event_uuid=event_uuid,
            cycle_options=cycle_options,
        )
        self.n_sigma = float(n_sigma)
        self.min_cycles = int(min_cycles)
        self.max_signals = int(max_signals)

    # ------------------------------------------------------------------
    # Curves
    # ------------------------------------------------------------------

    def _signals_used(self) -> list[str]:
        specs = self._profile().signal_specs()
        analog = [u for u, s in specs.items() if s.kind == "continuous"]
        if not analog:
            return []
        signals = self._signals()
        counts = sorted(
            ((len(signals.ns(u)), u) for u in sorted(analog) if u in signals),
            key=lambda k: -k[0],
        )
        return [u for _, u in counts[: self.max_signals]]

    @memo
    def _discrete_times(self) -> dict[str, np.ndarray]:
        """Sorted times of every discrete event label (one pass)."""
        events = self._events()
        # Only discrete milestones: an analog crossing moves with the curve's
        # own shape (a late rise would be warped away).
        events = events[events["signal_kind"] != "continuous"]
        return {
            str(label): np.sort(part["ns"].to_numpy(np.int64))
            for label, part in events.groupby("event", sort=False)
        }

    def _milestones(self, variant: str) -> list[tuple[float, np.ndarray]]:
        """``(template offset, sorted ns of the event)`` of the standard view."""
        view = self._profile().standard_view()
        if view.empty or "variant" not in view.columns:
            return []
        part = view[view["variant"].astype(str) == variant]
        times = self._discrete_times()
        out = []
        for label, offset in zip(
            part["event"].astype(str), part["median_offset_seconds"], strict=True
        ):
            ns = times.get(label)
            if ns is not None and len(ns):
                out.append((float(offset), ns))
        return sorted(out, key=lambda k: k[0])

    @staticmethod
    def _warp(
        offsets: np.ndarray,
        knots: list[tuple[float, np.ndarray]],
        start_ns: int,
        end_ns: int,
    ) -> np.ndarray:
        """Template offsets -> this cycle's offsets, piecewise linear between
        the milestones the cycle has (a later segment that runs long does not
        shift the earlier ones)."""
        xs, ys = [0.0], [0.0]
        for template, ns in knots:
            i = np.searchsorted(ns, start_ns, side="left")
            if i < len(ns) and ns[i] < end_ns:
                actual = (ns[i] - start_ns) / 1e9
                if template > xs[-1] and actual > ys[-1]:
                    xs.append(template)
                    ys.append(actual)
        if len(xs) < 2:
            return offsets
        # Beyond the last milestone the cycle runs at the template's pace.
        tail = offsets > xs[-1]
        out = np.interp(offsets, xs, ys)
        out[tail] = ys[-1] + (offsets[tail] - xs[-1])
        return np.asarray(out, dtype=float)

    def _phases(self, variant: str, span: float) -> list[tuple[float, str]]:
        """``(start offset, phase name)`` of the standard view's segments."""
        view = self._profile().standard_view()
        if view.empty or "variant" not in view.columns:
            return [(0.0, "cycle")]
        part = view[view["variant"].astype(str) == variant].sort_values(
            "median_offset_seconds"
        )
        if part.empty:
            return [(0.0, "cycle")]
        names = part["event"].astype(str).tolist()
        offsets = part["median_offset_seconds"].to_numpy(dtype=float)
        phases = [(0.0, f"start → {names[0]}")]
        for i in range(len(names)):
            nxt = names[i + 1] if i + 1 < len(names) else "end"
            phases.append((float(offsets[i]), f"{names[i]} → {nxt}"))
        return [p for p in phases if p[0] <= span] or [(0.0, "cycle")]

    @memo
    def _compute(self) -> dict[str, Any]:
        cycles = self._cycles()
        signals = self._signals()
        specs = self._profile().signal_specs()
        golden_rows: list[pd.DataFrame] = []
        score_rows: list[dict[str, Any]] = []
        residuals: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        evaluation: list[dict[str, Any]] = []
        use = cycles[cycles["cycle_class"] != "pause"]
        knots_of: dict[str, list[tuple[float, np.ndarray]]] = {}
        for uuid in self._signals_used():
            ns = signals.ns(uuid)
            values = signals.numeric(uuid, specs[uuid].value_column)
            spread = float(np.nanpercentile(values, 90) - np.nanpercentile(values, 10))
            for variant, part in use.groupby(use["variant"].astype(str), sort=True):
                normal = part[part["cycle_class"] == "normal"]
                if normal.empty:
                    continue
                span = float(normal["duration_seconds"].quantile(0.1))
                offsets = np.linspace(0.0, span, CURVE_POINTS, endpoint=False)
                if variant not in knots_of:
                    knots_of[variant] = self._milestones(variant)
                knots = knots_of[variant]
                curves, kept, counts = [], [], []
                for c in part.itertuples(index=False):
                    if c.cycle_class == "short":
                        continue  # cut-short cycles are classified short already
                    lo, hi = np.searchsorted(ns, [c.start_ns, c.end_ns])
                    counts.append(hi - lo)
                    warped = self._warp(offsets, knots, c.start_ns, c.end_ns)
                    grid = c.start_ns + (warped * 1e9).astype(np.int64)
                    curves.append(_curve(ns, values, grid))
                    kept.append(c)
                enough = (
                    int(np.median(counts)) >= CURVE_MIN_SAMPLES if counts else False
                )
                n_normal = int(sum(1 for c in kept if c.cycle_class == "normal"))
                evaluable, reason = evaluation_reason(n_normal, self.min_cycles)
                if not enough:
                    evaluable, reason = False, (
                        f"below_resolution: fewer than {CURVE_MIN_SAMPLES} readings "
                        "per cycle"
                    )
                evaluation.append(
                    {
                        "scope": f"{variant} | {uuid}",
                        "unit_count": n_normal,
                        "evaluable": evaluable,
                        "reason": reason,
                        "time_resolution_seconds": self._resolution(),
                        "variant": variant,
                        "signal": uuid,
                    }
                )
                if not evaluable:
                    continue
                matrix = np.vstack(curves)
                is_normal = np.array([c.cycle_class == "normal" for c in kept])
                ref = matrix[is_normal]
                median = np.nanmedian(ref, axis=0)
                p10 = np.nanpercentile(ref, 10, axis=0)
                p90 = np.nanpercentile(ref, 90, axis=0)
                mad = MAD_SCALE * np.nanmedian(np.abs(ref - median), axis=0)
                scale = np.maximum(mad, SCALE_FLOOR * max(spread, 1e-9))
                # A reading is known to one sample: an edge may sit one grid
                # point early or late without being a different shape.
                resid = _shift_tolerant(matrix, median) / scale
                # Mean deviation plus the peak (a short spike matters too).
                absr = np.abs(resid)
                score = (
                    np.nanmean(absr, axis=1)
                    + np.nanpercentile(absr, PEAK_PERCENTILE, axis=1) / PEAK_WEIGHT
                )
                s_med = float(np.nanmedian(score[is_normal]))
                s_mad = MAD_SCALE * float(
                    np.nanmedian(np.abs(score[is_normal] - s_med))
                )
                limit = max(s_med + self.n_sigma * max(s_mad, 1e-9), 2.0 * s_med)
                anomaly = np.isfinite(score) & (score > limit)
                phases = self._phases(variant, span)
                starts = np.array([p[0] for p in phases])
                phase_of = np.searchsorted(starts, offsets, side="right") - 1
                golden_rows.append(
                    pd.DataFrame(
                        {
                            "variant": variant,
                            "signal": uuid,
                            "offset_seconds": offsets,
                            "phase": [phases[i][1] for i in phase_of],
                            "p10": p10,
                            "median": median,
                            "p90": p90,
                        }
                    )
                )
                for k, c in enumerate(kept):
                    per_phase = [
                        (
                            np.nanmean(np.abs(resid[k, phase_of == j]))
                            if (phase_of == j).any()
                            else 0.0
                        )
                        for j in range(len(phases))
                    ]
                    finite = np.isfinite(per_phase)
                    j = (
                        int(np.nanargmax(np.where(finite, per_phase, -np.inf)))
                        if finite.any()
                        else 0
                    )
                    sign = (
                        float(np.nanmean(resid[k, phase_of == j]))
                        if finite.any()
                        else 0.0
                    )
                    sign = sign if np.isfinite(sign) else 0.0
                    score_rows.append(
                        {
                            "cycle_uuid": str(c.cycle_uuid),
                            "end_ns": int(c.end_ns),
                            "variant": variant,
                            "signal": uuid,
                            "cycle_class": c.cycle_class,
                            "duration_seconds": float(c.duration_seconds),
                            "deviation_score": float(score[k]),
                            "is_anomaly": bool(anomaly[k]),
                            "deviating_phase": phases[j][1],
                            "direction": "high" if sign >= 0 else "low",
                        }
                    )
                residuals[(variant, uuid)] = (
                    np.array([str(c.cycle_uuid) for c in kept]),
                    np.nan_to_num(resid),
                )
        scores = pd.DataFrame(score_rows)
        result = {
            "golden": (
                pd.concat(golden_rows, ignore_index=True)
                if golden_rows
                else pd.DataFrame()
            ),
            "scores": scores,
            "residuals": residuals,
            "evaluation": evaluation,
        }
        result["scores"] = self._cluster(result)
        return result

    def _cluster(self, result: dict[str, Any]) -> pd.DataFrame:
        """Pattern id of every anomalous cycle (``-1``: none)."""
        scores = result["scores"]
        if scores.empty:
            return scores.assign(
                pattern_id=pd.Series(dtype=np.int64),
                after_restart=pd.Series(dtype=bool),
            )
        scores = scores.copy()
        scores["pattern_id"] = -1
        # A curve off the band right after a pause or variant change is a
        # warm-up (see StartupProfileEvents), not an in-run fault pattern.
        scores["after_restart"] = self._warmup(scores)
        next_id = 0
        # One clustering per signal over all variants: the residuals are
        # scaled per variant, so the same fault looks alike in every variant.
        by_signal: dict[str, list[tuple[Any, np.ndarray]]] = {}
        for (_, uuid), (ids, resid) in result["residuals"].items():
            by_signal.setdefault(uuid, []).append((ids, resid))
        for uuid, parts in by_signal.items():
            mask = scores["signal"] == uuid
            anomalous = scores[mask & scores["is_anomaly"] & ~scores["after_restart"]]
            if len(anomalous) < PATTERN_MIN_CYCLES:
                continue
            row_of = {}
            stacked: list[np.ndarray] = []
            for ids, resid in parts:
                for i, u in enumerate(ids):
                    row_of[u] = len(stacked) + i
                stacked.extend(list(resid))
            matrix = np.asarray(stacked, dtype=float)
            take = [row_of[u] for u in anomalous["cycle_uuid"]]
            vectors = matrix[take]
            # Downsample to a few points, so a shape matters, not its noise.
            blocks = np.array_split(np.arange(vectors.shape[1]), PATTERN_POINTS)
            small = np.column_stack([vectors[:, b].mean(axis=1) for b in blocks])
            labels = _cluster_shapes(small)
            for label in np.unique(labels):
                members = anomalous.index[labels == label]
                if len(members) >= PATTERN_MIN_CYCLES:
                    scores.loc[members, "pattern_id"] = next_id
                    next_id += 1
        return scores

    def _warmup(self, scores: pd.DataFrame) -> pd.Series:
        """Anomalous cycles of a warm-up: a run of off-band cycles of one
        signal that begins within ``RESTART_CYCLES`` after a pause or a
        variant change (the signal coming back to its running state)."""
        units = self._cycles()
        out = pd.Series(False, index=scores.index)
        if units.empty or scores.empty:
            return out
        pos = pd.Series(np.arange(len(units)), index=units["cycle_uuid"].astype(str))
        pause = (units["cycle_class"] == "pause").to_numpy()
        starts = self._restarts()["position"].to_numpy(dtype=np.int64)
        where = scores["cycle_uuid"].astype(str).map(pos)
        for _, part in scores[scores["is_anomaly"].astype(bool)].groupby("signal"):
            at = dict(zip(where[part.index].astype(int), part.index, strict=True))
            for k in starts:
                j = next(
                    (j for j in range(int(k), int(k) + RESTART_CYCLES + 1) if j in at),
                    None,
                )
                while j is not None and j in at:
                    out[at[j]] = True
                    j += 1
                    while j < len(pause) and pause[j]:
                        j += 1  # a pause inside the warm-up does not end it
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curve_golden(self) -> pd.DataFrame:
        """The golden curve of every variant and analog signal.

        Returns:
            One row per variant, signal and grid point: ``variant``,
            ``signal``, ``offset_seconds`` (from the cycle start), ``phase``
            (segment of the standard view), ``p10``, ``median``, ``p90``.
        """
        golden = self._compute()["golden"]
        columns = [
            "variant",
            "signal",
            "offset_seconds",
            "phase",
            "p10",
            "median",
            "p90",
        ]
        return golden[columns] if len(golden) else self._empty(columns)

    def curve_cycle_scores(self) -> pd.DataFrame:
        """How far every cycle's curve leaves the golden curve.

        Returns:
            Point events at the cycle end: ``systime``, ``uuid``,
            ``source_uuid`` (the signal), ``cycle_uuid``, ``variant``,
            ``cycle_class``, ``duration_seconds``, ``deviation_score`` (mean
            |residual| in robust units plus a quarter of its 97.5th
            percentile), ``is_anomaly``,
            ``deviating_phase``, ``direction``, ``pattern_id`` (``-1``
            when the cycle belongs to no recurring pattern) and
            ``after_restart`` (part of a warm-up: a run of off-band cycles
            beginning right after a pause or variant change; left out of the
            patterns).
        """
        scores = self._compute()["scores"]
        columns = [
            "systime",
            "uuid",
            "source_uuid",
            "cycle_uuid",
            "variant",
            "cycle_class",
            "duration_seconds",
            "deviation_score",
            "is_anomaly",
            "deviating_phase",
            "direction",
            "pattern_id",
            "after_restart",
        ]
        if scores.empty:
            return self._empty(columns)
        out = scores.assign(
            systime=self._times(scores["end_ns"].to_numpy()).to_numpy(),
            uuid=self.event_uuid,
            source_uuid=scores["signal"],
        )
        return out[columns].reset_index(drop=True)

    def curve_patterns(self) -> pd.DataFrame:
        """The recurring fault shapes, ranked by the time they cost.

        Returns:
            One row per pattern: ``pattern_id``, ``pattern`` (``"<signal>:
            <direction> in <phase>"``), ``variant`` (the variants it was
            seen in; patterns are clustered per signal over all variants),
            ``signal``,
            ``cycle_count``, ``long_share_ratio`` (share of its cycles that
            are long), ``lost_seconds`` (cycle time beyond the variant's
            median, summed), ``median_deviation_score`` and
            ``potential_rank``. Warm-up cycles (``after_restart``) are not
            clustered.
        """
        scores = self._compute()["scores"]
        columns = [
            "pattern_id",
            "pattern",
            "variant",
            "signal",
            "cycle_count",
            "long_share_ratio",
            "lost_seconds",
            "median_deviation_score",
            "potential_rank",
        ]
        if scores.empty or not (scores["pattern_id"] >= 0).any():
            return pd.DataFrame(columns=columns)
        limits = self._profile().variant_limits()
        rows = []
        for pid, part in scores[scores["pattern_id"] >= 0].groupby("pattern_id"):
            variants = sorted(set(part["variant"].astype(str)))
            variant = ", ".join(variants) if len(variants) <= 3 else "several"
            median = (
                part["variant"]
                .astype(str)
                .map(lambda v: limits.get(v, {}).get("median", np.nan))
            )
            phase = part["deviating_phase"].mode().iloc[0]
            direction = part["direction"].mode().iloc[0]
            rows.append(
                {
                    "pattern_id": int(pid),
                    "pattern": f"{part['signal'].iloc[0]}: {direction} in {phase}",
                    "variant": variant,
                    "signal": str(part["signal"].iloc[0]),
                    "cycle_count": int(len(part)),
                    "long_share_ratio": float((part["cycle_class"] == "long").mean()),
                    "lost_seconds": float(
                        (part["duration_seconds"] - median).clip(lower=0).sum()
                    ),
                    "median_deviation_score": float(part["deviation_score"].median()),
                }
            )
        out = pd.DataFrame(rows).sort_values(
            ["lost_seconds", "cycle_count"], ascending=[False, False], kind="stable"
        )
        out["potential_rank"] = np.arange(1, len(out) + 1)
        return out[columns].reset_index(drop=True)

    def evaluation_window(self) -> pd.DataFrame:
        """Whether each variant and signal can be trusted, and why not.

        Returns:
            One row per variant and analog signal (``scope`` =
            ``"<variant> | <signal>"``): ``unit_count`` (normal cycles),
            ``evaluable``, ``reason`` (``below_resolution`` with fewer than
            :data:`CURVE_MIN_SAMPLES` readings per cycle), ``variant`` and
            ``signal``; one ``all`` row without analog signals or cycles.
        """
        rows = self._compute()["evaluation"]
        if not rows:
            return self._evaluation_row(
                0, self.min_cycles, missing="analog signal", variant=None, signal=None
            )
        return evaluation_frame(rows, ["variant", "signal"])

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable`` (any variant and signal), ``reason``,
            ``anomalous_cycle_count`` (in-run cycles off the band),
            ``warmup_cycle_count`` (off the band right after a restart),
            ``pattern_count`` and ``top_pattern``.
        """
        scores = self.curve_cycle_scores()
        patterns = self.curve_patterns()
        return {
            **self._summary_head(),
            "anomalous_cycle_count": _count(scores, restart=False),
            "warmup_cycle_count": _count(scores, restart=True),
            "pattern_count": int(len(patterns)),
            "top_pattern": None if patterns.empty else str(patterns["pattern"].iloc[0]),
        }


def _count(scores: pd.DataFrame, *, restart: bool) -> int:
    """Distinct anomalous cycles, in-run or right after a restart."""
    if scores.empty:
        return 0
    sel = scores["is_anomaly"].astype(bool) & (
        scores["after_restart"].astype(bool) == restart
    )
    return int(scores.loc[sel, "cycle_uuid"].nunique())
