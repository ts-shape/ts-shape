"""Shared building blocks of the mapping-free unit profilers.

The cycle profiler (:class:`~ts_shape.events.production.cycle_profiling.
CycleProfileEvents`) answers one question from the raw signals of a PLC:
where does the cycle time go? The unit profilers ask the same kind of
question about other *units of analysis* -- a stop, a changeover, a start-up,
a handshake, an order, a bad part, a setting change, a control loop -- and
run them through the same steps:

    units -> learned standard -> classify -> register (potential) ->
    causes / influencers -> evaluability -> one record per unit

This module holds what they share:

* :func:`robust_classes` -- ``pause`` / ``long`` / ``short`` / ``normal`` on
  the log duration, per group (the cycle profiler's rule);
* :func:`potential_register` -- per part: count, median, P10, P90, spread,
  and the potential (median - P10 per unit, and the total over all units);
* :func:`enrichment` -- which labels occur more often in the units than in a
  baseline (lift, Fisher exact test, Benjamini-Hochberg q-value);
* :func:`effect_size` -- before/after shift (Hodges-Lehmann), Cliff's delta
  and Mann-Whitney p-value;
* :func:`window_labels` -- the event labels inside each of many windows;
* :func:`silent_gaps` -- holes in which no signal logged anything (unknown
  time, never a stop);
* :func:`evaluation_frame` and the ``REASON_*`` vocabulary every profiler's
  ``evaluation_window()`` uses.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ts_shape.events.production.cycle_profiling import (
    MAD_SCALE,
    MIN_LOG_SIGMA,
    RESOLUTION_MAX_SHARE,
    CycleProfileEvents,
    _bh_qvalues,
)
from ts_shape.utils.base import Base

__all__ = [
    "EVALUATION_COLUMNS",
    "REASON_BELOW_RESOLUTION",
    "REASON_NO_ANCHOR",
    "REASON_NO_CANDIDATE",
    "REASON_NOT_APPLICABLE",
    "REASON_OK",
    "REASON_TOO_FEW",
    "REASON_UPSTREAM",
    "effect_size",
    "enrichment",
    "evaluation_frame",
    "potential_register",
    "robust_classes",
    "silent_gaps",
    "window_labels",
]

#: Reasons of ``evaluation_window()`` (the text after ``:`` is detail).
REASON_OK = "ok"
REASON_TOO_FEW = "too_few_units"
REASON_BELOW_RESOLUTION = "below_resolution"
REASON_NO_ANCHOR = "no_cycle_anchor"
REASON_UPSTREAM = "upstream_not_evaluable"
REASON_NO_CANDIDATE = "no_candidate_signal"
REASON_NOT_APPLICABLE = "not_applicable"

#: Columns every profiler's ``evaluation_window()`` starts with.
EVALUATION_COLUMNS = [
    "scope",
    "unit_count",
    "evaluable",
    "reason",
    "time_resolution_seconds",
]

#: Significance level of :func:`enrichment` and :func:`effect_size` users.
FDR = 0.05

_REGISTER_COLUMNS = [
    "unit_count",
    "median_seconds",
    "p10_seconds",
    "p90_seconds",
    "spread_seconds",
    "mean_seconds",
    "total_seconds",
    "potential_seconds",
    "total_potential_seconds",
]


def evaluation_frame(
    rows: Sequence[dict[str, Any]], extra: Sequence[str] = ()
) -> pd.DataFrame:
    """``evaluation_window()`` rows with the shared columns first."""
    columns = [*EVALUATION_COLUMNS, *extra]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(list(rows))
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[columns + [c for c in out.columns if c not in columns]]


def evaluation_reason(
    unit_count: int,
    min_units: int,
    *,
    resolution: float = 0.0,
    median_seconds: float = float("nan"),
    upstream: str | None = None,
) -> tuple[bool, str]:
    """``(evaluable, reason)`` by the shared rules.

    The upstream reason (the cycle profile's, when the units are cut from
    its cycles) wins; then too few units; then a time resolution that is at
    least :data:`RESOLUTION_MAX_SHARE` of the median unit.
    """
    if upstream is not None and upstream != REASON_OK:
        return False, f"{REASON_UPSTREAM}: {upstream}"
    if unit_count < min_units:
        return False, f"{REASON_TOO_FEW}: {unit_count} < {min_units}"
    if (
        resolution > 0
        and np.isfinite(median_seconds)
        and median_seconds > 0
        and resolution >= RESOLUTION_MAX_SHARE * median_seconds
    ):
        return False, (
            f"{REASON_BELOW_RESOLUTION}: the signals are seen every {resolution:g} s, "
            f"at least {RESOLUTION_MAX_SHARE:g} of the median unit ({median_seconds:g} s)"
        )
    return True, REASON_OK


def robust_classes(
    durations: np.ndarray,
    groups: np.ndarray,
    *,
    n_sigma: float = 3.0,
    pause_factor: float | None = 3.0,
    quantum: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Class (``pause/long/short/normal``) and robust z of every unit, per group.

    A unit at least ``pause_factor`` times its group's median is a
    ``pause`` (``None``: no pauses); the others are classified on the robust
    z of their log duration (median, MAD scaled, sigma at least
    :data:`MIN_LOG_SIGMA` and the ``quantum`` share of the median).

    Returns:
        ``(labels, z, limits)`` with ``limits[group]`` holding ``lower``,
        ``upper``, ``pause`` and ``median`` seconds.
    """
    d = np.asarray(durations, dtype=float)
    g = np.asarray(groups, dtype=object)
    labels = np.full(len(d), "normal", dtype=object)
    z = np.full(len(d), np.nan)
    limits: dict[str, dict[str, float]] = {}
    for group in pd.unique(g):
        idx = np.flatnonzero(g == group)
        values = d[idx]
        ok = np.isfinite(values) & (values > 0)
        if not ok.any():
            continue
        median_all = float(np.median(values[ok]))
        threshold = pause_factor * median_all if pause_factor is not None else np.inf
        pause = values >= threshold
        base = (
            np.log(values[ok & ~pause]) if (ok & ~pause).any() else np.log(values[ok])
        )
        m = float(np.median(base))
        s = max(
            MAD_SCALE * float(np.median(np.abs(base - m))),
            MIN_LOG_SIGMA,
            quantum / max(float(np.exp(m)), 1e-9),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = (np.log(values) - m) / s
        cls = np.where(
            pause,
            "pause",
            np.where(zz > n_sigma, "long", np.where(zz < -n_sigma, "short", "normal")),
        )
        labels[idx] = cls
        z[idx] = zz
        limits[str(group)] = {
            "lower": float(np.exp(m - n_sigma * s)),
            "upper": float(np.exp(m + n_sigma * s)),
            "pause": float(threshold),
            "median": float(np.exp(m)),
        }
    return labels, z, limits


def potential_register(
    table: pd.DataFrame,
    *,
    keys: Sequence[str],
    value: str = "seconds",
) -> pd.DataFrame:
    """Per ``keys`` group: count, median, P10, P90, spread and potential.

    The potential of a unit is the time it took beyond the best demonstrated
    (P10) of its group; ``potential_seconds`` is that per typical unit
    (median - P10), ``total_potential_seconds`` the sum over all units. The
    result is sorted by ``total_potential_seconds`` (largest first) and
    ranked (``potential_rank``).

    Args:
        table: One row per unit (and part), with ``keys`` and ``value``.
        keys: Grouping columns (e.g. ``["variant", "step"]``).
        value: Column with the seconds.
    """
    columns = [*keys, *_REGISTER_COLUMNS, "potential_rank"]
    data = table.dropna(subset=[value]) if len(table) else table
    if data.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key, part in data.groupby(list(keys), sort=True, dropna=False):
        v = part[value].to_numpy(dtype=float)
        p10, median, p90 = np.percentile(v, [10, 50, 90])
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row.update(
            {
                "unit_count": int(len(v)),
                "median_seconds": float(median),
                "p10_seconds": float(p10),
                "p90_seconds": float(p90),
                "spread_seconds": float(p90 - p10),
                "mean_seconds": float(v.mean()),
                "total_seconds": float(v.sum()),
                "potential_seconds": float(median - p10),
                "total_potential_seconds": float(np.clip(v - p10, 0, None).sum()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        "total_potential_seconds", ascending=False, kind="stable"
    )
    out["potential_rank"] = np.arange(1, len(out) + 1)
    return out[columns].reset_index(drop=True)


def window_labels(
    events: pd.DataFrame,
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
    *,
    labels: Iterable[str] | None = None,
) -> list[set[str]]:
    """The event labels inside each window ``[start, end)``.

    Args:
        events: Event frame with ``ns`` (int64) and ``event`` columns.
        starts_ns: Window starts.
        ends_ns: Window ends (exclusive).
        labels: Only these labels (default: all).
    """
    ev = events if labels is None else events[events["event"].isin(set(labels))]
    ev = ev.sort_values("ns", kind="stable")
    ns = ev["ns"].to_numpy(dtype=np.int64)
    names = ev["event"].to_numpy(dtype=object)
    lo = np.searchsorted(ns, np.asarray(starts_ns, dtype=np.int64), side="left")
    hi = np.searchsorted(ns, np.asarray(ends_ns, dtype=np.int64), side="left")
    return [set(names[a:b]) for a, b in zip(lo, hi, strict=True)]


def first_label(
    events: pd.DataFrame,
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
    candidates: Iterable[str],
) -> list[str | None]:
    """The earliest of ``candidates`` inside each window, or None."""
    wanted = set(candidates)
    ev = events[events["event"].isin(wanted)].sort_values("ns", kind="stable")
    ns = ev["ns"].to_numpy(dtype=np.int64)
    names = ev["event"].to_numpy(dtype=object)
    lo = np.searchsorted(ns, np.asarray(starts_ns, dtype=np.int64), side="left")
    hi = np.searchsorted(ns, np.asarray(ends_ns, dtype=np.int64), side="left")
    return [str(names[a]) if b > a else None for a, b in zip(lo, hi, strict=True)]


def enrichment(
    unit_sets: Sequence[set[str]],
    baseline_sets: Sequence[set[str]],
    *,
    min_units: int = 3,
    min_lift: float = 3.0,
) -> pd.DataFrame:
    """Labels that occur more often in the units than in the baseline.

    For every label present in at least ``min_units`` units: its presence
    ratio in the units and in the baseline windows, the lift (ratio of the
    two, with a +0.5 continuity correction on the baseline), a one-sided
    Fisher exact p-value and its Benjamini-Hochberg q-value.
    ``is_significant`` needs ``q < 0.05`` and ``lift >= min_lift``.
    """
    columns = [
        "label",
        "unit_count",
        "unit_presence_ratio",
        "baseline_presence_ratio",
        "lift_factor",
        "p_value",
        "q_value",
        "is_significant",
    ]
    n_units, n_base = len(unit_sets), len(baseline_sets)
    if n_units == 0:
        return pd.DataFrame(columns=columns)
    counts: dict[str, int] = {}
    for labels in unit_sets:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    base_counts: dict[str, int] = {}
    for labels in baseline_sets:
        for label in labels:
            base_counts[label] = base_counts.get(label, 0) + 1
    rows = []
    for label, a in counts.items():
        if a < min_units:
            continue
        b = base_counts.get(label, 0)
        unit_ratio = a / n_units
        base_ratio = b / n_base if n_base else 0.0
        lift = unit_ratio / ((b + 0.5) / (n_base + 1.0))
        table = [[a, n_units - a], [b, max(n_base - b, 0)]]
        p = float(sps.fisher_exact(table, alternative="greater")[1]) if n_base else 1.0
        rows.append(
            {
                "label": label,
                "unit_count": int(a),
                "unit_presence_ratio": unit_ratio,
                "baseline_presence_ratio": base_ratio,
                "lift_factor": lift,
                "p_value": p,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out["q_value"] = _bh_qvalues(out["p_value"].to_numpy(dtype=float))
    out["is_significant"] = (out["q_value"] < FDR) & (out["lift_factor"] >= min_lift)
    out = out.sort_values(
        ["is_significant", "lift_factor", "unit_count"],
        ascending=[False, False, False],
        kind="stable",
    )
    return out[columns].reset_index(drop=True)


def effect_size(before: Sequence[float], after: Sequence[float]) -> dict[str, float]:
    """Shift and effect of ``after`` vs ``before``.

    Returns:
        ``before_median``, ``after_median``, ``shift`` (Hodges-Lehmann:
        median of all pairwise after - before differences),
        ``cliffs_delta_ratio`` (2 AUC - 1, -1..1: positive = after larger)
        and the two-sided Mann-Whitney ``p_value``. NaN with fewer than two
        values on a side.
    """
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    nan = float("nan")
    if len(a) < 2 or len(b) < 2:
        return {
            "before_median": float(np.median(a)) if len(a) else nan,
            "after_median": float(np.median(b)) if len(b) else nan,
            "shift": nan,
            "cliffs_delta_ratio": nan,
            "p_value": nan,
        }
    diffs = (b[:, None] - a[None, :]).ravel()
    test = sps.mannwhitneyu(b, a, alternative="two-sided")
    auc = float(test.statistic) / (len(a) * len(b))
    return {
        "before_median": float(np.median(a)),
        "after_median": float(np.median(b)),
        "shift": float(np.median(diffs)),
        "cliffs_delta_ratio": 2.0 * auc - 1.0,
        "p_value": float(test.pvalue),
    }


def silent_gaps(
    times_ns: np.ndarray, *, min_gap_seconds: float
) -> list[tuple[int, int]]:
    """``(start_ns, end_ns)`` of gaps in which no signal logged anything."""
    t = np.unique(np.asarray(times_ns, dtype=np.int64))
    if len(t) < 2:
        return []
    gaps = np.diff(t)
    idx = np.flatnonzero(gaps > min_gap_seconds * 1e9).tolist()
    return [(int(t[i]), int(t[i + 1])) for i in idx]


def overlaps_any(
    starts_ns: np.ndarray, ends_ns: np.ndarray, holes: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Whether each ``[start, end)`` overlaps one of ``holes``."""
    s = np.asarray(starts_ns, dtype=np.int64)
    e = np.asarray(ends_ns, dtype=np.int64)
    out = np.zeros(len(s), dtype=bool)
    for h0, h1 in holes:
        out |= (s < h1) & (e > h0)
    return out


class _UnitProfiler(Base):
    """Common constructor and cycle-profile access of the unit profilers.

    A unit profiler either reuses a computed
    :class:`~ts_shape.events.production.cycle_profiling.CycleProfileEvents`
    (``cycle_profile=``) or builds one from its own frame with
    ``cycle_options``; it reads it only through its public accessors.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: Sequence[str] | None = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:unit_profile",
        cycle_options: dict[str, Any] | None = None,
    ) -> None:
        """Keep the frame and the (optional) cycle profile.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` builds one on first use.
            signal_uuids: Signals to use (default: all).
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the emitted records.
            cycle_options: Keyword arguments for the cycle profile built when
                ``cycle_profile`` is None.
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.event_uuid = event_uuid
        self.signal_uuids = None if signal_uuids is None else list(signal_uuids)
        self._cycle_profile = cycle_profile
        self._cycle_options = dict(cycle_options or {})
        self._cache: dict[str, Any] = {}

    def _profile(self) -> CycleProfileEvents:
        """The cycle profile (built once from this frame when not given)."""
        if self._cycle_profile is None:
            self._cycle_profile = CycleProfileEvents(
                self.dataframe,
                signal_uuids=self.signal_uuids,
                time_column=self.time_column,
                uuid_column=self.uuid_column,
                **self._cycle_options,
            )
        return self._cycle_profile

    def _events(self) -> pd.DataFrame:
        if "events" not in self._cache:
            self._cache["events"] = self._profile().events_frame()
        return self._cache["events"]

    def _units(self) -> pd.DataFrame:
        if "cycles" not in self._cache:
            self._cache["cycles"] = self._profile().units()
        return self._cache["cycles"]

    def _tz(self) -> Any:
        times = self.dataframe[self.time_column] if len(self.dataframe) else None
        if times is None or not hasattr(times, "dt"):
            return None
        return getattr(times.dt, "tz", None)

    def _times(self, ns: np.ndarray) -> pd.Series:
        """Integer nanoseconds as timestamps in the data's timezone."""
        out = pd.to_datetime(np.asarray(ns, dtype=np.int64), unit="ns", utc=True)
        tz = self._tz()
        out = out.tz_convert(tz) if tz is not None else out.tz_localize(None)
        return pd.Series(out)
