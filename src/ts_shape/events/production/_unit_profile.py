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

* :class:`SignalIndex` -- every signal's readings, sorted, found without
  scanning the whole frame (one pass per frame);
* :func:`assign_to_cycles`, :func:`restart_points`, :func:`cycles_since` --
  which cycle a reading falls in, where running restarts (a pause or a
  variant change) and how many cycles ago;
* :func:`robust_center_scale` -- median and scaled MAD (std as fallback);
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
  ``evaluation_window()`` uses;
* :class:`_ProfilerBase` / :class:`_UnitProfiler` -- the profiler base: a
  profiler declares ``PROFILE_KIND``, ``PROFILE_TABLES`` (and
  ``PROFILE_OPTIONS``, ``PROFILE_ORDER``, ``NEEDS_CYCLES``) in its class body
  and is then registered in :data:`PROFILER_CLASSES`, which
  :func:`ts_shape.unit_profiles.profile_units` runs.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, TypeVar

import numpy as np
import pandas as pd
from scipy import stats as sps

from ts_shape.events.production.cycle_profiling import (  # noqa: F401 (labels)
    FALL,
    INCREMENT,
    MAD_SCALE,
    RESOLUTION_MAX_SHARE,
    RISE,
    CycleProfileEvents,
    _bh_qvalues,
    _from_ns,
)
from ts_shape.utils.base import Base
from ts_shape.utils.frames import to_ns

__all__ = [
    "EVALUATION_COLUMNS",
    "PROFILER_CLASSES",
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
    "assign_to_cycles",
    "bh_qvalues",
    "cycles_since",
    "potential_register",
    "restart_points",
    "robust_center_scale",
    "silent_gaps",
    "window_labels",
]

#: Benjamini-Hochberg q-values (the cycle profiler's implementation).
bh_qvalues = _bh_qvalues

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
    # The label breaks ties: the labels come from sets, whose order changes
    # between runs (hash randomisation); the same input gives the same table.
    out = out.sort_values(
        ["is_significant", "lift_factor", "unit_count", "label"],
        ascending=[False, False, False, True],
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


# ----------------------------------------------------------------------
# Signals, cycles, restarts
# ----------------------------------------------------------------------


class SignalIndex:
    """Every signal's readings in time order, found without rescanning.

    One pass over the frame (factorize the uuids, sort by uuid and time)
    instead of one full-frame filter per signal. Rows with a missing
    timestamp (``NaT``) are left out.

    Args:
        frame: Long-format signals.
        uuid_column: Signal column.
        time_column: Timestamp column.
    """

    def __init__(self, frame: pd.DataFrame, uuid_column: str, time_column: str) -> None:
        self.frame = frame
        if frame.empty or uuid_column not in frame or time_column not in frame:
            self._order = np.zeros(0, dtype=np.int64)
            self._ns = np.zeros(0, dtype=np.int64)
            self._bounds: dict[str, tuple[int, int]] = {}
            return
        ns = np.asarray(to_ns(frame[time_column]), dtype=np.int64)
        codes, names = pd.factorize(frame[uuid_column].astype(str), sort=False)
        valid = np.flatnonzero((ns != np.iinfo(np.int64).min) & (codes >= 0))
        order = valid[np.lexsort((ns[valid], codes[valid]))]
        self._order = order
        self._ns = ns[order]
        sorted_codes = codes[order]
        cuts = np.flatnonzero(np.diff(sorted_codes)) + 1
        starts = np.r_[0, cuts] if len(order) else np.zeros(0, dtype=np.int64)
        ends = np.r_[cuts, len(order)] if len(order) else np.zeros(0, dtype=np.int64)
        self._bounds = {
            str(names[sorted_codes[a]]): (int(a), int(b))
            for a, b in zip(starts, ends, strict=True)
        }

    def __contains__(self, uuid: object) -> bool:
        return str(uuid) in self._bounds

    @property
    def uuids(self) -> list[str]:
        """The signals with at least one timed reading."""
        return list(self._bounds)

    @property
    def all_ns(self) -> np.ndarray:
        """Every timed reading (int64 ns), grouped by signal (not sorted)."""
        return self._ns

    def rows(self, uuid: str) -> np.ndarray:
        """Positional row numbers of ``uuid`` in the frame, in time order.

        Args:
            uuid: The signal.
        """
        a, b = self._bounds.get(str(uuid), (0, 0))
        return self._order[a:b]

    def ns(self, uuid: str) -> np.ndarray:
        """Timestamps (int64 ns) of ``uuid``, sorted.

        Args:
            uuid: The signal.
        """
        a, b = self._bounds.get(str(uuid), (0, 0))
        return self._ns[a:b]

    def column(self, uuid: str, column: str) -> pd.Series:
        """The ``column`` values of ``uuid`` in time order (original dtype).

        Args:
            uuid: The signal.
            column: The value column.
        """
        if column not in self.frame.columns:
            return pd.Series(dtype=float)
        return self.frame[column].iloc[self.rows(uuid)].reset_index(drop=True)

    def numeric(self, uuid: str, column: str) -> np.ndarray:
        """The ``column`` values of ``uuid`` as floats (NaN when not numeric).

        Args:
            uuid: The signal.
            column: The value column.
        """
        values = pd.to_numeric(self.column(uuid, column), errors="coerce")
        return np.asarray(values, dtype=float)


def assign_to_cycles(
    ns: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    closed: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """``(position, inside)``: the cycle every time falls in.

    Args:
        ns: Times (int64 ns), any order.
        starts: Cycle starts (sorted).
        ends: Cycle ends (same order).
        closed: ``"left"`` -- a cycle holds ``[start, end)`` (an event that
            happens in it); ``"right"`` -- ``(start, end]`` (a count reported
            at a boundary belongs to the cycle that just ended).

    Returns:
        ``position`` (index into the cycles, -1 before the first) and
        ``inside`` (the time lies in that cycle, not in a gap after it).
    """
    t = np.asarray(ns, dtype=np.int64)
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    if closed == "left":
        pos = np.searchsorted(s, t, side="right") - 1
    elif closed == "right":
        pos = np.searchsorted(s, t, side="left") - 1
    else:
        raise ValueError(f"closed must be 'left' or 'right', not {closed!r}")
    inside = pos >= 0
    if closed == "left":
        inside[inside] &= t[inside] < e[pos[inside]]
    else:
        inside[inside] &= t[inside] <= e[pos[inside]]
    return pos, inside


def restart_points(cycles: pd.DataFrame) -> pd.DataFrame:
    """Where running restarts: a pause cycle or the first cycle of a new variant.

    Args:
        cycles: The cycle profile's ``units()`` (time order, with
            ``cycle_class`` and ``variant``).

    Returns:
        ``position`` (index into ``cycles``) and ``trigger`` (``changeover``
        when the variant changes there, else ``pause``).
    """
    if cycles.empty:
        return pd.DataFrame({"position": pd.Series(dtype=np.int64), "trigger": []})
    classes = cycles["cycle_class"].to_numpy(dtype=object)
    variant = cycles["variant"].astype(str).to_numpy()
    change = np.r_[False, variant[1:] != variant[:-1]]
    idx = np.flatnonzero((classes == "pause") | change)
    return pd.DataFrame(
        {
            "position": idx.astype(np.int64),
            "trigger": np.where(change[idx], "changeover", "pause"),
        }
    )


def cycles_since(positions: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    """Per cycle: cycles since the last of ``positions`` at or before it.

    ``0`` at a position itself, a large number (``2**31``) before the first.
    """
    out = np.full(n, 2**31, dtype=np.int64)
    if n == 0:
        return out
    marks = np.zeros(n, dtype=bool)
    p = np.asarray(positions, dtype=np.int64)
    marks[p[(p >= 0) & (p < n)]] = True
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(marks, idx, -1))
    has = last >= 0
    out[has] = idx[has] - last[has]
    return out


def robust_center_scale(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """``(median, scale)``: the scaled MAD, the std when the MAD is 0.

    ``scale`` is NaN when neither is positive (a constant or empty input).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    median = float(np.median(v))
    scale = MAD_SCALE * float(np.median(np.abs(v - median)))
    if not scale > 0:
        scale = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
    return median, scale if scale > 0 else float("nan")


def cycle_signal_stats(
    signals: SignalIndex,
    cycles: pd.DataFrame,
    specs: dict[str, Any],
    *,
    stats: Sequence[str] = ("mean", "max"),
) -> pd.DataFrame:
    """Per cycle and continuous signal: ``<uuid>:<stat>`` of its readings.

    A reading belongs to the cycle ``[start, end)`` it falls in.

    Returns:
        One row per cycle (index aligned with ``cycles``), one column per
        continuous signal and stat; NaN for a cycle without a reading.
    """
    columns: dict[str, pd.Series] = {}
    if cycles.empty:
        return pd.DataFrame(index=cycles.index)
    starts = cycles["start_ns"].to_numpy(dtype=np.int64)
    ends = cycles["end_ns"].to_numpy(dtype=np.int64)
    for uuid, spec in specs.items():
        if spec.kind != "continuous" or uuid not in signals:
            continue
        values = signals.numeric(uuid, spec.value_column)
        pos, inside = assign_to_cycles(signals.ns(uuid), starts, ends)
        ok = inside & np.isfinite(values)
        if not ok.any():
            continue
        grouped = pd.Series(values[ok]).groupby(pos[ok])
        index = cycles.index[np.unique(pos[ok])]
        for stat in stats:
            columns[f"{uuid}:{stat}"] = pd.Series(
                grouped.agg(stat).to_numpy(), index=index
            )
    return pd.DataFrame(columns, index=cycles.index)


# ----------------------------------------------------------------------
# Profiler base
# ----------------------------------------------------------------------

#: Profiler kind (``"stops"``, ``"loops"`` ...) -> class; filled by the
#: classes that declare ``PROFILE_KIND`` (see :class:`_ProfilerBase`).
PROFILER_CLASSES: dict[str, type[_ProfilerBase]] = {}

_S = TypeVar("_S", bound="_ProfilerBase")
_T = TypeVar("_T")


def memo(fn: Callable[[_S], _T]) -> Callable[[_S], _T]:
    """Compute a private, argument-free profiler step once per instance."""
    key = f"memo:{fn.__name__}"

    @functools.wraps(fn)
    def wrapper(self: _S) -> _T:
        cache = self._cache
        if key not in cache:
            cache[key] = fn(self)
        value: _T = cache[key]
        return value

    return wrapper


class _ProfilerBase(Base):
    """What every mapping-free profiler shares: frame, cache, records, verdicts.

    A profiler declares in its own class body (not inherited):

    * ``PROFILE_KIND`` -- its name in :func:`~ts_shape.unit_profiles.
      profile_units` (``"stops"``); setting it registers the class in
      :data:`PROFILER_CLASSES`;
    * ``PROFILE_TABLES`` -- its public table methods (``evaluation_window``
      is always added);
    * ``PROFILE_OPTIONS`` -- constructor arguments ``profile_units`` passes
      through (``("plan",)``);
    * ``PROFILE_ORDER`` -- the run order (dependencies first);
    * ``NEEDS_CYCLES`` -- whether it takes ``cycle_profile=``.
    """

    PROFILE_KIND: ClassVar[str | None] = None
    PROFILE_TABLES: ClassVar[tuple[str, ...]] = ()
    PROFILE_OPTIONS: ClassVar[tuple[str, ...]] = ()
    PROFILE_ORDER: ClassVar[int] = 100
    NEEDS_CYCLES: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = vars(cls).get("PROFILE_KIND")
        if kind:
            PROFILER_CLASSES[str(kind)] = cls

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        signal_uuids: Sequence[str] | None = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:unit_profile",
    ) -> None:
        """Keep the frame.

        Args:
            dataframe: Long-format signals.
            signal_uuids: Signals to use (default: all).
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the emitted records.
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.event_uuid = event_uuid
        self.signal_uuids = None if signal_uuids is None else list(signal_uuids)
        self._cache: dict[str, Any] = {}

    @classmethod
    def profile_tables(cls) -> tuple[str, ...]:
        """The table methods :func:`~ts_shape.unit_profiles.profile_units` saves."""
        tables = tuple(cls.PROFILE_TABLES)
        return (
            tables if "evaluation_window" in tables else (*tables, "evaluation_window")
        )

    # -- frame access ----------------------------------------------------

    @memo
    def _signals(self) -> SignalIndex:
        """The frame's signals in time order (one pass)."""
        frame = self.dataframe
        if self.signal_uuids is not None and self.uuid_column in frame:
            frame = frame[frame[self.uuid_column].astype(str).isin(self.signal_uuids)]
        return SignalIndex(frame, self.uuid_column, self.time_column)

    def _tz(self) -> Any:
        times = self.dataframe[self.time_column] if len(self.dataframe) else None
        if times is None or not hasattr(times, "dt"):
            return None
        return getattr(times.dt, "tz", None)

    def _times(self, ns: np.ndarray | pd.Series) -> pd.Series:
        """Integer nanoseconds as timestamps in the data's timezone."""
        return _from_ns(np.asarray(ns, dtype=np.int64), self._tz())

    # -- outputs -----------------------------------------------------------

    def _empty(self, columns: Sequence[str]) -> pd.DataFrame:
        """An empty table with ``columns`` (``systime`` as a datetime)."""
        out = pd.DataFrame(columns=list(columns))
        if "systime" in out.columns:
            out["systime"] = self._times(np.zeros(0, dtype=np.int64))
        return out

    def _point_records(
        self,
        units: pd.DataFrame,
        *,
        columns: Sequence[str],
        source: Any,
        at: str = "end_ns",
        **extra: Any,
    ) -> pd.DataFrame:
        """One point event per unit: the shared record prefix plus ``extra``.

        Args:
            units: One row per unit with ``unit_uuid``, ``start_ns`` and
                ``end_ns``.
            columns: The output columns, in order.
            source: ``source_uuid`` (one value or one per unit).
            at: The column with the event time (default the unit end).
            **extra: More columns (one value per unit).
        """
        if units.empty:
            return self._empty(columns)
        start = units["start_ns"].to_numpy(dtype=np.int64)
        end = units["end_ns"].to_numpy(dtype=np.int64)
        data: dict[str, Any] = {
            "systime": self._times(units[at].to_numpy(dtype=np.int64)),
            "uuid": self.event_uuid,
            "source_uuid": source,
            "unit_uuid": units["unit_uuid"].to_numpy(),
            "start_ns": start,
            "end_ns": end,
            "duration_seconds": (end - start) / 1e9,
        }
        data.update(
            {
                k: (v.to_numpy() if isinstance(v, pd.Series) else v)
                for k, v in extra.items()
            }
        )
        return pd.DataFrame(data)[list(columns)]

    # -- verdicts ----------------------------------------------------------

    def _resolution(self) -> float:
        """Seconds within which the data cannot order events (0: unknown)."""
        return 0.0

    def _upstream(self) -> str | None:
        """The reason of an upstream analysis that blocks this one, or None."""
        return None

    def _evaluation_row(
        self,
        unit_count: int,
        min_units: int,
        *,
        median_seconds: float = float("nan"),
        missing: str | None = None,
        scope: str = "all",
        upstream: str | None = None,
        resolution: float | None = None,
        **extra: Any,
    ) -> pd.DataFrame:
        """A one-row ``evaluation_window()`` by the shared rules.

        The upstream reason wins, then a missing candidate signal
        (``no_candidate_signal: <missing>``), then
        :func:`evaluation_reason` (count and resolution).
        """
        resolution = self._resolution() if resolution is None else resolution
        upstream = self._upstream() if upstream is None else upstream
        if upstream is not None and upstream != REASON_OK:
            evaluable, reason = False, f"{REASON_UPSTREAM}: {upstream}"
        elif missing is not None:
            evaluable, reason = False, f"{REASON_NO_CANDIDATE}: {missing}"
        else:
            evaluable, reason = evaluation_reason(
                unit_count,
                min_units,
                resolution=resolution,
                median_seconds=median_seconds,
            )
        return evaluation_frame(
            [
                {
                    "scope": scope,
                    "unit_count": int(unit_count),
                    "evaluable": evaluable,
                    "reason": reason,
                    "time_resolution_seconds": resolution,
                    **extra,
                }
            ],
            list(extra),
        )

    def evaluation_window(self) -> pd.DataFrame:  # pragma: no cover - abstract
        """Whether the profile can be trusted (one row per scope)."""
        raise NotImplementedError

    def _summary_head(self, count_key: str | None = None) -> dict[str, Any]:
        """``evaluable`` (any scope), ``reason`` and the unit count."""
        window = self.evaluation_window()
        evaluable = (
            bool(window["evaluable"].astype(bool).any()) if len(window) else False
        )
        if evaluable:
            reason = REASON_OK
        else:
            reason = str(window["reason"].iloc[0]) if len(window) else REASON_TOO_FEW
        head: dict[str, Any] = {"evaluable": evaluable, "reason": reason}
        if count_key is not None:
            head[count_key] = int(window["unit_count"].iloc[0]) if len(window) else 0
        return head


class _UnitProfiler(_ProfilerBase):
    """A profiler of units cut from the cycle profile.

    It either reuses a computed
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
        super().__init__(
            dataframe,
            signal_uuids=signal_uuids,
            time_column=time_column,
            uuid_column=uuid_column,
            event_uuid=event_uuid,
        )
        self._cycle_profile = cycle_profile
        self._cycle_options = dict(cycle_options or {})

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

    @memo
    def _events(self) -> pd.DataFrame:
        return self._profile().events_frame()

    @memo
    def _units(self) -> pd.DataFrame:
        return self._profile().units()

    @memo
    def _cycles(self) -> pd.DataFrame:
        """The cycles with a 0..n-1 index (positions)."""
        return self._units().reset_index(drop=True)

    @memo
    def _restarts(self) -> pd.DataFrame:
        """:func:`restart_points` of the cycles."""
        return restart_points(self._cycles())

    def _resolution(self) -> float:
        return self._profile().resolution_seconds()

    def _upstream(self) -> str | None:
        return self._profile().evaluation_reason() if self._units().empty else None
