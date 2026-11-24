"""Mapping-free cycle time profiling of one PLC.

:class:`CycleProfileEvents` takes the raw long-format signals of one machine
controller and answers, without any configuration of signal roles:

1. *What must be optimized first to shorten the cycle time?* --
   :meth:`~CycleProfileEvents.contribution_register` ranks the segments of the
   learned standard cycle by their potential (median - best demonstrated).
2. *What is unstable within a cycle, and how is the variation distributed?*
   -- the same register reports each segment's spread, skew, bimodality and
   share of the cycle time variance.

The pipeline (computed once, on first use, and cached):

1. every signal becomes discrete events, by the kind
   :func:`ts_shape.signals.infer.infer_spec` infers for it: edges of a
   boolean (``uuid↑``, ``uuid↓``), entry into a state value (``uuid=v``), a
   counter increment (``uuid+1``), or a hysteresis crossing of an analog
   level (``uuid>``, ``uuid<``);
2. the cycle anchor is the event that repeats in step with the most other
   signals (a heartbeat repeats regularly but alone, so it never wins); the
   cycles are cut at it with :class:`~ts_shape.features.cycles.CycleExtractor`;
3. a product variant is a rarely changing text or state signal that explains
   the cycle time (eta squared of the log duration of at least 0.2);
4. per variant, every cycle is classified robustly on its log duration:
   ``pause``, ``long``, ``short`` or ``normal``;
5. the *standard view* of a variant is the events that occur exactly once in
   almost every cycle, ordered by their median phase; consecutive events of
   it split each cycle into segments that add up to the cycle time.

Signal names are used for labels only. Spec IDs in the method docstrings
(``US-1`` ...) refer to the cycle time profiling requirements, see
``docs/modules/production/cycle-profiling.md``.
"""

from __future__ import annotations

import logging
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.signals.infer import infer_spec
from ts_shape.utils.base import Base
from ts_shape.utils.frames import to_ns

logger = logging.getLogger(__name__)

#: Event label suffixes.
RISE, FALL, INCREMENT, HIGH, LOW = "↑", "↓", "+1", ">", "<"
#: A variant signal changes in at most this share of the cycles.
VARIANT_MAX_CHANGE_RATIO = 0.2
#: Smallest eta squared of log(duration) for a variant signal.
VARIANT_MIN_ETA_SQUARED = 0.2
#: Event counts within this fraction of each other "repeat in step".
SUPPORT_TOLERANCE = 0.10
#: A partner of an anchor candidate has this many events per candidate event
#: (re-tries allowed) ...
SUPPORT_MIN_RATIO, SUPPORT_MAX_RATIO = 0.8, 1.5
#: ... and is phase-locked to it (mean resultant length of its phases).
SUPPORT_MIN_LOCKING = 0.7
#: An anchor candidate covers at least this share of the best candidate's
#: coverage (span of its regular intervals over the recording). A quarter:
#: a machine down half the time (stops, changeovers) still has its cycle
#: compete against a slow event that spans the whole recording.
MIN_RELATIVE_COVERAGE = 0.25
#: Among variant candidates explaining at least this share of the best
#: candidate's adjusted eta squared, the one with the fewest values wins.
VARIANT_ETA_TOLERANCE = 0.9
#: A view whose time resolution is at least this share of the median cycle
#: cannot resolve its cycles (``evaluable`` is False).
RESOLUTION_MAX_SHARE = 0.25
#: Share of timestamp gaps that must be multiples of one period for the
#: timestamps to count as a polling grid.
GRID_SHARE = 0.9
#: Scale of the median absolute deviation to a normal sigma.
MAD_SCALE = 1.4826
#: Smallest robust sigma of log(duration) (1 %): perfectly regular cycles
#: would otherwise flag any jitter.
MIN_LOG_SIGMA = 0.01
#: Sarle's bimodality coefficient above this suggests two modes.
BIMODALITY_THRESHOLD = 5.0 / 9.0
#: Segments whose durations vary less than this (timestamp resolution) get
#: no skewness or bimodality.
MIN_SPREAD_SECONDS = 1e-3
#: An event that occurs more than this many times per cycle on average is a
#: clock (heartbeat, sampling toggle): its count per cycle only measures the
#: cycle duration and is no influencer feature.
MAX_EVENTS_PER_CYCLE = 2.0
#: Bootstrap resamples for the confidence intervals of the register (fixed
#: seed: the same data give the same intervals).
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 20261118
#: False discovery rate under which an influencer is significant.
INFLUENCER_FDR = 0.05
#: Largest number of learned levels of an analog signal (analog_levels="auto").
MAX_ANALOG_LEVELS = 4
#: Decimals of the interval CV that count when ranking anchor candidates.
CV_DECIMALS = 2
#: Hysteresis of an analog level, as fractions of its P10..P90 range.
HYSTERESIS_HIGH, HYSTERESIS_LOW = 0.6, 0.4

_EVENT_COLUMNS = ["systime", "uuid", "source_uuid", "event", "signal_kind"]
_ANCHOR_COLUMNS = [
    "rank",
    "event",
    "source_uuid",
    "signal_kind",
    "event_count",
    "support_count",
    "is_counter",
    "median_interval_seconds",
    "interval_cv_factor",
    "coverage_ratio",
    "is_anchor",
]
_WINDOW_COLUMNS = [
    "start",
    "end",
    "duration_seconds",
    "variant",
    "cycle_count",
    "pause_count",
    "longest_coherent_cycle_count",
    "evaluable",
    "reason",
    "anchor_event",
    "anchor_uuid",
    "variant_uuid",
    "time_resolution_seconds",
]
_VIEW_COLUMNS = [
    "variant",
    "position",
    "milestone",
    "event",
    "source_uuid",
    "presence_ratio",
    "median_phase_ratio",
    "median_offset_seconds",
    "cycle_count",
]
_SEGMENT_COLUMNS = [
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
    "cycle_uuid",
    "variant",
    "cycle_class",
    "position",
    "segment",
    "from_event",
    "to_event",
]
_REGISTER_COLUMNS = [
    "variant",
    "position",
    "segment",
    "from_event",
    "to_event",
    "cycle_count",
    "deviation_cycle_count",
    "median_seconds",
    "p10_seconds",
    "p90_seconds",
    "spread_seconds",
    "mad_seconds",
    "mean_seconds",
    "std_seconds",
    "skewness",
    "bimodality_coefficient",
    "is_bimodal",
    "median_share_ratio",
    "potential_seconds",
    "potential_ci_low_seconds",
    "potential_ci_high_seconds",
    "potential_rank",
    "variance_contribution_factor",
    "variance_contribution_ci_low_factor",
    "variance_contribution_ci_high_factor",
]
_STATISTICS_COLUMNS = [
    "variant",
    "cycle_count",
    "normal_count",
    "long_count",
    "short_count",
    "pause_count",
    "median_seconds",
    "p10_seconds",
    "p90_seconds",
    "mad_seconds",
    "mean_seconds",
    "std_seconds",
    "min_seconds",
    "max_seconds",
    "target_seconds",
    "potential_seconds",
    "lower_limit_seconds",
    "upper_limit_seconds",
    "pause_threshold_seconds",
]
_INFLUENCER_COLUMNS = [
    "comparison",
    "rank",
    "feature",
    "source_uuid",
    "feature_kind",
    "importance_ratio",
    "auc_ratio",
    "direction",
    "group_median",
    "normal_median",
    "group_cycle_count",
    "normal_cycle_count",
    "p_value",
    "q_value",
    "is_significant",
]
_RECORD_COLUMNS = [
    "systime",
    "uuid",
    "source_uuid",
    "cycle_uuid",
    "start_ns",
    "end_ns",
    "duration_seconds",
    "cycle_class",
    "variant",
    "median_seconds",
    "deviation_seconds",
    "sequence_deviation",
    "dominant_segment",
    "dominant_segment_seconds",
    "dominant_excess_seconds",
    "anchor_event",
]

#: Variant label when no variant signal explains the cycle time.
ALL = "all"


def _seconds(value: str | float | pd.Timedelta | None) -> float | None:
    """An offset string, Timedelta or number of seconds as float seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return float(value)
    return float(pd.Timedelta(value).total_seconds())


def _value_label(value: Any) -> str:
    """A state value as text: integral numbers without a decimal point."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return str(value)


def _robust_cv(intervals: np.ndarray) -> float:
    """Median absolute deviation over median (NaN with no intervals)."""
    if len(intervals) == 0:
        return float("nan")
    median = float(np.median(intervals))
    if median <= 0:
        return float("nan")
    return float(np.median(np.abs(intervals - median)) / median)


def _bimodality(values: np.ndarray) -> tuple[float, float]:
    """Sample skewness and Sarle's bimodality coefficient (NaN when undefined)."""
    n = len(values)
    if n < 4 or float(np.std(values)) < MIN_SPREAD_SECONDS:
        return float("nan"), float("nan")
    skew = float(sps.skew(values, bias=False))
    kurt = float(sps.kurtosis(values, fisher=True, bias=False))
    coefficient = (skew**2 + 1.0) / (kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return skew, float(coefficient)


def _analog_levels(values: np.ndarray, levels: int | str) -> np.ndarray:
    """Sorted centres of the operating levels of an analog signal (1-D k-means).

    ``levels="auto"`` tries 2 to :data:`MAX_ANALOG_LEVELS` levels and keeps a
    further level only while it cuts the within-level sum of squares by more
    than half. Deterministic: quantile start, fixed iterations.
    """
    x = np.sort(values[np.isfinite(values)])
    if len(x) < 4 or x[-1] == x[0]:
        return np.array([])
    if len(x) > 20000:
        x = x[np.linspace(0, len(x) - 1, 20000).astype(int)]

    def fit(k: int) -> tuple[np.ndarray, float]:
        centers = np.quantile(x, (np.arange(k) + 0.5) / k)
        for _ in range(50):
            edges = 0.5 * (centers[1:] + centers[:-1])
            idx = np.searchsorted(edges, x)
            new = np.array(
                [
                    x[idx == j].mean() if (idx == j).any() else centers[j]
                    for j in range(k)
                ]
            )
            if np.allclose(new, centers):
                break
            centers = new
        edges = 0.5 * (centers[1:] + centers[:-1])
        idx = np.searchsorted(edges, x)
        return np.sort(centers), float(((x - centers[idx]) ** 2).sum())

    if levels != "auto":
        return fit(int(levels))[0]
    best, sse = fit(2)
    for k in range(3, MAX_ANALOG_LEVELS + 1):
        centers, new_sse = fit(k)
        if new_sse > 0.5 * sse:
            break
        best, sse = centers, new_sse
    return best


def _phase_locking(anchor_ns: np.ndarray, partner_ns: np.ndarray) -> float:
    """How firmly ``partner`` events sit at one point of the anchor's intervals.

    Mean resultant length of the partner's phases within the intervals
    between consecutive anchor events (1: always at the same point, 0:
    anywhere). Intervals longer than three times the median (pauses) are
    left out.
    """
    if len(anchor_ns) < 3 or len(partner_ns) == 0:
        return 0.0
    gaps = np.diff(anchor_ns).astype(float)
    pos = np.searchsorted(anchor_ns, partner_ns, side="right") - 1
    ok = (pos >= 0) & (pos < len(gaps))
    if not ok.any():
        return 0.0
    gap = gaps[pos[ok]]
    keep = gap <= 3.0 * float(np.median(gaps))
    phase = (partner_ns[ok][keep] - anchor_ns[pos[ok]][keep]) / gap[keep]
    if len(phase) == 0:
        return 0.0
    angle = 2 * np.pi * phase
    return float(np.hypot(np.cos(angle).mean(), np.sin(angle).mean()))


def _bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), NaN kept."""
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(m)
    out[order] = adjusted
    q[ok] = out
    return q


def _bootstrap_register(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    """90 % bootstrap intervals (over cycles) of potential and variance share.

    ``matrix`` is cycles x segments. Returns potential low / high and
    variance-contribution low / high per segment; NaN with fewer than 10
    cycles.
    """
    n, k = matrix.shape
    nan = np.full(k, np.nan)
    if n < 10:
        return nan, nan, nan, nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pot = np.empty((BOOTSTRAP_RESAMPLES, k))
    share = np.empty((BOOTSTRAP_RESAMPLES, k))
    for b in range(BOOTSTRAP_RESAMPLES):
        sample = matrix[rng.integers(0, n, n)]
        q10, q50 = np.quantile(sample, [0.1, 0.5], axis=0)
        pot[b] = q50 - q10
        total = sample.sum(axis=1)
        var_t = total.var(ddof=1)
        centered = sample - sample.mean(axis=0)
        cov = (centered * (total - total.mean())[:, None]).sum(axis=0) / (n - 1)
        share[b] = cov / var_t if var_t > 0 else np.nan
    lo, hi = 5, 95
    return (
        np.percentile(pot, lo, axis=0),
        np.percentile(pot, hi, axis=0),
        np.nanpercentile(share, lo, axis=0),
        np.nanpercentile(share, hi, axis=0),
    )


def _from_ns(ns: np.ndarray, tz: Any) -> pd.Series:
    """Int64 nanoseconds back to timestamps in the input's timezone."""
    times = pd.to_datetime(pd.Series(np.asarray(ns, dtype=np.int64)), unit="ns")
    if tz is not None:
        times = times.dt.tz_localize("UTC").dt.tz_convert(tz)
    return times


def _grid_period(t_ns: np.ndarray) -> float:
    """The polling grid the timestamps lie on, in seconds (0 when none).

    The distinct timestamps of a poller lie on multiples of its period even
    when only changes are kept. The period is the smallest common gap: when
    at least :data:`GRID_SHARE` of the gaps are within 5 % of a whole
    multiple of it (and it is at least one second), it is the grid.
    """
    t = np.unique(np.asarray(t_ns, dtype=np.int64))
    if len(t) < 5:
        return 0.0
    gaps = np.diff(t).astype(np.float64)
    gaps = gaps[gaps > 0]
    if len(gaps) < 4:
        return 0.0
    base = float(np.percentile(gaps, 10))
    if base < 1e9:
        return 0.0
    ratio = gaps / base
    multiple = np.abs(ratio - np.round(ratio)) <= 0.05 * np.maximum(np.round(ratio), 1)
    share = float((multiple & (np.round(ratio) >= 1)).mean())
    return base / 1e9 if share >= GRID_SHARE else 0.0


class CycleProfileEvents(Base):
    """Production: mapping-free cycle time profiling of one PLC.

    Give it every signal of one machine controller. It finds the cycle by
    itself, learns the standard sequence of a cycle, splits each cycle into
    segments and reports what limits and what destabilizes the cycle time --
    per product variant, without a role mapping or master data.

    Methods:
    - signal_events: The discrete events every signal was turned into.
    - cycle_anchor_candidates: Ranking of the events that could cut cycles.
    - evaluation_window: Data readiness per variant (enough coherent cycles?).
    - standard_view: The learned standard sequence of a cycle per variant.
    - segment_durations: Each cycle split into segments (they add up to it).
    - contribution_register: Per segment: potential, spread, skew,
      bimodality, variance contribution.
    - cycle_time_statistics: Robust statistics and class counts per variant.
    - detect_cycle_influencers: Features that separate long / short cycles
      from normal ones.
    - cycle_records: One record per cycle, every field at the same timestamp.
    - detect_unusual_cycles: The long and short cycle records.
    - units: The classified cycles (for unit profilers).
    - events_frame: Every discrete event (for unit profilers).
    - signal_specs: Inferred spec per signal.
    - variant_limits: Classification limits per variant.
    - cycle_anchor: The anchor event and its signal.
    - variant_signal: The variant signal.
    - resolution_seconds: Time resolution of the cycle view.
    - evaluation_reason: Reason of the all-variants row.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
        >>> profile = ts_shape.CycleProfileEvents(frame)
        >>> profile.cycle_anchor_candidates().loc[0, "event"]
        'plc:part_counter+1'
        >>> register = profile.contribution_register()
        >>> top = register.sort_values("potential_rank").iloc[0]
        >>> top["variant"], top["segment"]
        ('A', 'plc:step=60 → plc:step=70')
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        signal_uuids: list[str] | None = None,
        anchor_uuid: str | None = None,
        min_duration: str | float | None = None,
        max_duration: str | float | None = None,
        min_cycles: int = 50,
        presence_ratio: float = 0.9,
        n_sigma: float = 3.0,
        pause_factor: float = 3.0,
        time_resolution: str | float | None = None,
        analog_levels: int | str = 2,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:cycle_profile",
    ) -> None:
        """Initialize the profiler.

        Args:
            dataframe: Long-format signals of one PLC (``systime``, ``uuid``,
                ``value_*``).
            signal_uuids: Signals to use; ``None`` (default) uses every
                signal of the frame.
            anchor_uuid: Signal whose best-ranked event cuts the cycles; the
                default ``None`` discovers the anchor (US-1).
            min_duration: Optional plausibility bound (offset string or
                seconds): a shorter cycle is ``short`` whatever its z-score.
                Cycle limits are derived from the data; this only adds to
                them (DQ-4).
            max_duration: Optional plausibility bound: a longer cycle is a
                ``pause`` (DQ-4).
            min_cycles: Non-pause cycles a variant needs to be evaluable
                (DQ-3; default 50).
            presence_ratio: Share of the non-pause cycles in which an event
                must occur exactly once to join the standard view.
            n_sigma: Robust z-score of log(duration) beyond which a cycle is
                ``long`` or ``short``.
            pause_factor: A cycle of at least this many times the variant's
                median duration is a ``pause`` (DQ-2).
            time_resolution: How far apart (offset string or seconds) two
                events must be to be told apart. Polled signals (an OPC UA
                client reading every 10 s) put events of one poll in random
                order; standard-view events closer than this become one
                *milestone* that counts at its earliest event. The default
                ``None`` uses the median sample period of the view's sampled
                signals (0 for change-encoded signals).
            analog_levels: How an analog signal becomes events. ``2``
                (default): one hysteresis band between 40 % and 60 % of its
                P10..P90 range (``uuid>``, ``uuid<``). An integer from 3 or
                ``"auto"``: the signal's operating levels are learned (1-D
                k-means; ``"auto"`` picks 2 to 4 levels by the drop of the
                within-level spread) and entering a level is an event
                (``uuid≈<level>``), with hysteresis of a quarter of the gap
                to the neighbouring level.
            time_column: Timestamp column.
            uuid_column: Signal identifier column.
            event_uuid: ``uuid`` of the point and interval outputs.

        Raises:
            ValueError: If a numeric parameter is out of range.
        """
        super().__init__(dataframe, time_column=time_column)
        if min_cycles < 1:
            raise ValueError("min_cycles must be at least 1")
        if not 0.0 < presence_ratio <= 1.0:
            raise ValueError("presence_ratio must be in (0, 1]")
        if n_sigma <= 0:
            raise ValueError("n_sigma must be positive")
        if pause_factor <= 1.0:
            raise ValueError("pause_factor must be greater than 1")
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.event_uuid = event_uuid
        self.signal_uuids = None if signal_uuids is None else list(signal_uuids)
        self.anchor_uuid = anchor_uuid
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_cycles = int(min_cycles)
        self.presence_ratio = float(presence_ratio)
        self.n_sigma = float(n_sigma)
        self.pause_factor = float(pause_factor)
        self.time_resolution = time_resolution
        if not (
            analog_levels == "auto"
            or (isinstance(analog_levels, int) and analog_levels >= 2)
        ):
            raise ValueError("analog_levels must be 'auto' or an integer of at least 2")
        self.analog_levels = analog_levels
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Pipeline (computed once)
    # ------------------------------------------------------------------

    @property
    def _tz(self) -> Any:
        times = self.dataframe[self.time_column] if not self.dataframe.empty else None
        if times is None or not isinstance(times.dtype, pd.DatetimeTZDtype):
            return None
        return times.dt.tz

    def _signals(self) -> pd.DataFrame:
        """Rows of the profiled signals with a timestamp."""
        df = self.dataframe
        if df.empty or self.uuid_column not in df.columns:
            return df.iloc[0:0]
        if self.signal_uuids is not None:
            wanted = {str(u) for u in self.signal_uuids}
            df = df[df[self.uuid_column].astype(str).isin(wanted)]
        return df[df[self.time_column].notna()]

    def _events(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Discrete events of every signal and the inferred spec per signal."""
        if "events" in self._cache:
            return self._cache["events"], self._cache["specs"]
        frames: list[pd.DataFrame] = []
        specs: dict[str, Any] = {}
        signals = self._signals()
        for key, group in signals.groupby(self.uuid_column, sort=True, observed=True):
            uuid = str(key)
            if len(group) < 2:
                continue
            spec = infer_spec(group, uuid, self.time_column)
            specs[uuid] = spec
            events = self._signal_to_events(group, uuid, spec)
            if not events.empty:
                frames.append(events)
        if frames:
            events = pd.concat(frames, ignore_index=True)
            events["ns"] = to_ns(events["systime"])
            events = events.sort_values(["ns", "event"], kind="stable").reset_index(
                drop=True
            )
        else:
            events = pd.DataFrame(
                {
                    "systime": pd.Series([], dtype="datetime64[ns]"),
                    "source_uuid": pd.Series([], dtype=object),
                    "event": pd.Series([], dtype=object),
                    "signal_kind": pd.Series([], dtype=object),
                    "ns": pd.Series([], dtype=np.int64),
                }
            )
        self._cache["events"], self._cache["specs"] = events, specs
        return events, specs

    def _signal_to_events(
        self, group: pd.DataFrame, uuid: str, spec: Any
    ) -> pd.DataFrame:
        """The events of one signal (LL-1: the kind decides how)."""
        column = spec.value_column
        if column not in group.columns:
            return pd.DataFrame()
        g = group[[self.time_column, column]].copy()
        if column == "value_bool":
            g["v"] = g[column].astype("boolean")
        elif column == "value_string":
            g["v"] = g[column].where(g[column].notna(), None)
        else:
            g["v"] = pd.to_numeric(g[column], errors="coerce")
        g = g[g["v"].notna()]
        if len(g) < 2:
            return pd.DataFrame()
        times = g[self.time_column].reset_index(drop=True)
        if column == "value_bool":
            state = g["v"].astype(bool).to_numpy()
            change = np.r_[False, state[1:] != state[:-1]]
            labels = np.where(state, f"{uuid}{RISE}", f"{uuid}{FALL}")
            kind = "bool"
        elif column == "value_string" or spec.kind == "state":
            text = g["v"].map(_value_label).to_numpy(dtype=object)
            change = np.r_[False, text[1:] != text[:-1]]
            labels = np.array([f"{uuid}={v}" for v in text], dtype=object)
            kind = "text" if column == "value_string" else "state"
        elif spec.kind == "counter":
            values = g["v"].to_numpy(dtype=float)
            change = np.r_[False, values[1:] > values[:-1]]
            labels = np.full(len(values), f"{uuid}{INCREMENT}", dtype=object)
            kind = "counter"
        elif self.analog_levels != 2:
            values = g["v"].to_numpy(dtype=float)
            return self._level_events(times, values, uuid)
        else:
            values = g["v"].to_numpy(dtype=float)
            p10, p90 = np.quantile(values, [0.1, 0.9])
            span = float(p90 - p10)
            if not span > 0:
                return pd.DataFrame()
            high = p10 + HYSTERESIS_HIGH * span
            low = p10 + HYSTERESIS_LOW * span
            level = pd.Series(
                np.where(values > high, 1.0, np.where(values < low, 0.0, np.nan))
            ).ffill()
            known = level.notna().to_numpy()
            level_values = level.to_numpy()
            prev = np.r_[np.nan, level_values[:-1]]
            change = known & ~np.isnan(prev) & (level_values != prev)
            labels = np.where(level_values == 1.0, f"{uuid}{HIGH}", f"{uuid}{LOW}")
            kind = "continuous"
        if not change.any():
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "systime": times[change].reset_index(drop=True),
                "source_uuid": uuid,
                "event": labels[change],
                "signal_kind": kind,
            }
        )

    def _level_events(
        self, times: pd.Series, values: np.ndarray, uuid: str
    ) -> pd.DataFrame:
        """Entries into the learned operating levels of an analog signal."""
        centers = _analog_levels(values, self.analog_levels)
        if len(centers) < 2:
            return pd.DataFrame()
        gaps = np.diff(centers)
        state = np.full(len(values), -1)
        current = int(np.argmin(np.abs(centers - values[0])))
        for i, v in enumerate(values):
            nearest = int(np.argmin(np.abs(centers - v)))
            if nearest != current:
                # hysteresis: a quarter of the gap past the midpoint
                lo, hi = sorted((nearest, current))
                margin = 0.25 * gaps[lo] if hi - lo == 1 else 0.0
                mid = 0.5 * (centers[lo] + centers[hi])
                beyond = v > mid + margin if nearest > current else v < mid - margin
                if beyond:
                    current = nearest
            state[i] = current
        change = np.r_[False, state[1:] != state[:-1]]
        if not change.any():
            return pd.DataFrame()
        labels = np.array(
            [f"{uuid}≈{float(centers[k]):.3g}" for k in state],
            dtype=object,
        )
        return pd.DataFrame(
            {
                "systime": times[change].reset_index(drop=True),
                "source_uuid": uuid,
                "event": labels[change],
                "signal_kind": "continuous",
            }
        )

    def _anchor_table(self) -> pd.DataFrame:
        """Ranked anchor candidates (US-1, DQ-1)."""
        if "anchors" in self._cache:
            return self._cache["anchors"]
        events, _ = self._events()
        if events.empty:
            table = pd.DataFrame(columns=_ANCHOR_COLUMNS)
            self._cache["anchors"] = table
            return table
        grouped = events.groupby("event", sort=True)
        table = grouped.agg(
            source_uuid=("source_uuid", "first"),
            signal_kind=("signal_kind", "first"),
            event_count=("ns", "size"),
        ).reset_index()
        table = table[table["event_count"] >= 3].reset_index(drop=True)
        if table.empty:
            table = pd.DataFrame(columns=_ANCHOR_COLUMNS)
            self._cache["anchors"] = table
            return table
        counts = table["event_count"].to_numpy(dtype=float)
        sources = table["source_uuid"].to_numpy(dtype=object)
        # A partner repeats about as often as the candidate (re-tries allowed)
        # and is phase-locked to it: its events fall at the same point of the
        # candidate's intervals. A heartbeat repeats too often; an independent
        # machine on the same controller (a parallel lane) drifts.
        ratio = counts[None, :] / counts[:, None]
        near = (ratio >= SUPPORT_MIN_RATIO) & (ratio <= SUPPORT_MAX_RATIO)
        other = sources[None, :] != sources[:, None]
        times = {
            label: np.unique(ns)
            for label, ns in events.groupby("event")["ns"].apply(np.asarray).items()
            if label in set(table["event"])
        }
        labels = table["event"].to_numpy(dtype=object)
        support = []
        for i in range(len(table)):
            anchor_ns = times[labels[i]]
            locked: set[str] = set()
            for j in np.flatnonzero(near[i] & other[i]):
                if sources[j] in locked:
                    continue
                if _phase_locking(anchor_ns, times[labels[j]]) >= SUPPORT_MIN_LOCKING:
                    locked.add(str(sources[j]))
            support.append(len(locked))
        table["support_count"] = support
        table["is_counter"] = table["signal_kind"] == "counter"
        medians, cvs, coverage = [], [], []
        span = max(float(events["ns"].max() - events["ns"].min()) / 1e9, 1e-9)
        for label in table["event"]:
            ns = events.loc[events["event"] == label, "ns"].to_numpy()
            intervals = np.diff(np.unique(ns)) / 1e9
            median = float(np.median(intervals)) if len(intervals) else np.nan
            medians.append(median)
            cvs.append(_robust_cv(intervals))
            regular = (
                intervals[intervals <= 3.0 * median] if len(intervals) else intervals
            )
            coverage.append(float(regular.sum()) / span if len(regular) else 0.0)
        table["median_interval_seconds"] = medians
        table["interval_cv_factor"] = cvs
        table["coverage_ratio"] = np.clip(coverage, 0.0, 1.0)
        # Regularity breaks ties at CV_RESOLUTION; finer differences are noise,
        # so the event that misses fewer cycles (more events) wins next: a
        # shift counter that skips its reset loses to the part counter.
        table["_cv"] = table["interval_cv_factor"].round(CV_DECIMALS)
        # Only events that cycle through most of the recording compete: a
        # burst of a rare event can have locked partners within its few
        # intervals but explains little of the data.
        best = float(table["coverage_ratio"].max()) if len(table) else 0.0
        table["_covers"] = table["coverage_ratio"] >= MIN_RELATIVE_COVERAGE * best
        table = (
            table.sort_values(
                [
                    "_covers",
                    "support_count",
                    "is_counter",
                    "_cv",
                    "event_count",
                    "event",
                ],
                ascending=[False, False, False, True, False, True],
                kind="stable",
                na_position="last",
            )
            .drop(columns=["_cv", "_covers"])
            .reset_index(drop=True)
        )
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        eligible = table["median_interval_seconds"] > 0
        if self.anchor_uuid is not None:
            eligible &= table["source_uuid"] == str(self.anchor_uuid)
        table["is_anchor"] = False
        if eligible.any():
            table.loc[eligible.idxmax(), "is_anchor"] = True
        table = table[_ANCHOR_COLUMNS]
        self._cache["anchors"] = table
        return table

    def _anchor(self) -> tuple[str, str] | None:
        """``(event, source_uuid)`` of the anchor, or None."""
        table = self._anchor_table()
        chosen = table[table["is_anchor"].astype(bool)]
        if chosen.empty:
            return None
        row = chosen.iloc[0]
        return str(row["event"]), str(row["source_uuid"])

    def _cycles(self) -> pd.DataFrame:
        """Complete cycles between consecutive anchor events, classified."""
        if "cycles" in self._cache:
            return self._cache["cycles"]
        columns = [
            "cycle_uuid",
            "start_ns",
            "end_ns",
            "duration_seconds",
            "variant",
            "cycle_class",
            "z",
        ]
        anchor = self._anchor()
        events, _ = self._events()
        if anchor is None:
            cycles = pd.DataFrame(columns=columns)
            self._cache["cycles"] = cycles
            self._cache["variant_uuid"] = None
            self._cache["limits"] = {}
            return cycles
        label, _source = anchor
        times = events.loc[events["event"] == label, "systime"]
        pulse = pd.DataFrame({"systime": times.to_numpy(), "uuid": label})
        pulse["systime"] = times.reset_index(drop=True)
        pulse["value_bool"] = True
        extracted = CycleExtractor(pulse, label).process_trigger_cycle()
        extracted = extracted[extracted["is_complete"].astype(bool)].reset_index(
            drop=True
        )
        cycles = pd.DataFrame(
            {
                "cycle_uuid": extracted["cycle_uuid"].astype(str),
                "start_ns": to_ns(extracted["start"]),
                "end_ns": to_ns(extracted["end"]),
            }
        )
        cycles = cycles[cycles["end_ns"] > cycles["start_ns"]].reset_index(drop=True)
        cycles["duration_seconds"] = (cycles["end_ns"] - cycles["start_ns"]) / 1e9
        variant, variant_uuid = self._variant(cycles)
        cycles["variant"] = variant
        self._cache["variant_uuid"] = variant_uuid
        self._classify(cycles)
        cycles = cycles[columns]
        self._cache["cycles"] = cycles
        return cycles

    def _variant(self, cycles: pd.DataFrame) -> tuple[pd.Series, str | None]:
        """The variant of every cycle and the signal it comes from (LL-1)."""
        default = pd.Series([ALL] * len(cycles), index=cycles.index, dtype=object)
        if len(cycles) < 4:
            return default, None
        events, specs = self._events()
        anchor = self._anchor()
        durations = cycles["duration_seconds"].to_numpy(dtype=float)
        log_d = np.log(durations)
        # The variant must explain the bulk of the cycles: pauses, aborted and
        # disturbed cycles (outside n_sigma robust sigmas over all cycles,
        # whose spread the variants widen) are left out of eta squared.
        core = durations < self.pause_factor * float(np.median(durations))
        center = float(np.median(log_d[core]))
        sigma = max(
            MAD_SCALE * float(np.median(np.abs(log_d[core] - center))), MIN_LOG_SIGMA
        )
        preliminary = core & (np.abs(log_d - center) <= self.n_sigma * sigma)
        signals = self._signals()
        found: list[tuple[float, int, str, pd.Series]] = []
        for uuid, spec in sorted(specs.items()):
            if anchor is not None and uuid == anchor[1]:
                continue
            textual = spec.value_column == "value_string"
            state = spec.kind == "state" and spec.value_column != "value_bool"
            if not (textual or state):
                continue
            changes = int((events["source_uuid"] == uuid).sum())
            if changes > VARIANT_MAX_CHANGE_RATIO * len(cycles):
                continue
            rows = signals[signals[self.uuid_column].astype(str) == uuid]
            column = spec.value_column
            rows = rows[rows[column].notna()]
            if rows.empty:
                continue
            values = rows[column]
            if not textual:
                values = pd.to_numeric(values, errors="coerce")
            lookup = pd.DataFrame(
                {
                    "ns": to_ns(rows[self.time_column]),
                    "value": values.map(_value_label).to_numpy(dtype=object),
                }
            ).sort_values("ns", kind="stable")
            lookup = lookup.drop_duplicates("ns", keep="last")
            at_start = pd.merge_asof(
                cycles[["start_ns"]].rename(columns={"start_ns": "ns"}),
                lookup,
                on="ns",
                direction="backward",
            )["value"]
            at_start = at_start.bfill()
            labels = at_start.to_numpy(dtype=object)
            if pd.isna(labels).any():
                continue
            groups = pd.Series(log_d[preliminary]).groupby(labels[preliminary])
            if groups.ngroups < 2 or (groups.size() < 2).any():
                continue
            total = float(((log_d[preliminary] - log_d[preliminary].mean()) ** 2).sum())
            if total <= 0:
                continue
            between = float(
                (groups.size() * (groups.mean() - log_d[preliminary].mean()) ** 2).sum()
            )
            eta = between / total
            if eta >= VARIANT_MIN_ETA_SQUARED:
                # Adjusted for the number of groups (as adjusted R squared):
                # a finer grouping always explains a little more by chance.
                n, k = int(preliminary.sum()), int(groups.ngroups)
                adjusted = 1.0 - (1.0 - eta) * (n - 1) / max(n - k, 1)
                found.append(
                    (
                        adjusted,
                        int(groups.ngroups),
                        uuid,
                        pd.Series(labels, index=cycles.index, dtype=object),
                    )
                )
        if not found:
            return default, None
        # A finer signal (a fixture or order id that changes with every
        # article block) explains at least as much as the article: ranked by
        # the adjusted eta squared, and the simplest signal that explains
        # almost as much as the best is the variant.
        top = max(f[0] for f in found)
        near = [f for f in found if f[0] >= VARIANT_ETA_TOLERANCE * top]
        eta, _groups, uuid, labels = min(near, key=lambda f: (f[1], -f[0], f[2]))
        return labels, uuid

    def _classify(self, cycles: pd.DataFrame) -> None:
        """Class and robust z of every cycle, per variant (DQ-2, US-2)."""
        min_d = _seconds(self.min_duration)
        max_d = _seconds(self.max_duration)
        cycles["cycle_class"] = "normal"
        cycles["z"] = np.nan
        limits: dict[str, dict[str, float]] = {}
        for variant, idx in cycles.groupby("variant", sort=True).groups.items():
            d = cycles.loc[idx, "duration_seconds"].to_numpy(dtype=float)
            pause_threshold = self.pause_factor * float(np.median(d))
            if max_d is not None:
                pause_threshold = min(pause_threshold, max_d)
            pause = d >= pause_threshold
            if max_d is not None:
                pause |= d > max_d
            base = np.log(d[~pause]) if (~pause).any() else np.log(d)
            m = float(np.median(base))
            # A cycle is known to one sample of its cycle signal at each end:
            # the sigma floor covers that quantization (polled signals).
            quantum = self._anchor_period() / max(float(np.exp(m)), 1e-9)
            s = max(
                MAD_SCALE * float(np.median(np.abs(base - m))),
                MIN_LOG_SIGMA,
                quantum,
            )
            z = (np.log(d) - m) / s
            short = z < -self.n_sigma
            if min_d is not None:
                short |= d < min_d
            label = np.where(
                pause,
                "pause",
                np.where(z > self.n_sigma, "long", np.where(short, "short", "normal")),
            )
            cycles.loc[idx, "cycle_class"] = label
            cycles.loc[idx, "z"] = z
            limits[str(variant)] = {
                "lower": float(np.exp(m - self.n_sigma * s)),
                "upper": float(np.exp(m + self.n_sigma * s)),
                "pause": pause_threshold,
                "median": float(np.median(d[~pause] if (~pause).any() else d)),
            }
        self._cache["limits"] = limits

    def _anchor_period(self) -> float:
        """Sample period (s) of the anchor's signal when it is polled, else 0.

        A polled signal has many more readings than changes (at least
        three per anchor event); a change-encoded counter that changes once
        per cycle is not polled, however regular it is.
        """
        anchor = self._anchor()
        if anchor is None:
            return 0.0
        events, _ = self._events()
        signals = self._signals()
        rows = signals.loc[
            signals[self.uuid_column].astype(str) == anchor[1], self.time_column
        ]
        changes = int((events["event"] == anchor[0]).sum())
        if changes == 0 or len(rows) < 3 * changes:
            return 0.0
        gaps = np.diff(np.unique(to_ns(rows))) / 1e9
        return float(np.median(gaps)) if len(gaps) else 0.0

    def _cycle_events(self) -> pd.DataFrame:
        """Events with the cycle they fall in (anchor events left out)."""
        if "cycle_events" in self._cache:
            return self._cache["cycle_events"]
        events, _ = self._events()
        cycles = self._cycles()
        anchor = self._anchor()
        if cycles.empty or anchor is None:
            out = events.iloc[0:0].assign(cycle=pd.Series([], dtype=np.int64))
            self._cache["cycle_events"] = out
            return out
        starts = cycles["start_ns"].to_numpy(dtype=np.int64)
        ends = cycles["end_ns"].to_numpy(dtype=np.int64)
        ev = events[events["event"] != anchor[0]]
        ns = ev["ns"].to_numpy(dtype=np.int64)
        pos = np.searchsorted(starts, ns, side="right") - 1
        valid = pos >= 0
        valid[valid] &= ns[valid] < ends[pos[valid]]
        out = ev[valid].assign(cycle=pos[valid]).reset_index(drop=True)
        self._cache["cycle_events"] = out
        return out

    def _views(self) -> pd.DataFrame:
        """Standard view of every variant."""
        if "views" in self._cache:
            return self._cache["views"]
        cycles = self._cycles()
        ce = self._cycle_events()
        rows: list[pd.DataFrame] = []
        for variant, idx in cycles.groupby("variant", sort=True).groups.items():
            members = cycles.loc[idx]
            base = members[members["cycle_class"] != "pause"]
            n = len(base)
            if n == 0:
                continue
            sub = ce[ce["cycle"].isin(base.index)]
            if sub.empty:
                continue
            per = sub.groupby(["event", "cycle"], sort=True).agg(
                count=("ns", "size"), first_ns=("ns", "min")
            )
            per = per.reset_index()
            once = per[per["count"] == 1]
            presence = once.groupby("event").size() / n
            keep = presence[presence >= self.presence_ratio - 1e-12].index
            if len(keep) == 0:
                continue
            first = per[per["event"].isin(keep)].copy()
            start = cycles.loc[first["cycle"], "start_ns"].to_numpy()
            dur = cycles.loc[first["cycle"], "duration_seconds"].to_numpy()
            first["offset"] = (first["first_ns"].to_numpy() - start) / 1e9
            first["phase"] = first["offset"] / dur
            agg = first.groupby("event").agg(
                median_phase_ratio=("phase", "median"),
                median_offset_seconds=("offset", "median"),
            )
            agg["presence_ratio"] = presence.loc[agg.index]
            agg = agg.reset_index().sort_values(
                ["median_phase_ratio", "event"], kind="stable"
            )
            source = sub.drop_duplicates("event").set_index("event")["source_uuid"]
            agg["source_uuid"] = agg["event"].map(source)
            agg["variant"] = str(variant)
            agg["position"] = np.arange(1, len(agg) + 1)
            resolution = self._resolution(agg["source_uuid"])
            milestone, first_offset, groups = [], None, 0
            for offset in agg["median_offset_seconds"].to_numpy(dtype=float):
                if first_offset is None or offset - first_offset >= resolution:
                    groups += 1
                    first_offset = offset
                milestone.append(groups)
            agg["milestone"] = milestone
            agg["cycle_count"] = n
            rows.append(agg[_VIEW_COLUMNS])
        views = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=_VIEW_COLUMNS)
        )
        self._cache["views"] = views
        return views

    def _resolution(self, sources: pd.Series) -> float:
        """Seconds within which events cannot be ordered (see __init__)."""
        given = _seconds(self.time_resolution)
        if given is not None:
            resolution = max(given, 0.0)
        else:
            _, specs = self._events()
            periods = []
            for uuid in sources:
                spec = specs.get(str(uuid))
                sampled = spec is not None and spec.encoding == "sampled"
                period = spec.nominal_period if sampled and spec is not None else None
                periods.append(period.total_seconds() if period is not None else 0.0)
            resolution = float(np.median(periods)) if periods else 0.0
        self._cache["resolution"] = max(resolution, self._cache.get("resolution", 0.0))
        return resolution

    def _segment_table(self) -> pd.DataFrame:
        """Segments of every in-sequence cycle (internal: ns and indices)."""
        if "segments" in self._cache:
            return self._cache["segments"]
        cycles = self._cycles()
        views = self._views()
        ce = self._cycle_events()
        anchor = self._anchor()
        columns = [
            "cycle",
            "variant",
            "cycle_class",
            "position",
            "segment",
            "from_event",
            "to_event",
            "start_ns",
            "end_ns",
            "duration_seconds",
        ]
        frames: list[pd.DataFrame] = []
        deviation = pd.Series(False, index=cycles.index)
        for variant, idx in cycles.groupby("variant", sort=True).groups.items():
            view = views[views["variant"] == str(variant)]
            members = cycles.loc[idx]
            if view.empty or anchor is None:
                deviation.loc[idx] = True
                continue
            order = view["event"].tolist()
            sub = ce[ce["cycle"].isin(members.index) & ce["event"].isin(order)]
            per_event = sub.groupby(["cycle", "event"])["ns"].min().unstack("event")
            per_event = per_event.reindex(index=members.index, columns=order)
            # A milestone counts at its earliest event; it is missing when
            # one of its events is.
            groups = view.groupby("milestone", sort=True)["event"].agg(list)
            first = pd.DataFrame(
                {
                    events[0]: per_event[events].min(axis=1, skipna=False)
                    for events in groups
                },
                index=members.index,
            )
            order = list(first.columns)
            matrix = first.to_numpy(dtype=float)
            bounds = np.column_stack(
                [
                    members["start_ns"].to_numpy(dtype=float),
                    matrix,
                    members["end_ns"].to_numpy(dtype=float),
                ]
            )
            missing = np.isnan(bounds).any(axis=1)
            filled = np.where(np.isnan(bounds), 0.0, bounds)
            ordered = (np.diff(filled, axis=1) >= 0).all(axis=1)
            ok = ~missing & ordered
            deviation.loc[members.index[~ok]] = True
            if not ok.any():
                continue
            good = members[ok]
            # Exact integer nanoseconds: the segments add up to the cycle.
            ints = np.column_stack(
                [
                    good["start_ns"].to_numpy(dtype=np.int64),
                    first.loc[good.index].to_numpy(dtype=np.int64),
                    good["end_ns"].to_numpy(dtype=np.int64),
                ]
            )
            names = [anchor[0], *order, anchor[0]]
            k = len(names) - 1
            n = len(good)
            frames.append(
                pd.DataFrame(
                    {
                        "cycle": np.repeat(good.index.to_numpy(), k),
                        "variant": str(variant),
                        "cycle_class": np.repeat(good["cycle_class"].to_numpy(), k),
                        "position": np.tile(np.arange(1, k + 1), n),
                        "segment": np.tile(
                            [f"{a} → {b}" for a, b in pairwise(names)],
                            n,
                        ),
                        "from_event": np.tile(names[:-1], n),
                        "to_event": np.tile(names[1:], n),
                        "start_ns": ints[:, :-1].ravel(),
                        "end_ns": ints[:, 1:].ravel(),
                    }
                )
            )
        if frames:
            table = pd.concat(frames, ignore_index=True)
            table["duration_seconds"] = (table["end_ns"] - table["start_ns"]) / 1e9
            table = table.sort_values(["cycle", "position"], kind="stable")
            table = table.reset_index(drop=True)[columns]
        else:
            table = pd.DataFrame(columns=columns)
        self._cache["segments"] = table
        self._cache["deviation"] = deviation
        return table

    def _register(self) -> pd.DataFrame:
        if "register" in self._cache:
            return self._cache["register"]
        seg = self._segment_table()
        cycles = self._cycles()
        deviation: pd.Series = self._cache.get(
            "deviation", pd.Series(False, index=cycles.index)
        )
        rows: list[dict[str, Any]] = []
        base = seg[seg["cycle_class"] != "pause"]
        resolution = float(self._cache.get("resolution", 0.0))
        for variant, part in base.groupby("variant", sort=True):
            pivot = part.pivot(
                index="cycle", columns="position", values="duration_seconds"
            )
            total = pivot.sum(axis=1).to_numpy(dtype=float)
            var_t = float(np.var(total, ddof=1)) if len(total) > 1 else float("nan")
            meta = part.drop_duplicates("position").set_index("position")
            members = cycles[
                (cycles["variant"] == variant) & (cycles["cycle_class"] != "pause")
            ]
            deviating = int(deviation.reindex(members.index).fillna(False).sum())
            medians = pivot.median(axis=0)
            median_sum = float(medians.sum())
            ci = _bootstrap_register(pivot.to_numpy(dtype=float))
            for col, position in enumerate(pivot.columns):
                x = pivot[position].to_numpy(dtype=float)
                p10, median, p90 = np.quantile(x, [0.1, 0.5, 0.9])
                skew, coefficient = _bimodality(x)
                cov = float(np.cov(x, total, ddof=1)[0, 1]) if len(x) > 1 else np.nan
                rows.append(
                    {
                        "variant": str(variant),
                        "position": int(position),
                        "segment": meta.loc[position, "segment"],
                        "from_event": meta.loc[position, "from_event"],
                        "to_event": meta.loc[position, "to_event"],
                        "cycle_count": len(x),
                        "deviation_cycle_count": deviating,
                        "median_seconds": float(median),
                        "p10_seconds": float(p10),
                        "p90_seconds": float(p90),
                        "spread_seconds": float(p90 - p10),
                        "mad_seconds": float(np.median(np.abs(x - median))),
                        "mean_seconds": float(np.mean(x)),
                        "std_seconds": (
                            float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
                        ),
                        "skewness": skew,
                        "bimodality_coefficient": coefficient,
                        # Two values one poll apart are quantization, not
                        # two modes: a mode needs a spread of more than two
                        # time-resolution steps.
                        "is_bimodal": bool(
                            not np.isnan(coefficient)
                            and coefficient > BIMODALITY_THRESHOLD
                            and float(p90 - p10) > 2.0 * resolution
                        ),
                        "median_share_ratio": (
                            float(median) / median_sum if median_sum > 0 else np.nan
                        ),
                        "potential_seconds": float(median - p10),
                        "potential_ci_low_seconds": ci[0][col],
                        "potential_ci_high_seconds": ci[1][col],
                        "variance_contribution_factor": (
                            cov / var_t if var_t and var_t > 0 else np.nan
                        ),
                        "variance_contribution_ci_low_factor": ci[2][col],
                        "variance_contribution_ci_high_factor": ci[3][col],
                    }
                )
        if not rows:
            register = pd.DataFrame(columns=_REGISTER_COLUMNS)
        else:
            register = pd.DataFrame(rows)
            register["potential_rank"] = (
                register.groupby("variant")["potential_seconds"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            register = register[_REGISTER_COLUMNS]
        self._cache["register"] = register
        return register

    # ------------------------------------------------------------------
    # Public outputs
    # ------------------------------------------------------------------

    def signal_events(self) -> pd.DataFrame:
        """The discrete events every signal was turned into (LL-1).

        Implements: LL-1 (the kind of a signal decides its events; analog
        values that are really codes or counters are treated as such).

        A boolean gives its edges (``uuid↑``, ``uuid↓``), a text or integer
        state its entries into a value (``uuid=v``), a counter its increments
        (``uuid+1``) and an analog level its crossings of a hysteresis band
        between 40 % and 60 % of its P10..P90 range (``uuid>``, ``uuid<``).
        The first sample of a signal gives no event.

        Returns:
            Point events: ``systime``, ``uuid`` (``event_uuid``),
            ``source_uuid`` (the signal), ``event`` (the label) and
            ``signal_kind`` (``bool``, ``state``, ``text``, ``counter`` or
            ``continuous``).

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=60, seed=1)
            >>> events = ts_shape.CycleProfileEvents(frame).signal_events()
            >>> events.groupby("signal_kind")["event"].nunique().to_dict()
            {'bool': 10, 'continuous': 2, 'counter': 1, 'state': 8, 'text': 1}
        """
        events, _ = self._events()
        out = events.drop(columns=["ns"]).copy()
        out.insert(1, "uuid", self.event_uuid)
        if out.empty:
            out["systime"] = out["systime"].astype(
                self.dataframe[self.time_column].dtype
            )
        return out[_EVENT_COLUMNS].reset_index(drop=True)

    def cycle_anchor_candidates(self) -> pd.DataFrame:
        """Rank the events that could mark the cycle (the anchor).

        Implements: US-1 (find the unknown cycle signal from raw data only),
        DQ-1 (one start event; its recurrence is the end).

        Every event that occurs at least three times is ranked by
        coverage first: only events whose regular intervals (gaps of at
        most three times their median) cover at least half as much of the
        recording as the best candidate compete (``coverage_ratio``); then

        1. ``support_count``: the number of *other* signals with an event
           that repeats 0.8 to 1.5 times per candidate event and is
           phase-locked to it (its events fall at the same point of the
           candidate's intervals, mean resultant length >= 0.7) -- a cycle
           event repeats in step with the rest of the machine; a heartbeat
           repeats too often and an independent machine on the same
           controller drifts;
        2. ``is_counter``: a part counter is the natural cycle boundary;
        3. ``interval_cv_factor``: robust coefficient of variation of the
           intervals (median absolute deviation / median), ascending;
        4. the event label (a deterministic tie-break).

        The first row is the anchor unless ``anchor_uuid`` names a signal,
        then the best-ranked event of that signal is.

        Returns:
            Static table: ``rank``, ``event``, ``source_uuid``,
            ``signal_kind``, ``event_count``, ``support_count``,
            ``is_counter``, ``median_interval_seconds``,
            ``interval_cv_factor``, ``coverage_ratio``, ``is_anchor``.

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=60, seed=1)
            >>> table = ts_shape.CycleProfileEvents(frame).cycle_anchor_candidates()
            >>> table[["event", "support_count", "is_anchor"]].head(2)
                            event  support_count  is_anchor
            0  plc:part_counter+1              5       True
            1       plc:pressure<              5      False
            >>> table.loc[table["event"] == "plc:heartbeat↑", "support_count"].item()
            0
        """
        return self._anchor_table().copy()

    def evaluation_window(self) -> pd.DataFrame:
        """Data readiness: are there enough coherent cycles per variant?

        Implements: DQ-3 (50-100 coherent cycles, otherwise optimize
        organizationally first; cycles may be stitched together).

        Pauses break a run of coherent cycles; the non-pause cycles are
        stitched together, so ``cycle_count`` counts all of them and
        ``longest_coherent_cycle_count`` the longest uninterrupted run.

        Returns:
            Summary, one row per variant plus ``all``: ``start``, ``end``,
            ``duration_seconds`` (the cycles covered), ``variant``,
            ``cycle_count`` (non-pause), ``pause_count``,
            ``longest_coherent_cycle_count``, ``evaluable``, ``reason``
            (``ok``, ``too_few_cycles: optimize organizationally first`` or
            ``no_cycle_anchor``), ``anchor_event``, ``anchor_uuid``,
            ``variant_uuid`` and ``time_resolution_seconds`` (events closer
            than this are not ordered; see ``time_resolution``). Without an anchor, one ``all`` row covers the
            data.

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
            >>> window = ts_shape.CycleProfileEvents(frame).evaluation_window()
            >>> window[["variant", "cycle_count", "evaluable", "reason"]]
              variant  cycle_count  evaluable reason
            0     all          118       True     ok
            1       A           59       True     ok
            2       B           59       True     ok
        """
        cycles = self._cycles()
        anchor = self._anchor()
        tz = self._tz
        variant_uuid = self._cache.get("variant_uuid")
        if anchor is None or cycles.empty:
            signals = self._signals()
            if signals.empty:
                return pd.DataFrame(columns=_WINDOW_COLUMNS)
            ns = to_ns(signals[self.time_column])
            first, last = int(np.min(ns)), int(np.max(ns))
            start = _from_ns(np.array([first]), tz)
            end = _from_ns(np.array([last]), tz)
            return pd.DataFrame(
                {
                    "start": start,
                    "end": end,
                    "duration_seconds": [(last - first) / 1e9],
                    "variant": [ALL],
                    "cycle_count": [0],
                    "pause_count": [0],
                    "longest_coherent_cycle_count": [0],
                    "evaluable": [False],
                    "reason": ["no_cycle_anchor"],
                    "anchor_event": [None if anchor is None else anchor[0]],
                    "anchor_uuid": [None if anchor is None else anchor[1]],
                    "variant_uuid": [variant_uuid],
                    "time_resolution_seconds": [np.nan],
                }
            )[_WINDOW_COLUMNS]
        rows = []
        groups = [(ALL, cycles)] + [
            (str(v), part) for v, part in cycles.groupby("variant", sort=True)
        ]
        if variant_uuid is None:
            groups = groups[:1]
        pause = (cycles["cycle_class"] == "pause").to_numpy()
        variants = cycles["variant"].to_numpy(dtype=object)
        for name, part in groups:
            run = longest = 0
            for p, v in zip(pause, variants):
                if not p and (name == ALL or v == name):
                    run += 1
                    longest = max(longest, run)
                else:
                    run = 0
            count = int((part["cycle_class"] != "pause").sum())
            evaluable = count >= self.min_cycles
            rows.append(
                {
                    "start_ns": int(part["start_ns"].min()),
                    "end_ns": int(part["end_ns"].max()),
                    "variant": name,
                    "cycle_count": count,
                    "pause_count": int((part["cycle_class"] == "pause").sum()),
                    "longest_coherent_cycle_count": longest,
                    "evaluable": evaluable,
                    "reason": (
                        "ok"
                        if evaluable
                        else "too_few_cycles: optimize organizationally first"
                    ),
                    "anchor_event": anchor[0],
                    "anchor_uuid": anchor[1],
                    "variant_uuid": variant_uuid,
                }
            )
        self._views()
        out = pd.DataFrame(rows)
        resolution = float(self._cache.get("resolution", 0.0) or 0.0)
        if not resolution > 0 and self.time_resolution is None:
            # Change-encoded rows recorded by a poller still lie on its grid:
            # events cannot be ordered within one poll period.
            resolution = _grid_period(to_ns(self._signals()[self.time_column]))
            self._cache["resolution"] = resolution
        out["time_resolution_seconds"] = resolution
        median_cycle = (
            float(
                np.median(
                    (cycles["end_ns"] - cycles["start_ns"]).to_numpy()[~pause] / 1e9
                )
            )
            if (~pause).any()
            else np.nan
        )
        too_coarse = (
            resolution > 0
            and np.isfinite(median_cycle)
            and resolution >= RESOLUTION_MAX_SHARE * median_cycle
        )
        if too_coarse:
            out["evaluable"] = False
            out["reason"] = (
                f"anchor_below_resolution: the signals are seen every "
                f"{resolution:g} s, at least {RESOLUTION_MAX_SHARE:g} of the "
                f"median cycle ({median_cycle:g} s); cycles cannot be resolved"
            )
        out["start"] = _from_ns(out["start_ns"].to_numpy(), tz)
        out["end"] = _from_ns(out["end_ns"].to_numpy(), tz)
        out["duration_seconds"] = (out["end_ns"] - out["start_ns"]) / 1e9
        return out[_WINDOW_COLUMNS]

    def standard_view(self) -> pd.DataFrame:
        """The learned standard sequence of a cycle, per variant.

        Implements: the "Standard View on every PLC" use case (learned, not
        configured).

        An event joins the view when it occurs exactly once in at least
        ``presence_ratio`` of a variant's non-pause cycles; the view is
        ordered by the median phase (0 at the anchor, 1 at the next one).

        Returns:
            Static table: ``variant``, ``position``, ``milestone`` (events
            within ``time_resolution`` of each other share one; segments run
            between milestones), ``event``, ``source_uuid``,
            ``presence_ratio``, ``median_phase_ratio``,
            ``median_offset_seconds``, ``cycle_count``.

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
            >>> view = ts_shape.CycleProfileEvents(frame).standard_view()
            >>> view.loc[view["variant"] == "A", "event"].head(4).tolist()
            ['plc:step=10', 'plc:step=20', 'plc:clamp_closed↑', 'plc:pressure>']
        """
        return self._views().copy()

    def segment_durations(self) -> pd.DataFrame:
        """Every cycle split into the segments of its standard view.

        Implements: event pairing ("which pair of events has the longest
        time between them") for consecutive events of the standard view.

        A cycle's segments run from the anchor to the first view event,
        between consecutive view events and from the last to the next anchor,
        so they add up exactly to the cycle duration. A cycle in which a view
        event is missing or out of order is a sequence deviation and has no
        segments (see ``sequence_deviation`` in :meth:`cycle_records`). Each
        view event counts at its first occurrence in the cycle; repeats (a
        re-inspection) stay inside their segment.

        Returns:
            Intervals: ``start``, ``end``, ``duration_seconds``, ``uuid``,
            ``source_uuid`` (the anchor signal), ``cycle_uuid``, ``variant``,
            ``cycle_class``, ``position``, ``segment``, ``from_event``,
            ``to_event``.

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=60, seed=1)
            >>> profile = ts_shape.CycleProfileEvents(frame)
            >>> seg = profile.segment_durations()
            >>> total = seg.groupby("cycle_uuid")["duration_seconds"].sum()
            >>> rec = profile.cycle_records().set_index("cycle_uuid")
            >>> bool(((total - rec.loc[total.index, "duration_seconds"]).abs() < 1e-6).all())
            True
        """
        seg = self._segment_table()
        anchor = self._anchor()
        if seg.empty or anchor is None:
            return self._empty_intervals(_SEGMENT_COLUMNS)
        cycles = self._cycles()
        tz = self._tz
        out = pd.DataFrame(
            {
                "start": _from_ns(seg["start_ns"].to_numpy(), tz),
                "end": _from_ns(seg["end_ns"].to_numpy(), tz),
                "duration_seconds": seg["duration_seconds"].to_numpy(dtype=float),
                "uuid": self.event_uuid,
                "source_uuid": anchor[1],
                "cycle_uuid": cycles.loc[seg["cycle"], "cycle_uuid"].to_numpy(),
            }
        )
        for column in _SEGMENT_COLUMNS[6:]:
            out[column] = seg[column].to_numpy()
        return out[_SEGMENT_COLUMNS]

    def contribution_register(self) -> pd.DataFrame:
        """What to optimize first, and what varies, per variant and segment.

        Implements: Contribution Register and Schwankungsbreiten (spread)
        from the use-case notes; US-2 (statistical, transferable KPIs).

        On the non-pause cycles of a variant without a sequence deviation:

        - ``median_seconds``, ``p10_seconds``, ``p90_seconds``,
          ``spread_seconds`` (P90 - P10), ``mad_seconds`` (median absolute
          deviation), ``mean_seconds``, ``std_seconds``;
        - ``skewness`` and Sarle's ``bimodality_coefficient``
          (``is_bimodal`` above 5/9 and a spread of more than two
          time-resolution steps: the segment has two typical durations);
        - ``median_share_ratio``: the segment's share of the summed medians;
        - ``potential_seconds`` = median - P10, the time the segment loses
          against its own best demonstrated performance, and its
          ``potential_rank`` within the variant (question 1: what first);
        - ``variance_contribution_factor`` = Cov(segment, cycle) / Var(cycle):
          the segment's share of the cycle time variance. The shares add up
          to exactly 1; a negative share compensates other segments
          (question 2: what is unstable).

        Returns:
            Static table, one row per variant and segment, with the columns
            above plus ``variant``, ``position``, ``segment``,
            ``from_event``, ``to_event``, ``cycle_count`` and
            ``deviation_cycle_count`` (cycles left out for a sequence
            deviation).

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
            >>> reg = ts_shape.CycleProfileEvents(frame).contribution_register()
            >>> reg.groupby("variant")["variance_contribution_factor"].sum().round(6).to_dict()
            {'A': 1.0, 'B': 1.0}
        """
        return self._register().copy()

    def cycle_time_statistics(self) -> pd.DataFrame:
        """Robust cycle time statistics and class counts per variant.

        Implements: US-2 (short and long cycles found statistically, KPIs
        transferable across plants), VIZ-1 (target from the data: the best
        demonstrated cycle time, P10, instead of a master-data target).

        A cycle is a ``pause`` at ``pause_factor`` times the median or more
        (or above ``max_duration``); otherwise ``long`` / ``short`` beyond
        ``n_sigma`` robust sigmas of log(duration) (median and scaled median
        absolute deviation; floor 1 %, and at least one sample period of a
        polled cycle signal relative to the median), or below
        ``min_duration``; else ``normal``. Statistics other than the counts leave pauses out.

        Returns:
            Static table, one row per variant: ``variant``, ``cycle_count``
            (all classes), ``normal_count``, ``long_count``, ``short_count``,
            ``pause_count``, ``median_seconds``, ``p10_seconds``,
            ``p90_seconds``, ``mad_seconds``, ``mean_seconds``,
            ``std_seconds``, ``min_seconds``, ``max_seconds``,
            ``target_seconds`` (P10), ``potential_seconds`` (median - P10),
            ``lower_limit_seconds`` / ``upper_limit_seconds`` (short / long
            bounds) and ``pause_threshold_seconds``.

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
            >>> st = ts_shape.CycleProfileEvents(frame).cycle_time_statistics()
            >>> st[["variant", "long_count", "short_count", "pause_count"]]
              variant  long_count  short_count  pause_count
            0       A           1            3            0
            1       B           1            1            1
        """
        cycles = self._cycles()
        limits: dict[str, dict[str, float]] = self._cache.get("limits", {})
        rows = []
        for variant, part in cycles.groupby("variant", sort=True):
            cls = part["cycle_class"]
            d = part.loc[cls != "pause", "duration_seconds"].to_numpy(dtype=float)
            lim = limits.get(str(variant), {})
            if len(d):
                p10, median, p90 = np.quantile(d, [0.1, 0.5, 0.9])
                values = {
                    "median_seconds": float(median),
                    "p10_seconds": float(p10),
                    "p90_seconds": float(p90),
                    "mad_seconds": float(np.median(np.abs(d - median))),
                    "mean_seconds": float(np.mean(d)),
                    "std_seconds": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
                    "min_seconds": float(np.min(d)),
                    "max_seconds": float(np.max(d)),
                    "target_seconds": float(p10),
                    "potential_seconds": float(median - p10),
                }
            else:
                values = dict.fromkeys(_STATISTICS_COLUMNS[6:16], np.nan)
            rows.append(
                {
                    "variant": str(variant),
                    "cycle_count": len(part),
                    "normal_count": int((cls == "normal").sum()),
                    "long_count": int((cls == "long").sum()),
                    "short_count": int((cls == "short").sum()),
                    "pause_count": int((cls == "pause").sum()),
                    **values,
                    "lower_limit_seconds": lim.get("lower", np.nan),
                    "upper_limit_seconds": lim.get("upper", np.nan),
                    "pause_threshold_seconds": lim.get("pause", np.nan),
                }
            )
        if not rows:
            return pd.DataFrame(columns=_STATISTICS_COLUMNS)
        return pd.DataFrame(rows)[_STATISTICS_COLUMNS]

    def detect_cycle_influencers(self, *, top_n: int = 10) -> pd.DataFrame:
        """The signals that separate long and short cycles from normal ones.

        Implements: US-3 (state information that explains why a cycle was
        long or short), OUT-1 (top-10 influencers with uuid, label and
        importance, as the earlier Cycle Analyser published them).

        Features per cycle: the count of every event (``event_count``), the
        mean and maximum of every analog signal (``mean``, ``max``) and the
        state of every boolean at the cycle start (``start_state``). For
        ``long_vs_normal`` and ``short_vs_normal`` each feature is scored by
        the Mann-Whitney AUC of the group against the normal cycles:
        ``importance_ratio`` = |AUC - 0.5| x 2 (0: no separation, 1:
        complete), ``direction`` says whether the group is ``higher`` or
        ``lower``. Each comparison needs at least two cycles on both sides.
        ``p_value`` is the two-sided Mann-Whitney p-value, ``q_value`` the
        Benjamini-Hochberg adjusted value over every feature tested in the
        comparison, and ``is_significant`` is ``q_value < 0.05``.

        Analog means and maxima are taken over the same window in every
        cycle (from the cycle start for the variant's best demonstrated cycle
        time, P10), so a longer cycle does not lower its mean just by waiting
        longer.

        Args:
            top_n: Features kept per comparison.

        Returns:
            Static table: ``comparison``, ``rank``, ``feature``,
            ``source_uuid``, ``feature_kind``, ``importance_ratio``,
            ``auc_ratio``, ``direction``, ``group_median``,
            ``normal_median``, ``group_cycle_count``, ``normal_cycle_count``,
            ``p_value``, ``q_value``, ``is_significant``.

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=200, seed=3)
            >>> inf = ts_shape.CycleProfileEvents(frame).detect_cycle_influencers(top_n=3)
            >>> inf.loc[inf["comparison"] == "long_vs_normal", "feature"].tolist()
            ['event_count:plc:camera_done↑', 'event_count:plc:camera_done↓', 'event_count:plc:camera_nok↑']
        """
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        cycles = self._cycles()
        if cycles.empty:
            return pd.DataFrame(columns=_INFLUENCER_COLUMNS)
        features, meta = self._features(cycles)
        cls = cycles["cycle_class"].to_numpy(dtype=object)
        rows: list[dict[str, Any]] = []
        normal = cls == "normal"
        for group in ("long", "short"):
            member = cls == group
            if member.sum() < 2 or normal.sum() < 2 or features.shape[1] == 0:
                continue
            x = features.to_numpy(dtype=float)
            sel = member | normal
            sub = x[sel]
            is_group = member[sel]
            usable = ~np.isnan(sub).any(axis=0)
            sub = np.where(np.isnan(sub), 0.0, sub)
            ranks = sps.rankdata(sub, axis=0)
            n1, n0 = int(is_group.sum()), int((~is_group).sum())
            u = ranks[is_group].sum(axis=0) - n1 * (n1 + 1) / 2.0
            auc = u / (n1 * n0)
            importance = np.abs(auc - 0.5) * 2.0
            constant = sub.min(axis=0) == sub.max(axis=0)
            tested = usable & ~constant
            p_values = np.full(sub.shape[1], np.nan)
            if tested.any():
                p_values[tested] = sps.mannwhitneyu(
                    sub[is_group][:, tested],
                    sub[~is_group][:, tested],
                    axis=0,
                    method="asymptotic",
                ).pvalue
            q_values = _bh_qvalues(p_values)
            keep = tested & (importance > 0)
            order = sorted(
                np.flatnonzero(keep),
                key=lambda j: (-importance[j], features.columns[j]),
            )[:top_n]
            for rank, j in enumerate(order, start=1):
                name = features.columns[j]
                rows.append(
                    {
                        "comparison": f"{group}_vs_normal",
                        "rank": rank,
                        "feature": name,
                        "source_uuid": meta[name][0],
                        "feature_kind": meta[name][1],
                        "importance_ratio": float(importance[j]),
                        "auc_ratio": float(auc[j]),
                        "direction": "higher" if auc[j] > 0.5 else "lower",
                        "group_median": float(np.median(sub[is_group, j])),
                        "normal_median": float(np.median(sub[~is_group, j])),
                        "group_cycle_count": n1,
                        "normal_cycle_count": n0,
                        "p_value": float(p_values[j]),
                        "q_value": float(q_values[j]),
                        "is_significant": bool(q_values[j] < INFLUENCER_FDR),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=_INFLUENCER_COLUMNS)
        return pd.DataFrame(rows)[_INFLUENCER_COLUMNS]

    def _features(
        self, cycles: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
        """Per-cycle feature matrix and ``feature -> (source_uuid, kind)``."""
        if "features" in self._cache:
            cached: tuple[pd.DataFrame, dict[str, tuple[str, str]]]
            cached = self._cache["features"]
            return cached
        meta: dict[str, tuple[str, str]] = {}
        parts: list[pd.DataFrame] = []
        ce = self._cycle_events()
        if not ce.empty:
            counts = (
                ce.groupby(["cycle", "event"]).size().unstack("event", fill_value=0)
            )
            # An event seen in a single cycle cannot separate groups; a clock
            # event (heartbeat) only measures the duration.
            seen = (counts > 0).sum(axis=0) >= 2
            rate = counts.sum(axis=0) / max(len(cycles), 1)
            counts = counts.loc[:, seen & (rate <= MAX_EVENTS_PER_CYCLE)]
            counts = counts.reindex(index=cycles.index, fill_value=0)
            source = ce.drop_duplicates("event").set_index("event")["source_uuid"]
            names = {e: f"event_count:{e}" for e in counts.columns}
            for e, name in names.items():
                meta[name] = (str(source[e]), "event_count")
            parts.append(counts.rename(columns=names).astype(float))
        _, specs = self._events()
        signals = self._signals()
        starts = cycles["start_ns"].to_numpy(dtype=np.int64)
        ends = cycles["end_ns"].to_numpy(dtype=np.int64)
        # Comparable window for analog features: from the cycle start for the
        # variant's best demonstrated cycle time (P10 of non-pause cycles).
        body = cycles[cycles["cycle_class"] != "pause"]
        p10 = body.groupby("variant")["duration_seconds"].quantile(0.1)
        window = cycles["variant"].map(p10).fillna(cycles["duration_seconds"])
        window_end = starts + (
            np.minimum(
                window.to_numpy(dtype=float),
                cycles["duration_seconds"].to_numpy(dtype=float),
            )
            * 1e9
        ).astype(np.int64)
        for uuid, spec in sorted(specs.items()):
            rows = signals[signals[self.uuid_column].astype(str) == uuid]
            if spec.value_column not in rows.columns:
                continue
            ns = to_ns(rows[self.time_column])
            if spec.kind == "continuous" and spec.value_column != "value_bool":
                values = pd.to_numeric(
                    rows[spec.value_column], errors="coerce"
                ).to_numpy(dtype=float)
                pos = np.searchsorted(starts, ns, side="right") - 1
                ok = (pos >= 0) & ~np.isnan(values)
                ok[ok] &= ns[ok] < np.minimum(ends, window_end)[pos[ok]]
                if not ok.any():
                    continue
                agg = (
                    pd.DataFrame({"cycle": pos[ok], "v": values[ok]})
                    .groupby("cycle")["v"]
                    .agg(["mean", "max"])
                    .reindex(range(len(cycles)))
                )
                for stat in ("mean", "max"):
                    name = f"{stat}:{uuid}"
                    meta[name] = (uuid, stat)
                    parts.append(
                        pd.DataFrame({name: agg[stat].to_numpy()}, index=cycles.index)
                    )
            elif spec.value_column == "value_bool":
                state = rows["value_bool"].astype("boolean")
                lookup = pd.DataFrame({"ns": ns, "v": state.to_numpy()})
                lookup = lookup[lookup["v"].notna()].drop_duplicates("ns", keep="last")
                if lookup.empty:
                    continue
                at = pd.merge_asof(
                    pd.DataFrame({"ns": starts}), lookup, on="ns", direction="backward"
                )["v"]
                name = f"start_state:{uuid}"
                meta[name] = (uuid, "start_state")
                parts.append(
                    pd.DataFrame(
                        {
                            name: at.astype("Float64").to_numpy(
                                dtype=float, na_value=np.nan
                            )
                        },
                        index=cycles.index,
                    )
                )
        features = (
            pd.concat(parts, axis=1) if parts else pd.DataFrame(index=cycles.index)
        )
        features = features.reindex(sorted(features.columns), axis=1)
        self._cache["features"] = (features, meta)
        return features, meta

    def cycle_records(self) -> pd.DataFrame:
        """One record per cycle; every field carries the same timestamp.

        Implements: OUT-1 (all outputs of a cycle share one timestamp, so
        they can be matched and charted, e.g. in Grafana), DQ-2 (pauses are
        kept, as their own class, not dropped).

        ``systime`` is the cycle end. ``dominant_segment`` is the segment that
        exceeded its variant median by the most seconds in this cycle
        (``dominant_excess_seconds``), i.e. where the cycle lost its time;
        it is empty for a cycle with a sequence deviation.

        Returns:
            Point events, one per complete cycle including pauses:
            ``systime``, ``uuid``, ``source_uuid`` (anchor signal),
            ``cycle_uuid``, ``start_ns``, ``end_ns`` (Unix nanoseconds, UTC
            for tz-aware input), ``duration_seconds``, ``cycle_class``,
            ``variant``, ``median_seconds`` (variant median),
            ``deviation_seconds`` (duration - median), ``sequence_deviation``,
            ``dominant_segment``, ``dominant_segment_seconds``,
            ``dominant_excess_seconds``, ``anchor_event``.

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
            >>> rec = ts_shape.CycleProfileEvents(frame).cycle_records()
            >>> rec["cycle_class"].value_counts().sort_index().to_dict()
            {'long': 2, 'normal': 112, 'pause': 1, 'short': 4}
        """
        cycles = self._cycles()
        anchor = self._anchor()
        if cycles.empty or anchor is None:
            return self._empty_points(_RECORD_COLUMNS)
        seg = self._segment_table()
        deviation: pd.Series = self._cache.get(
            "deviation", pd.Series(True, index=cycles.index)
        )
        register = self._register()
        limits: dict[str, dict[str, float]] = self._cache.get("limits", {})
        out = pd.DataFrame(
            {
                "systime": _from_ns(cycles["end_ns"].to_numpy(), self._tz),
                "uuid": self.event_uuid,
                "source_uuid": anchor[1],
                "cycle_uuid": cycles["cycle_uuid"].to_numpy(),
                "start_ns": cycles["start_ns"].to_numpy(dtype=np.int64),
                "end_ns": cycles["end_ns"].to_numpy(dtype=np.int64),
                "duration_seconds": cycles["duration_seconds"].to_numpy(dtype=float),
                "cycle_class": cycles["cycle_class"].to_numpy(dtype=object),
                "variant": cycles["variant"].to_numpy(dtype=object),
            }
        )
        out["median_seconds"] = [
            limits.get(str(v), {}).get("median", np.nan) for v in out["variant"]
        ]
        out["deviation_seconds"] = out["duration_seconds"] - out["median_seconds"]
        out["sequence_deviation"] = (
            deviation.reindex(cycles.index).fillna(True).astype(bool).to_numpy()
        )
        out["dominant_segment"] = pd.Series([None] * len(out), dtype=object)
        out["dominant_segment_seconds"] = np.nan
        out["dominant_excess_seconds"] = np.nan
        if not seg.empty and not register.empty:
            medians = register.set_index(["variant", "position"])["median_seconds"]
            key = pd.MultiIndex.from_arrays([seg["variant"], seg["position"]])
            excess = seg["duration_seconds"].to_numpy(dtype=float) - medians.reindex(
                key
            ).to_numpy(dtype=float)
            ranked = seg.assign(excess=excess).dropna(subset=["excess"])
            ranked = ranked.sort_values(
                ["cycle", "excess", "position"],
                ascending=[True, False, True],
                kind="stable",
            ).drop_duplicates("cycle")
            idx = ranked["cycle"].to_numpy()
            out.loc[idx, "dominant_segment"] = ranked["segment"].to_numpy()
            out.loc[idx, "dominant_segment_seconds"] = ranked[
                "duration_seconds"
            ].to_numpy(dtype=float)
            out.loc[idx, "dominant_excess_seconds"] = ranked["excess"].to_numpy(
                dtype=float
            )
        out["anchor_event"] = anchor[0]
        return out[_RECORD_COLUMNS]

    def detect_unusual_cycles(self) -> pd.DataFrame:
        """The long and short cycle records, for alerts and the event log.

        Implements: US-2 (short and long cycles detected statistically and
        handed to staff for investigation).

        Returns:
            The rows of :meth:`cycle_records` with ``cycle_class`` ``long``
            or ``short``, plus ``direction`` (``above`` / ``below`` the
            robust band).

        Example:
            >>> import ts_shape
            >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
            >>> unusual = ts_shape.CycleProfileEvents(frame).detect_unusual_cycles()
            >>> sorted(unusual["cycle_class"].unique())
            ['long', 'short']
        """
        records = self.cycle_records()
        mask = records["cycle_class"].isin(["long", "short"])
        out = records[mask].reset_index(drop=True)
        out["direction"] = np.where(out["cycle_class"] == "long", "above", "below")
        return out

    # ------------------------------------------------------------------
    # Empty outputs
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Read-only access for the unit profilers (events.production.*_profiling)
    # ------------------------------------------------------------------

    def units(self) -> pd.DataFrame:
        """The classified cycles: one row per complete cycle.

        Columns: ``cycle_uuid``, ``start_ns``, ``end_ns``, ``duration_seconds``,
        ``variant``, ``cycle_class`` (``pause`` / ``long`` / ``short`` /
        ``normal``) and ``z`` (robust z of the log duration). Used by the unit
        profilers (stops, changeovers, start-ups, orders ...) to cut their
        own units from the cycles.

        Returns:
            A copy of the cycle table (empty without a cycle anchor).
        """
        return self._cycles().copy()

    def events_frame(self) -> pd.DataFrame:
        """Every discrete event of every signal (see :meth:`signal_events`).

        Columns: ``systime``, ``source_uuid``, ``event``, ``signal_kind``,
        ``ns`` (integer nanoseconds), sorted by time.

        Returns:
            A copy of the event table.
        """
        events, _ = self._events()
        return events.copy()

    def signal_specs(self) -> dict[str, Any]:
        """The inferred :class:`~ts_shape.signals.spec.SignalSpec` per signal.

        Returns:
            ``{uuid: SignalSpec}`` (a copy of the mapping).
        """
        _, specs = self._events()
        return dict(specs)

    def variant_limits(self) -> dict[str, dict[str, float]]:
        """Classification limits per variant (``lower``, ``upper``, ``pause``, ``median`` seconds).

        Returns:
            ``{variant: {...}}`` (empty without cycles).
        """
        self._cycles()
        return {k: dict(v) for k, v in self._cache.get("limits", {}).items()}

    def cycle_anchor(self) -> tuple[str, str] | None:
        """``(event label, source uuid)`` of the cycle anchor, or None."""
        return self._anchor()

    def variant_signal(self) -> str | None:
        """The uuid of the signal that defines the product variant, or None."""
        self._cycles()
        value = self._cache.get("variant_uuid")
        return None if value is None else str(value)

    def resolution_seconds(self) -> float:
        """Seconds within which events cannot be ordered (as in :meth:`evaluation_window`)."""
        window = self.evaluation_window()
        if window.empty:
            return 0.0
        value = float(window["time_resolution_seconds"].iloc[0])
        return value if np.isfinite(value) else 0.0

    def evaluation_reason(self) -> str:
        """The ``reason`` of the ``all`` row of :meth:`evaluation_window`."""
        window = self.evaluation_window()
        if window.empty:
            return "no_cycle_anchor"
        rows = window[window["variant"] == ALL]
        row = rows.iloc[0] if len(rows) else window.iloc[0]
        return str(row["reason"])

    def _time_dtype(self) -> Any:
        if self.time_column in self.dataframe.columns:
            dtype = self.dataframe[self.time_column].dtype
            if isinstance(dtype, pd.DatetimeTZDtype) or (
                isinstance(dtype, np.dtype) and dtype.kind == "M"
            ):
                return dtype
        return "datetime64[ns]"

    def _empty_points(self, columns: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(columns=columns)
        out["systime"] = out["systime"].astype(self._time_dtype())
        return out

    def _empty_intervals(self, columns: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(columns=columns)
        for column in ("start", "end"):
            out[column] = out[column].astype(self._time_dtype())
        out["duration_seconds"] = out["duration_seconds"].astype(float)
        return out
