"""Change-point detection: where did the process level (or spread) shift?

Offline segmentation of one signal into stretches with constant mean (and
optionally constant variance) using PELT, the exact pruned dynamic program
of Killick, Fearnhead & Eckley (2012). Unlike a CUSUM or a rolling-window
test it finds *all* change points at once, places them at the exact sample,
and needs no window length. Typical uses: a tool change, a new raw-material
lot, a recalibration, a drifting setpoint that was corrected.

The implementation is plain numpy (no extra dependency). Costs are Gaussian
negative log-likelihoods computed from cumulative sums, so each candidate
segment costs O(1).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
)
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only

logger = logging.getLogger(__name__)

Cost = Literal["normal_mean", "normal_meanvar"]
Method = Literal["pelt", "binseg"]

_POINT_COLS = [
    "value",
    "mean_before",
    "mean_after",
    "std_before",
    "std_after",
    "shift",
    "shift_sigma",
]
_SEGMENT_COLS = ["segment", "mean", "std", "sample_count"]


def _robust_sigma(x: np.ndarray) -> float:
    """Noise sigma from the MAD of first differences (immune to level shifts)."""
    if len(x) < 3:
        return float(np.std(x)) or 1.0
    d = np.diff(x)
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = mad / (0.6744897501960817 * np.sqrt(2.0))
    if sigma <= 0:
        sigma = float(np.std(d)) / np.sqrt(2.0)
    return sigma if sigma > 0 else 1.0


def _cost_function(x: np.ndarray, cost: Cost) -> Callable[[Any, Any], Any]:
    """Vectorised segment cost ``C(s, t)`` of ``x[s:t]`` from cumulative sums.

    ``cost`` is validated by the caller (:func:`_setup`).
    """
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def sse(s: Any, t: Any) -> Any:
        return (s2[t] - s2[s]) - (s1[t] - s1[s]) ** 2 / (t - s)

    if cost == "normal_mean":
        inv_var = 1.0 / _robust_sigma(x) ** 2
        return lambda s, t: np.maximum(sse(s, t), 0.0) * inv_var
    var_floor = max(float(np.var(x)), 1e-12) * 1e-6
    return lambda s, t: (t - s) * np.log(np.maximum(sse(s, t) / (t - s), var_floor))


def _setup(
    x: np.ndarray, cost: Cost, penalty: float | None, min_size: int
) -> tuple[np.ndarray, int, float]:
    x = np.asarray(x, dtype=float)
    if cost not in ("normal_mean", "normal_meanvar"):
        raise ValueError(
            f"cost must be 'normal_mean' or 'normal_meanvar', got {cost!r}"
        )
    min_size = max(int(min_size), 2 if cost == "normal_meanvar" else 1)
    if penalty is None:
        n = max(len(x), 2)
        penalty = (2.0 if cost == "normal_mean" else 3.0) * float(np.log(n))
    return x, min_size, float(penalty)


def pelt(
    x: np.ndarray,
    *,
    cost: Cost = "normal_mean",
    penalty: float | None = None,
    min_size: int = 2,
) -> list[int]:
    """Optimal change points of ``x`` under a penalized Gaussian cost (PELT).

    Exact: returns the segmentation minimizing total cost plus ``penalty``
    per change. Runtime is near-linear when changes are frequent and
    quadratic when they are rare; for long series with few changes use
    :func:`binseg` or resample first.

    Args:
        x: 1-D finite values.
        cost: ``"normal_mean"`` (mean changes, common noise level estimated
            robustly) or ``"normal_meanvar"`` (mean and variance change).
            ``normal_mean`` assumes the noise level is constant: on a signal
            whose spread changes it splits the noisy stretch into many
            segments, so use ``normal_meanvar`` there.
        penalty: Cost of one extra change point. ``None`` uses the BIC:
            ``2 log n`` for ``normal_mean`` and ``3 log n`` for
            ``normal_meanvar`` (parameters added per change: location +
            mean, or location + mean + variance).
        min_size: Minimum samples per segment.

    Returns:
        Sorted indices ``i`` where a new segment starts at ``x[i]``.
    """
    x, min_size, penalty = _setup(x, cost, penalty, min_size)
    n = len(x)
    if n < 2 * min_size:
        return []
    seg_cost = _cost_function(x, cost)

    f = np.full(n + 1, np.inf)
    f[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    # Candidate segment starts and the time each was pruned at (n + 1 = not
    # pruned). A start s with f[s] + C(s, t) > f[t] can never beat t as the
    # last change for an end T >= t + min_size (K = 0 holds for Gaussian
    # likelihood costs), but for t < T < t + min_size the segment [t, T) is
    # too short, so s stays eligible until then.
    cand = np.array([0], dtype=int)
    pruned_at = np.array([n + 1], dtype=int)
    for t in range(min_size, n + 1):
        alive = pruned_at > t - min_size
        cand, pruned_at = cand[alive], pruned_at[alive]
        evaluable = t - cand >= min_size
        if not evaluable.any():
            continue
        eligible = cand[evaluable]
        costs = f[eligible] + seg_cost(eligible, t)
        best = int(np.argmin(costs))
        f[t] = costs[best] + penalty
        last[t] = eligible[best]
        newly = np.zeros(len(cand), dtype=bool)
        newly[evaluable] = costs > f[t]
        pruned_at[newly & (pruned_at > n)] = t
        cand = np.append(cand, t)
        pruned_at = np.append(pruned_at, n + 1)

    points = []
    t = n
    while t > 0:
        s = int(last[t])
        if s > 0:
            points.append(s)
        t = s
    return sorted(points)


def binseg(
    x: np.ndarray,
    *,
    cost: Cost = "normal_mean",
    penalty: float | None = None,
    min_size: int = 2,
) -> list[int]:
    """Change points by binary segmentation: fast (O(n log n)), approximate.

    Splits a segment at its best point while the cost reduction exceeds
    ``penalty``, then recurses into both halves. Same arguments and return
    value as :func:`pelt`; close changes may be merged or misplaced.
    """
    x, min_size, penalty = _setup(x, cost, penalty, min_size)
    n = len(x)
    if n < 2 * min_size:
        return []
    seg_cost = _cost_function(x, cost)
    points: list[int] = []
    stack = [(0, n)]
    while stack:
        a, b = stack.pop()
        if b - a < 2 * min_size:
            continue
        ks = np.arange(a + min_size, b - min_size + 1)
        split = seg_cost(np.full(len(ks), a), ks) + seg_cost(ks, np.full(len(ks), b))
        i = int(np.argmin(split))
        if seg_cost(a, b) - split[i] > penalty:
            k = int(ks[i])
            points.append(k)
            stack += [(a, k), (k, b)]
    return sorted(points)


#: Above this many samples ``method="pelt"`` warns that it may be slow.
PELT_WARN_SAMPLES = 20_000


class ChangePointEvents(Base):
    """Quality: change-point detection (PELT).

    Methods:
    - detect_change_points: One point event per detected shift, with the
      level and spread before and after.
    - segment_statistics: The stable stretches between change points as
      intervals with their mean, std and sample count.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        value_column: str = "value_double",
        event_uuid: str = "quality:change_point",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_column = time_column
        rows = self.dataframe[self.dataframe["uuid"] == signal_uuid]
        sig = pd.DataFrame(
            {
                "t": pd.to_datetime(rows[time_column]),
                "x": pd.to_numeric(rows[value_column], errors="coerce"),
            }
        ).dropna()
        self.signal = sig.sort_values("t", kind="stable").reset_index(drop=True)

    def _segments(
        self, cost: Cost, penalty: float | None, min_size: int, method: Method
    ) -> list[tuple[int, int]]:
        x = self.signal["x"].to_numpy(dtype=float)
        if method == "pelt":
            if len(x) > PELT_WARN_SAMPLES:
                warnings.warn(
                    f"PELT on {len(x)} samples may be slow when changes are "
                    "rare; use method='binseg' or resample the signal",
                    UserWarning,
                    stacklevel=3,
                )
            found = pelt(x, cost=cost, penalty=penalty, min_size=min_size)
        elif method == "binseg":
            found = binseg(x, cost=cost, penalty=penalty, min_size=min_size)
        else:
            raise ValueError(f"method must be 'pelt' or 'binseg', got {method!r}")
        bounds = [0, *found, len(x)]
        return list(zip(bounds[:-1], bounds[1:], strict=True))

    @keyword_only(1)
    def detect_change_points(
        self,
        cost: Cost = "normal_mean",
        penalty: float | None = None,
        min_size: int = 5,
        method: Method = "pelt",
    ) -> pd.DataFrame:
        """Detect level (and optionally spread) shifts.

        Args:
            cost: ``"normal_mean"`` finds mean shifts; ``"normal_meanvar"``
                also finds changes in variance.
            penalty: Cost of one change point; ``None`` = BIC (see
                :func:`pelt`). Larger values find fewer, larger changes.
            min_size: Minimum samples per segment.
            method: ``"pelt"`` (exact) or ``"binseg"`` (fast, approximate;
                for long series with few changes).

        Returns:
            Point events at the first sample of each new segment: ``systime``,
            ``uuid``, ``source_uuid``, ``value`` (the new level),
            ``mean_before``, ``mean_after``, ``std_before``, ``std_after``,
            ``shift`` (``mean_after - mean_before``) and ``shift_sigma``
            (the shift in units of the pooled segment std).
        """
        if len(self.signal) < 2 * min_size:
            return empty_event_df("point", _POINT_COLS)
        x = self.signal["x"].to_numpy(dtype=float)
        segs = self._segments(cost, penalty, min_size, method)
        rows = []
        for (a, b), (c, d) in zip(segs[:-1], segs[1:], strict=True):
            before, after = x[a:b], x[c:d]
            sd_b, sd_a = float(np.std(before, ddof=1)), float(np.std(after, ddof=1))
            pooled = float(np.sqrt((sd_b**2 + sd_a**2) / 2))
            shift = float(after.mean() - before.mean())
            rows.append(
                {
                    "systime": self.signal["t"].iloc[c],
                    "value": float(after.mean()),
                    "mean_before": float(before.mean()),
                    "mean_after": float(after.mean()),
                    "std_before": sd_b,
                    "std_after": sd_a,
                    "shift": shift,
                    "shift_sigma": shift / pooled if pooled > 0 else np.inf,
                }
            )
        if not rows:
            return empty_event_df("point", _POINT_COLS)
        return finalize_point_df(
            pd.DataFrame(rows), uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    def segment_statistics(
        self,
        cost: Cost = "normal_mean",
        penalty: float | None = None,
        min_size: int = 5,
        method: Method = "pelt",
    ) -> pd.DataFrame:
        """The stable segments between change points.

        A segment holds until the first sample of the next one; the last
        segment ends at the last sample.

        Args:
            cost: As in :meth:`detect_change_points`.
            penalty: As in :meth:`detect_change_points`.
            min_size: As in :meth:`detect_change_points`.
            method: As in :meth:`detect_change_points`.

        Returns:
            Intervals: ``start``, ``end``, ``duration_seconds``, ``uuid``,
            ``source_uuid``, ``segment``, ``mean``, ``std``, ``sample_count``.
        """
        if self.signal.empty:
            return empty_event_df("interval", _SEGMENT_COLS)
        x = self.signal["x"].to_numpy(dtype=float)
        t = self.signal["t"]
        segs = (
            self._segments(cost, penalty, min_size, method)
            if len(x) >= 2 * min_size
            else [(0, len(x))]
        )
        rows = []
        for i, (a, b) in enumerate(segs):
            vals = x[a:b]
            rows.append(
                {
                    "start": t.iloc[a],
                    "end": t.iloc[b] if b < len(x) else t.iloc[-1],
                    "segment": i,
                    "mean": float(vals.mean()),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "sample_count": int(len(vals)),
                }
            )
        out = pd.DataFrame(rows)
        out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
        return finalize_interval_df(
            out, uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
