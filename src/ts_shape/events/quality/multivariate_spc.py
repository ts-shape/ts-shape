"""Multivariate SPC: Hotelling T² across correlated process variables.

Univariate control charts check each variable against its own limits. On a
process whose variables move together (temperature and pressure, feed and
torque) a fault often breaks the *relationship* while every variable stays
inside its own 3-sigma band. Hotelling's T² measures the distance of each
observation from the in-control mean in units of the in-control covariance,
so it sees exactly that.

Phase I estimates the mean vector and covariance matrix on a reference
period. Phase II scores every observation against them, with the exact
F-distribution control limit for a covariance estimated from ``m``
reference observations (Montgomery, *Introduction to Statistical Quality
Control*, ch. 11). Reference observations themselves are judged against
the Phase I beta-distribution limit.

Signals are aligned on the timestamps of the first signal; every other
signal contributes the value in force at that time (held backward, never
across more than ``max_gap``: no look-ahead).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
)
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only

logger = logging.getLogger(__name__)

_T2_COLS = ["t2", "ucl", "phase", "out_of_control"]
_CONTRIB_COLS = ["t2", "contribution", "contribution_pct", "z_score"]
_OOC_COLS = ["max_t2", "ucl", "sample_count", "top_contributor"]


def phase2_ucl(p: int, m: int, alpha: float) -> float:
    """Phase II T² upper control limit for new observations.

    ``p (m + 1)(m - 1) / (m (m - p)) * F(1 - alpha; p, m - p)`` for a mean
    and covariance estimated from ``m`` reference observations.
    """
    if m <= p:
        raise ValueError(f"need more reference observations ({m}) than variables ({p})")
    return float(
        p * (m + 1) * (m - 1) / (m * (m - p)) * stats.f.ppf(1 - alpha, p, m - p)
    )


def phase1_ucl(p: int, m: int, alpha: float) -> float:
    """Phase I T² limit for the reference observations themselves.

    ``(m - 1)^2 / m * Beta(1 - alpha; p / 2, (m - p - 1) / 2)``.
    """
    if m <= p + 1:
        raise ValueError(
            f"need at least p + 2 = {p + 2} reference observations, got {m}"
        )
    return float((m - 1) ** 2 / m * stats.beta.ppf(1 - alpha, p / 2, (m - p - 1) / 2))


class MultivariateSPCEvents(Base):
    """Quality: multivariate SPC with Hotelling T².

    Methods:
    - hotelling_t2: T² and its control limit for every aligned observation.
    - contributions: Which variables drive each out-of-control T².
    - detect_out_of_control: Runs of out-of-control observations as
      intervals, with the top contributing variable.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuids: list[str],
        *,
        value_column: str = "value_double",
        event_uuid: str = "quality:hotelling_t2",
        max_gap: str | pd.Timedelta | None = None,
        time_column: str = "systime",
    ) -> None:
        """
        Args:
            dataframe: Long-format data.
            signal_uuids: The process variables (at least two).
            value_column: Column holding the values.
            event_uuid: ``uuid`` of the emitted events.
            max_gap: Longest time a value of the second and later signals is
                held to align it with the first signal; ``None`` = no cap.
            time_column: Timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        if len(signal_uuids) < 2:
            raise ValueError("multivariate SPC needs at least two signal_uuids")
        self.signal_uuids = list(signal_uuids)
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_column = time_column
        self.max_gap = pd.Timedelta(max_gap) if max_gap is not None else None
        self.matrix = self._align()

    @property
    def _source(self) -> str:
        return ",".join(self.signal_uuids)

    def _series(self, uuid: str) -> pd.DataFrame:
        rows = self.dataframe[self.dataframe["uuid"] == uuid]
        out = pd.DataFrame(
            {
                "systime": pd.to_datetime(rows[self.time_column]),
                uuid: pd.to_numeric(rows[self.value_column], errors="coerce"),
            }
        ).dropna()
        out = out.sort_values("systime", kind="stable")
        return out.drop_duplicates("systime", keep="last")

    def _align(self) -> pd.DataFrame:
        merged = self._series(self.signal_uuids[0])
        for uuid in self.signal_uuids[1:]:
            merged = pd.merge_asof(
                merged,
                self._series(uuid),
                on="systime",
                direction="backward",
                tolerance=self.max_gap,
            )
        return merged.dropna().reset_index(drop=True)

    def _reference_mask(
        self, reference_period: tuple[object, object] | None
    ) -> np.ndarray:
        t = self.matrix["systime"]
        if reference_period is None:
            return np.ones(len(t), dtype=bool)
        start, end = (pd.Timestamp(v) for v in reference_period)  # type: ignore[arg-type]
        tz = getattr(t.dt, "tz", None)
        if tz is not None:
            start = start.tz_localize(tz) if start.tzinfo is None else start
            end = end.tz_localize(tz) if end.tzinfo is None else end
        return ((t >= start) & (t < end)).to_numpy()

    def _fit(
        self, reference_period: tuple[object, object] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self.matrix[self.signal_uuids].to_numpy(dtype=float)
        ref = self._reference_mask(reference_period)
        ref_x = x[ref]
        p, m = x.shape[1], len(ref_x)
        if m <= p + 1:
            raise ValueError(
                f"reference period has {m} aligned observations; need at "
                f"least {p + 2} for {p} variables"
            )
        mean = ref_x.mean(axis=0)
        cov = np.cov(ref_x, rowvar=False, ddof=1)
        inv = np.linalg.pinv(cov)
        if np.linalg.matrix_rank(cov) < p:
            logger.warning(
                "reference covariance is singular (constant or collinear "
                "variables); T² uses the pseudo-inverse"
            )
        return x, ref, mean, inv

    @keyword_only(1)
    def hotelling_t2(
        self,
        reference_period: tuple[object, object] | None = None,
        alpha: float = 0.0027,
    ) -> pd.DataFrame:
        """Hotelling T² for every aligned observation.

        Args:
            reference_period: ``(start, end)`` of the in-control reference
                data (end exclusive). ``None`` uses all data (a retrospective
                Phase I study).
            alpha: False-alarm probability per observation (0.0027 matches
                3-sigma limits).

        Returns:
            Point events: ``systime``, ``uuid``, ``source_uuid`` (the signal
            uuids, comma-joined), ``t2``, ``ucl``, ``phase`` (``"I"`` for
            reference observations, ``"II"`` otherwise) and
            ``out_of_control``.
        """
        if len(self.matrix) == 0:
            return empty_event_df("point", _T2_COLS)
        x, ref, mean, inv = self._fit(reference_period)
        d = x - mean
        t2 = np.einsum("ij,jk,ik->i", d, inv, d)
        p, m = x.shape[1], int(ref.sum())
        ucl = np.where(ref, phase1_ucl(p, m, alpha), phase2_ucl(p, m, alpha))
        out = pd.DataFrame(
            {
                "systime": self.matrix["systime"],
                "t2": t2,
                "ucl": ucl,
                "phase": np.where(ref, "I", "II"),
                "out_of_control": t2 > ucl,
            }
        )
        return finalize_point_df(out, uuid=self.event_uuid, source_uuid=self._source)

    @keyword_only(1)
    def contributions(
        self,
        reference_period: tuple[object, object] | None = None,
        alpha: float = 0.0027,
        only_out_of_control: bool = True,
    ) -> pd.DataFrame:
        """Per-variable decomposition of T².

        Contribution of variable ``j``: ``d_j * (S^-1 d)_j`` with ``d`` the
        deviation from the reference mean. The contributions of one
        observation sum to its T²; a large share points at the variable (or
        broken relationship) behind the signal. ``z_score`` is the plain
        univariate deviation, for comparison.

        Args:
            reference_period: As in :meth:`hotelling_t2`.
            alpha: As in :meth:`hotelling_t2`.
            only_out_of_control: Keep only observations flagged out of control by
                :meth:`hotelling_t2` (default). False returns every observation.

        Returns:
            Point events, one row per (observation, variable): ``systime``,
            ``uuid``, ``source_uuid`` (the variable), ``t2``,
            ``contribution``, ``contribution_pct`` and ``z_score``.
        """
        t2 = self.hotelling_t2(reference_period=reference_period, alpha=alpha)
        if t2.empty:
            return empty_event_df("point", _CONTRIB_COLS)
        x, ref, mean, inv = self._fit(reference_period)
        d = x - mean
        contrib = d * (d @ inv.T)
        std = x[ref].std(axis=0, ddof=1)
        z = np.divide(d, std, out=np.zeros_like(d), where=std > 0)
        keep = t2["out_of_control"].to_numpy() if only_out_of_control else None
        rows = []
        for j, uuid in enumerate(self.signal_uuids):
            part = pd.DataFrame(
                {
                    "systime": self.matrix["systime"],
                    "source_uuid": uuid,
                    "t2": t2["t2"].to_numpy(),
                    "contribution": contrib[:, j],
                    "contribution_pct": np.divide(
                        100 * contrib[:, j],
                        t2["t2"].to_numpy(),
                        out=np.zeros(len(d)),
                        where=t2["t2"].to_numpy() > 0,
                    ),
                    "z_score": z[:, j],
                }
            )
            rows.append(part[keep] if keep is not None else part)
        out = pd.concat(rows, ignore_index=True)
        if out.empty:
            return empty_event_df("point", _CONTRIB_COLS)
        out = out.sort_values(["systime", "source_uuid"], kind="stable")
        return finalize_point_df(
            out.reset_index(drop=True), uuid=self.event_uuid, source_uuid=None
        )

    @keyword_only(1)
    def detect_out_of_control(
        self,
        reference_period: tuple[object, object] | None = None,
        alpha: float = 0.0027,
        min_samples: int = 1,
    ) -> pd.DataFrame:
        """Runs of consecutive out-of-control observations.

        Args:
            reference_period: As in :meth:`hotelling_t2`.
            alpha: As in :meth:`hotelling_t2`.
            min_samples: Minimum observations in a run.

        Returns:
            Intervals from the first to the last out-of-control observation
            of each run: ``start``, ``end``, ``duration_seconds``, ``uuid``,
            ``source_uuid``, ``max_t2``, ``ucl`` (at the peak),
            ``sample_count`` and ``top_contributor`` (the variable with the
            largest summed contribution over the run).
        """
        t2 = self.hotelling_t2(reference_period=reference_period, alpha=alpha)
        if t2.empty or not t2["out_of_control"].any():
            return empty_event_df("interval", _OOC_COLS)
        x, _, mean, inv = self._fit(reference_period)
        d = x - mean
        contrib = d * (d @ inv.T)
        ooc = t2["out_of_control"].to_numpy()
        run_id = np.cumsum(np.r_[True, ooc[1:] != ooc[:-1]])
        rows = []
        for rid in np.unique(run_id[ooc]):
            idx = np.flatnonzero((run_id == rid) & ooc)
            if len(idx) < min_samples:
                continue
            peak = idx[np.argmax(t2["t2"].to_numpy()[idx])]
            top = int(np.argmax(contrib[idx].sum(axis=0)))
            rows.append(
                {
                    "start": t2["systime"].iloc[idx[0]],
                    "end": t2["systime"].iloc[idx[-1]],
                    "max_t2": float(t2["t2"].iloc[peak]),
                    "ucl": float(t2["ucl"].iloc[peak]),
                    "sample_count": int(len(idx)),
                    "top_contributor": self.signal_uuids[top],
                }
            )
        if not rows:
            return empty_event_df("interval", _OOC_COLS)
        out = pd.DataFrame(rows)
        out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
        return finalize_interval_df(out, uuid=self.event_uuid, source_uuid=self._source)
