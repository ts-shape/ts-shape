"""Reliability: MTBF, MTTR and Weibull life analysis from a failure-state signal.

Input is one held state signal whose value says whether the asset is failed
(down for repair) -- a fault bit, a breakdown status code, a maintenance
mode. The timeline is built with :func:`ts_shape.signals.primitives.intervals`,
so each state holds until the next change and a logging gap longer than
``max_gap`` is "no data": it counts as neither operating nor repair time.

Definitions (hours throughout):

* **operating time** -- time in a non-failed state.
* **repair time (TTR)** -- time in a failed state. Consecutive failed
  intervals separated only by a data gap are one failure.
* **TBF** -- operating time between the end of one repair and the next
  failure.
* ``mtbf_hours`` = operating time / failures (the common definition).
* ``mtbf_iso22400_hours`` = operating time / (failures + 1): ISO 22400-2
  averages over the FE + 1 operating periods, including the one still
  running at the end of the window.
* ``mttr_hours`` = repair time / failures.
* ``availability_pct`` = operating / (operating + repair): inherent
  availability, the share of known time the asset was not failed.

The Weibull fit treats the operating period still running at the end of the
data as right-censored (it has not failed *yet*), which is what keeps the
life estimate from being biased low.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_summary_df,
)
from ts_shape.signals.primitives import intervals, windows
from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs
from ts_shape.utils.normalize import _caller_stacklevel

logger = logging.getLogger(__name__)

_FAILURE_COLS = ["repair_hours", "operating_hours_before", "first_observed"]
_SUMMARY_COLS = [
    "operating_hours",
    "repair_hours",
    "no_data_hours",
    "failures",
    "mtbf_hours",
    "mtbf_iso22400_hours",
    "mttr_hours",
    "availability_pct",
    "failure_rate_per_hour",
]
_WEIBULL_COLS = [
    "beta",
    "eta_hours",
    "mttf_hours",
    "b10_hours",
    "n_failures",
    "n_censored",
    "failure_pattern",
]
_CURVE_COLS = [
    "hours",
    "age_hours",
    "reliability",
    "failure_probability",
    "hazard_per_hour",
]


#: A fitted shape above this means the times between failures have
#: (almost) no spread; the likelihood has no finite maximum there.
_MAX_BETA = 1e3


def _pattern(beta: float) -> str:
    """Bathtub-curve reading of the Weibull shape."""
    if not np.isfinite(beta):
        return "unknown"
    if beta < 0.9:
        return "infant_mortality"
    if beta <= 1.1:
        return "random"
    return "wear_out"


class ReliabilityEvents(Base):
    """Maintenance: reliability KPIs from a failure-state signal.

    Methods:
    - failure_events: One interval per failure, with its repair time and the
      operating time before it.
    - mtbf_mttr: MTBF, MTTR, availability and failure rate, overall or per
      calendar window.
    - weibull_fit: Weibull life model of the times between failures, with
      right censoring; shape ``beta`` tells infant mortality / random /
      wear-out.
    - reliability_at: Reliability and hazard at given operating times from
      the fitted model, optionally conditional on the current age.

    Example:
        >>> import pandas as pd, ts_shape
        >>> t = pd.date_range("2025-01-01", periods=8, freq="1h")
        >>> df = pd.DataFrame({"systime": t, "uuid": "fault", "value_bool":
        ...                    [False, False, True, False, False, False, True, False]})
        >>> kpi = ts_shape.ReliabilityEvents(df, "fault").mtbf_mttr()
        >>> kpi[["failures", "mtbf_hours", "mttr_hours", "availability_pct"]].round(1)
           failures  mtbf_hours  mttr_hours  availability_pct
        0         2         2.5         1.0              71.4
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        state_uuid: str,
        *,
        failure_states: Sequence[object] = (True,),
        value_column: str = "value_bool",
        max_gap: str | pd.Timedelta | None = None,
        data_end: str | pd.Timestamp | None = None,
        event_uuid: str = "maint:reliability",
        time_column: str = "systime",
    ) -> None:
        """
        Args:
            dataframe: Long-format data.
            state_uuid: The failure-state signal.
            failure_states: Values meaning "failed / down for repair". The
                default ``(True,)`` fits a fault bit; for a status code pass
                e.g. ``("BREAKDOWN", "REPAIR")``.
            value_column: Column holding the state.
            max_gap: Longest time a state is held; longer gaps are no data.
                Default None: every state is held until the next sample, however
                long the gap.
            data_end: End of the observation (e.g. "now"); the last state
                holds until then. Default: the last sample.
            event_uuid: ``uuid`` of the emitted events.
            time_column: Timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.state_uuid = state_uuid
        self.failure_states = list(failure_states)
        self.event_uuid = event_uuid
        self.time_column = time_column
        spec = SignalSpec(
            uuid=state_uuid,
            kind="state",
            value_column=value_column,
            max_gap=None if max_gap is None else pd.Timedelta(max_gap),
        )
        end = pd.Timestamp(data_end) if data_end is not None else None
        timeline = intervals(
            self.dataframe, spec, data_end=end, time_column=time_column
        )
        timeline["failed"] = timeline["state"].isin(self.failure_states)
        self.timeline = timeline.reset_index(drop=True)
        self._episodes = self._build_episodes()

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _build_episodes(self) -> pd.DataFrame:
        """Failures (merged across data gaps) with TTR and preceding TBF."""
        cols = ["start", "end", "repair_s", "operating_before_s", "first_observed"]
        rows: list[dict[str, Any]] = []
        operating = 0.0
        current: dict[str, Any] | None = None
        seen_repair_end = False
        for iv in self.timeline.itertuples(index=False):
            dur = float(iv.duration_seconds)
            if iv.failed:
                if current is None:
                    current = {
                        "start": iv.start,
                        "end": iv.end,
                        "repair_s": 0.0,
                        "operating_before_s": operating,
                        "first_observed": not seen_repair_end,
                    }
                current["end"] = iv.end
                current["repair_s"] += dur
            else:
                if current is not None:
                    rows.append(current)
                    current = None
                    seen_repair_end = True
                    operating = 0.0
                operating += dur
        if current is not None:
            rows.append(current)
            operating = 0.0
        self._trailing_operating_s = operating
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def failure_events(self) -> pd.DataFrame:
        """One interval per failure.

        Returns:
            Intervals ``start`` (failure) to ``end`` (back in service) with
            ``repair_hours`` (known failed time; data gaps excluded),
            ``operating_hours_before`` (operating time since the previous
            repair) and ``first_observed`` (True when that operating period
            began before the data did, so its length is a lower bound).
        """
        ep = self._episodes
        if ep.empty:
            return empty_event_df("interval", _FAILURE_COLS)
        out = pd.DataFrame(
            {
                "start": ep["start"],
                "end": ep["end"],
                "repair_hours": ep["repair_s"].astype(float) / 3600,
                "operating_hours_before": ep["operating_before_s"].astype(float) / 3600,
                "first_observed": ep["first_observed"].astype(bool),
            }
        )
        out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
        return finalize_interval_df(
            out, uuid=self.event_uuid, source_uuid=self.state_uuid
        )

    @keyword_only(1)
    def mtbf_mttr(
        self, window: str | None = None, tz: str | None = None
    ) -> pd.DataFrame:
        """MTBF, MTTR, availability and failure rate.

        Args:
            window: Calendar window (e.g. ``"1D"``, ``"7D"``, ``"MS"``);
                ``None`` = one row over the whole observation. A failure
                belongs to the window it starts in; operating and repair
                time are split exactly at window boundaries.
            tz: Timezone for the window grid (DST-correct day lengths).
                Default: the data's timezone.

        Returns:
            Summary rows (``start``, ``end``, ``duration_seconds``, ``uuid``,
            ``source_uuid``) with
            ``operating_hours``, ``repair_hours``, ``no_data_hours``,
            ``failures``, ``mtbf_hours``, ``mtbf_iso22400_hours``,
            ``mttr_hours``, ``availability_pct`` and
            ``failure_rate_per_hour``. Ratios with no failures (or no
            known time) are NaN.
        """
        tl = self.timeline
        if tl.empty:
            return empty_event_df("summary", ["uuid", "source_uuid", *_SUMMARY_COLS])
        t0, t1 = tl["start"].iloc[0], tl["end"].iloc[-1]
        if window is None:
            grid = pd.DataFrame({"start": [t0], "end": [t1]})
        else:
            grid = windows(t0, t1, window, tz=tz)[["start", "end"]]
            if grid.empty:
                return empty_event_df(
                    "summary", ["uuid", "source_uuid", *_SUMMARY_COLS]
                )

        failed = tl["failed"].to_numpy()
        ep_starts = pd.to_datetime(self._episodes["start"])
        rows = []
        for w_start, w_end in zip(grid["start"], grid["end"], strict=True):
            lo = tl["start"].clip(lower=w_start)
            hi = tl["end"].clip(upper=w_end)
            overlap = (hi - lo).dt.total_seconds().clip(lower=0).to_numpy()
            operating = float(overlap[~failed].sum())
            repair = float(overlap[failed].sum())
            span = (w_end - w_start).total_seconds()
            n = (
                int(((ep_starts >= w_start) & (ep_starts < w_end)).sum())
                if len(ep_starts)
                else 0
            )
            rows.append(
                {
                    "start": w_start,
                    "end": w_end,
                    "operating_hours": operating / 3600,
                    "repair_hours": repair / 3600,
                    "no_data_hours": max(span - operating - repair, 0.0) / 3600,
                    "failures": n,
                    "mtbf_hours": operating / 3600 / n if n else np.nan,
                    "mtbf_iso22400_hours": operating / 3600 / (n + 1),
                    "mttr_hours": repair / 3600 / n if n else np.nan,
                    "availability_pct": (
                        100 * operating / (operating + repair)
                        if operating + repair > 0
                        else np.nan
                    ),
                    "failure_rate_per_hour": (
                        n / (operating / 3600) if operating > 0 else np.nan
                    ),
                }
            )
        out = pd.DataFrame(rows)
        out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
        return finalize_summary_df(
            out, uuid=self.event_uuid, source_uuid=self.state_uuid
        )

    @keyword_only(1)
    @renamed_kwargs(include_first="include_initial")
    def weibull_fit(self, include_initial: bool = False) -> pd.DataFrame:
        """Two-parameter Weibull model of the times between failures.

        Fitted by maximum likelihood (``scipy.stats.weibull_min``, location
        fixed at 0). The operating period still running at the end of the
        data is right-censored.

        Args:
            include_initial: Also use the operating time before the first
                failure as a complete observation. Off by default: that
                period started before the data did, so the asset's age at
                the first failure is unknown.

        Returns:
            One row: ``beta`` (shape), ``eta_hours`` (scale: 63.2 % have
            failed by then), ``mttf_hours`` (``eta * Gamma(1 + 1/beta)``),
            ``b10_hours`` (10 % have failed by then), ``n_failures``,
            ``n_censored`` and ``failure_pattern`` (``infant_mortality``
            for beta < 0.9, ``random`` up to 1.1, else ``wear_out``).
            Empty when fewer than two failure intervals are available.
            When the times between failures (almost) all have the same
            length, the model has no finite fit: the parameters are NaN,
            ``failure_pattern`` is ``"unknown"`` and a
            :class:`~ts_shape.errors.DataQualityWarning` is emitted.
        """
        ep = self._episodes
        keep = ep if include_initial else ep[~ep["first_observed"].astype(bool)]
        tbf = keep["operating_before_s"].astype(float).to_numpy() / 3600
        tbf = tbf[tbf > 0]
        censored = self._trailing_operating_s / 3600
        right = np.array([censored]) if censored > 0 else np.array([])
        if len(tbf) < 2:
            return pd.DataFrame(columns=_WEIBULL_COLS)
        beta, eta = np.nan, np.nan
        if float(np.ptp(tbf)) > 1e-9 * float(tbf.max()):
            data = stats.CensoredData(uncensored=tbf, right=right)
            try:
                with np.errstate(all="ignore"):
                    beta, _, eta = stats.weibull_min.fit(data, floc=0)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                logger.warning("Weibull fit failed: %s", exc)
        beta, eta = float(beta), float(eta)
        ok = bool(np.isfinite(beta) and np.isfinite(eta) and 0 < beta <= _MAX_BETA)
        if not ok:
            warnings.warn(
                f"weibull_fit: no finite Weibull fit for {len(tbf)} times between "
                "failures (e.g. all of the same length); the parameters are NaN",
                DataQualityWarning,
                stacklevel=_caller_stacklevel(),
            )
        return pd.DataFrame(
            [
                {
                    "beta": beta if ok else np.nan,
                    "eta_hours": eta if ok else np.nan,
                    "mttf_hours": eta * math.gamma(1 + 1 / beta) if ok else np.nan,
                    "b10_hours": (
                        eta * (-math.log(0.9)) ** (1 / beta) if ok else np.nan
                    ),
                    "n_failures": len(tbf),
                    "n_censored": len(right),
                    "failure_pattern": _pattern(beta) if ok else "unknown",
                }
            ],
            columns=_WEIBULL_COLS,
        )

    @keyword_only(2)
    @renamed_kwargs(include_first="include_initial")
    def reliability_at(
        self,
        hours: Sequence[float],
        age_hours: float = 0.0,
        include_initial: bool = False,
    ) -> pd.DataFrame:
        """Reliability and hazard from the fitted Weibull model.

        Args:
            hours: Operating hours ahead to evaluate.
            age_hours: Operating hours already survived since the last
                repair. Reliability is then conditional:
                ``R(age + t) / R(age)`` -- "given it has run 200 h, how
                likely is it to run another 100 h?".
            include_initial: Passed to :meth:`weibull_fit`.

        Returns:
            One row per value of ``hours``: ``hours``, ``age_hours``,
            ``reliability``, ``failure_probability`` and
            ``hazard_per_hour`` (instantaneous failure rate at age + t).
            Empty when there are too few failures to fit; NaN values when
            the fit is degenerate (see :meth:`weibull_fit`).
        """
        fit = self.weibull_fit(include_initial=include_initial)
        if fit.empty:
            return pd.DataFrame(columns=_CURVE_COLS)
        beta, eta = float(fit["beta"].iloc[0]), float(fit["eta_hours"].iloc[0])
        t = np.asarray(list(hours), dtype=float)

        def surv(x: np.ndarray | float) -> np.ndarray:
            return np.exp(-((np.asarray(x) / eta) ** beta))

        with np.errstate(all="ignore"):
            rel = surv(age_hours + t) / surv(age_hours)
            hazard = beta / eta * ((age_hours + t) / eta) ** (beta - 1)
        # A degenerate fit (NaN parameters) or an overflow gives NaN, never inf.
        rel = np.where(np.isfinite(rel), rel, np.nan)
        hazard = np.where(np.isfinite(hazard), hazard, np.nan)
        return pd.DataFrame(
            {
                "hours": t,
                "age_hours": float(age_hours),
                "reliability": rel,
                "failure_probability": 1 - rel,
                "hazard_per_hour": hazard,
            },
            columns=_CURVE_COLS,
        )
