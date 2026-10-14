"""Event-driven root-cause ranking: which signal led the events we detected?

:class:`~ts_shape.events.correlation.anomaly_correlation.AnomalyCorrelationEvents`
works signal-to-signal: it z-scores every signal, finds anomalies, and asks
which signal's anomalies tend to come first. That is undirected discovery, and
it can only rank signals whose own anomalies were detectable.

This detector runs the directed version. It takes the events another detector
already produced -- an SPC violation, a downtime interval, a scrap booking --
and asks, for each candidate signal, whether that signal moves *before* those
events, and by how much lead time. Only non-negative lags are tested: a cause
precedes its effect, and searching negative lags would happily "explain" a
stoppage with the temperature drop it caused.

Correlation is computed against a binary onset series on a resampled grid.
Scanning many lags and keeping the strongest is a multiple comparison, so the
p-value of the winning lag is Bonferroni-corrected by the number of lags
tested: without that, noise scanned over 30 lags looks "significant" about as
often as not, and every candidate would be promoted to root cause.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)

# Columns a trigger-event frame may carry its onset timestamp in, in the order
# the canonical event schemas prefer them.
_ONSET_COLUMNS = ("systime", "start")


class RootCauseCorrelationEvents(Base):
    """Correlation: rank candidate causes for a set of detected events.

    ``trigger_events`` is any detector's output — the onset timestamp is read
    from ``systime`` (point events) or ``start`` (interval/summary events)::

        stops = LongDowntimeEvents(df, ...).detect_long_downtime(min_gap="10min")
        rc = RootCauseCorrelationEvents(df)
        rc.rank_causes(stops, ["temp:spindle", "press:hydraulic", "flow:coolant"])

    Methods:
    - rank_causes: Ranked causal candidates with lag and p-value.
    - top_cause: Single-row verdict naming the best-supported candidate.
    - lag_profile: Correlation-vs-lag curve for one candidate signal.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "corr:root_cause",
        value_column: str = "value_double",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        """Initialize the root-cause ranker.

        Args:
            dataframe: Timeseries frame holding the candidate signals.
            event_uuid: UUID stamped on emitted rows.
            value_column: Column holding the numeric readings.
            time_column: Timestamp column.
            uuid_column: Column identifying the signal.
        """
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.uuid_column = uuid_column

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _onset_times(trigger_events: pd.DataFrame) -> pd.Series:
        """Extract event onset timestamps from any canonical event frame.

        Raises:
            TypeError: If ``trigger_events`` is not a DataFrame.
            ValueError: If no recognized onset column is present.
        """
        if not isinstance(trigger_events, pd.DataFrame):
            raise TypeError(
                "trigger_events must be a pandas DataFrame, got "
                f"{type(trigger_events).__name__}"
            )
        if trigger_events.empty:
            return pd.Series(dtype="datetime64[ns]")
        for col in _ONSET_COLUMNS:
            if col in trigger_events.columns:
                return pd.to_datetime(trigger_events[col]).dropna().sort_values()
        raise ValueError(
            "trigger_events carries no onset column; expected one of "
            f"{list(_ONSET_COLUMNS)}, got {list(trigger_events.columns)}"
        )

    def _candidate_series(self, uuid: str, window: str) -> pd.Series:
        """Resample one candidate signal onto the analysis grid."""
        sel = self.dataframe[self.dataframe[self.uuid_column] == uuid]
        if sel.empty:
            return pd.Series(dtype="float64")
        frame = sel[[self.time_column, self.value_column]].copy()
        frame[self.time_column] = pd.to_datetime(frame[self.time_column])
        frame[self.value_column] = to_numeric_checked(
            frame[self.value_column], what=f"signal {uuid!r}"
        )
        frame = frame.dropna(subset=[self.value_column])
        if frame.empty:
            return pd.Series(dtype="float64")
        return (
            frame.set_index(self.time_column)[self.value_column].resample(window).mean()
        )

    @staticmethod
    def _onset_series(onsets: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Binary series marking the grid bins in which an event started."""
        marks = pd.Series(0.0, index=index)
        if index.empty:
            return marks
        positions = index.searchsorted(onsets, side="right") - 1
        for pos in positions:
            if 0 <= pos < len(index):
                marks.iloc[pos] = 1.0
        return marks

    @staticmethod
    def _correlate_at_lag(
        candidate: np.ndarray, onsets: np.ndarray, lag: int
    ) -> tuple[float, float, int]:
        """Pearson r / p-value between the candidate led by ``lag`` bins.

        ``lag`` is how far the candidate *precedes* the event, so the
        candidate values at ``t`` are paired with the onset flags at
        ``t + lag``.
        """
        if lag > 0:
            left, right = candidate[:-lag], onsets[lag:]
        else:
            left, right = candidate, onsets
        mask = ~np.isnan(left) & ~np.isnan(right)
        left, right = left[mask], right[mask]
        n = len(left)
        if n < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
            return float("nan"), float("nan"), n
        r, p = stats.pearsonr(left, right)
        return float(r), float(p), n

    def _best_lag(
        self,
        candidate: pd.Series,
        onsets: pd.Series,
        *,
        max_lag: int,
    ) -> tuple[float, float, int, int]:
        """Return ``(r, p, lag_bins, n)`` for the candidate's strongest lag."""
        cand = candidate.to_numpy(dtype=float)
        marks = onsets.to_numpy(dtype=float)
        best: tuple[float, float, int, int] = (float("nan"), float("nan"), 0, 0)
        for lag in range(max_lag + 1):
            r, p, n = self._correlate_at_lag(cand, marks, lag)
            if np.isnan(r):
                continue
            if np.isnan(best[0]) or abs(r) > abs(best[0]):
                best = (r, p, lag, n)
        return best

    def _grid(
        self,
        trigger_events: pd.DataFrame,
        candidate_uuids: list[str],
        *,
        window: str,
    ) -> tuple[pd.Series, dict[str, pd.Series]]:
        """Align every candidate and the event onsets on one common grid."""
        onsets = self._onset_times(trigger_events)
        series = {
            uuid: self._candidate_series(uuid, window) for uuid in candidate_uuids
        }
        series = {u: s for u, s in series.items() if not s.empty}
        if onsets.empty or not series:
            return pd.Series(dtype="float64"), {}

        index = series[next(iter(series))].index
        for s in series.values():
            index = index.union(s.index)
        aligned = {u: s.reindex(index) for u, s in series.items()}
        return self._onset_series(onsets, index), aligned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @renamed_kwargs(resample="window", significant_only="only_significant")
    def rank_causes(
        self,
        trigger_events: pd.DataFrame,
        candidate_uuids: list[str],
        *,
        window: str = "1min",
        max_lag: int = 30,
        p_threshold: float = 0.05,
        trigger_uuid: str | None = None,
        trigger_event_type: str | None = None,
        only_significant: bool = False,
    ) -> pd.DataFrame:
        """Rank candidate signals by how strongly they lead the trigger events.

        Args:
            trigger_events: Event frame whose onsets are being explained.
            candidate_uuids: Signals to test as possible causes.
            window: Grid resolution (offset string); also the unit of ``max_lag``.
            max_lag: Largest lead tested, in ``window`` periods.
            p_threshold: Bonferroni-corrected p-value at or below which a
                candidate is marked ``significant``.
            trigger_uuid: Identifier of the triggering signal, for the audit
                trail. Defaults to the ``source_uuid`` carried by
                ``trigger_events`` when it has one.
            trigger_event_type: Label of the triggering event. Defaults to the
                ``uuid`` carried by ``trigger_events`` when it has one.
            only_significant: Drop candidates above ``p_threshold``.

        Returns:
            Static-shape DataFrame, ranked by absolute correlation:
            ``uuid``, ``trigger_event_type``, ``trigger_uuid``,
            ``candidate_uuid``, ``correlation_coefficient``, ``lag_seconds``,
            ``p_value`` (raw, at the winning lag), ``p_value_adjusted``
            (Bonferroni over the lag scan), ``significant``, ``sample_count``,
            ``trigger_count``, ``rank``, ``method``.
        """
        cols = [
            "uuid",
            "trigger_event_type",
            "trigger_uuid",
            "candidate_uuid",
            "correlation_coefficient",
            "lag_seconds",
            "p_value",
            "p_value_adjusted",
            "significant",
            "sample_count",
            "trigger_count",
            "rank",
            "method",
        ]
        if max_lag < 0:
            raise ValueError(f"max_lag must be non-negative, got {max_lag}")

        onsets, aligned = self._grid(trigger_events, candidate_uuids, window=window)
        if not aligned:
            return pd.DataFrame(columns=cols)

        bin_seconds = pd.to_timedelta(window).total_seconds()
        event_type = trigger_event_type or self._frame_scalar(trigger_events, "uuid")
        source = trigger_uuid or self._frame_scalar(trigger_events, "source_uuid")
        trigger_count = int(onsets.sum())

        # Bonferroni over the lags scanned; keeping the strongest of many lags
        # is a multiple comparison, so the winning lag's raw p-value overstates
        # the evidence by roughly the number of lags tried.
        n_lags = max_lag + 1

        rows: list[dict[str, Any]] = []
        for uuid, series in aligned.items():
            r, p, lag_bins, n = self._best_lag(series, onsets, max_lag=max_lag)
            if np.isnan(r):
                continue
            p_adjusted = min(1.0, p * n_lags)
            rows.append(
                {
                    "uuid": self.event_uuid,
                    "trigger_event_type": event_type,
                    "trigger_uuid": source,
                    "candidate_uuid": uuid,
                    "correlation_coefficient": float(r),
                    "lag_seconds": lag_bins * bin_seconds,
                    "p_value": float(p),
                    "p_value_adjusted": float(p_adjusted),
                    "significant": bool(p_adjusted <= p_threshold),
                    "sample_count": n,
                    "trigger_count": trigger_count,
                    "method": "lagged_onset_correlation",
                }
            )
        if not rows:
            return pd.DataFrame(columns=cols)

        out = pd.DataFrame(rows)
        if only_significant:
            out = out[out["significant"]]
            if out.empty:
                return pd.DataFrame(columns=cols)
        out["_abs"] = out["correlation_coefficient"].abs()
        out = out.sort_values(["_abs", "candidate_uuid"], ascending=[False, True]).drop(
            columns="_abs"
        )
        out = out.reset_index(drop=True)
        out["rank"] = range(1, len(out) + 1)
        return out[cols]

    @renamed_kwargs(resample="window")
    def top_cause(
        self,
        trigger_events: pd.DataFrame,
        candidate_uuids: list[str],
        *,
        window: str = "1min",
        max_lag: int = 30,
        p_threshold: float = 0.05,
        trigger_uuid: str | None = None,
        trigger_event_type: str | None = None,
    ) -> pd.DataFrame:
        """Name the best-supported cause, or report that none qualifies.

        Only significant candidates are eligible: when no candidate clears
        ``p_threshold`` the row still comes back — with ``top_cause_uuid`` set
        to ``None`` and the tested counts filled in — so a search that found
        nothing is visible rather than silently empty.

        Args:
            trigger_events: Event frame whose onsets are being explained.
            candidate_uuids: Signals to test as possible causes.
            window: Grid resolution (offset string); also the unit of ``max_lag``.
            max_lag: Largest lead tested, in ``window`` periods.
            p_threshold: Significance level for eligibility.
            trigger_uuid: See :meth:`rank_causes`.
            trigger_event_type: See :meth:`rank_causes`.

        Returns:
            Static-shape DataFrame with a single row: ``uuid``,
            ``trigger_event_type``, ``trigger_uuid``, ``top_cause_uuid``,
            ``top_cause_lag_seconds``, ``top_cause_correlation``, ``p_value``,
            ``p_value_adjusted``, ``confidence``, ``candidates_tested``,
            ``candidates_significant``, ``trigger_count``, ``method``.
        """
        cols = [
            "uuid",
            "trigger_event_type",
            "trigger_uuid",
            "top_cause_uuid",
            "top_cause_lag_seconds",
            "top_cause_correlation",
            "p_value",
            "p_value_adjusted",
            "confidence",
            "candidates_tested",
            "candidates_significant",
            "trigger_count",
            "method",
        ]
        ranked = self.rank_causes(
            trigger_events,
            candidate_uuids,
            window=window,
            max_lag=max_lag,
            p_threshold=p_threshold,
            trigger_uuid=trigger_uuid,
            trigger_event_type=trigger_event_type,
        )
        if ranked.empty:
            return pd.DataFrame(columns=cols)

        significant = ranked[ranked["significant"]]
        best = significant.iloc[0] if not significant.empty else None
        return pd.DataFrame(
            [
                {
                    "uuid": self.event_uuid,
                    "trigger_event_type": ranked["trigger_event_type"].iloc[0],
                    "trigger_uuid": ranked["trigger_uuid"].iloc[0],
                    "top_cause_uuid": None if best is None else best["candidate_uuid"],
                    "top_cause_lag_seconds": (
                        float("nan") if best is None else best["lag_seconds"]
                    ),
                    "top_cause_correlation": (
                        float("nan")
                        if best is None
                        else best["correlation_coefficient"]
                    ),
                    "p_value": float("nan") if best is None else best["p_value"],
                    "p_value_adjusted": (
                        float("nan") if best is None else best["p_value_adjusted"]
                    ),
                    "confidence": (
                        0.0 if best is None else abs(best["correlation_coefficient"])
                    ),
                    "candidates_tested": len(ranked),
                    "candidates_significant": len(significant),
                    "trigger_count": int(ranked["trigger_count"].iloc[0]),
                    "method": "lagged_onset_correlation",
                }
            ],
            columns=cols,
        )

    @renamed_kwargs(resample="window")
    def lag_profile(
        self,
        trigger_events: pd.DataFrame,
        candidate_uuid: str,
        *,
        window: str = "1min",
        max_lag: int = 30,
    ) -> pd.DataFrame:
        """Correlation against the trigger onsets at every tested lag.

        The curve is the diagnostic behind :meth:`rank_causes`: a genuine
        cause peaks at one lead time and falls away, while a spurious match
        tends to wander without a clear peak. ``p_value`` here is the raw
        per-lag value — the whole point of the profile is to see every lag, so
        no multiple-comparison correction is applied; use
        :meth:`rank_causes` for the corrected verdict.

        Args:
            trigger_events: Event frame whose onsets are being explained.
            candidate_uuid: The signal to profile.
            window: Grid resolution (offset string); also the unit of ``max_lag``.
            max_lag: Largest lead tested, in ``window`` periods.

        Returns:
            Static-shape DataFrame, one row per lag: ``uuid``,
            ``candidate_uuid``, ``lag_seconds``, ``correlation``,
            ``p_value``, ``sample_count``, ``is_best_lag``, ``method``.
        """
        cols = [
            "uuid",
            "candidate_uuid",
            "lag_seconds",
            "correlation",
            "p_value",
            "sample_count",
            "is_best_lag",
            "method",
        ]
        if max_lag < 0:
            raise ValueError(f"max_lag must be non-negative, got {max_lag}")

        onsets, aligned = self._grid(trigger_events, [candidate_uuid], window=window)
        if not aligned:
            return pd.DataFrame(columns=cols)

        bin_seconds = pd.to_timedelta(window).total_seconds()
        candidate = aligned[candidate_uuid].to_numpy(dtype=float)
        marks = onsets.to_numpy(dtype=float)

        rows: list[dict[str, Any]] = []
        for lag in range(max_lag + 1):
            r, p, n = self._correlate_at_lag(candidate, marks, lag)
            if np.isnan(r):
                continue
            rows.append(
                {
                    "uuid": self.event_uuid,
                    "candidate_uuid": candidate_uuid,
                    "lag_seconds": lag * bin_seconds,
                    "correlation": float(r),
                    "p_value": float(p),
                    "sample_count": n,
                    "method": "lagged_onset_correlation",
                }
            )
        if not rows:
            return pd.DataFrame(columns=cols)

        out = pd.DataFrame(rows)
        out["is_best_lag"] = False
        out.loc[out["correlation"].abs().idxmax(), "is_best_lag"] = True
        return out[cols]

    # ------------------------------------------------------------------

    @staticmethod
    def _frame_scalar(frame: pd.DataFrame, column: str) -> str | None:
        """First non-null value of ``column``, or ``None`` when unavailable."""
        if column not in frame.columns or frame.empty:
            return None
        values = frame[column].dropna()
        return str(values.iloc[0]) if not values.empty else None
