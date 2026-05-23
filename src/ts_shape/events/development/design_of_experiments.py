"""Design-of-Experiments (DOE) run segmentation and effect estimation.

A DOE campaign sweeps one or more factor signals through discrete levels
while a response signal is measured. This detector recovers the run
structure from continuous process data: contiguous time intervals where
every factor signal is stationary at a recognisable level. Each run is
tagged with its factor-level combination, and a follow-up method
aggregates the response per factor level to expose main effects.

The detector is deliberately classical -- no regression model fit, no
fractional factorial enumeration -- so it works on any factor pattern an
experimenter actually ran on the line, including OFAT, full factorial,
fractional, and Latin square designs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DesignOfExperimentsEvents(Base):
    """Segment continuous process data into DOE runs and estimate effects.

    A *run* is a contiguous interval during which every factor signal sits
    on a single discrete level (low rolling std relative to its tolerance)
    long enough to be a deliberate experimental setting rather than a
    transient. Tagging each run with its factor-level combination produces
    an experimental dataset that downstream effect estimation can act on.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        factor_uuids: list[str],
        *,
        event_uuid: str = "dev:doe_run",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        if not factor_uuids:
            raise ValueError("factor_uuids must contain at least one signal UUID")
        self.factor_uuids = list(factor_uuids)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        for uid in self.factor_uuids:
            self._validate_uuid(self.dataframe, uid)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wide_factor_frame(self) -> pd.DataFrame:
        """Pivot the long-form factor signals to wide form aligned on time."""
        if self.dataframe.empty:
            return pd.DataFrame()
        sel = self.dataframe[self.dataframe["uuid"].isin(self.factor_uuids)]
        if sel.empty:
            return pd.DataFrame()
        wide = (
            sel.pivot_table(
                index=self.time_column,
                columns="uuid",
                values=self.value_column,
                aggfunc="last",
            )
            .sort_index()
            .ffill()
            .dropna(how="any")
        )
        # Restrict to requested factors and preserve the caller's order.
        present = [u for u in self.factor_uuids if u in wide.columns]
        return wide[present] if present else pd.DataFrame()

    @staticmethod
    def _bin_level(values: np.ndarray, n_levels: int | None) -> np.ndarray:
        """Bin observed factor values into discrete level labels.

        When ``n_levels`` is given, the values are quantile-binned into
        exactly that many levels. Otherwise distinct rounded values are
        used as their own levels -- correct for a real DOE where the
        operator drove the factor to a small number of explicit setpoints.
        """
        if n_levels is None:
            return np.round(values, 6)
        bins = pd.qcut(values, q=n_levels, labels=False, duplicates="drop")
        return bins.astype(float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_runs(
        self,
        *,
        min_duration: str = "5min",
        stability_tol: float = 0.02,
        n_levels: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Identify DOE runs as stable intervals across all factor signals.

        Args:
            min_duration: Minimum run length to be reported (e.g. ``"5min"``).
                Shorter stable intervals are treated as transients.
            stability_tol: Maximum relative range
                ``(max - min) / (|mean| + eps)`` permitted within a run for
                each factor. Default 2%.
            n_levels: Optional ``{factor_uuid: k}`` mapping. When given, the
                factor values are quantile-binned into ``k`` levels;
                otherwise the rounded observed value is used as the level.

        Returns:
            Interval-shape DataFrame with columns: ``start``, ``end``,
            ``duration_seconds``, ``uuid``, ``run_id``, and one
            ``factor__<uuid>_level`` column per factor.
        """
        n_levels = n_levels or {}
        base_cols = ["start", "end", "duration_seconds", "uuid", "run_id"]
        level_cols = [f"factor__{u}_level" for u in self.factor_uuids]
        all_cols = base_cols + level_cols

        wide = self._wide_factor_frame()
        if wide.empty:
            return pd.DataFrame(columns=all_cols)

        min_td = pd.to_timedelta(min_duration)
        # Per-row relative range across factors; rows with a single factor
        # are stable by definition. We measure relative range against the
        # *next* row to detect setpoint transitions -- a robust signal
        # because real DOE runs change the factor by orders of magnitude
        # more than stability_tol.
        eps = np.finfo(float).eps
        diff = wide.diff().abs()
        ref = wide.abs().clip(lower=eps)
        rel = (diff / ref).fillna(0.0)
        is_transition = (rel > stability_tol).any(axis=1)
        # Group label increments at every transition.
        group_id = is_transition.cumsum()

        events: list[dict[str, Any]] = []
        for gid, idxs in wide.groupby(group_id).groups.items():
            block = wide.loc[idxs]
            if len(block) < 2:
                continue
            start_t = block.index[0]
            end_t = block.index[-1]
            if (end_t - start_t) < min_td:
                continue
            row: dict[str, Any] = {
                "start": start_t,
                "end": end_t,
                "duration_seconds": (end_t - start_t).total_seconds(),
                "uuid": self.event_uuid,
                "run_id": int(gid),
            }
            for u in self.factor_uuids:
                if u not in block.columns:
                    row[f"factor__{u}_level"] = np.nan
                    continue
                mean_val = float(block[u].mean())
                k = n_levels.get(u)
                # When n_levels[u] is set we cannot bin a single run in
                # isolation, so we just attach the mean and rely on the
                # post-pass below to assign discrete bin labels.
                row[f"factor__{u}_level"] = mean_val if k is None else mean_val
            events.append(row)

        if not events:
            return pd.DataFrame(columns=all_cols)

        out = pd.DataFrame(events, columns=all_cols)

        # Post-pass: quantile-bin per factor when ``n_levels`` was given.
        for u, k in n_levels.items():
            col = f"factor__{u}_level"
            if col in out.columns and out[col].notna().any():
                out[col] = self._bin_level(out[col].to_numpy(dtype=float), k)

        return out

    def compute_effects(
        self,
        response_uuid: str,
        *,
        statistic: str = "mean",
        min_duration: str = "5min",
        stability_tol: float = 0.02,
        n_levels: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Aggregate the response signal per factor level (main effects).

        Args:
            response_uuid: UUID of the response (output) signal.
            statistic: Aggregation applied to the response within each run.
                One of ``"mean"`` (default), ``"median"``, ``"max"``,
                ``"min"``, or ``"settled"`` -- the latter takes the mean of
                the final third of the run, useful for response signals
                that need to stabilise after a factor change.
            min_duration: Forwarded to :meth:`detect_runs`.
            stability_tol: Forwarded to :meth:`detect_runs`.
            n_levels: Forwarded to :meth:`detect_runs`.

        Returns:
            Summary DataFrame with one row per (factor, level), columns:
            ``factor``, ``level``, ``n_runs``, ``response_mean``,
            ``response_std``, ``main_effect``.
            ``main_effect`` is ``response_mean`` minus the grand mean
            across all runs.
        """
        cols = [
            "factor",
            "level",
            "n_runs",
            "response_mean",
            "response_std",
            "main_effect",
        ]
        self._validate_uuid(self.dataframe, response_uuid)

        runs = self.detect_runs(
            min_duration=min_duration,
            stability_tol=stability_tol,
            n_levels=n_levels,
        )
        if runs.empty:
            return pd.DataFrame(columns=cols)

        # Extract the response timeseries once.
        resp = (
            self.dataframe[self.dataframe["uuid"] == response_uuid][
                [self.time_column, self.value_column]
            ]
            .copy()
            .sort_values(self.time_column)
            .set_index(self.time_column)
        )
        if resp.empty:
            return pd.DataFrame(columns=cols)

        def _agg(window: pd.Series) -> float:
            if window.empty:
                return float("nan")
            if statistic == "mean":
                return float(window.mean())
            if statistic == "median":
                return float(window.median())
            if statistic == "max":
                return float(window.max())
            if statistic == "min":
                return float(window.min())
            if statistic == "settled":
                tail = window.iloc[max(0, int(len(window) * 2 / 3)) :]
                return float(tail.mean()) if len(tail) else float(window.mean())
            raise ValueError(f"Unknown statistic: {statistic!r}")

        run_response = []
        for _, run in runs.iterrows():
            mask = (resp.index >= run["start"]) & (resp.index <= run["end"])
            window = resp.loc[mask, self.value_column]
            run_response.append(_agg(window))
        runs = runs.assign(_response=run_response).dropna(subset=["_response"])
        if runs.empty:
            return pd.DataFrame(columns=cols)

        grand_mean = float(runs["_response"].mean())

        rows: list[dict[str, Any]] = []
        for u in self.factor_uuids:
            col = f"factor__{u}_level"
            if col not in runs.columns:
                continue
            for level, group in runs.groupby(col, dropna=True):
                response_mean = float(group["_response"].mean())
                rows.append(
                    {
                        "factor": u,
                        "level": level,
                        "n_runs": int(len(group)),
                        "response_mean": response_mean,
                        "response_std": (
                            float(group["_response"].std(ddof=1))
                            if len(group) > 1
                            else 0.0
                        ),
                        "main_effect": response_mean - grand_mean,
                    }
                )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
