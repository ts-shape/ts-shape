"""Outcome-driven ranking of candidate critical process parameters (CPPs).

During process development the question is rarely "is this signal in
spec?" -- it is "*which* of these signals drives the outcome we care
about?" This detector ranks candidate input parameters by their
statistical association with a per-run quality outcome (yield, pass/fail,
KPI), so the process development engineer can shortlist a small set of
true CPPs for tighter control.

Three association measures are supported, all from ``scipy.stats``:

* ``pearson``   -- linear correlation (best when both inputs and outcome
  are continuous and approximately linear).
* ``spearman``  -- rank correlation (robust to non-linear monotonic
  relationships and outliers).
* ``anova_f``   -- one-way ANOVA F-statistic between groups defined by
  discrete levels of the parameter (the right test when factors were
  swept at discrete levels in a DOE).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class CriticalParameterRankingEvents(Base):
    """Rank input parameters by their statistical link to a quality outcome.

    The detector operates on a *per-run* table where each row is a
    completed run (batch, DOE point, shift), each candidate column is the
    aggregated value of an input parameter during that run, and an
    outcome column holds the quality / yield measurement of interest.

    The constructor still accepts a long-form ``dataframe`` for symmetry
    with the rest of ts-shape, but :meth:`rank` requires the wide-format
    per-run table directly because aggregating "the right number" for
    each parameter is a development-engineer judgment call (mean of the
    hold phase? settled value? peak?) that varies by parameter.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "dev:cpp_ranking",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        per_run_df: pd.DataFrame,
        candidate_columns: list[str],
        outcome_column: str,
        *,
        method: str = "spearman",
        anova_bins: int = 3,
    ) -> pd.DataFrame:
        """Rank candidate parameters by statistical association with the outcome.

        Args:
            per_run_df: One row per run, one column per candidate
                parameter, plus an outcome column.
            candidate_columns: Columns to evaluate as candidate CPPs.
            outcome_column: Outcome (response) column.
            method: ``"pearson"``, ``"spearman"``, or ``"anova_f"``.
            anova_bins: Only used for ``method="anova_f"``. Each candidate
                column is quantile-binned into this many groups before the
                one-way ANOVA is run.

        Returns:
            Summary DataFrame: ``parameter``, ``method``, ``statistic``,
            ``p_value``, ``abs_effect_size``, ``rank``. Sorted by
            ``abs_effect_size`` descending so the top driver is the first
            row.
        """
        from scipy import stats  # local import: keeps top-level import cheap

        cols = ["parameter", "method", "statistic", "p_value", "abs_effect_size", "rank"]
        if outcome_column not in per_run_df.columns:
            raise ValueError(
                f"Outcome column {outcome_column!r} missing from per_run_df; "
                f"available: {list(per_run_df.columns)}"
            )
        if not candidate_columns:
            return pd.DataFrame(columns=cols)

        y_full = pd.to_numeric(per_run_df[outcome_column], errors="coerce")
        if y_full.dropna().empty:
            return pd.DataFrame(columns=cols)

        rows: list[dict[str, Any]] = []
        for col in candidate_columns:
            if col not in per_run_df.columns:
                logger.debug("Skipping unknown candidate column %s", col)
                continue
            x = pd.to_numeric(per_run_df[col], errors="coerce")
            xy = pd.concat([x, y_full], axis=1).dropna()
            if len(xy) < 3:
                continue
            x_arr = xy.iloc[:, 0].to_numpy(dtype=float)
            y_arr = xy.iloc[:, 1].to_numpy(dtype=float)

            stat: float
            pval: float
            effect: float

            if method == "pearson":
                res = stats.pearsonr(x_arr, y_arr)
                stat, pval = float(res.statistic), float(res.pvalue)
                effect = abs(stat)
            elif method == "spearman":
                res = stats.spearmanr(x_arr, y_arr)
                stat, pval = float(res.statistic), float(res.pvalue)
                effect = abs(stat)
            elif method == "anova_f":
                # Quantile-bin the candidate into `anova_bins` groups.
                try:
                    bins = pd.qcut(x_arr, q=anova_bins, labels=False, duplicates="drop")
                except ValueError:
                    continue
                groups = [y_arr[bins == b] for b in np.unique(bins) if (bins == b).any()]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) < 2:
                    continue
                res = stats.f_oneway(*groups)
                stat, pval = float(res.statistic), float(res.pvalue)
                # Effect size: eta-squared = SS_between / SS_total.
                grand = float(np.mean(np.concatenate(groups)))
                ss_between = sum(
                    len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups
                )
                ss_total = float(np.sum((y_arr - grand) ** 2))
                effect = ss_between / ss_total if ss_total > 0 else 0.0
            else:
                raise ValueError(
                    f"Unknown method {method!r}; "
                    f"choose 'pearson', 'spearman', or 'anova_f'."
                )

            rows.append(
                {
                    "parameter": col,
                    "method": method,
                    "statistic": stat,
                    "p_value": pval,
                    "abs_effect_size": effect,
                }
            )

        if not rows:
            return pd.DataFrame(columns=cols)

        out = (
            pd.DataFrame(rows)
            .sort_values("abs_effect_size", ascending=False)
            .reset_index(drop=True)
        )
        out["rank"] = np.arange(1, len(out) + 1)
        return out[cols]

    def top_drivers(
        self,
        per_run_df: pd.DataFrame,
        candidate_columns: list[str],
        outcome_column: str,
        *,
        method: str = "spearman",
        k: int = 5,
        alpha: float = 0.05,
        anova_bins: int = 3,
    ) -> pd.DataFrame:
        """Return the top-``k`` candidates with ``p_value <= alpha``.

        Wraps :meth:`rank` and filters by significance. The result keeps
        the same column schema as :meth:`rank`.
        """
        full = self.rank(
            per_run_df,
            candidate_columns,
            outcome_column,
            method=method,
            anova_bins=anova_bins,
        )
        if full.empty:
            return full
        return full[full["p_value"] <= alpha].head(k).reset_index(drop=True)
