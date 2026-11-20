"""Pareto ranking of categorical values per time window.

A generic Pareto roll-up: group a long-format frame by a time window and a
category column (typically ``value_string`` — downtime reasons, alarm codes,
scrap codes, operator ids), then rank categories by count or by a summed
value column. For each window the output carries ``rank``,
``share_of_total_pct`` and ``cumulative_pct`` so the 80/20 cut is a filter,
not a hand calculation.

This is the "which reasons took the most hours this shift?" building block.
It composes with :class:`~ts_shape.events.production.DowntimeEvents`
(whose ``top_downtime_reasons`` is downtime-specific) and with the
loader / filter / event-log chain exactly like the other detectors.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)

_PARETO_COLS = [
    "start",
    "category",
    "total",
    "share_of_total_pct",
    "cumulative_pct",
    "rank",
]


class ParetoEvents(Base):
    """Rank categorical values per window by count or by a summed amount.

    The returned frame is long-format and Grafana-ready: one row per
    ``(start, category)`` with the category's share of the window and the
    cumulative share up to its rank, sorted highest-first within each
    window.

    Pipeline example::

        pareto = ParetoEvents(df).rank_by_count(
            category_column="value_string", window="1D", top_n=5
        )
        # → the five biggest reason codes per day with share_of_total_pct.

    Each UUID in the input represents one signal; the categorical values are
    read from ``category_column`` (default ``value_string``). The constructor
    inherits :class:`~ts_shape.utils.base.Base`, so the frame is sorted,
    uuids and timestamps are validated, and the usual signal-check warnings
    fire.
    """

    @keyword_only(2)
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
    ) -> None:
        """Keep a sorted copy of ``dataframe``.

        Args:
            dataframe: Long-format input (``systime``, ``uuid``, ``value_*``
                and the category column — by default ``value_string``).
            time_column: Timestamp column to window by. Default ``"systime"``.
        """
        super().__init__(dataframe, time_column=time_column)

    def rank_by_count(
        self,
        *,
        category_column: str = "value_string",
        time_column: str = "systime",
        window: str = "1D",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Count rows per ``(window, category)`` and rank by count.

        Args:
            category_column: The column holding the category label. Default
                ``"value_string"``.
            time_column: Time column to window by. Default ``"systime"``.
            window: Pandas offset string (``"1h"``, ``"1D"``, ``"MS"``).
                Default ``"1D"``.
            top_n: Keep only the top-N rows per window. ``None`` (default)
                keeps every category.

        Returns:
            A flat long-format DataFrame with columns ``start``, ``category``,
            ``total``, ``share_of_total_pct``, ``cumulative_pct``, ``rank``.
            Rows are sorted by ``(start, rank)``.
        """
        return self._pareto(
            category_column=category_column,
            time_column=time_column,
            window=window,
            top_n=top_n,
            how="count",
            value_column=None,
        )

    def rank_by_sum(
        self,
        *,
        category_column: str = "value_string",
        value_column: str = "value_double",
        time_column: str = "systime",
        window: str = "1D",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Sum ``value_column`` per ``(window, category)`` and rank by sum.

        Args:
            category_column: The column holding the category label. Default
                ``"value_string"``.
            value_column: The numeric column to sum. Default
                ``"value_double"``.
            time_column: Time column to window by. Default ``"systime"``.
            window: Pandas offset string. Default ``"1D"``.
            top_n: Keep only the top-N rows per window.

        Returns:
            A flat long-format DataFrame with the same columns as
            :meth:`rank_by_count`; ``total`` is the summed value.
        """
        return self._pareto(
            category_column=category_column,
            time_column=time_column,
            window=window,
            top_n=top_n,
            how="sum",
            value_column=value_column,
        )

    def _pareto(
        self,
        *,
        category_column: str,
        time_column: str,
        window: str,
        top_n: int | None,
        how: str,
        value_column: str | None,
    ) -> pd.DataFrame:
        where = f"ParetoEvents.rank_by_{how}"
        required = [time_column, category_column]
        if value_column is not None:
            required.append(value_column)
        require_columns(self.dataframe, required, where=where)
        if top_n is not None and top_n <= 0:
            raise ValueError(f"{where}: top_n must be positive, got {top_n}")

        df = self.dataframe
        if df.empty:
            tz = (
                getattr(df[time_column].dt, "tz", None)
                if time_column in df.columns
                and pd.api.types.is_datetime64_any_dtype(df[time_column].dtype)
                else None
            )
            return _empty_pareto(tz)

        time = df[time_column]
        # Window grouper; pandas picks calendar/tick behaviour per the
        # offset string, matching PowerIntegrator's convention.
        grouper = pd.Grouper(key=time_column, freq=window)
        categories = df[category_column]
        mask = categories.notna()
        if value_column is not None:
            mask &= df[value_column].notna()
        work = df.loc[mask, :]
        if work.empty:
            return _empty_pareto(time.dt.tz)

        if how == "count":
            totals: pd.DataFrame = (
                work.groupby([grouper, category_column], observed=True, dropna=False)
                .size()
                .reset_index(name="total")
            )
        else:
            assert value_column is not None
            totals = (
                work.groupby([grouper, category_column], observed=True, dropna=False)[
                    value_column
                ]
                .sum()
                .reset_index(name="total")
            )
        # A zero-weight row carries no Pareto information and would also give
        # a 0 % cumulative share — drop before ranking so the cut-off is
        # meaningful.
        totals = totals[totals["total"] > 0].copy()
        if totals.empty:
            return _empty_pareto(time.dt.tz)

        totals = totals.rename(
            columns={time_column: "start", category_column: "category"}
        )
        totals["category"] = totals["category"].astype(object)
        # Rank highest-first per window, then compute share and cumulative
        # share against the window's full total (before top_n trimming).
        totals = totals.sort_values(
            ["start", "total"], ascending=[True, False], kind="stable"
        ).reset_index(drop=True)
        window_total = totals.groupby("start", observed=True)["total"].transform("sum")
        totals["share_of_total_pct"] = (totals["total"] / window_total * 100.0).astype(
            "float64"
        )
        totals["cumulative_pct"] = totals.groupby("start", observed=True)[
            "share_of_total_pct"
        ].cumsum()
        totals["rank"] = (
            totals.groupby("start", observed=True).cumcount().add(1).astype("int64")
        )
        if top_n is not None:
            totals = totals[totals["rank"] <= top_n].reset_index(drop=True)
        return totals[_PARETO_COLS]


def _empty_pareto(tz: Any) -> pd.DataFrame:
    """Typed, empty frame with the canonical Pareto columns."""
    dtype = f"datetime64[ns, {tz}]" if tz is not None else "datetime64[ns]"
    return pd.DataFrame(
        {
            "start": pd.Series(dtype=dtype),
            "category": pd.Series(dtype=object),
            "total": pd.Series(dtype="float64"),
            "share_of_total_pct": pd.Series(dtype="float64"),
            "cumulative_pct": pd.Series(dtype="float64"),
            "rank": pd.Series(dtype="int64"),
        }
    )
