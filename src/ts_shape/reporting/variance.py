"""Explain a change in a fleet number: who moved, and why the average moved.

A global KPI moved two points. The review needs to know two different things:

* **Which sites moved it.** :meth:`VarianceBridge.facility_bridge` splits the
  fleet delta into per-facility contributions -- the data behind a waterfall.
* **Whether performance or mix moved it.** A fleet-weighted rate can fall while
  every single plant improves, purely because volume shifted to a weaker site.
  :meth:`VarianceBridge.rate_bridge` separates the *rate effect* (sites got
  better or worse) from the *mix effect* (the weighting changed) and the
  interaction between them. The three add up exactly to the observed change.

Plan comparisons use the same machinery:
:meth:`VarianceBridge.plan_vs_actual` measures each site against its own target
rather than against the fleet.

Example::

    from ts_shape.reporting import VarianceBridge

    bridge = VarianceBridge(kpi_df)
    bridge.facility_bridge("parts_produced", base="2026-01", compare="2026-02")
    bridge.rate_bridge("oee_pct", weight_column="planned_minutes",
                       base="2026-01", compare="2026-02")
    bridge.plan_vs_actual("parts_produced", targets={"DE-STR": 120_000})
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from ts_shape.reporting.aggregation import MetricSpec, aggregate_metrics
from ts_shape.reporting.facility import FACILITY_COLUMN

__all__ = ["VarianceBridge"]

logger = logging.getLogger(__name__)


class VarianceBridge:
    """Decompose changes in a consolidated metric across facilities and time.

    Args:
        dataframe: Long KPI frame with a facility column and a period column.
        facility_column: Column holding the site id.
        period_column: Column holding the reporting period.

    Raises:
        ValueError: If either key column is missing.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str = FACILITY_COLUMN,
        period_column: str = "period",
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(dataframe).__name__}"
            )
        for column in (facility_column, period_column):
            if column not in dataframe.columns:
                raise ValueError(
                    f"Column {column!r} not found. "
                    f"Available: {list(dataframe.columns)}"
                )
        self.dataframe = dataframe.copy()
        self.facility_column = facility_column
        self.period_column = period_column

    def __repr__(self) -> str:
        return (
            f"VarianceBridge(rows={len(self.dataframe)}, "
            f"periods={self.dataframe[self.period_column].nunique()})"
        )

    # -- internals -------------------------------------------------------

    @property
    def periods(self) -> list[object]:
        """Distinct periods in the frame, in sorted order."""
        return sorted(self.dataframe[self.period_column].dropna().unique())

    def _resolve_periods(
        self, base: object | None, compare: object | None
    ) -> tuple[object, object]:
        """Default to the first and last period present."""
        available = self.periods
        if len(available) < 2 and (base is None or compare is None):
            raise ValueError(
                "Need at least two periods to build a bridge; the frame has "
                f"{len(available)}."
            )
        resolved_base = available[0] if base is None else base
        resolved_compare = available[-1] if compare is None else compare
        for period in (resolved_base, resolved_compare):
            if period not in available:
                raise ValueError(f"Period {period!r} not found. Available: {available}")
        if resolved_base == resolved_compare:
            raise ValueError(
                f"base and compare are the same period ({resolved_base!r})."
            )
        return resolved_base, resolved_compare

    def _period_values(self, period: object, specs: list[MetricSpec]) -> pd.DataFrame:
        subset = self.dataframe[self.dataframe[self.period_column] == period]
        return aggregate_metrics(
            subset,
            by=[self.facility_column],
            specs=specs,
            include_rows=False,
        )

    # -- bridges ---------------------------------------------------------

    def facility_bridge(
        self,
        metric: str,
        *,
        base: object | None = None,
        compare: object | None = None,
        spec: MetricSpec | None = None,
    ) -> pd.DataFrame:
        """Split a fleet-level change into per-facility contributions.

        Only meaningful for additive metrics -- the fleet figure must be the
        sum of the site figures for the contributions to add up. Use
        :meth:`rate_bridge` for rates.

        Args:
            metric: Metric column.
            base: Baseline period. Defaults to the earliest present.
            compare: Comparison period. Defaults to the latest present.
            spec: Aggregation used within each period per facility. Defaults to
                a sum.

        Returns:
            One row per facility with ``base``, ``compare``, ``delta``,
            ``delta_pct``, ``contribution_pct`` (share of the total fleet
            change, signed) and ``direction`` (``"up"`` / ``"down"`` /
            ``"flat"``), largest absolute mover first. A site present in only
            one period reports the other side as 0 and is flagged in
            ``status`` as ``"entered"`` or ``"exited"``.

        Raises:
            ValueError: If the metric is missing or the periods cannot be
                resolved.
        """
        base_period, compare_period = self._resolve_periods(base, compare)
        used = spec or MetricSpec.sum(metric)
        specs = [MetricSpec(**{**used.__dict__, "output": "value"})]

        left = self._period_values(base_period, specs).rename(columns={"value": "base"})
        right = self._period_values(compare_period, specs).rename(
            columns={"value": "compare"}
        )
        merged = left.merge(right, on=self.facility_column, how="outer")

        merged["status"] = "present"
        merged.loc[merged["base"].isna(), "status"] = "entered"
        merged.loc[merged["compare"].isna(), "status"] = "exited"
        merged[["base", "compare"]] = merged[["base", "compare"]].fillna(0.0)

        merged["delta"] = merged["compare"] - merged["base"]
        merged["delta_pct"] = merged.apply(
            lambda row: (
                float("nan") if row["base"] == 0 else row["delta"] / row["base"] * 100.0
            ),
            axis=1,
        )
        total_delta = float(merged["delta"].sum())
        merged["contribution_pct"] = (
            merged["delta"] / total_delta * 100.0 if total_delta != 0 else float("nan")
        )
        merged["direction"] = merged["delta"].map(
            lambda d: "up" if d > 0 else ("down" if d < 0 else "flat")
        )
        merged["base_period"] = base_period
        merged["compare_period"] = compare_period

        return merged.reindex(
            merged["delta"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

    def rate_bridge(
        self,
        rate_metric: str,
        *,
        weight_column: str,
        base: object | None = None,
        compare: object | None = None,
    ) -> pd.DataFrame:
        """Split a change in a weighted fleet rate into rate, mix and interaction.

        With shares ``s`` (a facility's weight over the fleet weight) and rates
        ``r``, the fleet rate is ``R = sum(s * r)`` and the change decomposes
        exactly::

            R1 - R0 = sum(s0 * (r1 - r0))     # rate effect: sites improved
                    + sum((s1 - s0) * r0)     # mix effect: weight moved
                    + sum((s1 - s0) * (r1 - r0))   # interaction

        Args:
            rate_metric: The rate column (e.g. ``"oee_pct"``).
            weight_column: The exposure the rate is weighted by (e.g.
                ``"planned_minutes"``). Summed per facility within each period.
            base: Baseline period. Defaults to the earliest present.
            compare: Comparison period. Defaults to the latest present.

        Returns:
            One row per facility plus a final ``TOTAL`` row, with
            ``base_rate``, ``compare_rate``, ``base_share``, ``compare_share``,
            ``rate_effect``, ``mix_effect``, ``interaction`` and ``total_effect``.
            The ``TOTAL`` row's ``total_effect`` equals the observed change in
            the fleet rate; its ``base_rate`` and ``compare_rate`` are the
            weighted fleet rates themselves.

            A facility with no exposure in one of the periods has no defined
            rate there, so its rate is carried across from the other period:
            an entering or exiting site lands entirely in ``mix_effect``, which
            is what it is -- a change in the weighting, not in performance.

        Raises:
            ValueError: If a column is missing or the periods cannot be
                resolved.
        """
        for column in (rate_metric, weight_column):
            if column not in self.dataframe.columns:
                raise ValueError(
                    f"Column {column!r} not found. "
                    f"Available: {list(self.dataframe.columns)}"
                )
        base_period, compare_period = self._resolve_periods(base, compare)
        specs = [
            MetricSpec.weighted(rate_metric, weight=weight_column, output="rate"),
            MetricSpec.sum(weight_column, output="weight"),
        ]
        left = self._period_values(base_period, specs).rename(
            columns={"rate": "base_rate", "weight": "base_weight"}
        )
        right = self._period_values(compare_period, specs).rename(
            columns={"rate": "compare_rate", "weight": "compare_weight"}
        )
        merged = left.merge(right, on=self.facility_column, how="outer")
        merged[["base_weight", "compare_weight"]] = merged[
            ["base_weight", "compare_weight"]
        ].fillna(0.0)
        # A facility with no exposure in a period has no defined rate there.
        # Carrying the other period's rate keeps its entry or exit out of the
        # rate effect, where it would masquerade as a performance change.
        base_undefined = (merged["base_weight"] <= 0) | merged["base_rate"].isna()
        compare_undefined = (merged["compare_weight"] <= 0) | merged[
            "compare_rate"
        ].isna()
        merged.loc[base_undefined, "base_rate"] = merged.loc[
            base_undefined, "compare_rate"
        ]
        merged.loc[compare_undefined, "compare_rate"] = merged.loc[
            compare_undefined, "base_rate"
        ]
        merged[["base_rate", "compare_rate"]] = merged[
            ["base_rate", "compare_rate"]
        ].fillna(0.0)

        base_total = float(merged["base_weight"].sum())
        compare_total = float(merged["compare_weight"].sum())
        if base_total == 0 or compare_total == 0:
            raise ValueError(
                f"Total {weight_column!r} is zero in "
                f"{'base' if base_total == 0 else 'compare'} period; a weighted "
                "rate bridge is undefined."
            )
        merged["base_share"] = merged["base_weight"] / base_total
        merged["compare_share"] = merged["compare_weight"] / compare_total

        rate_delta = merged["compare_rate"] - merged["base_rate"]
        share_delta = merged["compare_share"] - merged["base_share"]
        merged["rate_effect"] = merged["base_share"] * rate_delta
        merged["mix_effect"] = share_delta * merged["base_rate"]
        merged["interaction"] = share_delta * rate_delta
        merged["total_effect"] = (
            merged["rate_effect"] + merged["mix_effect"] + merged["interaction"]
        )

        base_rate = float((merged["base_share"] * merged["base_rate"]).sum())
        compare_rate = float((merged["compare_share"] * merged["compare_rate"]).sum())
        total_row = {
            self.facility_column: "TOTAL",
            "base_rate": base_rate,
            "compare_rate": compare_rate,
            "base_weight": base_total,
            "compare_weight": compare_total,
            "base_share": 1.0,
            "compare_share": 1.0,
            "rate_effect": float(merged["rate_effect"].sum()),
            "mix_effect": float(merged["mix_effect"].sum()),
            "interaction": float(merged["interaction"].sum()),
            "total_effect": compare_rate - base_rate,
        }
        ordered = merged.reindex(
            merged["total_effect"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
        result = pd.concat([ordered, pd.DataFrame([total_row])], ignore_index=True)
        result["base_period"] = base_period
        result["compare_period"] = compare_period
        return result

    # -- plan comparison -------------------------------------------------

    def plan_vs_actual(
        self,
        metric: str,
        *,
        targets: Mapping[str, float] | float,
        period: object | None = None,
        higher_is_better: bool = True,
        spec: MetricSpec | None = None,
        tolerance_pct: float = 5.0,
    ) -> pd.DataFrame:
        """Compare each facility against its own plan.

        Args:
            metric: Metric column.
            targets: ``{facility: target}``, or one number applied to every
                facility.
            period: Restrict to one period. Defaults to every period in the
                frame, aggregated per facility.
            higher_is_better: Direction of goodness.
            spec: Aggregation per facility. Defaults to a sum.
            tolerance_pct: How far short of plan still counts as ``"amber"``
                rather than ``"red"``.

        Returns:
            One row per facility with ``actual``, ``target``, ``variance``
            (signed so positive is always good), ``variance_pct``,
            ``attainment_pct`` and ``status`` (``"green"`` / ``"amber"`` /
            ``"red"``; ``"no_target"`` when the site has no plan). Worst first.

        Raises:
            ValueError: If the metric or period is missing.
        """
        frame = self.dataframe
        if period is not None:
            if period not in self.periods:
                raise ValueError(
                    f"Period {period!r} not found. Available: {self.periods}"
                )
            frame = frame[frame[self.period_column] == period]
        if metric not in frame.columns:
            raise ValueError(
                f"Metric column {metric!r} not found. "
                f"Available: {list(self.dataframe.columns)}"
            )

        used = spec or MetricSpec.sum(metric)
        actual = aggregate_metrics(
            frame,
            by=[self.facility_column],
            specs=[MetricSpec(**{**used.__dict__, "output": "actual"})],
            include_rows=False,
        )

        if isinstance(targets, Mapping):
            actual["target"] = (
                actual[self.facility_column].astype(str).map(dict(targets))
            )
        else:
            actual["target"] = float(targets)
        actual["target"] = pd.to_numeric(actual["target"], errors="coerce")

        sign = 1.0 if higher_is_better else -1.0
        actual["variance"] = (actual["actual"] - actual["target"]) * sign
        actual["variance_pct"] = actual.apply(
            lambda row: (
                float("nan")
                if not row["target"]
                else row["variance"] / abs(row["target"]) * 100.0
            ),
            axis=1,
        )
        actual["attainment_pct"] = actual.apply(
            lambda row: (
                float("nan")
                if not row["target"]
                else (
                    row["actual"] / row["target"] * 100.0
                    if higher_is_better
                    else (
                        row["target"] / row["actual"] * 100.0
                        if row["actual"]
                        else float("nan")
                    )
                )
            ),
            axis=1,
        )
        actual["status"] = [
            _status(variance, variance_pct, tolerance_pct, target)
            for variance, variance_pct, target in zip(
                actual["variance"], actual["variance_pct"], actual["target"]
            )
        ]
        if period is not None:
            actual["period"] = period
        return actual.sort_values(
            "variance", ascending=True, na_position="first"
        ).reset_index(drop=True)

    def period_over_period(
        self,
        metric: str,
        *,
        spec: MetricSpec | None = None,
    ) -> pd.DataFrame:
        """Fleet-level change from each period to the next.

        Returns:
            One row per period with ``value``, ``previous``, ``delta``,
            ``delta_pct`` and ``contributors``. The first period has NaN
            deltas.
        """
        used = spec or MetricSpec.sum(metric)
        by_period = aggregate_metrics(
            self.dataframe,
            by=[self.period_column],
            specs=[MetricSpec(**{**used.__dict__, "output": "value"})],
            contributor_column=self.facility_column,
            include_rows=False,
        ).sort_values(self.period_column)
        by_period["previous"] = by_period["value"].shift(1)
        by_period["delta"] = by_period["value"] - by_period["previous"]
        by_period["delta_pct"] = by_period.apply(
            lambda row: (
                float("nan")
                if not row["previous"]
                else row["delta"] / abs(row["previous"]) * 100.0
            ),
            axis=1,
        )
        return by_period.reset_index(drop=True)


def _status(
    variance: float, variance_pct: float, tolerance_pct: float, target: float
) -> str:
    """RAG status for a plan variance."""
    if pd.isna(target):
        return "no_target"
    if pd.isna(variance):
        return "no_data"
    if variance >= 0:
        return "green"
    if not pd.isna(variance_pct) and variance_pct >= -abs(tolerance_pct):
        return "amber"
    return "red"
