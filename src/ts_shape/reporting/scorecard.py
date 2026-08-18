"""The management view: every site, every KPI, against target, in one frame.

A global operations review runs off one table -- sites down the side, KPIs
across the top, a status per cell and a single number per site. Building it by
hand is where directions get inverted (scrap "above target" is bad) and where a
site with no data quietly reads as green.

:class:`GlobalScorecard` builds it from a KPI frame and a list of
:class:`KpiTarget` declarations. Each target carries its own direction,
tolerance, weight and aggregation, so a scrap target and an OEE target are
evaluated correctly by the same call, and a missing value is reported as
``"no_data"`` rather than passing.

Example::

    from ts_shape.reporting import GlobalScorecard, KpiTarget, MetricSpec

    targets = [
        KpiTarget("oee_pct", 75.0,
                  spec=MetricSpec.weighted("oee_pct", weight="planned_minutes")),
        KpiTarget("scrap_rate_pct", 2.0, higher_is_better=False),
        KpiTarget("parts_produced", 100_000, spec=MetricSpec.sum("parts_produced")),
    ]

    board = GlobalScorecard(kpi_df, registry=registry)
    board.evaluate(targets)          # long: facility x metric
    board.facility_summary(targets)  # wide: one row per site
    board.fleet_summary(targets)     # one row per metric, fleet level
    board.trend(targets)             # fleet movement between two periods
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ts_shape.reporting.aggregation import MetricSpec, aggregate_metrics
from ts_shape.reporting.facility import FACILITY_COLUMN, FacilityRegistry

__all__ = ["GlobalScorecard", "KpiTarget"]

logger = logging.getLogger(__name__)

#: Status ordering, worst first -- used to pick a facility's headline status.
_STATUS_ORDER: tuple[str, ...] = ("red", "amber", "no_data", "green")


@dataclass(frozen=True)
class KpiTarget:
    """A target a facility is measured against.

    Args:
        metric: Metric column in the KPI frame.
        target: The value to meet.
        higher_is_better: ``False`` for scrap, downtime, cost, energy per part.
        tolerance_pct: How far the wrong side of target still counts as amber
            rather than red, as a percentage of the target.
        weight: Relative weight in the composite score.
        spec: How the metric is aggregated across the rows of one facility.
            Defaults to a plain mean; declare it explicitly (``MetricSpec.sum``
            for quantities, ``MetricSpec.weighted`` for rates) for anything
            that leaves the desk.
        label: Display name for reports. Defaults to ``metric``.

    Raises:
        ValueError: If ``tolerance_pct`` is negative or ``weight`` is not
            positive.
    """

    metric: str
    target: float
    higher_is_better: bool = True
    tolerance_pct: float = 5.0
    weight: float = 1.0
    spec: MetricSpec | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.tolerance_pct < 0:
            raise ValueError(
                f"KpiTarget({self.metric!r}) tolerance_pct must be >= 0, "
                f"got {self.tolerance_pct}"
            )
        if self.weight <= 0:
            raise ValueError(
                f"KpiTarget({self.metric!r}) weight must be > 0, got {self.weight}"
            )

    @property
    def name(self) -> str:
        """Display name."""
        return self.label or self.metric

    def aggregation(self) -> MetricSpec:
        """The :class:`MetricSpec` used to collapse rows for this metric."""
        return self.spec or MetricSpec.mean(self.metric)

    def status(self, value: float) -> str:
        """RAG status of ``value`` against this target."""
        if value is None or pd.isna(value):
            return "no_data"
        meets = value >= self.target if self.higher_is_better else value <= self.target
        if meets:
            return "green"
        if self.target == 0:
            return "red"
        shortfall_pct = abs(value - self.target) / abs(self.target) * 100.0
        return "amber" if shortfall_pct <= self.tolerance_pct else "red"

    def attainment(self, value: float) -> float:
        """Attainment as a percentage of target, direction-aware.

        For a higher-is-better metric this is ``value / target``; for a
        lower-is-better metric it is ``target / value``, so 100 % always means
        "on plan" and above 100 % always means "better than plan".
        """
        if value is None or pd.isna(value) or self.target == 0:
            return float("nan")
        if self.higher_is_better:
            return float(value) / float(self.target) * 100.0
        if value == 0:
            return float("inf")
        return float(self.target) / float(value) * 100.0


class GlobalScorecard:
    """Evaluate facilities and the fleet against a set of KPI targets.

    Args:
        dataframe: Long KPI frame, one row per facility (and period).
        facility_column: Column holding the site id.
        period_column: Column holding the period, when present. Required by
            :meth:`trend`.
        registry: Optional master data, used to attach ``region`` so a
            scorecard can be grouped regionally.

    Raises:
        ValueError: If ``facility_column`` is missing.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str = FACILITY_COLUMN,
        period_column: str | None = None,
        registry: FacilityRegistry | None = None,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(dataframe).__name__}"
            )
        if facility_column not in dataframe.columns:
            raise ValueError(
                f"Facility column {facility_column!r} not found. "
                f"Available: {list(dataframe.columns)}"
            )
        self.dataframe = dataframe.copy()
        self.facility_column = facility_column
        self.period_column = period_column
        self.registry = registry

    def __repr__(self) -> str:
        return (
            f"GlobalScorecard(rows={len(self.dataframe)}, "
            f"facilities={self.dataframe[self.facility_column].nunique()})"
        )

    # -- internals -------------------------------------------------------

    @staticmethod
    def _check(targets: Sequence[KpiTarget]) -> list[KpiTarget]:
        if not targets:
            raise ValueError("GlobalScorecard needs at least one KpiTarget")
        return list(targets)

    def _values(
        self, targets: Sequence[KpiTarget], by: Sequence[str], frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate every target's metric to the ``by`` grain."""
        specs: list[MetricSpec] = []
        for target in targets:
            if target.metric not in frame.columns:
                raise ValueError(
                    f"Metric column {target.metric!r} not found. "
                    f"Available: {list(frame.columns)}"
                )
            base = target.aggregation()
            specs.append(MetricSpec(**{**base.__dict__, "output": target.name}))
        return aggregate_metrics(
            frame,
            by=list(by),
            specs=specs,
            contributor_column=self.facility_column if by else None,
            include_rows=False,
        )

    # -- views -----------------------------------------------------------

    def evaluate(
        self,
        targets: Sequence[KpiTarget],
        *,
        by: str | None = None,
    ) -> pd.DataFrame:
        """Evaluate every facility (or dimension value) against every target.

        Args:
            targets: The KPI targets.
            by: Grouping dimension. Defaults to the facility column; pass
                ``"region"`` for a regional scorecard (the column is attached
                from the registry when the frame lacks it).

        Returns:
            Long frame with one row per (node, metric): ``metric``, ``value``,
            ``target``, ``direction``, ``attainment_pct``, ``gap`` (signed so
            positive is always good), ``status`` and ``weight``.

        Raises:
            ValueError: If a metric or dimension column cannot be resolved.
        """
        checked = self._check(targets)
        dimension = by or self.facility_column
        frame = self._with_dimension(dimension)
        wide = self._values(checked, [dimension], frame)

        rows: list[dict[str, object]] = []
        for record in wide.to_dict(orient="records"):
            for target in checked:
                value = record.get(target.name)
                numeric = (
                    float("nan") if value is None or pd.isna(value) else float(value)
                )
                sign = 1.0 if target.higher_is_better else -1.0
                rows.append(
                    {
                        dimension: record[dimension],
                        "metric": target.name,
                        "value": numeric,
                        "target": float(target.target),
                        "direction": (
                            "higher_is_better"
                            if target.higher_is_better
                            else "lower_is_better"
                        ),
                        "attainment_pct": target.attainment(numeric),
                        "gap": (numeric - target.target) * sign,
                        "status": target.status(numeric),
                        "weight": float(target.weight),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                dimension,
                "metric",
                "value",
                "target",
                "direction",
                "attainment_pct",
                "gap",
                "status",
                "weight",
            ],
        )

    def _with_dimension(self, dimension: str) -> pd.DataFrame:
        if dimension in self.dataframe.columns:
            return self.dataframe
        if self.registry is None:
            raise ValueError(
                f"Dimension {dimension!r} is not in the frame and no "
                "FacilityRegistry was supplied to derive it."
            )
        return self.registry.attach(
            self.dataframe,
            facility_column=self.facility_column,
            columns=[dimension],
        )

    def facility_summary(
        self,
        targets: Sequence[KpiTarget],
        *,
        by: str | None = None,
    ) -> pd.DataFrame:
        """One row per facility: value and status per KPI, plus a score.

        Returns:
            Wide frame with ``<metric>`` and ``<metric>_status`` per target,
            ``score_pct`` (weighted share of targets met, ignoring
            ``no_data``), ``targets_met``, ``targets_evaluated``,
            ``worst_status`` and ``rank`` (best score first).
        """
        checked = self._check(targets)
        dimension = by or self.facility_column
        long = self.evaluate(checked, by=by)
        if long.empty:
            return pd.DataFrame(
                columns=[
                    dimension,
                    "score_pct",
                    "targets_met",
                    "targets_evaluated",
                    "worst_status",
                    "rank",
                ]
            )

        values = long.pivot(index=dimension, columns="metric", values="value")
        statuses = long.pivot(index=dimension, columns="metric", values="status")
        statuses.columns = [f"{c}_status" for c in statuses.columns]

        summary = pd.concat([values, statuses], axis=1).reset_index()
        scored = long[long["status"] != "no_data"]
        weights = scored.groupby(dimension)["weight"].sum()
        met = (
            scored.assign(_met=(scored["status"] == "green") * scored["weight"])
            .groupby(dimension)["_met"]
            .sum()
        )
        summary["score_pct"] = summary[dimension].map((met / weights * 100.0).to_dict())
        summary["targets_met"] = (
            summary[dimension]
            .map(
                scored[scored["status"] == "green"].groupby(dimension).size().to_dict()
            )
            .fillna(0)
            .astype(int)
        )
        summary["targets_evaluated"] = (
            summary[dimension]
            .map(scored.groupby(dimension).size().to_dict())
            .fillna(0)
            .astype(int)
        )
        summary["worst_status"] = summary[dimension].map(
            long.groupby(dimension)["status"]
            .apply(lambda s: _worst_status(list(s)))
            .to_dict()
        )
        summary["rank"] = summary["score_pct"].rank(ascending=False, method="min")

        ordered = [
            dimension,
            *[t.name for t in checked],
            *[f"{t.name}_status" for t in checked],
            "score_pct",
            "targets_met",
            "targets_evaluated",
            "worst_status",
            "rank",
        ]
        present = [c for c in ordered if c in summary.columns]
        return (
            summary[present]
            .sort_values(["rank", dimension], na_position="last")
            .reset_index(drop=True)
        )

    def fleet_summary(self, targets: Sequence[KpiTarget]) -> pd.DataFrame:
        """Fleet-level value and status per KPI.

        The fleet value uses each target's own aggregation across *all* rows,
        so a weighted OEE target consolidates by its weight rather than by
        averaging site averages.

        Returns:
            One row per metric with ``metric``, ``value``, ``target``,
            ``attainment_pct``, ``gap``, ``status``, ``facilities_meeting``,
            ``facilities_evaluated`` and ``facilities_at_risk`` (amber or red).
        """
        checked = self._check(targets)
        fleet = self._values(checked, [], self.dataframe)
        per_site = self.evaluate(checked)

        rows: list[dict[str, object]] = []
        for target in checked:
            raw = fleet[target.name].iloc[0] if not fleet.empty else float("nan")
            value = float("nan") if pd.isna(raw) else float(raw)
            sites = per_site[per_site["metric"] == target.name]
            evaluated = sites[sites["status"] != "no_data"]
            sign = 1.0 if target.higher_is_better else -1.0
            rows.append(
                {
                    "metric": target.name,
                    "value": value,
                    "target": float(target.target),
                    "attainment_pct": target.attainment(value),
                    "gap": (value - target.target) * sign,
                    "status": target.status(value),
                    "facilities_meeting": int((evaluated["status"] == "green").sum()),
                    "facilities_evaluated": int(len(evaluated)),
                    "facilities_at_risk": int(
                        evaluated["status"].isin(["amber", "red"]).sum()
                    ),
                }
            )
        return pd.DataFrame(rows)

    def trend(
        self,
        targets: Sequence[KpiTarget],
        *,
        base: object | None = None,
        compare: object | None = None,
        period_column: str | None = None,
    ) -> pd.DataFrame:
        """Fleet movement on every KPI between two periods.

        Args:
            targets: The KPI targets.
            base: Baseline period. Defaults to the earliest present.
            compare: Comparison period. Defaults to the latest present.
            period_column: Period column; defaults to the instance's.

        Returns:
            One row per metric with ``base``, ``compare``, ``delta``,
            ``delta_pct``, ``movement`` (``"improving"`` / ``"declining"`` /
            ``"flat"``, direction-aware), ``target``, ``status_base`` and
            ``status_compare`` -- so a KPI that improved but is still red is
            distinguishable from one that slipped into red.

        Raises:
            ValueError: If no period column is available or the periods cannot
                be resolved.
        """
        checked = self._check(targets)
        column = period_column or self.period_column
        if column is None:
            raise ValueError(
                "trend needs a period column: construct GlobalScorecard with "
                "period_column=... or pass period_column=..."
            )
        if column not in self.dataframe.columns:
            raise ValueError(
                f"Period column {column!r} not found. "
                f"Available: {list(self.dataframe.columns)}"
            )
        available = sorted(self.dataframe[column].dropna().unique())
        if len(available) < 2 and (base is None or compare is None):
            raise ValueError(
                f"Need at least two periods for a trend; found {len(available)}."
            )
        base_period = available[0] if base is None else base
        compare_period = available[-1] if compare is None else compare
        for period in (base_period, compare_period):
            if period not in available:
                raise ValueError(f"Period {period!r} not found. Available: {available}")

        base_values = self._values(
            checked, [], self.dataframe[self.dataframe[column] == base_period]
        )
        compare_values = self._values(
            checked, [], self.dataframe[self.dataframe[column] == compare_period]
        )

        rows: list[dict[str, object]] = []
        for target in checked:
            start = _first_value(base_values, target.name)
            end = _first_value(compare_values, target.name)
            delta = (
                end - start if not (pd.isna(start) or pd.isna(end)) else float("nan")
            )
            signed = delta * (1.0 if target.higher_is_better else -1.0)
            rows.append(
                {
                    "metric": target.name,
                    "base_period": base_period,
                    "compare_period": compare_period,
                    "base": start,
                    "compare": end,
                    "delta": delta,
                    "delta_pct": (
                        float("nan")
                        if not start or pd.isna(delta)
                        else delta / abs(start) * 100.0
                    ),
                    "movement": (
                        "flat"
                        if pd.isna(signed) or signed == 0
                        else ("improving" if signed > 0 else "declining")
                    ),
                    "target": float(target.target),
                    "status_base": target.status(start),
                    "status_compare": target.status(end),
                }
            )
        return pd.DataFrame(rows)


def _worst_status(statuses: Sequence[str]) -> str:
    """The most severe status present, per :data:`_STATUS_ORDER`."""
    for status in _STATUS_ORDER:
        if status in statuses:
            return status
    return "no_data"


def _first_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    value = frame[column].iloc[0]
    return float("nan") if pd.isna(value) else float(value)
