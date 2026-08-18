"""Consolidate per-facility KPIs into regional and global figures.

:class:`FleetRollup` is the reporting layer's workhorse: it takes a long KPI
frame -- one row per facility and period, as produced by the production, energy
and quality packs -- and aggregates it up the reporting dimensions with the
weighting semantics declared in
:class:`~ts_shape.reporting.aggregation.MetricSpec`.

It deliberately never *invents* a dimension: region and country come from the
:class:`~ts_shape.reporting.facility.FacilityRegistry`, deeper structure from an
:class:`~ts_shape.reporting.hierarchy.AssetHierarchy`. Every aggregate carries
``contributors`` -- how many facilities are behind the number -- so a global
figure that silently lost a site is visible in the output rather than in a
year-end reconciliation.

Example::

    from ts_shape.reporting import FleetRollup, MetricSpec

    rollup = FleetRollup(kpi_df, registry=registry)
    specs = [
        MetricSpec.sum("parts_produced"),
        MetricSpec.ratio("quality_pct", numerator="good", denominator="total",
                         scale=100.0),
        MetricSpec.weighted("oee_pct", weight="planned_minutes"),
    ]

    rollup.by_facility(specs)                 # per site, across all periods
    rollup.by_period(specs)                   # fleet total per period
    rollup.by(["region"], specs)              # per region
    rollup.consolidate(specs)                 # global + region + facility, stacked
    rollup.contribution("parts_produced")     # Pareto of sites by output
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from ts_shape.reporting.aggregation import MetricSpec, aggregate_metrics, infer_specs
from ts_shape.reporting.facility import FACILITY_COLUMN, FacilityRegistry
from ts_shape.reporting.hierarchy import AssetHierarchy

__all__ = ["FleetRollup"]

logger = logging.getLogger(__name__)

#: Label used for the single row that represents the whole fleet.
GLOBAL_NODE = "GLOBAL"


class FleetRollup:
    """Roll a per-facility KPI frame up to region and global level.

    Args:
        dataframe: Long KPI frame -- one row per facility (and period).
        facility_column: Column holding the site id.
        period_column: Column holding the reporting period. May be absent from
            the frame; period-aware methods then raise a clear error.
        registry: Facility master data, used to attach ``region`` / ``country``
            / ``currency`` when the frame does not already carry them.
        hierarchy: Optional asset hierarchy for levels below the site.

    Raises:
        ValueError: If ``facility_column`` is not in ``dataframe``.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str = FACILITY_COLUMN,
        period_column: str = "period",
        registry: FacilityRegistry | None = None,
        hierarchy: AssetHierarchy | None = None,
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
        self.hierarchy = hierarchy

    def __repr__(self) -> str:
        facilities = self.dataframe[self.facility_column].nunique(dropna=True)
        return f"FleetRollup(rows={len(self.dataframe)}, facilities={facilities})"

    # -- internals -------------------------------------------------------

    def _specs(self, specs: Sequence[MetricSpec] | None) -> list[MetricSpec]:
        if specs is not None:
            if not specs:
                raise ValueError("specs must not be empty")
            return list(specs)
        inferred = infer_specs(
            self.dataframe,
            exclude=[self.facility_column, self.period_column],
        )
        if not inferred:
            raise ValueError(
                "No numeric metric columns found to aggregate; pass specs "
                "explicitly."
            )
        logger.debug("Inferred %d metric spec(s) from column names", len(inferred))
        return inferred

    def _with_dimensions(self, needed: Sequence[str]) -> pd.DataFrame:
        """Return the frame with any missing master-data dimensions attached."""
        missing = [c for c in needed if c not in self.dataframe.columns]
        if not missing:
            return self.dataframe
        frame = self.dataframe
        if self.hierarchy is not None:
            hierarchy_levels = [c for c in missing if c in self.hierarchy.levels]
            if hierarchy_levels:
                frame = self.hierarchy.enrich(
                    frame, on=self.facility_column, levels=hierarchy_levels
                )
                missing = [c for c in missing if c not in frame.columns]
        if missing and self.registry is not None:
            master = {"region", "country", "currency", "timezone", "name"}
            attachable = [c for c in missing if c in master]
            if attachable:
                frame = self.registry.attach(
                    frame,
                    facility_column=self.facility_column,
                    columns=attachable,
                )
                missing = [c for c in missing if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Dimension column(s) {missing} are not in the frame and cannot "
                "be derived. Add them to the frame, or construct FleetRollup "
                "with a FacilityRegistry / AssetHierarchy that provides them."
            )
        return frame

    def _require_period(self) -> None:
        if self.period_column not in self.dataframe.columns:
            raise ValueError(
                f"Period column {self.period_column!r} not found. "
                f"Available: {list(self.dataframe.columns)}"
            )

    # -- rollups ---------------------------------------------------------

    def by_facility(
        self,
        specs: Sequence[MetricSpec] | None = None,
        *,
        per_period: bool = False,
    ) -> pd.DataFrame:
        """Aggregate to one row per facility.

        Args:
            specs: Aggregation semantics. Inferred from column names when
                omitted (sums for quantities, means for rate-like names) --
                convenient for exploration, explicit specs for anything shipped.
            per_period: Keep the period dimension, giving one row per facility
                and period.

        Returns:
            One row per facility (and period), with a ``rows`` count.
        """
        by = [self.facility_column]
        if per_period:
            self._require_period()
            by.append(self.period_column)
        return aggregate_metrics(self.dataframe, by=by, specs=self._specs(specs))

    def by_period(self, specs: Sequence[MetricSpec] | None = None) -> pd.DataFrame:
        """Aggregate all facilities to one fleet row per period."""
        self._require_period()
        return aggregate_metrics(
            self.dataframe,
            by=[self.period_column],
            specs=self._specs(specs),
            contributor_column=self.facility_column,
        )

    def by(
        self,
        dimensions: Sequence[str],
        specs: Sequence[MetricSpec] | None = None,
        *,
        per_period: bool = False,
    ) -> pd.DataFrame:
        """Aggregate to arbitrary dimensions (``region``, ``country``, ...).

        Dimensions absent from the frame are resolved from the hierarchy and
        then the registry, so ``by(["region"])`` works on a frame that only
        carries facility ids.

        Args:
            dimensions: Grouping columns.
            specs: Aggregation semantics; inferred when omitted.
            per_period: Also group by the period column.

        Returns:
            One row per dimension combination, with a ``contributors`` count of
            distinct facilities.

        Raises:
            ValueError: If a dimension cannot be resolved.
        """
        if not dimensions:
            raise ValueError("by() requires at least one dimension")
        frame = self._with_dimensions(dimensions)
        group = list(dimensions)
        if per_period:
            self._require_period()
            group.append(self.period_column)
        return aggregate_metrics(
            frame,
            by=group,
            specs=self._specs(specs),
            contributor_column=self.facility_column,
        )

    def total(
        self,
        specs: Sequence[MetricSpec] | None = None,
        *,
        per_period: bool = False,
    ) -> pd.DataFrame:
        """The single global row (or one row per period).

        This is the number that goes on the first slide, and the one most
        exposed to a bad rollup: check its ``contributors`` against the
        expected site count before quoting it.
        """
        if per_period:
            return self.by_period(specs)
        return aggregate_metrics(
            self.dataframe,
            by=[],
            specs=self._specs(specs),
            contributor_column=self.facility_column,
        )

    def consolidate(
        self,
        specs: Sequence[MetricSpec] | None = None,
        *,
        levels: Sequence[str] = ("region", FACILITY_COLUMN),
        per_period: bool = False,
        include_total: bool = True,
    ) -> pd.DataFrame:
        """Stack global, intermediate and facility figures into one frame.

        The shape a management report actually consumes: the fleet number, the
        regional numbers and the site numbers in a single long frame that a
        pivot table or dashboard can drill through.

        Args:
            specs: Aggregation semantics; inferred when omitted.
            levels: Reporting levels, coarse to fine. The facility column may
                be included as the finest level.
            per_period: Keep the period dimension on every level.
            include_total: Prepend the global row.

        Returns:
            Long frame with ``level`` (the dimension name, or ``"global"``),
            ``node`` (the dimension value, or ``"GLOBAL"``), the period column
            when requested, one column per metric, ``contributors`` and
            ``rows``.
        """
        resolved = self._specs(specs)
        frames: list[pd.DataFrame] = []

        if include_total:
            total = self.total(resolved, per_period=per_period)
            total.insert(0, "node", GLOBAL_NODE)
            total.insert(0, "level", "global")
            frames.append(total)

        for level in levels:
            if level == self.facility_column:
                part = aggregate_metrics(
                    self.dataframe,
                    by=(
                        [self.facility_column, self.period_column]
                        if per_period
                        else [self.facility_column]
                    ),
                    specs=resolved,
                    contributor_column=self.facility_column,
                )
            else:
                part = self.by([level], resolved, per_period=per_period)
            part = part.rename(columns={level: "node"})
            part.insert(0, "level", level)
            frames.append(part)

        report = pd.concat(frames, ignore_index=True, sort=False)
        ordered = ["level", "node"]
        if per_period:
            ordered.append(self.period_column)
        rest = [c for c in report.columns if c not in ordered]
        return report[ordered + rest]

    # -- distribution ----------------------------------------------------

    def contribution(
        self,
        metric: str,
        *,
        by: str | None = None,
        spec: MetricSpec | None = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Rank contributors to a fleet total, with cumulative share.

        A Pareto view across sites: which plants make up 80 % of output, of
        scrap cost, of downtime. Only meaningful for additive metrics -- pass a
        ``spec`` if the metric needs different treatment.

        Args:
            metric: Metric column to distribute.
            by: Dimension to attribute the total to. Defaults to the facility
                column; ``"region"`` gives a regional Pareto.
            spec: Aggregation for the metric. Defaults to a plain sum.
            ascending: Rank smallest-first (e.g. to find laggards).

        Returns:
            DataFrame with the dimension, ``value``, ``share_pct``,
            ``cumulative_share_pct`` and ``rank``. Shares are NaN when the
            total is zero, rather than dividing by zero.

        Raises:
            ValueError: If the metric column is missing.
        """
        dimension = by or self.facility_column
        used = spec or MetricSpec.sum(metric)
        frame = self._with_dimensions([dimension])
        if metric not in frame.columns and used.agg != "ratio":
            raise ValueError(
                f"Metric column {metric!r} not found. Available: {list(frame.columns)}"
            )
        aggregated = aggregate_metrics(
            frame,
            by=[dimension],
            specs=[MetricSpec(**{**used.__dict__, "output": "value"})],
            contributor_column=self.facility_column,
            include_rows=False,
        )
        aggregated = aggregated.sort_values(
            "value", ascending=ascending, na_position="last"
        ).reset_index(drop=True)

        total = float(pd.to_numeric(aggregated["value"], errors="coerce").sum())
        if total == 0 or pd.isna(total):
            aggregated["share_pct"] = float("nan")
            aggregated["cumulative_share_pct"] = float("nan")
        else:
            aggregated["share_pct"] = aggregated["value"] / total * 100.0
            aggregated["cumulative_share_pct"] = aggregated["share_pct"].cumsum()
        aggregated["rank"] = range(1, len(aggregated) + 1)
        return aggregated

    def pivot(
        self,
        metric: str,
        *,
        index: str | None = None,
        columns: str | None = None,
        spec: MetricSpec | None = None,
    ) -> pd.DataFrame:
        """Facility-by-period matrix for one metric.

        The heat-map view: rows are sites, columns are periods, so a site that
        stopped reporting or fell off a cliff is visible at a glance.

        Args:
            metric: Metric column.
            index: Row dimension. Defaults to the facility column.
            columns: Column dimension. Defaults to the period column.
            spec: Aggregation for the metric. Defaults to a plain sum.

        Returns:
            A wide DataFrame indexed by ``index`` with one column per
            ``columns`` value. Missing combinations are NaN.
        """
        row_dim = index or self.facility_column
        col_dim = columns or self.period_column
        frame = self._with_dimensions([d for d in (row_dim, col_dim)])
        if col_dim not in frame.columns:
            raise ValueError(
                f"Column dimension {col_dim!r} not found. "
                f"Available: {list(frame.columns)}"
            )
        used = spec or MetricSpec.sum(metric)
        long = aggregate_metrics(
            frame,
            by=[row_dim, col_dim],
            specs=[MetricSpec(**{**used.__dict__, "output": "value"})],
            include_rows=False,
        )
        if long.empty:
            return pd.DataFrame()
        return long.pivot(index=row_dim, columns=col_dim, values="value")
