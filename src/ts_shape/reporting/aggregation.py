"""Weighted metric aggregation -- the arithmetic a fleet rollup depends on.

Summing parts across sites is easy. Rolling up *rates* is where consolidated
reports go wrong: the unweighted mean of three plants' OEE (95 % on a line that
ran 30 minutes, 60 % on two that ran three shifts) reports a fleet OEE that no
plant experienced and that no amount of drill-down explains.

A :class:`MetricSpec` declares, per column, how it survives aggregation:

* ``sum`` -- counts, quantities, minutes, costs.
* ``weighted_mean`` -- a rate weighted by its own exposure (OEE by planned
  minutes, scrap rate by parts produced).
* ``ratio`` -- a rate *recomputed* from its components
  (``sum(good) / sum(total)``), which is exact rather than approximated.
* ``mean`` / ``median`` / ``min`` / ``max`` / ``first`` / ``last`` /
  ``count`` / ``nunique`` -- the plain pandas reductions, for the cases where
  they are genuinely what a report wants.

Prefer ``ratio`` when both components are available, ``weighted_mean`` when
only the rate and a defensible weight are, and ``mean`` only for metrics that
are already per-unit-comparable.

Example::

    from ts_shape.reporting import MetricSpec, aggregate_metrics

    specs = [
        MetricSpec.sum("parts_produced"),
        MetricSpec.ratio("quality_pct", numerator="good", denominator="total",
                         scale=100.0),
        MetricSpec.weighted("oee_pct", weight="planned_minutes"),
    ]
    fleet = aggregate_metrics(df, by=["region", "period"], specs=specs,
                              contributor_column="facility")
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = [
    "Aggregation",
    "MetricSpec",
    "aggregate_metrics",
    "infer_specs",
]

logger = logging.getLogger(__name__)

Aggregation = Literal[
    "sum",
    "mean",
    "weighted_mean",
    "ratio",
    "min",
    "max",
    "median",
    "first",
    "last",
    "count",
    "nunique",
]

#: Reductions that pandas can apply directly to a single column.
_SIMPLE: frozenset[str] = frozenset(
    {"sum", "mean", "min", "max", "median", "first", "last", "count", "nunique"}
)

#: Column-name fragments that mark a metric as a rate rather than a quantity.
_RATE_HINTS: tuple[str, ...] = ("_pct", "pct_", "_rate", "rate_", "_ratio", "ratio_")


@dataclass(frozen=True)
class MetricSpec:
    """How one metric column is aggregated when rows are rolled up.

    Args:
        column: Source column in the input frame. For ``ratio`` this names the
            *output* metric; the inputs are ``numerator`` and ``denominator``.
        agg: Aggregation to apply. See the module docstring.
        weight: Weight column, required for ``weighted_mean``.
        numerator: Numerator column, required for ``ratio``.
        denominator: Denominator column, required for ``ratio``.
        scale: Factor applied to the aggregated value -- ``100.0`` turns a
            ``ratio`` of good/total into a percentage.
        output: Output column name. Defaults to ``column``.

    Raises:
        ValueError: If the aggregation is unknown or its required companion
            columns are missing.
    """

    column: str
    agg: Aggregation = "sum"
    weight: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    scale: float = 1.0
    output: str | None = None

    def __post_init__(self) -> None:
        if self.agg not in _SIMPLE and self.agg not in {"weighted_mean", "ratio"}:
            raise ValueError(
                f"Unknown aggregation {self.agg!r} for column {self.column!r}. "
                f"Supported: {sorted(_SIMPLE | {'weighted_mean', 'ratio'})}"
            )
        if self.agg == "weighted_mean" and not self.weight:
            raise ValueError(
                f"MetricSpec({self.column!r}, agg='weighted_mean') requires a "
                "weight column -- an unweighted mean of a rate is what this "
                "class exists to prevent."
            )
        if self.agg == "ratio" and not (self.numerator and self.denominator):
            raise ValueError(
                f"MetricSpec({self.column!r}, agg='ratio') requires both "
                "numerator and denominator columns."
            )

    @property
    def name(self) -> str:
        """Output column name."""
        return self.output or self.column

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Input columns this spec reads."""
        if self.agg == "ratio":
            # Both are non-None; __post_init__ enforces it.
            return (str(self.numerator), str(self.denominator))
        if self.agg == "weighted_mean":
            return (self.column, str(self.weight))
        return (self.column,)

    # -- ergonomic constructors -----------------------------------------

    @classmethod
    def sum(
        cls, column: str, *, output: str | None = None, scale: float = 1.0
    ) -> MetricSpec:
        """Additive metric: counts, quantities, minutes, costs."""
        return cls(column=column, agg="sum", output=output, scale=scale)

    @classmethod
    def weighted(
        cls,
        column: str,
        *,
        weight: str,
        output: str | None = None,
        scale: float = 1.0,
    ) -> MetricSpec:
        """Rate metric averaged in proportion to ``weight``."""
        return cls(
            column=column,
            agg="weighted_mean",
            weight=weight,
            output=output,
            scale=scale,
        )

    @classmethod
    def ratio(
        cls,
        column: str,
        *,
        numerator: str,
        denominator: str,
        scale: float = 1.0,
        output: str | None = None,
    ) -> MetricSpec:
        """Rate metric recomputed from summed components -- the exact rollup."""
        return cls(
            column=column,
            agg="ratio",
            numerator=numerator,
            denominator=denominator,
            scale=scale,
            output=output,
        )

    @classmethod
    def mean(cls, column: str, *, output: str | None = None) -> MetricSpec:
        """Unweighted mean. Correct only for already-comparable metrics."""
        return cls(column=column, agg="mean", output=output)


def _weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """Weighted mean over rows where both value and weight are usable."""
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce")
    usable = values.notna() & weights.notna() & (weights > 0)
    if not usable.any():
        return float("nan")
    total_weight = float(weights[usable].sum())
    if total_weight == 0:
        return float("nan")
    return float((values[usable] * weights[usable]).sum() / total_weight)


def _ratio(group: pd.DataFrame, numerator: str, denominator: str) -> float:
    """Summed numerator over summed denominator; NaN on a zero denominator."""
    num = pd.to_numeric(group[numerator], errors="coerce").sum()
    den = pd.to_numeric(group[denominator], errors="coerce").sum()
    if den == 0 or pd.isna(den):
        return float("nan")
    return float(num) / float(den)


def aggregate_metrics(
    dataframe: pd.DataFrame,
    *,
    by: Sequence[str],
    specs: Sequence[MetricSpec],
    contributor_column: str | None = None,
    include_rows: bool = True,
    dropna_keys: bool = False,
) -> pd.DataFrame:
    """Aggregate metric columns to a coarser grain with per-metric semantics.

    Args:
        dataframe: Long KPI frame -- one row per (grouping keys, metrics).
        by: Columns to group by. Pass an empty sequence for a single
            fleet-wide total row.
        specs: One :class:`MetricSpec` per output metric.
        contributor_column: Optional column (typically ``"facility"``) whose
            distinct count is reported as ``contributors``. A rollup that
            silently lost a site is the failure mode this makes visible.
        include_rows: Add a ``rows`` column with the group's row count.
        dropna_keys: Drop rows whose grouping keys are NaN. Default ``False``
            keeps them as an explicit ``NaN`` group so unmapped sites surface.

    Returns:
        One row per group with the grouping columns, one column per spec, and
        the optional ``contributors`` / ``rows`` columns. An empty input
        returns an empty frame carrying the full column set.

    Raises:
        ValueError: If ``specs`` is empty or a required column is missing.
    """
    if not specs:
        raise ValueError("aggregate_metrics requires at least one MetricSpec")

    group_cols = list(by)
    out_cols = (
        group_cols
        + [spec.name for spec in specs]
        + ([contributor_column and "contributors"] if contributor_column else [])
        + (["rows"] if include_rows else [])
    )
    out_cols = [c for c in out_cols if c]

    missing_keys = [c for c in group_cols if c not in dataframe.columns]
    if missing_keys:
        raise ValueError(
            f"Grouping column(s) {missing_keys} not found. "
            f"Available: {list(dataframe.columns)}"
        )
    required: set[str] = set()
    for spec in specs:
        required.update(spec.required_columns)
    missing_metrics = sorted(c for c in required if c not in dataframe.columns)
    if missing_metrics:
        raise ValueError(
            f"Metric column(s) {missing_metrics} not found. "
            f"Available: {list(dataframe.columns)}"
        )
    if contributor_column and contributor_column not in dataframe.columns:
        raise ValueError(
            f"Contributor column {contributor_column!r} not found. "
            f"Available: {list(dataframe.columns)}"
        )

    if dataframe.empty:
        return pd.DataFrame(columns=out_cols)

    work = dataframe.copy()
    if group_cols and dropna_keys:
        work = work.dropna(subset=group_cols)
        if work.empty:
            return pd.DataFrame(columns=out_cols)

    if group_cols:
        groups: Iterable[tuple[tuple[object, ...], pd.DataFrame]] = (
            (key if isinstance(key, tuple) else (key,), grp)
            for key, grp in work.groupby(group_cols, dropna=False, sort=True)
        )
    else:
        groups = iter([((), work)])

    rows: list[dict[str, object]] = []
    for key, group in groups:
        row: dict[str, object] = dict(zip(group_cols, key))
        for spec in specs:
            row[spec.name] = _apply_spec(group, spec)
        if contributor_column:
            row["contributors"] = int(group[contributor_column].nunique(dropna=True))
        if include_rows:
            row["rows"] = int(len(group))
        rows.append(row)

    return pd.DataFrame(rows, columns=out_cols)


def _apply_spec(group: pd.DataFrame, spec: MetricSpec) -> float:
    """Reduce one group to a single value for ``spec``."""
    if spec.agg == "weighted_mean":
        value = _weighted_mean(group, spec.column, str(spec.weight))
    elif spec.agg == "ratio":
        value = _ratio(group, str(spec.numerator), str(spec.denominator))
    elif spec.agg in {"count", "nunique"}:
        series = group[spec.column]
        value = float(
            series.count() if spec.agg == "count" else series.nunique(dropna=True)
        )
    else:
        series = pd.to_numeric(group[spec.column], errors="coerce").dropna()
        if series.empty:
            return float("nan")
        if spec.agg == "first":
            value = float(series.iloc[0])
        elif spec.agg == "last":
            value = float(series.iloc[-1])
        else:
            value = float(getattr(series, spec.agg)())
    if pd.isna(value):
        return float("nan")
    return value * spec.scale


def infer_specs(
    dataframe: pd.DataFrame,
    *,
    exclude: Sequence[str] = (),
    weight: str | None = None,
) -> list[MetricSpec]:
    """Derive a reasonable :class:`MetricSpec` per numeric column.

    A convenience for exploratory rollups: columns whose name looks like a rate
    (``_pct``, ``_rate``, ``_ratio``) become ``weighted_mean`` when ``weight``
    is given and ``mean`` otherwise; every other numeric column becomes
    ``sum``. Declare specs explicitly for anything that ships to a stakeholder.

    Args:
        dataframe: Frame to inspect.
        exclude: Columns to skip (grouping keys, ids, weights).
        weight: Weight column for rate-like metrics.

    Returns:
        Specs in column order.
    """
    skip = set(exclude) | ({weight} if weight else set())
    specs: list[MetricSpec] = []
    for column in dataframe.columns:
        if column in skip or not pd.api.types.is_numeric_dtype(dataframe[column]):
            continue
        lowered = str(column).lower()
        is_rate = any(hint in lowered for hint in _RATE_HINTS)
        if is_rate and weight:
            specs.append(MetricSpec.weighted(column, weight=weight))
        elif is_rate:
            logger.debug(
                "Column %r looks like a rate but no weight was given; "
                "falling back to an unweighted mean.",
                column,
            )
            specs.append(MetricSpec.mean(column))
        else:
            specs.append(MetricSpec.sum(column))
    return specs
