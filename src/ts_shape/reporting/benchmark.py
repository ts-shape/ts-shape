"""Cross-facility benchmarking: rank, spread, and the size of the gap.

Once every site's KPIs are comparable, the questions a global operations review
asks are always the same four:

* Where does each plant stand against the fleet?
* What separates the top quartile from the rest?
* What would closing that gap be worth, in units rather than percentage points?
* Which sites are erratic, as opposed to merely low?

:class:`FacilityBenchmark` answers them on a KPI frame keyed by facility. It
never assumes a direction: every method takes ``higher_is_better`` because half
of a plant scorecard (scrap, downtime, energy per part) is better when lower.

Example::

    from ts_shape.reporting import FacilityBenchmark

    bench = FacilityBenchmark(site_kpis)
    bench.rank("oee_pct")
    bench.quartiles("oee_pct")
    bench.opportunity("oee_pct", exposure_column="planned_minutes")
    bench.scorecard({"oee_pct": True, "scrap_rate_pct": False})
    bench.stability("oee_pct", period_column="period")
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ts_shape.reporting.aggregation import MetricSpec, aggregate_metrics
from ts_shape.reporting.facility import FACILITY_COLUMN

__all__ = ["FacilityBenchmark"]

logger = logging.getLogger(__name__)


class FacilityBenchmark:
    """Rank, compare and size the gap between facilities on shared metrics.

    Args:
        dataframe: KPI frame keyed by facility. Frames with several rows per
            facility (one per period) are collapsed by each method using the
            supplied ``spec`` -- an unweighted mean by default, which is only
            appropriate for metrics that are already comparable across periods.
        facility_column: Column holding the site id.
        period_column: Column holding the period, when present. Required by
            :meth:`stability`.

    Raises:
        ValueError: If ``facility_column`` is missing.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str = FACILITY_COLUMN,
        period_column: str | None = None,
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

    def __repr__(self) -> str:
        return (
            f"FacilityBenchmark(rows={len(self.dataframe)}, "
            f"facilities={self.dataframe[self.facility_column].nunique()})"
        )

    # -- internals -------------------------------------------------------

    def _per_facility(
        self, metric: str, spec: MetricSpec | None, extra: Sequence[str] = ()
    ) -> pd.DataFrame:
        """Collapse to one row per facility with a ``value`` column."""
        if metric not in self.dataframe.columns:
            raise ValueError(
                f"Metric column {metric!r} not found. "
                f"Available: {list(self.dataframe.columns)}"
            )
        used = spec or MetricSpec.mean(metric)
        specs = [MetricSpec(**{**used.__dict__, "output": "value"})]
        for column in extra:
            if column not in self.dataframe.columns:
                raise ValueError(
                    f"Column {column!r} not found. "
                    f"Available: {list(self.dataframe.columns)}"
                )
            specs.append(MetricSpec.sum(column))
        return aggregate_metrics(
            self.dataframe,
            by=[self.facility_column],
            specs=specs,
            include_rows=False,
        )

    # -- ranking ---------------------------------------------------------

    def rank(
        self,
        metric: str,
        *,
        higher_is_better: bool = True,
        spec: MetricSpec | None = None,
    ) -> pd.DataFrame:
        """Rank facilities on one metric against the fleet.

        Args:
            metric: Metric column.
            higher_is_better: Direction of goodness. Set ``False`` for scrap,
                downtime, cost or energy per part.
            spec: How to collapse several rows per facility. Defaults to an
                unweighted mean; pass
                ``MetricSpec.weighted(metric, weight="planned_minutes")`` when
                the periods carry different exposure.

        Returns:
            One row per facility, best first, with columns ``value``, ``rank``
            (1 = best), ``percentile`` (share of the fleet this site is at
            least as good as), ``quartile`` (``"Q1"`` = top), ``z_score``,
            ``gap_to_best``, ``gap_to_median`` and ``vs_fleet_pct`` (percentage
            difference from the fleet mean). Facilities whose value is missing
            are kept with NaN ranks rather than dropped.
        """
        frame = self._per_facility(metric, spec)
        values = pd.to_numeric(frame["value"], errors="coerce")
        valid = values.dropna()

        frame["rank"] = values.rank(
            ascending=not higher_is_better, method="min"
        ).astype("float64")
        frame["percentile"] = values.rank(pct=True, ascending=higher_is_better) * 100.0

        if valid.empty:
            for column in (
                "quartile",
                "z_score",
                "gap_to_best",
                "gap_to_median",
                "vs_fleet_pct",
            ):
                frame[column] = float("nan") if column != "quartile" else None
            return frame.sort_values(self.facility_column).reset_index(drop=True)

        best = valid.max() if higher_is_better else valid.min()
        median = valid.median()
        mean = valid.mean()
        std = valid.std(ddof=0)

        frame["quartile"] = pd.cut(
            frame["percentile"],
            bins=[-0.01, 25.0, 50.0, 75.0, 100.0],
            labels=["Q4", "Q3", "Q2", "Q1"],
        ).astype("object")
        frame["z_score"] = (
            (values - mean) / std if std and not pd.isna(std) else float("nan")
        )
        sign = 1.0 if higher_is_better else -1.0
        frame["gap_to_best"] = (best - values) * sign
        frame["gap_to_median"] = (values - median) * sign
        frame["vs_fleet_pct"] = (
            (values - mean) / mean * 100.0 if mean not in (0, None) else float("nan")
        )
        return frame.sort_values(
            ["rank", self.facility_column], na_position="last"
        ).reset_index(drop=True)

    def quartiles(
        self,
        metric: str,
        *,
        higher_is_better: bool = True,
        spec: MetricSpec | None = None,
    ) -> pd.DataFrame:
        """Fleet distribution of a metric, one row.

        Returns:
            DataFrame with ``metric``, ``facilities``, ``min``, ``q1``,
            ``median``, ``q3``, ``max``, ``mean``, ``std``, ``iqr``,
            ``top_quartile`` (the threshold a site must reach to be in the best
            25 %, direction-aware) and ``spread_pct`` (best-to-worst spread as
            a percentage of the median -- the single number that says whether a
            fleet is consistent).
        """
        frame = self._per_facility(metric, spec)
        values = pd.to_numeric(frame["value"], errors="coerce").dropna()
        columns = [
            "metric",
            "facilities",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "mean",
            "std",
            "iqr",
            "top_quartile",
            "spread_pct",
        ]
        if values.empty:
            return pd.DataFrame([{c: float("nan") for c in columns}]).assign(
                metric=metric, facilities=0
            )[columns]

        q1, median, q3 = (
            float(values.quantile(0.25)),
            float(values.median()),
            float(values.quantile(0.75)),
        )
        best, worst = (
            (float(values.max()), float(values.min()))
            if higher_is_better
            else (float(values.min()), float(values.max()))
        )
        row = {
            "metric": metric,
            "facilities": int(values.size),
            "min": float(values.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "iqr": q3 - q1,
            "top_quartile": q3 if higher_is_better else q1,
            "spread_pct": (
                abs(best - worst) / abs(median) * 100.0 if median else float("nan")
            ),
        }
        return pd.DataFrame([row], columns=columns)

    # -- gap sizing ------------------------------------------------------

    def opportunity(
        self,
        metric: str,
        *,
        exposure_column: str,
        target: str | float = "top_quartile",
        higher_is_better: bool = True,
        spec: MetricSpec | None = None,
        metric_scale: float | None = None,
    ) -> pd.DataFrame:
        """Size the gap to a benchmark in units, not percentage points.

        "Six points of OEE" persuades nobody; "38 000 parts a year" funds a
        project. Each facility's gap to the benchmark is multiplied by its own
        exposure (planned minutes, parts, tonnes) to express the opportunity in
        the exposure's units.

        Args:
            metric: Metric column.
            exposure_column: Column giving each facility's exposure. Summed per
                facility.
            target: ``"top_quartile"``, ``"best"``, ``"median"`` or a fixed
                numeric benchmark.
            higher_is_better: Direction of goodness.
            spec: How to collapse several rows per facility (see :meth:`rank`).
            metric_scale: Factor converting a metric gap into a fraction of
                exposure. Defaults to ``0.01`` for percentage-style metrics
                (name contains ``pct``/``percent``/``rate``) and ``1.0``
                otherwise.

        Returns:
            One row per facility with ``value``, ``target_value``, ``gap``
            (zero for sites already at or beyond the benchmark), ``exposure``
            and ``opportunity`` in exposure units, plus a ``metric_scale``
            column recording the factor used. Sorted by opportunity, largest
            first.

        Raises:
            ValueError: If a column is missing or ``target`` is unknown.
        """
        frame = self._per_facility(metric, spec, extra=[exposure_column])
        values = pd.to_numeric(frame["value"], errors="coerce")
        valid = values.dropna()

        if isinstance(target, (int, float)) and not isinstance(target, bool):
            target_value = float(target)
        elif valid.empty:
            target_value = float("nan")
        elif target == "top_quartile":
            target_value = float(valid.quantile(0.75 if higher_is_better else 0.25))
        elif target == "best":
            target_value = float(valid.max() if higher_is_better else valid.min())
        elif target == "median":
            target_value = float(valid.median())
        else:
            raise ValueError(
                f"Unknown target {target!r}: expected 'top_quartile', 'best', "
                "'median' or a number."
            )

        scale = (
            metric_scale if metric_scale is not None else _default_metric_scale(metric)
        )
        gap = (
            (target_value - values) if higher_is_better else (values - target_value)
        ).clip(lower=0.0)

        frame["target_value"] = target_value
        frame["gap"] = gap
        frame["exposure"] = pd.to_numeric(frame[exposure_column], errors="coerce")
        frame["metric_scale"] = scale
        frame["opportunity"] = gap * frame["exposure"] * scale
        frame = frame.drop(columns=[exposure_column])
        return frame.sort_values(
            "opportunity", ascending=False, na_position="last"
        ).reset_index(drop=True)

    # -- multi-metric ----------------------------------------------------

    def scorecard(
        self,
        metrics: Mapping[str, bool],
        *,
        weights: Mapping[str, float] | None = None,
        specs: Mapping[str, MetricSpec] | None = None,
    ) -> pd.DataFrame:
        """Normalise several metrics onto 0-100 and combine into one score.

        Each metric is min-max normalised across the fleet in its own
        direction, so 100 is the best site on that metric and 0 the worst. The
        composite is the weighted mean of the normalised scores, skipping
        metrics a site is missing.

        Args:
            metrics: ``{column: higher_is_better}``.
            weights: Optional ``{column: weight}``; missing entries default to
                ``1.0``.
            specs: Optional ``{column: MetricSpec}`` for collapsing periods.

        Returns:
            One row per facility with ``<metric>`` (raw), ``<metric>_score``
            (0-100) per metric, ``composite_score`` and ``rank``, best first.

        Raises:
            ValueError: If ``metrics`` is empty or a column is missing.
        """
        if not metrics:
            raise ValueError("scorecard requires at least one metric")
        spec_map = dict(specs or {})
        weight_map = dict(weights or {})

        result: pd.DataFrame | None = None
        score_columns: list[str] = []
        applied_weights: list[float] = []

        for metric, higher_is_better in metrics.items():
            per_site = self._per_facility(metric, spec_map.get(metric)).rename(
                columns={"value": metric}
            )
            values = pd.to_numeric(per_site[metric], errors="coerce")
            span = values.max() - values.min()
            if pd.isna(span) or span == 0:
                # A fleet that is uniform on this metric scores flat rather
                # than dividing by zero; every site is equally good.
                per_site[f"{metric}_score"] = values.notna().map(
                    {True: 100.0, False: float("nan")}
                )
            else:
                normalised = (values - values.min()) / span * 100.0
                per_site[f"{metric}_score"] = (
                    normalised if higher_is_better else 100.0 - normalised
                )
            score_columns.append(f"{metric}_score")
            applied_weights.append(float(weight_map.get(metric, 1.0)))
            result = (
                per_site
                if result is None
                else result.merge(per_site, on=self.facility_column, how="outer")
            )

        assert result is not None  # metrics is non-empty
        scores = result[score_columns]
        weight_row = np.asarray(applied_weights, dtype="float64")
        mask = scores.notna().to_numpy()
        weighted = np.nansum(scores.to_numpy() * weight_row, axis=1)
        denominator = (mask * weight_row).sum(axis=1)
        result["composite_score"] = np.where(
            denominator > 0,
            weighted / np.where(denominator > 0, denominator, 1.0),
            np.nan,
        )
        result["rank"] = result["composite_score"].rank(ascending=False, method="min")
        return result.sort_values(
            ["rank", self.facility_column], na_position="last"
        ).reset_index(drop=True)

    # -- variability -----------------------------------------------------

    def stability(
        self,
        metric: str,
        *,
        period_column: str | None = None,
        min_periods: int = 2,
    ) -> pd.DataFrame:
        """Per-facility variability of a metric over time.

        A plant averaging 70 % OEE every week and one alternating 40 % and
        100 % rank identically on the mean and need entirely different
        interventions. This separates them.

        Args:
            metric: Metric column.
            period_column: Period column; defaults to the instance's.
            min_periods: Facilities with fewer periods report NaN for
                dispersion and trend rather than a meaningless value.

        Returns:
            One row per facility with ``periods``, ``mean``, ``std``,
            ``cv_pct`` (coefficient of variation), ``min``, ``max``, ``range``,
            ``last`` and ``trend_per_period`` (ordinary least-squares slope
            against period order -- positive means rising).

        Raises:
            ValueError: If no period column is available or a column is missing.
        """
        column = period_column or self.period_column
        if column is None:
            raise ValueError(
                "stability needs a period column: construct FacilityBenchmark "
                "with period_column=... or pass period_column=..."
            )
        for needed in (metric, column):
            if needed not in self.dataframe.columns:
                raise ValueError(
                    f"Column {needed!r} not found. "
                    f"Available: {list(self.dataframe.columns)}"
                )

        rows: list[dict[str, object]] = []
        for site, group in self.dataframe.groupby(
            self.dataframe[self.facility_column].astype(str), sort=True
        ):
            ordered = group.sort_values(column)
            values = pd.to_numeric(ordered[metric], errors="coerce").dropna()
            n = int(values.size)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n >= 2 else float("nan")
            rows.append(
                {
                    self.facility_column: site,
                    "periods": n,
                    "mean": mean,
                    "std": std if n >= min_periods else float("nan"),
                    "cv_pct": (
                        std / abs(mean) * 100.0
                        if n >= min_periods and mean not in (0.0,) and not pd.isna(mean)
                        else float("nan")
                    ),
                    "min": float(values.min()) if n else float("nan"),
                    "max": float(values.max()) if n else float("nan"),
                    "range": float(values.max() - values.min()) if n else float("nan"),
                    "last": float(values.iloc[-1]) if n else float("nan"),
                    "trend_per_period": (
                        _slope(values) if n >= max(2, min_periods) else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows)

    def outliers(
        self,
        metric: str,
        *,
        method: str = "iqr",
        threshold: float = 1.5,
        spec: MetricSpec | None = None,
    ) -> pd.DataFrame:
        """Flag facilities that sit outside the fleet distribution.

        Args:
            metric: Metric column.
            method: ``"iqr"`` (Tukey fences at ``threshold`` x IQR) or
                ``"zscore"`` (``|z| > threshold``).
            threshold: Fence multiplier or z-score cut-off.
            spec: How to collapse several rows per facility.

        Returns:
            One row per facility with ``value``, ``lower_fence``,
            ``upper_fence``, ``is_outlier`` and ``direction``
            (``"high"`` / ``"low"`` / ``""``).

        Raises:
            ValueError: If ``method`` is unknown.
        """
        frame = self._per_facility(metric, spec)
        values = pd.to_numeric(frame["value"], errors="coerce")
        valid = values.dropna()

        if method == "iqr":
            if valid.empty:
                lower = upper = float("nan")
            else:
                q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
                iqr = q3 - q1
                lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        elif method == "zscore":
            mean = float(valid.mean()) if not valid.empty else float("nan")
            std = float(valid.std(ddof=0)) if not valid.empty else float("nan")
            if not std or pd.isna(std):
                lower = upper = float("nan")
            else:
                lower, upper = mean - threshold * std, mean + threshold * std
        else:
            raise ValueError(f"Unknown method {method!r}: expected 'iqr' or 'zscore'.")

        frame["lower_fence"] = lower
        frame["upper_fence"] = upper
        high = (
            values > upper
            if not pd.isna(upper)
            else pd.Series(False, index=frame.index)
        )
        low = (
            values < lower
            if not pd.isna(lower)
            else pd.Series(False, index=frame.index)
        )
        frame["is_outlier"] = (high | low).fillna(False)
        frame["direction"] = np.where(high, "high", np.where(low, "low", ""))
        return frame.reset_index(drop=True)


def _default_metric_scale(metric: str) -> float:
    """0.01 for percentage-style metric names, 1.0 otherwise."""
    lowered = metric.lower()
    if any(hint in lowered for hint in ("pct", "percent", "rate", "ratio")):
        return 0.01
    return 1.0


def _slope(values: pd.Series) -> float:
    """OLS slope of ``values`` against 0, 1, 2, ... (NaN when degenerate)."""
    y = values.to_numpy(dtype="float64")
    x = np.arange(y.size, dtype="float64")
    if y.size < 2 or np.allclose(x, x[0]):
        return float("nan")
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)
