"""Make facility metrics comparable: currency, scale and exposure.

Two plants' numbers are not comparable just because they share a column name.
Cost sits in three currencies, energy in kWh at one site and MWh at another,
and output in absolute parts at sites with a factor-ten capacity difference.
Consolidating any of those without normalising produces a report that ranks
sites by size or by exchange rate rather than by performance.

:class:`MetricNormalizer` performs the four normalisations a consolidated
report needs:

* **Currency** -- convert monetary columns to one reporting currency, with
  either a flat rate table or period-dated rates applied as-of, so a
  restatement of last quarter uses last quarter's rate.
* **Scale** -- rescale units (kWh to MWh, pieces to thousands).
* **Exposure** -- divide by a denominator (per part, per operating hour, per
  unit of nameplate capacity) so a small site is judged on efficiency.
* **Index** -- restate a metric as an index against a baseline (fleet median,
  best site, a nominated site or a fixed value), which is how a scorecard
  compares metrics that share no unit.

Example::

    from ts_shape.reporting import MetricNormalizer

    norm = MetricNormalizer(kpi_df, registry=registry)
    df = norm.convert_currency(["scrap_cost"], to_currency="EUR",
                               rates={"USD": 0.92, "CNY": 0.13, "EUR": 1.0})
    df = MetricNormalizer(df).per_unit(["scrap_cost_EUR"], denominator="parts")
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pandas as pd

from ts_shape.reporting.facility import FACILITY_COLUMN, FacilityRegistry

__all__ = ["MetricNormalizer"]

logger = logging.getLogger(__name__)


class MetricNormalizer:
    """Currency, scale and exposure normalisation for a KPI frame.

    Every method returns a new DataFrame; the input is never mutated.

    Args:
        dataframe: KPI frame, typically one row per facility and period.
        facility_column: Column holding the site id.
        registry: Optional facility master data. Supplies each site's currency
            when the frame carries none, and nameplate capacity for
            :meth:`per_capacity`.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str = FACILITY_COLUMN,
        registry: FacilityRegistry | None = None,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(dataframe).__name__}"
            )
        self.dataframe = dataframe.copy()
        self.facility_column = facility_column
        self.registry = registry

    def __repr__(self) -> str:
        return (
            f"MetricNormalizer(rows={len(self.dataframe)}, "
            f"facility_column={self.facility_column!r})"
        )

    def _require(self, columns: Sequence[str]) -> None:
        missing = [c for c in columns if c not in self.dataframe.columns]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found. "
                f"Available: {list(self.dataframe.columns)}"
            )

    # -- currency --------------------------------------------------------

    def _source_currency(self, currency_column: str | None) -> pd.Series:
        """Per-row source currency, from a column or the registry."""
        if currency_column is not None:
            self._require([currency_column])
            return self.dataframe[currency_column].astype(str)
        if "currency" in self.dataframe.columns:
            return self.dataframe["currency"].astype(str)
        if self.registry is None:
            raise ValueError(
                "No currency information: pass currency_column, add a "
                "'currency' column, or construct with a FacilityRegistry."
            )
        self._require([self.facility_column])
        mapping = self.registry.currencies
        return (
            self.dataframe[self.facility_column]
            .astype(str)
            .map(mapping)
            .astype("object")
        )

    def convert_currency(
        self,
        columns: Sequence[str],
        *,
        to_currency: str,
        rates: Mapping[str, float] | pd.DataFrame,
        currency_column: str | None = None,
        period_column: str | None = None,
        suffix: str | None = None,
        keep_source: bool = True,
    ) -> pd.DataFrame:
        """Convert monetary columns into one reporting currency.

        Args:
            columns: Monetary columns to convert.
            to_currency: ISO code of the reporting currency.
            rates: Either a mapping ``{source_currency: rate}`` where ``rate``
                is how many units of ``to_currency`` one unit of the source
                buys, or a DataFrame with columns ``currency`` and ``rate``
                plus an optional date column for period-dated rates.
            currency_column: Column holding each row's source currency.
                Defaults to a ``currency`` column, then to the registry.
            period_column: Date/period column used to pick a period-dated rate
                as-of. Required when ``rates`` carries a date column.
            suffix: Output column suffix. Defaults to ``f"_{to_currency}"``.
            keep_source: Keep the original columns alongside the converted
                ones. A restated report that hides its source figures is not
                auditable, so this defaults to ``True``.

        Returns:
            A new frame with the converted columns plus an ``fx_rate`` column
            recording the rate applied to each row.

        Raises:
            ValueError: If a column is missing, currencies cannot be resolved,
                or ``rates`` is dated but ``period_column`` is not given.
        """
        self._require(list(columns))
        out_suffix = suffix if suffix is not None else f"_{to_currency}"
        source = self._source_currency(currency_column)

        result = self.dataframe.copy()
        rate = self._resolve_rates(source, rates, to_currency, period_column)
        unresolved = sorted(
            {str(c) for c, r in zip(source, rate) if pd.isna(r) and not pd.isna(c)}
        )
        if unresolved:
            logger.warning(
                "No %s rate for currency/currencies %s; those rows convert to NaN.",
                to_currency,
                unresolved,
            )

        result["fx_rate"] = rate.to_numpy()
        for column in columns:
            converted = (
                pd.to_numeric(result[column], errors="coerce") * result["fx_rate"]
            )
            result[f"{column}{out_suffix}"] = converted
            if not keep_source:
                result = result.drop(columns=[column])
        result["reporting_currency"] = to_currency
        return result

    def _resolve_rates(
        self,
        source: pd.Series,
        rates: Mapping[str, float] | pd.DataFrame,
        to_currency: str,
        period_column: str | None,
    ) -> pd.Series:
        """Per-row FX rate, flat or as-of a period-dated rate table."""
        if isinstance(rates, pd.DataFrame):
            return self._asof_rates(source, rates, to_currency, period_column)
        table = {str(k): float(v) for k, v in rates.items()}
        table.setdefault(to_currency, 1.0)
        return source.map(table).astype("float64")

    def _asof_rates(
        self,
        source: pd.Series,
        rates: pd.DataFrame,
        to_currency: str,
        period_column: str | None,
    ) -> pd.Series:
        required = {"currency", "rate"}
        missing = sorted(required - set(rates.columns))
        if missing:
            raise ValueError(
                f"Rate table is missing column(s) {missing}; "
                "expected 'currency', 'rate' and optionally a date column."
            )
        date_columns = [c for c in rates.columns if c not in required]
        if not date_columns:
            flat = dict(zip(rates["currency"].astype(str), rates["rate"].astype(float)))
            flat.setdefault(to_currency, 1.0)
            return source.map(flat).astype("float64")

        if period_column is None:
            raise ValueError(
                "The rate table is period-dated "
                f"(date column {date_columns[0]!r}); pass period_column so each "
                "row can be converted at the rate in force for its period."
            )
        self._require([period_column])
        date_column = date_columns[0]

        left = pd.DataFrame(
            {
                "_idx": range(len(source)),
                "currency": source.astype(str).to_numpy(),
                "_on": pd.to_datetime(
                    self.dataframe[period_column], errors="coerce"
                ).to_numpy(),
            }
        ).sort_values("_on")
        right = (
            rates.assign(
                currency=rates["currency"].astype(str),
                _on=pd.to_datetime(rates[date_column], errors="coerce"),
                rate=rates["rate"].astype(float),
            )[["currency", "_on", "rate"]]
            .dropna(subset=["_on"])
            .sort_values("_on")
        )
        merged = pd.merge_asof(
            left.dropna(subset=["_on"]),
            right,
            on="_on",
            by="currency",
            direction="backward",
        )
        resolved = pd.Series(float("nan"), index=range(len(source)), dtype="float64")
        resolved.loc[merged["_idx"].to_numpy()] = merged["rate"].to_numpy()
        # The reporting currency always converts one-for-one.
        resolved[source.astype(str).to_numpy() == to_currency] = 1.0
        resolved.index = source.index
        return resolved

    # -- scale and exposure ----------------------------------------------

    def rescale(
        self,
        columns: Sequence[str],
        *,
        factor: float,
        suffix: str = "_scaled",
    ) -> pd.DataFrame:
        """Multiply columns by ``factor`` (e.g. ``0.001`` for kWh to MWh)."""
        self._require(list(columns))
        result = self.dataframe.copy()
        for column in columns:
            result[f"{column}{suffix}"] = (
                pd.to_numeric(result[column], errors="coerce") * factor
            )
        return result

    def per_unit(
        self,
        columns: Sequence[str],
        *,
        denominator: str,
        suffix: str = "_per_unit",
    ) -> pd.DataFrame:
        """Divide columns by an exposure column (parts, hours, tonnes).

        Rows whose denominator is zero, negative or missing yield NaN rather
        than infinity, so one idle site cannot dominate a ranking.
        """
        self._require([*columns, denominator])
        result = self.dataframe.copy()
        base = pd.to_numeric(result[denominator], errors="coerce")
        base = base.where(base > 0)
        for column in columns:
            result[f"{column}{suffix}"] = (
                pd.to_numeric(result[column], errors="coerce") / base
            )
        return result

    def per_hour(
        self,
        columns: Sequence[str],
        *,
        hours_column: str,
        suffix: str = "_per_hour",
    ) -> pd.DataFrame:
        """Divide columns by operating hours -- a rate every site can be judged on."""
        return self.per_unit(columns, denominator=hours_column, suffix=suffix)

    def per_capacity(
        self,
        columns: Sequence[str],
        *,
        days_column: str | None = None,
        suffix: str = "_per_capacity",
    ) -> pd.DataFrame:
        """Express columns as a share of nameplate capacity.

        Capacity comes from :attr:`Facility.capacity_per_day` in the registry.
        With ``days_column`` the capacity is multiplied by the number of days
        the row covers, so a monthly figure is compared against monthly
        capacity.

        Raises:
            ValueError: If no registry was supplied, or no registered facility
                declares a capacity.
        """
        if self.registry is None:
            raise ValueError(
                "per_capacity requires a FacilityRegistry with capacity_per_day set."
            )
        self._require([*columns, self.facility_column])
        capacities = {
            f.id: f.capacity_per_day
            for f in self.registry
            if f.capacity_per_day is not None
        }
        if not capacities:
            raise ValueError("No facility in the registry declares capacity_per_day.")
        result = self.dataframe.copy()
        base = (
            result[self.facility_column].astype(str).map(capacities).astype("float64")
        )
        if days_column is not None:
            self._require([days_column])
            base = base * pd.to_numeric(result[days_column], errors="coerce")
        base = base.where(base > 0)
        for column in columns:
            result[f"{column}{suffix}"] = (
                pd.to_numeric(result[column], errors="coerce") / base
            )
        return result

    def index_to_baseline(
        self,
        columns: Sequence[str],
        *,
        baseline: str | float = "median",
        higher_is_better: bool = True,
        suffix: str = "_index",
        group_by: Sequence[str] = (),
    ) -> pd.DataFrame:
        """Restate columns as an index where the baseline is 100.

        Args:
            columns: Metric columns to index.
            baseline: ``"median"`` or ``"mean"`` of the fleet, ``"best"``
                (per ``higher_is_better``), a registered facility id, or a
                fixed numeric value.
            higher_is_better: Direction used by the ``"best"`` baseline.
            suffix: Output column suffix.
            group_by: Compute the baseline within these groups (e.g.
                ``["period"]`` so each period is indexed against itself).

        Returns:
            A new frame with the indexed columns added. A metric whose baseline
            is zero or missing yields NaN.

        Raises:
            ValueError: If a column is missing or ``baseline`` is unknown.
        """
        self._require(list(columns))
        if group_by:
            self._require(list(group_by))
        result = self.dataframe.copy()

        for column in columns:
            values = pd.to_numeric(result[column], errors="coerce")
            if group_by:
                base = values.groupby(
                    [result[c] for c in group_by], dropna=False
                ).transform(
                    lambda s: _baseline_value(s, baseline, higher_is_better, self)
                )
            else:
                base = pd.Series(
                    _baseline_value(values, baseline, higher_is_better, self),
                    index=result.index,
                    dtype="float64",
                )
            base = base.where(base != 0)
            result[f"{column}{suffix}"] = values / base * 100.0
        return result


def _baseline_value(
    values: pd.Series,
    baseline: str | float,
    higher_is_better: bool,
    normalizer: MetricNormalizer,
) -> float:
    """Resolve a baseline specification to a number for one series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        return float(baseline)
    if numeric.empty:
        return float("nan")
    if baseline == "median":
        return float(numeric.median())
    if baseline == "mean":
        return float(numeric.mean())
    if baseline == "best":
        return float(numeric.max() if higher_is_better else numeric.min())
    # Otherwise: a facility id.
    frame = normalizer.dataframe
    column = normalizer.facility_column
    if column not in frame.columns:
        raise ValueError(
            f"baseline={baseline!r} looks like a facility id but column "
            f"{column!r} is not in the frame."
        )
    mask = frame[column].astype(str) == str(baseline)
    selected = numeric[numeric.index.isin(frame.index[mask])]
    if selected.empty:
        raise ValueError(
            f"Unknown baseline {baseline!r}: not 'median', 'mean', 'best', "
            f"a number, or a facility present in {column!r}."
        )
    return float(selected.mean())
