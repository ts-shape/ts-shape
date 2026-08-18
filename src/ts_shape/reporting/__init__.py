"""Production reporting across global facilities.

Every other ts-shape pack answers a question about *one* asset: this line's
OEE, that station's scrap rate, this machine's downtime. ``ts_shape.reporting``
answers the questions a global operations function asks across dozens of sites
in a dozen timezones and half a dozen currencies:

* What is the fleet number, and how many plants are actually behind it?
* Which site moved it, and was it performance or volume mix?
* Who is in the top quartile, and what is closing the gap worth in units?
* Did everyone report, and how fresh is the data?

The layer is deliberately **downstream**: the event packs compute each site's
KPIs from its raw signals, and this package consolidates those KPI frames. It
never touches raw timeseries, so it works just as well on KPIs that arrived
from an MES, a data warehouse or another ts-shape run.

Modules
-------

:mod:`~ts_shape.reporting.facility`
    :class:`Facility` / :class:`FacilityRegistry` -- site master data
    (timezone, region, currency, shift pattern) and its validation.
:mod:`~ts_shape.reporting.calendar`
    :class:`FacilityCalendar` -- local production days, per-site shifts and
    DST-correct period windows.
:mod:`~ts_shape.reporting.hierarchy`
    :class:`AssetHierarchy` -- the ISA-95 equipment tree and roll-up along it.
:mod:`~ts_shape.reporting.aggregation`
    :class:`MetricSpec` / :func:`aggregate_metrics` -- weighted, ratio and
    additive aggregation semantics.
:mod:`~ts_shape.reporting.normalize`
    :class:`MetricNormalizer` -- currency, scale and exposure normalisation.
:mod:`~ts_shape.reporting.rollup`
    :class:`FleetRollup` -- facility, region and global consolidation.
:mod:`~ts_shape.reporting.benchmark`
    :class:`FacilityBenchmark` -- ranking, quartiles, opportunity sizing.
:mod:`~ts_shape.reporting.variance`
    :class:`VarianceBridge` -- who moved the number, and rate versus mix.
:mod:`~ts_shape.reporting.coverage`
    :class:`ReportingCoverage` -- did every site report, and how fresh is it.
:mod:`~ts_shape.reporting.scorecard`
    :class:`GlobalScorecard` / :class:`KpiTarget` -- targets, RAG status and
    the management table.

Example -- a full consolidation::

    from ts_shape.reporting import (
        Facility, FacilityRegistry, FacilityCalendar, FleetRollup,
        MetricSpec, ReportingCoverage,
    )

    registry = FacilityRegistry([
        Facility("DE-STR", timezone="Europe/Berlin", region="EMEA", currency="EUR"),
        Facility("US-DTW", timezone="America/Detroit", region="AMER", currency="USD"),
        Facility("CN-SHA", timezone="Asia/Shanghai", region="APAC", currency="CNY"),
    ])
    assert registry.validate().empty          # master data is sound

    calendar = FacilityCalendar(registry)
    expected = calendar.period_grid("2026-01-01", "2026-01-31")
    coverage = ReportingCoverage(site_kpis).summary(expected)

    specs = [
        MetricSpec.sum("parts_produced"),
        MetricSpec.weighted("oee_pct", weight="planned_minutes"),
    ]
    report = FleetRollup(site_kpis, registry=registry).consolidate(specs)
"""

from ts_shape.reporting.aggregation import (
    Aggregation,
    MetricSpec,
    aggregate_metrics,
    infer_specs,
)
from ts_shape.reporting.benchmark import FacilityBenchmark
from ts_shape.reporting.calendar import FacilityCalendar
from ts_shape.reporting.coverage import ReportingCoverage
from ts_shape.reporting.facility import Facility, FacilityRegistry
from ts_shape.reporting.hierarchy import AssetHierarchy
from ts_shape.reporting.normalize import MetricNormalizer
from ts_shape.reporting.rollup import FleetRollup
from ts_shape.reporting.scorecard import GlobalScorecard, KpiTarget
from ts_shape.reporting.variance import VarianceBridge

__all__ = [
    "Aggregation",
    "AssetHierarchy",
    "Facility",
    "FacilityBenchmark",
    "FacilityCalendar",
    "FacilityRegistry",
    "FleetRollup",
    "GlobalScorecard",
    "KpiTarget",
    "MetricNormalizer",
    "MetricSpec",
    "ReportingCoverage",
    "VarianceBridge",
    "aggregate_metrics",
    "infer_specs",
]
