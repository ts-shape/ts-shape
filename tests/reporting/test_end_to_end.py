"""End-to-end: raw multi-site timeseries -> local days -> global report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.reporting import (
    Facility,
    FacilityBenchmark,
    FacilityCalendar,
    FacilityRegistry,
    FleetRollup,
    GlobalScorecard,
    KpiTarget,
    MetricSpec,
    ReportingCoverage,
    VarianceBridge,
)


@pytest.fixture()
def fleet_registry() -> FacilityRegistry:
    return FacilityRegistry(
        [
            Facility("DE-STR", timezone="Europe/Berlin", region="EMEA", currency="EUR"),
            Facility(
                "US-DTW", timezone="America/Detroit", region="AMER", currency="USD"
            ),
            Facility("CN-SHA", timezone="Asia/Shanghai", region="APAC", currency="CNY"),
        ]
    )


@pytest.fixture()
def raw_readings() -> pd.DataFrame:
    """Hourly part counts for three sites over four UTC days."""
    rng = np.random.default_rng(7)
    stamps = pd.date_range("2026-01-01", periods=96, freq="1h", tz="UTC")
    frames = []
    for site, rate in (("DE-STR", 10), ("US-DTW", 6), ("CN-SHA", 20)):
        frames.append(
            pd.DataFrame(
                {
                    "facility": site,
                    "systime": stamps,
                    "parts_produced": rng.integers(rate - 2, rate + 3, len(stamps)),
                    "planned_minutes": 60.0,
                    "oee_pct": rng.uniform(55.0, 90.0, len(stamps)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_full_consolidation_pipeline(fleet_registry, raw_readings):
    calendar = FacilityCalendar(fleet_registry)

    # 1. Raw UTC readings become each site's own local production day.
    labelled = calendar.assign_shifts(raw_readings)
    assert {"shift", "production_date", "local_time"} <= set(labelled.columns)

    # A site east of UTC sees its local day start before the UTC day does, so
    # the local production days are genuinely different per site.
    per_site_days = labelled.groupby("facility")["production_date"].nunique()
    assert per_site_days.min() >= 4

    # 2. Per-site, per-day KPIs -- the grain every downstream report consumes.
    site_kpis = (
        labelled.groupby(["facility", "production_date"], as_index=False)
        .agg(
            parts_produced=("parts_produced", "sum"),
            planned_minutes=("planned_minutes", "sum"),
            oee_pct=("oee_pct", "mean"),
        )
        .rename(columns={"production_date": "period"})
    )
    site_kpis["period"] = pd.to_datetime(site_kpis["period"])

    # 3. Coverage before consolidation: how many sites are behind the number.
    coverage = ReportingCoverage(site_kpis).summary().iloc[0]
    assert coverage["facilities_reporting"] == 3

    # 4. Consolidate with the correct weighting semantics.
    specs = [
        MetricSpec.sum("parts_produced"),
        MetricSpec.weighted("oee_pct", weight="planned_minutes"),
    ]
    rollup = FleetRollup(site_kpis, registry=fleet_registry)
    total = rollup.total(specs).iloc[0]
    assert total["contributors"] == 3
    assert total["parts_produced"] == pytest.approx(site_kpis["parts_produced"].sum())
    # The weighted fleet OEE lies inside the range of the site values.
    by_site = rollup.by_facility(specs)
    assert by_site["oee_pct"].min() <= total["oee_pct"] <= by_site["oee_pct"].max()

    # 5. Regional and stacked views reconcile with the global row.
    stacked = rollup.consolidate(specs)
    regions = stacked[stacked["level"] == "region"]["parts_produced"].sum()
    assert regions == pytest.approx(total["parts_produced"])

    # 6. Benchmark, bridge and scorecard all run on the same frame.
    bench = FacilityBenchmark(site_kpis, period_column="period")
    ranked = bench.rank("oee_pct")
    assert list(ranked["rank"]) == [1.0, 2.0, 3.0]

    periods = sorted(site_kpis["period"].unique())
    bridge = VarianceBridge(site_kpis).facility_bridge(
        "parts_produced", base=periods[0], compare=periods[1]
    )
    assert bridge["delta"].sum() == pytest.approx(
        site_kpis[site_kpis["period"] == periods[1]]["parts_produced"].sum()
        - site_kpis[site_kpis["period"] == periods[0]]["parts_produced"].sum()
    )

    board = GlobalScorecard(site_kpis, period_column="period", registry=fleet_registry)
    fleet = board.fleet_summary(
        [
            KpiTarget(
                "oee_pct",
                70.0,
                spec=MetricSpec.weighted("oee_pct", weight="planned_minutes"),
            )
        ]
    ).iloc[0]
    assert fleet["facilities_evaluated"] == 3
    assert fleet["status"] in {"green", "amber", "red"}


def test_reporting_classes_are_exported_at_top_level():
    for name in (
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
    ):
        assert name in ts_shape.__all__, name
        assert getattr(ts_shape, name) is not None


def test_reporting_is_a_catalog_category():
    catalog = ts_shape.list_detectors("reporting")
    assert not catalog.empty
    assert set(catalog["category"]) == {"reporting"}
    assert "FleetRollup" in set(catalog["name"])


def test_top_level_alias_is_the_same_object():
    from ts_shape.reporting.rollup import FleetRollup as Deep

    assert ts_shape.FleetRollup is Deep
