"""Empty frames, alternate key names and single-facility fleets.

These are the shapes a scheduled report hits in production -- a site that has
not started reporting yet, a KPI table whose key column is called something
else, a pilot rollout with one plant. Every entry point must return a typed
empty frame rather than raising or silently returning an unusable object.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import (
    AssetHierarchy,
    Facility,
    FacilityBenchmark,
    FacilityCalendar,
    FacilityRegistry,
    FleetRollup,
    GlobalScorecard,
    KpiTarget,
    MetricNormalizer,
    MetricSpec,
    ReportingCoverage,
    VarianceBridge,
)


@pytest.fixture()
def empty_kpis() -> pd.DataFrame:
    return pd.DataFrame(columns=["facility", "period", "parts"])


def test_rollup_on_empty_frame(empty_kpis):
    result = FleetRollup(empty_kpis).by_facility([MetricSpec.sum("parts")])
    assert result.empty
    assert list(result.columns) == ["facility", "parts", "rows"]


def test_coverage_on_empty_frame(empty_kpis):
    result = ReportingCoverage(empty_kpis).completeness()
    assert result.empty
    assert "status" in result.columns
    assert ReportingCoverage(empty_kpis).missing().empty


def test_benchmark_on_empty_frame(empty_kpis):
    result = FacilityBenchmark(empty_kpis).rank("parts")
    assert result.empty
    assert {"rank", "percentile", "quartile"} <= set(result.columns)


def test_scorecard_on_empty_frame(empty_kpis):
    result = GlobalScorecard(empty_kpis).facility_summary([KpiTarget("parts", 1.0)])
    assert result.empty
    assert "worst_status" in result.columns


def test_alternate_facility_column_name():
    registry = FacilityRegistry(
        [Facility("A", region="EMEA"), Facility("B", region="APAC")]
    )
    kpis = pd.DataFrame(
        {"plant": ["A", "B"], "period": ["p1", "p1"], "parts": [1.0, 2.0]}
    )

    attached = registry.attach(kpis, facility_column="plant", columns=["region"])
    assert list(attached["region"]) == ["EMEA", "APAC"]
    assert "facility" not in attached.columns  # no stray join key left behind

    rollup = FleetRollup(kpis, facility_column="plant", registry=registry)
    by_region = rollup.by(["region"], [MetricSpec.sum("parts")]).set_index("region")
    assert by_region.loc["EMEA", "parts"] == 1.0

    stacked = rollup.consolidate([MetricSpec.sum("parts")], levels=["region", "plant"])
    assert set(stacked["level"]) == {"global", "region", "plant"}


def test_single_facility_fleet():
    kpis = pd.DataFrame(
        {
            "facility": ["only"],
            "period": ["p1"],
            "oee_pct": [70.0],
            "planned_minutes": [480.0],
        }
    )
    ranked = FacilityBenchmark(kpis).rank("oee_pct")
    assert len(ranked) == 1
    assert ranked.loc[0, "rank"] == 1.0
    assert ranked.loc[0, "gap_to_best"] == 0.0
    # A single-site fleet has no dispersion to speak of.
    assert pd.isna(ranked.loc[0, "z_score"])

    total = FleetRollup(kpis).total(
        [MetricSpec.weighted("oee_pct", weight="planned_minutes")]
    )
    assert total.loc[0, "oee_pct"] == 70.0
    assert total.loc[0, "contributors"] == 1


def test_calendar_on_empty_readings():
    registry = FacilityRegistry([Facility("A", timezone="UTC")])
    calendar = FacilityCalendar(registry)
    empty = pd.DataFrame({"facility": [], "systime": []})
    assert calendar.localize(empty).empty
    assert calendar.assign_shifts(empty).empty
    assert calendar.local_periods(empty).empty


def test_hierarchy_with_a_single_level():
    mapping = pd.DataFrame({"line": ["L1", "L2"]})
    hierarchy = AssetHierarchy(mapping, levels=["line"])
    kpis = pd.DataFrame({"line": ["L1", "L2"], "parts": [1.0, 2.0]})
    result = hierarchy.roll_up(kpis, to_level="line", specs=[MetricSpec.sum("parts")])
    assert dict(zip(result["line"], result["parts"])) == {"L1": 1.0, "L2": 2.0}


def test_normalizer_leaves_input_untouched():
    kpis = pd.DataFrame(
        {"facility": ["A"], "cost": [100.0], "currency": ["USD"], "parts": [10.0]}
    )
    normalizer = MetricNormalizer(kpis)
    normalizer.convert_currency(["cost"], to_currency="EUR", rates={"USD": 0.9})
    normalizer.per_unit(["cost"], denominator="parts")
    assert list(kpis.columns) == ["facility", "cost", "currency", "parts"]


def test_variance_needs_two_periods_but_reports_why():
    kpis = pd.DataFrame({"facility": ["A"], "period": ["p1"], "parts": [1.0]})
    with pytest.raises(ValueError, match="at least two periods"):
        VarianceBridge(kpis).facility_bridge("parts")
