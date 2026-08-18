"""Tests for facility, region and global consolidation."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import AssetHierarchy, FleetRollup, MetricSpec


@pytest.fixture()
def specs() -> list[MetricSpec]:
    return [
        MetricSpec.sum("parts_produced"),
        MetricSpec.ratio(
            "quality_pct", numerator="good", denominator="total", scale=100.0
        ),
        MetricSpec.weighted("oee_pct", weight="planned_minutes"),
    ]


@pytest.fixture()
def rollup(site_kpis, registry) -> FleetRollup:
    return FleetRollup(site_kpis, registry=registry)


def test_requires_facility_column(site_kpis):
    with pytest.raises(ValueError, match="Facility column"):
        FleetRollup(site_kpis, facility_column="plant")


def test_by_facility(rollup, specs):
    result = rollup.by_facility(specs).set_index("facility")
    assert result.loc["DE-STR", "parts_produced"] == 2100.0
    assert result.loc["DE-STR", "rows"] == 2
    assert result.loc["CN-SHA", "oee_pct"] == pytest.approx(
        (85.0 * 20000 + 86.0 * 30000) / 50000
    )


def test_by_facility_per_period(rollup, specs):
    result = rollup.by_facility(specs, per_period=True)
    assert len(result) == 6
    assert {"facility", "period"} <= set(result.columns)


def test_by_period_counts_contributors(rollup, specs):
    result = rollup.by_period(specs)
    assert list(result["contributors"]) == [3, 3]
    assert result.loc[0, "parts_produced"] == 3500.0


def test_by_region_uses_registry(rollup, specs):
    result = rollup.by(["region"], specs).set_index("region")
    assert set(result.index) == {"EMEA", "AMER", "APAC"}
    assert result.loc["EMEA", "parts_produced"] == 2100.0


def test_by_requires_a_dimension(rollup, specs):
    with pytest.raises(ValueError, match="at least one dimension"):
        rollup.by([], specs)


def test_unresolvable_dimension_is_reported(site_kpis, specs):
    plain = FleetRollup(site_kpis)  # no registry
    with pytest.raises(ValueError, match="cannot"):
        plain.by(["region"], specs)


def test_total_is_the_headline_number(rollup, specs):
    total = rollup.total(specs)
    assert len(total) == 1
    assert total.loc[0, "parts_produced"] == 7450.0
    assert total.loc[0, "contributors"] == 3


def test_total_quality_is_recomputed_not_averaged(rollup, specs):
    total = rollup.total(specs)
    good = 980 + 1078 + 475 + 423 + 1900 + 2304
    produced = 1000 + 1100 + 500 + 450 + 2000 + 2400
    assert total.loc[0, "quality_pct"] == pytest.approx(good / produced * 100.0)


def test_consolidate_stacks_levels(rollup, specs):
    report = rollup.consolidate(specs)
    assert list(report.columns[:2]) == ["level", "node"]
    assert set(report["level"]) == {"global", "region", "facility"}
    global_row = report[report["level"] == "global"].iloc[0]
    facility_sum = report[report["level"] == "facility"]["parts_produced"].sum()
    assert global_row["parts_produced"] == pytest.approx(facility_sum)


def test_consolidate_can_drop_the_total_and_keep_periods(rollup, specs):
    report = rollup.consolidate(
        specs, levels=["region"], per_period=True, include_total=False
    )
    assert set(report["level"]) == {"region"}
    assert "period" in report.columns
    assert len(report) == 6


def test_contribution_is_a_pareto(rollup):
    result = rollup.contribution("parts_produced")
    assert list(result["facility"]) == ["CN-SHA", "DE-STR", "US-DTW"]
    assert list(result["rank"]) == [1, 2, 3]
    assert result["cumulative_share_pct"].iloc[-1] == pytest.approx(100.0)


def test_contribution_by_region(rollup):
    result = rollup.contribution("parts_produced", by="region")
    assert set(result["region"]) == {"EMEA", "AMER", "APAC"}


def test_contribution_ascending_finds_laggards(rollup):
    result = rollup.contribution("parts_produced", ascending=True)
    assert result.loc[0, "facility"] == "US-DTW"


def test_contribution_with_zero_total_does_not_divide_by_zero():
    frame = pd.DataFrame({"facility": ["A", "B"], "parts": [0.0, 0.0]})
    result = FleetRollup(frame).contribution("parts")
    assert result["share_pct"].isna().all()


def test_contribution_missing_metric(rollup):
    with pytest.raises(ValueError, match="ghost"):
        rollup.contribution("ghost")


def test_pivot_matrix(rollup):
    matrix = rollup.pivot("parts_produced")
    assert list(matrix.index) == ["CN-SHA", "DE-STR", "US-DTW"]
    assert matrix.shape == (3, 2)
    assert matrix.loc["CN-SHA"].sum() == 4400.0


def test_pivot_empty_frame():
    frame = pd.DataFrame(columns=["facility", "period", "parts"])
    assert FleetRollup(frame).pivot("parts").empty


def test_specs_are_inferred_when_omitted(site_kpis):
    result = FleetRollup(site_kpis).by_facility()
    assert "parts_produced" in result.columns
    # A rate-looking column falls back to an unweighted mean.
    assert result.set_index("facility").loc["DE-STR", "oee_pct"] == pytest.approx(71.0)


def test_inference_fails_loudly_without_numeric_columns():
    frame = pd.DataFrame({"facility": ["A"], "period": ["2026-01"], "note": ["x"]})
    with pytest.raises(ValueError, match="No numeric metric columns"):
        FleetRollup(frame).by_facility()


def test_missing_period_column_is_reported(site_kpis, registry, specs):
    no_period = site_kpis.drop(columns=["period"])
    with pytest.raises(ValueError, match="Period column"):
        FleetRollup(no_period, registry=registry).by_period(specs)


def test_hierarchy_dimensions_take_precedence(site_kpis):
    mapping = pd.DataFrame(
        {
            "cluster": ["west", "west", "east"],
            "facility": ["DE-STR", "US-DTW", "CN-SHA"],
        }
    )
    hierarchy = AssetHierarchy(mapping, levels=["cluster", "facility"])
    rollup = FleetRollup(site_kpis, hierarchy=hierarchy)
    result = rollup.by(["cluster"], [MetricSpec.sum("parts_produced")]).set_index(
        "cluster"
    )
    assert result.loc["west", "parts_produced"] == 3050.0
    assert result.loc["east", "parts_produced"] == 4400.0


def test_repr_reports_scale(rollup):
    assert "facilities=3" in repr(rollup)
