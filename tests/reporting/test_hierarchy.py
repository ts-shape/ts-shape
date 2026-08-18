"""Tests for the asset hierarchy and roll-up along it."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import AssetHierarchy, MetricSpec


@pytest.fixture()
def mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["EMEA", "EMEA", "AMER", "AMER"],
            "site": ["DE-STR", "DE-STR", "US-DTW", "US-DTW"],
            "line": ["L1", "L2", "L3", "L4"],
        }
    )


@pytest.fixture()
def hierarchy(mapping) -> AssetHierarchy:
    return AssetHierarchy(mapping, levels=["region", "site", "line"])


@pytest.fixture()
def line_kpis() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line": ["L1", "L2", "L3", "L4"],
            "period": ["2026-01"] * 4,
            "parts": [10.0, 20.0, 30.0, 40.0],
            "oee_pct": [50.0, 70.0, 60.0, 80.0],
            "planned_minutes": [100.0, 100.0, 300.0, 100.0],
        }
    )


def test_levels_and_leaf(hierarchy):
    assert hierarchy.levels == ["region", "site", "line"]
    assert hierarchy.leaf_level == "line"
    assert hierarchy.levels_up_to("site") == ["region", "site"]


def test_requires_levels_present(mapping):
    with pytest.raises(ValueError, match="not found in mapping"):
        AssetHierarchy(mapping, levels=["region", "cell"])
    with pytest.raises(ValueError, match="at least one level"):
        AssetHierarchy(mapping, levels=[])


def test_type_is_checked():
    with pytest.raises(TypeError, match="DataFrame"):
        AssetHierarchy({"line": ["L1"]}, levels=["line"])  # type: ignore[arg-type]


def test_nodes_children_parent_path(hierarchy):
    assert hierarchy.nodes("site") == ["DE-STR", "US-DTW"]
    assert hierarchy.children("EMEA", level="region", child_level="line") == [
        "L1",
        "L2",
    ]
    assert hierarchy.parent("L1", level="line", parent_level="region") == "EMEA"
    assert hierarchy.path("L3") == {"region": "AMER", "site": "US-DTW", "line": "L3"}
    assert hierarchy.path("GHOST")["site"] is None


def test_direction_is_enforced(hierarchy):
    with pytest.raises(ValueError, match="finer"):
        hierarchy.children("L1", level="line", child_level="region")
    with pytest.raises(ValueError, match="coarser"):
        hierarchy.parent("EMEA", level="region", parent_level="line")


def test_validate_clean_hierarchy(hierarchy):
    issues = hierarchy.validate()
    assert issues.empty
    assert list(issues.columns) == ["level", "node", "issue", "detail"]


def test_validate_detects_double_counting():
    mapping = pd.DataFrame(
        {
            "region": ["EMEA", "AMER"],
            "site": ["DE-STR", "DE-STR"],  # one site, two regions
            "line": ["L1", "L1"],
        }
    )
    issues = AssetHierarchy(mapping, levels=["region", "site", "line"]).validate()
    assert "multiple_parents" in set(issues["issue"])
    assert "duplicate_leaf" in set(issues["issue"])


def test_validate_detects_missing_parent():
    mapping = pd.DataFrame(
        {"region": ["EMEA", None], "site": ["A", "B"], "line": ["L1", "L2"]}
    )
    issues = AssetHierarchy(mapping, levels=["region", "site", "line"]).validate()
    assert set(issues["issue"]) == {"missing_parent"}
    assert list(issues["node"]) == ["L2"]


def test_enrich_attaches_ancestors(hierarchy, line_kpis):
    enriched = hierarchy.enrich(line_kpis)
    assert list(enriched["region"]) == ["EMEA", "EMEA", "AMER", "AMER"]
    assert list(enriched["site"]) == ["DE-STR", "DE-STR", "US-DTW", "US-DTW"]
    assert len(enriched) == len(line_kpis)


def test_enrich_keeps_unmapped_nodes(hierarchy, line_kpis):
    extra = pd.concat(
        [line_kpis, pd.DataFrame([{"line": "L9", "period": "2026-01", "parts": 5.0}])],
        ignore_index=True,
    )
    enriched = hierarchy.enrich(extra)
    assert len(enriched) == 5
    assert pd.isna(enriched.iloc[-1]["site"])


def test_enrich_rejects_unknown_join_column(hierarchy, line_kpis):
    with pytest.raises(ValueError, match="Join column"):
        hierarchy.enrich(line_kpis, on="workcell")


def test_enrich_rejects_unknown_level(hierarchy, line_kpis):
    with pytest.raises(ValueError, match="Unknown level"):
        hierarchy.enrich(line_kpis, levels=["planet"])


def test_roll_up_to_site(hierarchy, line_kpis):
    result = hierarchy.roll_up(
        line_kpis,
        to_level="site",
        specs=[
            MetricSpec.sum("parts"),
            MetricSpec.weighted("oee_pct", weight="planned_minutes"),
        ],
        group_extra=["period"],
    )
    by_site = result.set_index("site")
    assert list(result.columns[:3]) == ["region", "site", "period"]
    assert by_site.loc["DE-STR", "parts"] == 30.0
    assert by_site.loc["DE-STR", "oee_pct"] == pytest.approx(60.0)
    # US-DTW's L3 ran three times the minutes of L4, so the weighted OEE
    # leans toward L3's 60 % rather than the naive 70 %.
    assert by_site.loc["US-DTW", "oee_pct"] == pytest.approx(65.0)
    assert by_site.loc["US-DTW", "contributors"] == 2


def test_roll_up_to_region(hierarchy, line_kpis):
    result = hierarchy.roll_up(
        line_kpis, to_level="region", specs=[MetricSpec.sum("parts")]
    )
    assert dict(zip(result["region"], result["parts"])) == {"EMEA": 30.0, "AMER": 70.0}


def test_roll_up_to_leaf_level_is_identity_grained(hierarchy, line_kpis):
    result = hierarchy.roll_up(
        line_kpis, to_level="line", specs=[MetricSpec.sum("parts")]
    )
    assert len(result) == 4
    assert set(result["line"]) == {"L1", "L2", "L3", "L4"}


def test_roll_up_unknown_level(hierarchy, line_kpis):
    with pytest.raises(ValueError, match="Unknown level"):
        hierarchy.roll_up(line_kpis, to_level="planet", specs=[MetricSpec.sum("parts")])


def test_duplicate_mapping_rows_do_not_duplicate_kpis(line_kpis):
    mapping = pd.DataFrame(
        {
            "site": ["DE-STR", "DE-STR", "DE-STR"],
            "line": ["L1", "L1", "L2"],  # L1 listed twice, same ancestry
        }
    )
    hierarchy = AssetHierarchy(mapping, levels=["site", "line"])
    enriched = hierarchy.enrich(line_kpis[line_kpis["line"].isin(["L1", "L2"])])
    assert len(enriched) == 2


def test_repr_mentions_levels(hierarchy):
    assert "levels=" in repr(hierarchy)
