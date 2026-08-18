"""Tests for variance bridges: who moved the number, and rate versus mix."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import MetricSpec, VarianceBridge


@pytest.fixture()
def bridge(site_kpis) -> VarianceBridge:
    return VarianceBridge(site_kpis)


def test_requires_key_columns(site_kpis):
    with pytest.raises(ValueError, match="not found"):
        VarianceBridge(site_kpis, period_column="week")


def test_facility_bridge_contributions_sum_to_the_total(bridge):
    result = bridge.facility_bridge("parts_produced")
    assert result["delta"].sum() == pytest.approx(450.0)  # 3950 - 3500
    assert result["contribution_pct"].sum() == pytest.approx(100.0)
    assert result.loc[0, "facility"] == "CN-SHA"  # biggest mover first


def test_facility_bridge_direction_and_pct(bridge):
    result = bridge.facility_bridge("parts_produced").set_index("facility")
    assert result.loc["US-DTW", "direction"] == "down"
    assert result.loc["US-DTW", "delta"] == -50.0
    assert result.loc["US-DTW", "delta_pct"] == pytest.approx(-10.0)


def test_facility_bridge_flags_entrants_and_exits():
    frame = pd.DataFrame(
        {
            "facility": ["A", "A", "B", "C"],
            "period": ["p1", "p2", "p1", "p2"],
            "parts": [10.0, 12.0, 5.0, 7.0],
        }
    )
    result = VarianceBridge(frame).facility_bridge("parts").set_index("facility")
    assert result.loc["B", "status"] == "exited"
    assert result.loc["C", "status"] == "entered"
    assert result.loc["B", "compare"] == 0.0


def test_facility_bridge_explicit_periods(bridge, site_kpis):
    result = bridge.facility_bridge(
        "parts_produced",
        base=pd.Timestamp("2026-01-01"),
        compare=pd.Timestamp("2026-02-01"),
    )
    assert result["base_period"].iloc[0] == pd.Timestamp("2026-01-01")


def test_facility_bridge_rejects_unknown_or_equal_periods(bridge):
    with pytest.raises(ValueError, match="not found"):
        bridge.facility_bridge("parts_produced", base="1999-01-01")
    with pytest.raises(ValueError, match="same period"):
        bridge.facility_bridge(
            "parts_produced",
            base=pd.Timestamp("2026-01-01"),
            compare=pd.Timestamp("2026-01-01"),
        )


def test_single_period_frame_cannot_bridge():
    frame = pd.DataFrame({"facility": ["A"], "period": ["p1"], "parts": [1.0]})
    with pytest.raises(ValueError, match="at least two periods"):
        VarianceBridge(frame).facility_bridge("parts")


def test_rate_bridge_effects_reconcile_exactly(bridge):
    result = bridge.rate_bridge("oee_pct", weight_column="planned_minutes")
    total = result[result["facility"] == "TOTAL"].iloc[0]
    assert total["total_effect"] == pytest.approx(
        total["compare_rate"] - total["base_rate"]
    )
    assert total["total_effect"] == pytest.approx(
        total["rate_effect"] + total["mix_effect"] + total["interaction"]
    )
    per_site = result[result["facility"] != "TOTAL"]
    assert per_site["total_effect"].sum() == pytest.approx(total["total_effect"])


def test_rate_bridge_isolates_a_pure_mix_shift():
    # Neither plant changed; volume simply moved to the weaker one.
    frame = pd.DataFrame(
        {
            "facility": ["good", "bad", "good", "bad"],
            "period": ["p1", "p1", "p2", "p2"],
            "oee_pct": [90.0, 50.0, 90.0, 50.0],
            "planned_minutes": [1000.0, 0.0, 0.0, 1000.0],
        }
    )
    result = VarianceBridge(frame).rate_bridge(
        "oee_pct", weight_column="planned_minutes"
    )
    total = result[result["facility"] == "TOTAL"].iloc[0]
    assert total["total_effect"] == pytest.approx(-40.0)
    assert total["rate_effect"] == pytest.approx(0.0)
    assert total["mix_effect"] + total["interaction"] == pytest.approx(-40.0)


def test_rate_bridge_isolates_a_pure_rate_change():
    frame = pd.DataFrame(
        {
            "facility": ["a", "b", "a", "b"],
            "period": ["p1", "p1", "p2", "p2"],
            "oee_pct": [60.0, 80.0, 65.0, 85.0],
            "planned_minutes": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    total = (
        VarianceBridge(frame)
        .rate_bridge("oee_pct", weight_column="planned_minutes")
        .set_index("facility")
        .loc["TOTAL"]
    )
    assert total["rate_effect"] == pytest.approx(5.0)
    assert total["mix_effect"] == pytest.approx(0.0)
    assert total["interaction"] == pytest.approx(0.0)


def test_rate_bridge_requires_weight_in_both_periods():
    frame = pd.DataFrame(
        {
            "facility": ["a", "a"],
            "period": ["p1", "p2"],
            "oee_pct": [60.0, 65.0],
            "planned_minutes": [0.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="is zero"):
        VarianceBridge(frame).rate_bridge("oee_pct", weight_column="planned_minutes")


def test_rate_bridge_missing_columns(bridge):
    with pytest.raises(ValueError, match="ghost"):
        bridge.rate_bridge("ghost", weight_column="planned_minutes")


def test_plan_vs_actual_per_site_targets(bridge):
    result = bridge.plan_vs_actual(
        "parts_produced",
        targets={"DE-STR": 2000.0, "US-DTW": 2000.0, "CN-SHA": 4000.0},
    ).set_index("facility")
    assert result.loc["DE-STR", "status"] == "green"
    assert result.loc["US-DTW", "status"] == "red"
    assert result.loc["CN-SHA", "attainment_pct"] == pytest.approx(110.0)


def test_plan_vs_actual_scalar_target_and_tolerance(bridge):
    result = bridge.plan_vs_actual(
        "parts_produced", targets=2200.0, tolerance_pct=10.0
    ).set_index("facility")
    # 2100 against 2200 is a 4.5 % shortfall -> amber, not red.
    assert result.loc["DE-STR", "status"] == "amber"
    assert result.loc["US-DTW", "status"] == "red"


def test_plan_vs_actual_lower_is_better(bridge):
    result = bridge.plan_vs_actual(
        "scrap_rate_pct",
        targets=5.0,
        higher_is_better=False,
        spec=MetricSpec.mean("scrap_rate_pct"),
    ).set_index("facility")
    assert result.loc["DE-STR", "status"] == "green"
    assert result.loc["US-DTW", "variance"] < 0


def test_plan_vs_actual_without_target(bridge):
    result = bridge.plan_vs_actual(
        "parts_produced", targets={"DE-STR": 100.0}
    ).set_index("facility")
    assert result.loc["CN-SHA", "status"] == "no_target"


def test_plan_vs_actual_for_one_period(bridge):
    result = bridge.plan_vs_actual(
        "parts_produced", targets=1000.0, period=pd.Timestamp("2026-01-01")
    )
    assert (result["period"] == pd.Timestamp("2026-01-01")).all()
    assert result.set_index("facility").loc["DE-STR", "actual"] == 1000.0


def test_plan_vs_actual_unknown_period(bridge):
    with pytest.raises(ValueError, match="not found"):
        bridge.plan_vs_actual("parts_produced", targets=1.0, period="never")


def test_period_over_period(bridge):
    result = bridge.period_over_period("parts_produced")
    assert pd.isna(result.loc[0, "delta"])
    assert result.loc[1, "delta"] == pytest.approx(450.0)
    assert result.loc[1, "delta_pct"] == pytest.approx(450 / 3500 * 100.0)
    assert list(result["contributors"]) == [3, 3]


def test_repr_reports_periods(bridge):
    assert "periods=2" in repr(bridge)
