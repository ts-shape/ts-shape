"""Tests for the global KPI scorecard."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import GlobalScorecard, KpiTarget, MetricSpec


@pytest.fixture()
def targets() -> list[KpiTarget]:
    return [
        KpiTarget(
            "oee_pct",
            75.0,
            spec=MetricSpec.weighted("oee_pct", weight="planned_minutes"),
        ),
        KpiTarget("scrap_rate_pct", 2.0, higher_is_better=False),
        KpiTarget(
            "parts_produced",
            2000.0,
            spec=MetricSpec.sum("parts_produced"),
            label="output",
        ),
    ]


@pytest.fixture()
def board(site_kpis, registry) -> GlobalScorecard:
    return GlobalScorecard(site_kpis, period_column="period", registry=registry)


def test_target_validation():
    with pytest.raises(ValueError, match="tolerance_pct"):
        KpiTarget("m", 1.0, tolerance_pct=-1.0)
    with pytest.raises(ValueError, match="weight"):
        KpiTarget("m", 1.0, weight=0.0)


def test_target_status_and_attainment():
    higher = KpiTarget("oee_pct", 80.0, tolerance_pct=5.0)
    assert higher.status(85.0) == "green"
    assert higher.status(77.0) == "amber"  # 3.75 % short
    assert higher.status(60.0) == "red"
    assert higher.status(float("nan")) == "no_data"
    assert higher.attainment(40.0) == pytest.approx(50.0)

    lower = KpiTarget("scrap_pct", 2.0, higher_is_better=False, tolerance_pct=10.0)
    assert lower.status(1.5) == "green"
    assert lower.status(2.1) == "amber"
    assert lower.status(4.0) == "red"
    assert lower.attainment(4.0) == pytest.approx(50.0)


def test_zero_target_never_reads_amber():
    assert KpiTarget("defects", 0.0, higher_is_better=False).status(3.0) == "red"


def test_evaluate_is_long_and_direction_aware(board, targets):
    result = board.evaluate(targets)
    assert len(result) == 9
    by_key = result.set_index(["facility", "metric"])
    assert by_key.loc[("US-DTW", "scrap_rate_pct"), "status"] == "red"
    assert by_key.loc[("US-DTW", "scrap_rate_pct"), "gap"] < 0
    assert by_key.loc[("CN-SHA", "oee_pct"), "status"] == "green"
    assert by_key.loc[("DE-STR", "output"), "value"] == 2100.0


def test_evaluate_by_region_uses_the_registry(board, targets):
    result = board.evaluate(targets, by="region")
    assert set(result["region"]) == {"EMEA", "AMER", "APAC"}


def test_evaluate_needs_a_registry_for_derived_dimensions(site_kpis, targets):
    with pytest.raises(ValueError, match="no FacilityRegistry"):
        GlobalScorecard(site_kpis).evaluate(targets, by="region")


def test_evaluate_requires_targets(board):
    with pytest.raises(ValueError, match="at least one KpiTarget"):
        board.evaluate([])


def test_evaluate_missing_metric(board):
    with pytest.raises(ValueError, match="ghost"):
        board.evaluate([KpiTarget("ghost", 1.0)])


def test_facility_summary_scores_and_ranks(board, targets):
    summary = board.facility_summary(targets).set_index("facility")
    assert summary.loc["CN-SHA", "rank"] == 1.0
    assert summary.loc["US-DTW", "worst_status"] == "red"
    assert summary.loc["US-DTW", "targets_met"] == 0
    assert summary.loc["CN-SHA", "targets_evaluated"] == 3
    assert 0.0 <= summary.loc["DE-STR", "score_pct"] <= 100.0
    assert "oee_pct_status" in summary.columns


def test_facility_summary_weights_the_score():
    frame = pd.DataFrame({"facility": ["A"], "hit": [10.0], "miss": [0.0]})
    board = GlobalScorecard(frame)
    even = board.facility_summary([KpiTarget("hit", 5.0), KpiTarget("miss", 5.0)]).loc[
        0, "score_pct"
    ]
    tilted = board.facility_summary(
        [KpiTarget("hit", 5.0, weight=9.0), KpiTarget("miss", 5.0)]
    ).loc[0, "score_pct"]
    assert even == pytest.approx(50.0)
    assert tilted == pytest.approx(90.0)


def test_no_data_targets_do_not_count_as_met():
    frame = pd.DataFrame({"facility": ["A"], "m": [float("nan")]})
    summary = GlobalScorecard(frame).facility_summary([KpiTarget("m", 1.0)])
    assert summary.loc[0, "targets_evaluated"] == 0
    assert summary.loc[0, "worst_status"] == "no_data"
    assert pd.isna(summary.loc[0, "score_pct"])


def test_fleet_summary_uses_target_aggregations(board, targets):
    fleet = board.fleet_summary(targets).set_index("metric")
    assert fleet.loc["output", "value"] == 7450.0
    assert fleet.loc["oee_pct", "value"] == pytest.approx(
        (70 * 10000 + 72 * 10000 + 60 * 5000 + 58 * 5000 + 85 * 20000 + 86 * 30000)
        / 80000
    )
    # The fleet meets the OEE target while two of three sites do not -- exactly
    # the situation facilities_at_risk exists to surface.
    assert fleet.loc["oee_pct", "status"] == "green"
    assert fleet.loc["oee_pct", "facilities_at_risk"] == 2
    assert fleet.loc["oee_pct", "facilities_evaluated"] == 3


def test_trend_reports_movement(board, targets):
    trend = board.trend(targets).set_index("metric")
    assert trend.loc["output", "base"] == 3500.0
    assert trend.loc["output", "compare"] == 3950.0
    assert trend.loc["output", "movement"] == "improving"
    # The fleet scrap rate is unchanged between the two periods.
    assert trend.loc["scrap_rate_pct", "delta"] == pytest.approx(0.0)
    assert trend.loc["scrap_rate_pct", "movement"] == "flat"


def test_trend_is_direction_aware():
    frame = pd.DataFrame(
        {
            "facility": ["A", "A"],
            "period": ["p1", "p2"],
            "scrap_pct": [5.0, 3.0],
        }
    )
    trend = GlobalScorecard(frame, period_column="period").trend(
        [KpiTarget("scrap_pct", 2.0, higher_is_better=False)]
    )
    # A falling value on a lower-is-better metric is an improvement.
    assert trend.loc[0, "delta"] == pytest.approx(-2.0)
    assert trend.loc[0, "movement"] == "improving"
    assert trend.loc[0, "status_base"] == "red"


def test_trend_needs_a_period_column(site_kpis, targets):
    with pytest.raises(ValueError, match="period column"):
        GlobalScorecard(site_kpis).trend(targets)


def test_trend_rejects_unknown_periods(board, targets):
    with pytest.raises(ValueError, match="not found"):
        board.trend(targets, base="1999")


def test_trend_needs_two_periods(targets):
    frame = pd.DataFrame(
        {
            "facility": ["A"],
            "period": ["p1"],
            "oee_pct": [70.0],
            "scrap_rate_pct": [1.0],
            "parts_produced": [1.0],
            "planned_minutes": [10.0],
        }
    )
    with pytest.raises(ValueError, match="at least two periods"):
        GlobalScorecard(frame, period_column="period").trend(targets)


def test_requires_facility_column(site_kpis):
    with pytest.raises(ValueError, match="Facility column"):
        GlobalScorecard(site_kpis, facility_column="plant")


def test_repr(board):
    assert "facilities=3" in repr(board)
