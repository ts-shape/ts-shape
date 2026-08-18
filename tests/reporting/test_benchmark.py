"""Tests for cross-facility benchmarking."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import FacilityBenchmark, MetricSpec


@pytest.fixture()
def bench(site_kpis) -> FacilityBenchmark:
    return FacilityBenchmark(site_kpis, period_column="period")


def test_requires_facility_column(site_kpis):
    with pytest.raises(ValueError, match="Facility column"):
        FacilityBenchmark(site_kpis, facility_column="plant")


def test_rank_orders_best_first(bench):
    result = bench.rank("oee_pct")
    assert list(result["facility"]) == ["CN-SHA", "DE-STR", "US-DTW"]
    assert list(result["rank"]) == [1.0, 2.0, 3.0]
    assert result.loc[0, "gap_to_best"] == 0.0
    assert result.loc[0, "quartile"] == "Q1"


def test_rank_direction_can_be_inverted(bench):
    result = bench.rank("scrap_rate_pct", higher_is_better=False)
    assert result.loc[0, "facility"] == "DE-STR"  # lowest scrap wins
    assert result.loc[0, "gap_to_best"] == 0.0


def test_rank_uses_the_supplied_spec(bench):
    weighted = bench.rank(
        "oee_pct", spec=MetricSpec.weighted("oee_pct", weight="planned_minutes")
    ).set_index("facility")
    assert weighted.loc["CN-SHA", "value"] == pytest.approx(
        (85.0 * 20000 + 86.0 * 30000) / 50000
    )


def test_rank_keeps_facilities_without_data():
    frame = pd.DataFrame({"facility": ["A", "B"], "oee_pct": [70.0, float("nan")]})
    result = FacilityBenchmark(frame).rank("oee_pct")
    assert set(result["facility"]) == {"A", "B"}
    assert result["value"].isna().sum() == 1


def test_rank_with_no_data_at_all():
    frame = pd.DataFrame({"facility": ["A"], "oee_pct": [float("nan")]})
    result = FacilityBenchmark(frame).rank("oee_pct")
    assert len(result) == 1
    assert pd.isna(result.loc[0, "z_score"])


def test_rank_missing_metric(bench):
    with pytest.raises(ValueError, match="ghost"):
        bench.rank("ghost")


def test_quartiles_describe_the_fleet(bench):
    stats = bench.quartiles("oee_pct").iloc[0]
    assert stats["facilities"] == 3
    assert stats["min"] == pytest.approx(59.0)
    assert stats["max"] == pytest.approx(85.5)
    assert stats["top_quartile"] == pytest.approx(stats["q3"])


def test_quartiles_direction_flips_the_threshold(bench):
    lower_better = bench.quartiles("scrap_rate_pct", higher_is_better=False).iloc[0]
    assert lower_better["top_quartile"] == pytest.approx(lower_better["q1"])


def test_quartiles_of_empty_metric():
    frame = pd.DataFrame({"facility": ["A"], "m": [float("nan")]})
    stats = FacilityBenchmark(frame).quartiles("m").iloc[0]
    assert stats["facilities"] == 0
    assert pd.isna(stats["median"])


def test_opportunity_is_expressed_in_exposure_units(bench):
    result = bench.opportunity("oee_pct", exposure_column="planned_minutes").set_index(
        "facility"
    )
    # The leader has nothing to gain; the laggard has the biggest prize.
    assert result.loc["CN-SHA", "opportunity"] == 0.0
    assert result.loc["US-DTW", "opportunity"] > result.loc["DE-STR", "opportunity"]
    assert result.loc["US-DTW", "metric_scale"] == 0.01


def test_opportunity_against_best_and_fixed_targets(bench):
    best = bench.opportunity(
        "oee_pct", exposure_column="planned_minutes", target="best"
    ).set_index("facility")
    fixed = bench.opportunity(
        "oee_pct", exposure_column="planned_minutes", target=90.0
    ).set_index("facility")
    assert best.loc["CN-SHA", "opportunity"] == 0.0
    assert fixed.loc["CN-SHA", "opportunity"] > 0.0
    assert (fixed["target_value"] == 90.0).all()


def test_opportunity_lower_is_better(bench):
    result = bench.opportunity(
        "scrap_rate_pct",
        exposure_column="parts_produced",
        higher_is_better=False,
        target="best",
    ).set_index("facility")
    assert result.loc["DE-STR", "gap"] == 0.0
    assert result.loc["US-DTW", "gap"] > 0.0


def test_opportunity_scale_override(bench):
    result = bench.opportunity(
        "oee_pct",
        exposure_column="planned_minutes",
        metric_scale=1.0,
    )
    assert (result["metric_scale"] == 1.0).all()


def test_opportunity_unknown_target(bench):
    with pytest.raises(ValueError, match="Unknown target"):
        bench.opportunity("oee_pct", exposure_column="planned_minutes", target="vibes")


def test_scorecard_normalises_direction_aware(bench):
    board = bench.scorecard({"oee_pct": True, "scrap_rate_pct": False}).set_index(
        "facility"
    )
    assert board.loc["CN-SHA", "oee_pct_score"] == 100.0
    assert board.loc["DE-STR", "scrap_rate_pct_score"] == 100.0  # lowest scrap
    assert board.loc["US-DTW", "rank"] == 3.0


def test_scorecard_weights_shift_the_composite(bench):
    even = bench.scorecard({"oee_pct": True, "scrap_rate_pct": False}).set_index(
        "facility"
    )
    tilted = bench.scorecard(
        {"oee_pct": True, "scrap_rate_pct": False},
        weights={"scrap_rate_pct": 9.0},
    ).set_index("facility")
    assert (
        tilted.loc["DE-STR", "composite_score"] > even.loc["DE-STR", "composite_score"]
    )


def test_scorecard_uniform_metric_scores_flat():
    frame = pd.DataFrame({"facility": ["A", "B"], "m": [5.0, 5.0]})
    board = FacilityBenchmark(frame).scorecard({"m": True})
    assert list(board["m_score"]) == [100.0, 100.0]


def test_scorecard_requires_metrics(bench):
    with pytest.raises(ValueError, match="at least one metric"):
        bench.scorecard({})


def test_stability_separates_erratic_from_low():
    frame = pd.DataFrame(
        {
            "facility": ["steady"] * 4 + ["erratic"] * 4,
            "period": list(range(4)) * 2,
            "oee_pct": [70.0, 70.0, 70.0, 70.0, 40.0, 100.0, 40.0, 100.0],
        }
    )
    result = (
        FacilityBenchmark(frame, period_column="period")
        .stability("oee_pct")
        .set_index("facility")
    )
    assert result.loc["steady", "mean"] == result.loc["erratic", "mean"]
    assert result.loc["steady", "cv_pct"] == 0.0
    assert result.loc["erratic", "cv_pct"] > 40.0


def test_stability_trend_slope():
    frame = pd.DataFrame(
        {
            "facility": ["A"] * 3,
            "period": [1, 2, 3],
            "m": [10.0, 20.0, 30.0],
        }
    )
    result = FacilityBenchmark(frame, period_column="period").stability("m")
    assert result.loc[0, "trend_per_period"] == pytest.approx(10.0)
    assert result.loc[0, "last"] == 30.0


def test_stability_needs_a_period_column(site_kpis):
    with pytest.raises(ValueError, match="period column"):
        FacilityBenchmark(site_kpis).stability("oee_pct")


def test_stability_single_period_reports_nan_dispersion():
    frame = pd.DataFrame({"facility": ["A"], "period": [1], "m": [5.0]})
    result = FacilityBenchmark(frame, period_column="period").stability("m")
    assert result.loc[0, "periods"] == 1
    assert pd.isna(result.loc[0, "std"])
    assert pd.isna(result.loc[0, "trend_per_period"])


def test_outliers_iqr_and_zscore():
    frame = pd.DataFrame(
        {
            "facility": ["A", "B", "C", "D", "E"],
            "m": [10.0, 11.0, 10.5, 10.2, 100.0],
        }
    )
    bench = FacilityBenchmark(frame)
    iqr = bench.outliers("m").set_index("facility")
    assert bool(iqr.loc["E", "is_outlier"]) is True
    assert iqr.loc["E", "direction"] == "high"
    assert not iqr.loc["A", "is_outlier"]
    z = bench.outliers("m", method="zscore", threshold=1.5).set_index("facility")
    assert bool(z.loc["E", "is_outlier"]) is True


def test_outliers_unknown_method(bench):
    with pytest.raises(ValueError, match="Unknown method"):
        bench.outliers("oee_pct", method="magic")
