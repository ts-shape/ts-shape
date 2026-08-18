"""Tests for weighted metric aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import MetricSpec, aggregate_metrics, infer_specs


@pytest.fixture()
def uneven() -> pd.DataFrame:
    """One site ran three shifts at 60 %, another ran 10 minutes at 95 %."""
    return pd.DataFrame(
        {
            "facility": ["big", "small"],
            "oee_pct": [60.0, 95.0],
            "planned_minutes": [1440.0, 10.0],
            "good": [900.0, 10.0],
            "total": [1000.0, 10.0],
            "parts": [1000.0, 10.0],
        }
    )


def test_weighted_mean_beats_naive_mean(uneven):
    result = aggregate_metrics(
        uneven,
        by=[],
        specs=[
            MetricSpec.weighted("oee_pct", weight="planned_minutes"),
            MetricSpec.mean("oee_pct", output="naive_pct"),
        ],
    )
    # The naive mean claims 77.5 %; only 10 minutes of the fleet ran at 95 %.
    assert result.loc[0, "naive_pct"] == pytest.approx(77.5)
    assert result.loc[0, "oee_pct"] == pytest.approx(60.241, abs=1e-3)


def test_ratio_recomputes_from_components(uneven):
    result = aggregate_metrics(
        uneven,
        by=[],
        specs=[
            MetricSpec.ratio(
                "quality_pct", numerator="good", denominator="total", scale=100.0
            )
        ],
    )
    assert result.loc[0, "quality_pct"] == pytest.approx(910 / 1010 * 100.0)


def test_sum_and_contributors(uneven):
    result = aggregate_metrics(
        uneven,
        by=[],
        specs=[MetricSpec.sum("parts")],
        contributor_column="facility",
    )
    assert result.loc[0, "parts"] == 1010.0
    assert result.loc[0, "contributors"] == 2
    assert result.loc[0, "rows"] == 2


def test_grouping_produces_one_row_per_group():
    frame = pd.DataFrame(
        {
            "region": ["EMEA", "EMEA", "APAC"],
            "parts": [1.0, 2.0, 4.0],
        }
    )
    result = aggregate_metrics(frame, by=["region"], specs=[MetricSpec.sum("parts")])
    assert dict(zip(result["region"], result["parts"])) == {"EMEA": 3.0, "APAC": 4.0}


def test_nan_group_keys_are_kept_by_default():
    frame = pd.DataFrame({"region": ["EMEA", None], "parts": [1.0, 2.0]})
    kept = aggregate_metrics(frame, by=["region"], specs=[MetricSpec.sum("parts")])
    assert len(kept) == 2
    dropped = aggregate_metrics(
        frame, by=["region"], specs=[MetricSpec.sum("parts")], dropna_keys=True
    )
    assert len(dropped) == 1


def test_weighted_mean_ignores_zero_and_missing_weights():
    frame = pd.DataFrame(
        {
            "oee_pct": [50.0, 100.0, 80.0],
            "planned_minutes": [100.0, 0.0, float("nan")],
        }
    )
    result = aggregate_metrics(
        frame,
        by=[],
        specs=[MetricSpec.weighted("oee_pct", weight="planned_minutes")],
    )
    assert result.loc[0, "oee_pct"] == pytest.approx(50.0)


def test_weighted_mean_with_no_usable_weight_is_nan():
    frame = pd.DataFrame({"oee_pct": [50.0], "planned_minutes": [0.0]})
    result = aggregate_metrics(
        frame, by=[], specs=[MetricSpec.weighted("oee_pct", weight="planned_minutes")]
    )
    assert pd.isna(result.loc[0, "oee_pct"])


def test_ratio_with_zero_denominator_is_nan():
    frame = pd.DataFrame({"good": [0.0], "total": [0.0]})
    result = aggregate_metrics(
        frame,
        by=[],
        specs=[MetricSpec.ratio("q", numerator="good", denominator="total")],
    )
    assert pd.isna(result.loc[0, "q"])


def test_empty_input_returns_typed_frame():
    frame = pd.DataFrame(columns=["facility", "parts"])
    result = aggregate_metrics(
        frame,
        by=["facility"],
        specs=[MetricSpec.sum("parts")],
        contributor_column="facility",
    )
    assert result.empty
    assert list(result.columns) == ["facility", "parts", "contributors", "rows"]


def test_missing_columns_are_named_in_the_error():
    frame = pd.DataFrame({"facility": ["A"], "parts": [1.0]})
    with pytest.raises(ValueError, match="ghost"):
        aggregate_metrics(frame, by=["facility"], specs=[MetricSpec.sum("ghost")])
    with pytest.raises(ValueError, match="region"):
        aggregate_metrics(frame, by=["region"], specs=[MetricSpec.sum("parts")])


def test_weighted_mean_requires_a_weight():
    with pytest.raises(ValueError, match="requires a weight"):
        MetricSpec("oee_pct", agg="weighted_mean")


def test_ratio_requires_components():
    with pytest.raises(ValueError, match="numerator and denominator"):
        MetricSpec("q", agg="ratio")


def test_unknown_aggregation_rejected():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        MetricSpec("parts", agg="mode")  # type: ignore[arg-type]


def test_specs_required():
    with pytest.raises(ValueError, match="at least one MetricSpec"):
        aggregate_metrics(pd.DataFrame({"a": [1]}), by=[], specs=[])


def test_simple_reductions_and_scale():
    frame = pd.DataFrame({"v": [1.0, 5.0, 3.0], "label": ["a", "a", "b"]})
    result = aggregate_metrics(
        frame,
        by=[],
        specs=[
            MetricSpec("v", agg="min", output="lo"),
            MetricSpec("v", agg="max", output="hi"),
            MetricSpec("v", agg="median", output="mid"),
            MetricSpec("v", agg="first", output="first"),
            MetricSpec("v", agg="last", output="last"),
            MetricSpec("v", agg="count", output="n"),
            MetricSpec("label", agg="nunique", output="labels"),
            MetricSpec("v", agg="sum", output="scaled", scale=0.5),
        ],
    )
    row = result.iloc[0]
    assert (row["lo"], row["hi"], row["mid"]) == (1.0, 5.0, 3.0)
    assert (row["first"], row["last"]) == (1.0, 3.0)
    assert row["n"] == 3.0
    assert row["labels"] == 2.0
    assert row["scaled"] == 4.5


def test_all_nan_metric_is_nan_not_zero():
    frame = pd.DataFrame({"v": [float("nan"), float("nan")]})
    result = aggregate_metrics(frame, by=[], specs=[MetricSpec.sum("v")])
    assert pd.isna(result.loc[0, "v"])


def test_infer_specs_classifies_rates_and_quantities():
    frame = pd.DataFrame(
        {
            "facility": ["A"],
            "parts": [1.0],
            "oee_pct": [50.0],
            "scrap_rate": [1.0],
            "planned_minutes": [10.0],
        }
    )
    specs = {
        spec.column: spec
        for spec in infer_specs(frame, exclude=["facility"], weight="planned_minutes")
    }
    assert specs["parts"].agg == "sum"
    assert specs["oee_pct"].agg == "weighted_mean"
    assert specs["scrap_rate"].weight == "planned_minutes"
    assert "planned_minutes" not in specs  # the weight itself is not a metric


def test_infer_specs_without_weight_falls_back_to_mean():
    frame = pd.DataFrame({"oee_pct": [50.0]})
    assert infer_specs(frame)[0].agg == "mean"
