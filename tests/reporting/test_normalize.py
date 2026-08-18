"""Tests for currency, scale and exposure normalisation."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import MetricNormalizer


@pytest.fixture()
def normalizer(site_kpis, registry) -> MetricNormalizer:
    return MetricNormalizer(site_kpis, registry=registry)


def test_type_is_checked():
    with pytest.raises(TypeError, match="DataFrame"):
        MetricNormalizer({"facility": ["A"]})  # type: ignore[arg-type]


def test_convert_currency_uses_registry_currencies(normalizer):
    result = normalizer.convert_currency(
        ["scrap_cost"],
        to_currency="EUR",
        rates={"EUR": 1.0, "USD": 0.92, "CNY": 0.13},
    ).set_index("facility")
    assert result.loc["DE-STR", "scrap_cost_EUR"].iloc[0] == 1000.0
    assert result.loc["US-DTW", "scrap_cost_EUR"].iloc[0] == pytest.approx(1840.0)
    assert result.loc["CN-SHA", "fx_rate"].iloc[0] == 0.13
    assert (result["reporting_currency"] == "EUR").all()


def test_convert_currency_keeps_source_by_default(normalizer):
    result = normalizer.convert_currency(
        ["scrap_cost"], to_currency="EUR", rates={"EUR": 1.0, "USD": 0.92, "CNY": 0.13}
    )
    assert "scrap_cost" in result.columns
    dropped = normalizer.convert_currency(
        ["scrap_cost"],
        to_currency="EUR",
        rates={"EUR": 1.0, "USD": 0.92, "CNY": 0.13},
        keep_source=False,
    )
    assert "scrap_cost" not in dropped.columns


def test_convert_currency_prefers_an_explicit_column():
    frame = pd.DataFrame(
        {"facility": ["A", "B"], "cost": [100.0, 100.0], "ccy": ["USD", "EUR"]}
    )
    result = MetricNormalizer(frame).convert_currency(
        ["cost"], to_currency="EUR", rates={"USD": 0.9}, currency_column="ccy"
    )
    assert list(result["cost_EUR"]) == [90.0, 100.0]


def test_convert_currency_without_currency_information():
    frame = pd.DataFrame({"facility": ["A"], "cost": [1.0]})
    with pytest.raises(ValueError, match="No currency information"):
        MetricNormalizer(frame).convert_currency(
            ["cost"], to_currency="EUR", rates={"USD": 1.0}
        )


def test_unmapped_currency_becomes_nan_not_zero():
    frame = pd.DataFrame({"facility": ["A"], "cost": [100.0], "currency": ["JPY"]})
    result = MetricNormalizer(frame).convert_currency(
        ["cost"], to_currency="EUR", rates={"USD": 0.9}
    )
    assert pd.isna(result.loc[0, "cost_EUR"])


def test_period_dated_rates_are_applied_asof():
    frame = pd.DataFrame(
        {
            "facility": ["A", "A"],
            "period": pd.to_datetime(["2026-01-15", "2026-02-15"]),
            "cost": [100.0, 100.0],
            "currency": ["USD", "USD"],
        }
    )
    rates = pd.DataFrame(
        {
            "currency": ["USD", "USD"],
            "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "rate": [0.90, 0.95],
        }
    )
    result = MetricNormalizer(frame).convert_currency(
        ["cost"], to_currency="EUR", rates=rates, period_column="period"
    )
    assert list(result["cost_EUR"]) == [90.0, 95.0]


def test_dated_rate_table_requires_a_period_column():
    frame = pd.DataFrame({"facility": ["A"], "cost": [1.0], "currency": ["USD"]})
    rates = pd.DataFrame(
        {"currency": ["USD"], "date": pd.to_datetime(["2026-01-01"]), "rate": [0.9]}
    )
    with pytest.raises(ValueError, match="period-dated"):
        MetricNormalizer(frame).convert_currency(
            ["cost"], to_currency="EUR", rates=rates
        )


def test_flat_rate_table_as_dataframe():
    frame = pd.DataFrame({"facility": ["A"], "cost": [100.0], "currency": ["USD"]})
    rates = pd.DataFrame({"currency": ["USD"], "rate": [0.9]})
    result = MetricNormalizer(frame).convert_currency(
        ["cost"], to_currency="EUR", rates=rates
    )
    assert result.loc[0, "cost_EUR"] == 90.0


def test_rate_table_needs_the_right_columns():
    frame = pd.DataFrame({"facility": ["A"], "cost": [1.0], "currency": ["USD"]})
    with pytest.raises(ValueError, match="missing column"):
        MetricNormalizer(frame).convert_currency(
            ["cost"], to_currency="EUR", rates=pd.DataFrame({"ccy": ["USD"]})
        )


def test_rescale(normalizer):
    result = normalizer.rescale(["parts_produced"], factor=0.001, suffix="_k")
    assert result.loc[0, "parts_produced_k"] == 1.0


def test_per_unit_guards_against_zero_denominators():
    frame = pd.DataFrame(
        {"facility": ["A", "B"], "cost": [10.0, 10.0], "parts": [5.0, 0.0]}
    )
    result = MetricNormalizer(frame).per_unit(["cost"], denominator="parts")
    assert result.loc[0, "cost_per_unit"] == 2.0
    assert pd.isna(result.loc[1, "cost_per_unit"])


def test_per_hour_is_per_unit_with_a_clearer_name(normalizer):
    result = normalizer.per_hour(["parts_produced"], hours_column="planned_minutes")
    assert "parts_produced_per_hour" in result.columns


def test_per_capacity_uses_master_data(normalizer):
    result = normalizer.per_capacity(["parts_produced"]).set_index("facility")
    assert result.loc["DE-STR", "parts_produced_per_capacity"].iloc[0] == 1.0
    assert result.loc["CN-SHA", "parts_produced_per_capacity"].iloc[0] == 1.0


def test_per_capacity_scales_by_days(normalizer, site_kpis):
    with_days = site_kpis.assign(days=30.0)
    result = (
        MetricNormalizer(with_days, registry=normalizer.registry)
        .per_capacity(["parts_produced"], days_column="days")
        .set_index("facility")
    )
    assert result.loc["DE-STR", "parts_produced_per_capacity"].iloc[0] == pytest.approx(
        1000 / 30000
    )


def test_per_capacity_requires_a_registry(site_kpis):
    with pytest.raises(ValueError, match="requires a FacilityRegistry"):
        MetricNormalizer(site_kpis).per_capacity(["parts_produced"])


def test_index_to_baseline_variants(normalizer):
    best = normalizer.index_to_baseline(["oee_pct"], baseline="best").set_index(
        "facility"
    )
    assert best.loc["CN-SHA", "oee_pct_index"].max() == pytest.approx(100.0)

    median = normalizer.index_to_baseline(["oee_pct"], baseline="median")
    assert median["oee_pct_index"].median() == pytest.approx(100.0)

    fixed = normalizer.index_to_baseline(["oee_pct"], baseline=100.0)
    assert fixed.loc[0, "oee_pct_index"] == pytest.approx(70.0)

    site = normalizer.index_to_baseline(["oee_pct"], baseline="DE-STR").set_index(
        "facility"
    )
    assert site.loc["DE-STR", "oee_pct_index"].mean() == pytest.approx(100.0)


def test_index_to_baseline_lower_is_better(normalizer):
    result = normalizer.index_to_baseline(
        ["scrap_rate_pct"], baseline="best", higher_is_better=False
    )
    # The best (lowest) scrap rate becomes the 100 baseline; worse sites index above.
    assert result["scrap_rate_pct_index"].min() == pytest.approx(100.0)


def test_index_to_baseline_per_group(normalizer):
    result = normalizer.index_to_baseline(
        ["oee_pct"], baseline="best", group_by=["period"]
    )
    per_period = result.groupby("period")["oee_pct_index"].max()
    assert per_period.tolist() == pytest.approx([100.0, 100.0])


def test_index_to_baseline_unknown_baseline(normalizer):
    with pytest.raises(ValueError, match="Unknown baseline"):
        normalizer.index_to_baseline(["oee_pct"], baseline="nowhere")


def test_missing_columns_are_reported(normalizer):
    with pytest.raises(ValueError, match="ghost"):
        normalizer.rescale(["ghost"], factor=1.0)
