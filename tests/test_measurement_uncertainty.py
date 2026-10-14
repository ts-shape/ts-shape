import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.measurement_uncertainty import MeasurementUncertaintyEvents


@pytest.fixture
def repeated_measurements_df():
    """Repeated measurements with known std=0.1, mean=50.0."""
    np.random.seed(42)
    n = 500
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.randn() * 0.1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def with_part_uuid_df():
    """Measurements with a separate part UUID signal."""
    np.random.seed(42)
    n = 200
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.randn() * 0.1,
            }
        )
        rows.append(
            {
                "systime": base + pd.Timedelta(minutes=i),
                "uuid": "part_id",
                "value_double": float(i % 3),  # 3 parts cycling
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def type_b_sources():
    return [
        {"source": "calibration", "half_width": 0.05, "distribution": "normal"},
        {"source": "resolution", "half_width": 0.005, "distribution": "rectangular"},
        {"source": "temperature", "half_width": 0.02, "distribution": "triangular"},
    ]


class TestTypeAUncertainty:
    def test_basic(self, repeated_measurements_df):
        mu = MeasurementUncertaintyEvents(
            repeated_measurements_df,
            "sensor_1",
        )
        result = mu.type_a_uncertainty(window="4h")
        assert len(result) > 0
        # With std ≈ 0.1 and n ≈ 240 per 4h, u_A ≈ 0.1/sqrt(240) ≈ 0.0065
        first = result.iloc[0]
        assert 0.001 < first["type_a_uncertainty"] < 0.02
        assert first["n_samples"] > 100
        assert abs(first["mean"] - 50.0) < 0.1

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        mu = MeasurementUncertaintyEvents(df, "sensor_1")
        assert len(mu.type_a_uncertainty()) == 0


class TestTypeBUncertainty:
    def test_multiple_sources(self, type_b_sources):
        result = MeasurementUncertaintyEvents.type_b_uncertainty(type_b_sources)
        assert len(result) == 3
        # Calibration (normal, k=2): 0.05/2 = 0.025
        cal_row = result[result["source"] == "calibration"].iloc[0]
        assert abs(cal_row["type_b_uncertainty"] - 0.025) < 0.001
        # Resolution (rectangular): 0.005/√3 ≈ 0.002887
        res_row = result[result["source"] == "resolution"].iloc[0]
        assert abs(res_row["type_b_uncertainty"] - 0.005 / np.sqrt(3)) < 0.0001

    def test_empty_sources(self):
        result = MeasurementUncertaintyEvents.type_b_uncertainty([])
        assert len(result) == 0


class TestCombinedUncertainty:
    def test_with_type_b(self, repeated_measurements_df, type_b_sources):
        mu = MeasurementUncertaintyEvents(
            repeated_measurements_df,
            "sensor_1",
        )
        result = mu.combined_uncertainty(window="4h", type_b_sources=type_b_sources)
        assert len(result) > 0
        first = result.iloc[0]
        assert first["combined_uncertainty"] > first["type_a"]
        assert first["expanded_uncertainty"] > first["combined_uncertainty"]
        assert first["coverage_factor"] == 2.0
        assert (
            abs(first["expanded_uncertainty"] - 2.0 * first["combined_uncertainty"])
            < 1e-6
        )

    def test_without_type_b(self, repeated_measurements_df):
        mu = MeasurementUncertaintyEvents(
            repeated_measurements_df,
            "sensor_1",
        )
        result = mu.combined_uncertainty(window="4h")
        assert len(result) > 0
        first = result.iloc[0]
        # Without Type B, combined = Type A
        assert abs(first["combined_uncertainty"] - first["type_a"]) < 1e-8
        assert first["type_b_combined"] == 0.0

    def test_custom_coverage(self, repeated_measurements_df):
        mu = MeasurementUncertaintyEvents(
            repeated_measurements_df,
            "sensor_1",
        )
        result = mu.combined_uncertainty(window="4h", coverage_factor=3.0)
        first = result.iloc[0]
        assert first["coverage_factor"] == 3.0
        assert (
            abs(first["expanded_uncertainty"] - 3.0 * first["combined_uncertainty"])
            < 1e-6
        )

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        mu = MeasurementUncertaintyEvents(df, "sensor_1")
        assert len(mu.combined_uncertainty()) == 0


class TestUncertaintyBudget:
    def test_budget_sums_to_100(self, repeated_measurements_df, type_b_sources):
        mu = MeasurementUncertaintyEvents(
            repeated_measurements_df,
            "sensor_1",
        )
        result = mu.uncertainty_budget(window="4h", type_b_sources=type_b_sources)
        assert len(result) == 4  # 1 Type A + 3 Type B
        total_pct = result["pct_contribution"].sum()
        assert abs(total_pct - 100.0) < 0.1

    def test_budget_without_type_b(self, repeated_measurements_df):
        mu = MeasurementUncertaintyEvents(
            repeated_measurements_df,
            "sensor_1",
        )
        result = mu.uncertainty_budget(window="4h")
        assert len(result) == 1
        assert result.iloc[0]["source"] == "repeatability (Type A)"
        assert result.iloc[0]["pct_contribution"] == 100.0

    def test_dominant_source(self, repeated_measurements_df):
        # Large calibration uncertainty should dominate
        sources = [
            {"source": "calibration", "half_width": 1.0, "distribution": "rectangular"},
        ]
        mu = MeasurementUncertaintyEvents(
            repeated_measurements_df,
            "sensor_1",
        )
        result = mu.uncertainty_budget(window="4h", type_b_sources=sources)
        cal_row = result[result["source"] == "calibration (Type B)"].iloc[0]
        assert cal_row["pct_contribution"] > 90.0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        mu = MeasurementUncertaintyEvents(df, "sensor_1")
        assert len(mu.uncertainty_budget()) == 0
