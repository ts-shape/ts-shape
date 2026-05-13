import pytest
import pandas as pd
import numpy as np
from ts_shape.events.quality.gauge_repeatability import GaugeRepeatabilityEvents


@pytest.fixture
def gauge_rr_df():
    """3 parts × 3 operators × 5 trials with controlled variance."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []
    part_targets = {"part_A": 10.0, "part_B": 20.0, "part_C": 30.0}
    operators = ["op_1", "op_2", "op_3"]
    n_trials = 5
    idx = 0

    for part, target in part_targets.items():
        for op in operators:
            op_bias = {"op_1": 0.0, "op_2": 0.1, "op_3": -0.05}[op]
            for trial in range(n_trials):
                rows.append(
                    {
                        "systime": base + pd.Timedelta(seconds=idx),
                        "uuid": "gauge_1",
                        "value_double": target + op_bias + np.random.randn() * 0.05,
                        "value_string": part,
                        "operator": op,
                    }
                )
                idx += 1

    return pd.DataFrame(rows)


@pytest.fixture
def simple_gauge_df():
    """Simple 2-part dataset without operators."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(20):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "gauge_1",
                "value_double": 50.0 + np.random.randn() * 0.1,
                "value_string": "part_X",
            }
        )
    for i in range(20):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=20 + i),
                "uuid": "gauge_1",
                "value_double": 60.0 + np.random.randn() * 0.1,
                "value_string": "part_Y",
            }
        )
    return pd.DataFrame(rows)


class TestRepeatability:
    def test_per_part_ev(self, gauge_rr_df):
        grr = GaugeRepeatabilityEvents(
            gauge_rr_df,
            "gauge_1",
            part_column="value_string",
        )
        result = grr.repeatability()
        assert len(result) == 3
        assert set(result["part"].tolist()) == {"part_A", "part_B", "part_C"}
        # EV should be small (controlled noise ~0.05 std)
        assert all(ev < 1.0 for ev in result["EV"].tolist())

    def test_simple(self, simple_gauge_df):
        grr = GaugeRepeatabilityEvents(
            simple_gauge_df,
            "gauge_1",
            part_column="value_string",
        )
        result = grr.repeatability()
        assert len(result) == 2

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double", "value_string"])
        grr = GaugeRepeatabilityEvents(df, "gauge_1")
        assert len(grr.repeatability()) == 0


class TestReproducibility:
    def test_with_operators(self, gauge_rr_df):
        grr = GaugeRepeatabilityEvents(
            gauge_rr_df,
            "gauge_1",
            part_column="value_string",
            operator_column="operator",
        )
        result = grr.reproducibility()
        assert len(result) == 3
        assert set(result["operator"].tolist()) == {"op_1", "op_2", "op_3"}
        # AV should be > 0 since operators have different biases
        assert result.iloc[0]["AV"] > 0

    def test_no_operator_column(self, simple_gauge_df):
        grr = GaugeRepeatabilityEvents(
            simple_gauge_df,
            "gauge_1",
            part_column="value_string",
        )
        result = grr.reproducibility()
        assert len(result) == 0


class TestGaugeRRSummary:
    def test_full_summary(self, gauge_rr_df):
        grr = GaugeRepeatabilityEvents(
            gauge_rr_df,
            "gauge_1",
            part_column="value_string",
            operator_column="operator",
        )
        result = grr.gauge_rr_summary()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["EV"] > 0
        assert row["GRR"] > 0
        assert row["TV"] > 0
        assert 0 <= row["pct_GRR"] <= 100
        assert row["ndc"] >= 0

    def test_with_tolerance(self, gauge_rr_df):
        grr = GaugeRepeatabilityEvents(
            gauge_rr_df,
            "gauge_1",
            part_column="value_string",
            operator_column="operator",
        )
        result = grr.gauge_rr_summary(tolerance_range=5.0)
        assert "pct_GRR_tolerance" in result.columns
        assert result.iloc[0]["pct_GRR_tolerance"] > 0

    def test_without_operator(self, simple_gauge_df):
        grr = GaugeRepeatabilityEvents(
            simple_gauge_df,
            "gauge_1",
            part_column="value_string",
        )
        result = grr.gauge_rr_summary()
        assert len(result) == 1
        # Without operators, AV should be 0
        assert result.iloc[0]["AV"] == 0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double", "value_string"])
        grr = GaugeRepeatabilityEvents(df, "gauge_1")
        assert len(grr.gauge_rr_summary()) == 0


class TestMeasurementBias:
    def test_bias_calculation(self, gauge_rr_df):
        grr = GaugeRepeatabilityEvents(
            gauge_rr_df,
            "gauge_1",
            part_column="value_string",
        )
        refs = {"part_A": 10.0, "part_B": 20.0, "part_C": 30.0}
        result = grr.measurement_bias(refs)
        assert len(result) == 3
        # Bias should be small (noise is ~0.05 std, operator bias ~0.05)
        for _, row in result.iterrows():
            assert abs(row["bias"]) < 0.5

    def test_partial_references(self, gauge_rr_df):
        grr = GaugeRepeatabilityEvents(
            gauge_rr_df,
            "gauge_1",
            part_column="value_string",
        )
        refs = {"part_A": 10.0}
        result = grr.measurement_bias(refs)
        assert len(result) == 1
        assert result.iloc[0]["part"] == "part_A"

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double", "value_string"])
        grr = GaugeRepeatabilityEvents(df, "gauge_1")
        assert len(grr.measurement_bias({"p1": 1.0})) == 0
