import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.environmental_compensation import (
    EnvironmentalCompensationEvents,
)


@pytest.fixture
def temp_correlated_df():
    """Signal with 0.5 µm/°C temperature sensitivity. Temp varies 18-25°C."""
    np.random.seed(42)
    n = 480  # 8 hours at 1/min
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        t = base + pd.Timedelta(minutes=i)
        temp = 20.0 + 3.0 * np.sin(2 * np.pi * i / 480) + np.random.randn() * 0.1
        # measurement = 100 + 0.5 * temp + noise
        measurement = 100.0 + 0.5 * temp + np.random.randn() * 0.05
        rows.append({"systime": t, "uuid": "sensor_1", "value_double": measurement})
        rows.append({"systime": t, "uuid": "thermocouple", "value_double": temp})
    return pd.DataFrame(rows)


@pytest.fixture
def stable_env_df():
    """Signal with no temperature sensitivity. Temp is constant 20°C."""
    np.random.seed(42)
    n = 300
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        t = base + pd.Timedelta(minutes=i)
        rows.append(
            {
                "systime": t,
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.randn() * 0.1,
            }
        )
        rows.append(
            {
                "systime": t,
                "uuid": "thermocouple",
                "value_double": 20.0 + np.random.randn() * 0.01,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def exceedance_df():
    """Temperature signal that goes out of range at the end."""
    np.random.seed(42)
    n = 300
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        t = base + pd.Timedelta(minutes=i)
        if i < 200:
            temp = 20.0 + np.random.randn() * 0.5
        else:
            temp = 28.0 + np.random.randn() * 0.5  # exceeds valid range
        rows.append({"systime": t, "uuid": "sensor_1", "value_double": 50.0})
        rows.append({"systime": t, "uuid": "thermocouple", "value_double": temp})
    return pd.DataFrame(rows)


class TestDetectCorrelation:
    def test_correlated_signal(self, temp_correlated_df):
        ec = EnvironmentalCompensationEvents(
            temp_correlated_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.detect_correlation(window="4h")
        assert len(result) > 0
        # Should detect strong positive correlation
        assert result.iloc[0]["correlation"] > 0.8
        assert bool(result.iloc[0]["significant"])

    def test_uncorrelated_signal(self, stable_env_df):
        ec = EnvironmentalCompensationEvents(
            stable_env_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.detect_correlation(window="2h")
        assert len(result) > 0
        # With constant temp, correlation should be weak
        assert abs(result.iloc[0]["correlation"]) < 0.5

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        ec = EnvironmentalCompensationEvents(df, "sensor_1", env_uuid="thermo")
        assert len(ec.detect_correlation()) == 0


class TestEstimateSensitivity:
    def test_known_coefficient(self, temp_correlated_df):
        ec = EnvironmentalCompensationEvents(
            temp_correlated_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.estimate_sensitivity(window="8h")
        assert len(result) > 0
        # Coefficient should be close to 0.5
        assert abs(result.iloc[0]["coefficient"] - 0.5) < 0.1
        assert result.iloc[0]["r_squared"] > 0.5

    def test_no_sensitivity(self, stable_env_df):
        ec = EnvironmentalCompensationEvents(
            stable_env_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.estimate_sensitivity(window="4h")
        assert len(result) > 0
        # R² should be low
        assert result.iloc[0]["r_squared"] < 0.5


class TestCompensatedSignal:
    def test_compensation_reduces_variation(self, temp_correlated_df):
        ec = EnvironmentalCompensationEvents(
            temp_correlated_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.compensated_signal()
        assert len(result) > 0
        assert "compensated_value" in result.columns
        assert "correction_applied" in result.columns
        # Compensated signal should have lower std than original
        original_std = result["value_double"].std()
        compensated_std = result["compensated_value"].std()
        assert compensated_std < original_std

    def test_with_explicit_coefficient(self, temp_correlated_df):
        ec = EnvironmentalCompensationEvents(
            temp_correlated_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.compensated_signal(coefficient=0.5)
        assert len(result) > 0
        compensated_std = result["compensated_value"].std()
        original_std = result["value_double"].std()
        assert compensated_std < original_std

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        ec = EnvironmentalCompensationEvents(df, "sensor_1", env_uuid="thermo")
        assert len(ec.compensated_signal()) == 0


class TestDetectEnvExceedance:
    def test_detects_high_temp(self, exceedance_df):
        ec = EnvironmentalCompensationEvents(
            exceedance_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.detect_env_exceedance(valid_range=(15.0, 25.0), window="1h")
        assert len(result) > 0
        # Should flag "high" exceedance in later windows
        high_events = result[result["exceedance_type"] == "high"]
        assert len(high_events) > 0

    def test_no_exceedance(self, stable_env_df):
        ec = EnvironmentalCompensationEvents(
            stable_env_df,
            "sensor_1",
            env_uuid="thermocouple",
        )
        result = ec.detect_env_exceedance(valid_range=(15.0, 25.0), window="1h")
        assert len(result) == 0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        ec = EnvironmentalCompensationEvents(df, "sensor_1", env_uuid="thermo")
        assert len(ec.detect_env_exceedance(valid_range=(15.0, 25.0))) == 0


class TestConstructorValidation:
    def test_missing_env_raises(self):
        df = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01")],
                "uuid": ["s1"],
                "value_double": [1.0],
            }
        )
        with pytest.raises(ValueError, match="env_uuid"):
            EnvironmentalCompensationEvents(df, "s1")
