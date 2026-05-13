"""Tests for the energy events pack."""

import pandas as pd
import numpy as np
import pytest

from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents
from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_energy_df(hours: int = 48) -> pd.DataFrame:
    """Create a realistic energy + production timeseries."""
    base = pd.Timestamp("2024-01-01")
    rows = []

    for i in range(hours * 60):  # per-minute readings
        t = base + pd.Timedelta(minutes=i)
        hour = t.hour

        # Energy meter: higher during day shifts, lower at night
        if 6 <= hour < 22:
            energy = 50 + np.random.normal(0, 5)  # ~50 kWh/min during shifts
        else:
            energy = 10 + np.random.normal(0, 2)  # ~10 kWh/min idle

        rows.append(
            {
                "systime": t,
                "uuid": "meter:main",
                "value_double": max(0, energy),
                "value_integer": None,
                "value_bool": None,
                "value_string": None,
                "is_delta": True,
            }
        )

        # Production counter (monotonically increasing during day shifts)
        if 6 <= hour < 22:
            counter = 100 + i  # increment every minute
        else:
            counter = 100 + (hour * 60 if hour < 6 else 22 * 60)

        rows.append(
            {
                "systime": t,
                "uuid": "counter:line1",
                "value_double": None,
                "value_integer": counter,
                "value_bool": None,
                "value_string": None,
                "is_delta": True,
            }
        )

        # Machine state (boolean: True=running)
        running = 6 <= hour < 22
        rows.append(
            {
                "systime": t,
                "uuid": "state:machine1",
                "value_double": None,
                "value_integer": None,
                "value_bool": running,
                "value_string": None,
                "is_delta": True,
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def energy_df():
    np.random.seed(42)
    return _make_energy_df(hours=48)


# ---------------------------------------------------------------------------
# EnergyConsumptionEvents
# ---------------------------------------------------------------------------


class TestEnergyConsumptionEvents:

    def test_consumption_by_window(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.consumption_by_window("meter:main", window="1h")
        assert not result.empty
        assert "consumption" in result.columns
        assert "window_start" in result.columns
        assert result["uuid"].iloc[0] == "energy:consumption"
        assert result["source_uuid"].iloc[0] == "meter:main"

    def test_consumption_by_window_empty(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.consumption_by_window("nonexistent:uuid")
        assert result.empty
        assert list(result.columns) == [
            "window_start",
            "uuid",
            "source_uuid",
            "is_delta",
            "consumption",
        ]

    def test_consumption_agg_mean(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.consumption_by_window("meter:main", window="1h", agg="mean")
        assert not result.empty

    def test_peak_demand_detection(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.peak_demand_detection("meter:main", window="1h")
        assert not result.empty
        assert "is_peak" in result.columns
        assert "threshold" in result.columns
        # Some peaks should be detected
        assert result["is_peak"].any()

    def test_peak_demand_with_fixed_threshold(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.peak_demand_detection("meter:main", window="1h", threshold=2000)
        assert not result.empty
        assert result["threshold"].iloc[0] == 2000

    def test_consumption_baseline_deviation(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.consumption_baseline_deviation(
            "meter:main", window="1h", baseline_periods=6
        )
        assert not result.empty
        assert "baseline" in result.columns
        assert "deviation_pct" in result.columns
        assert "is_anomaly" in result.columns

    def test_energy_per_unit(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.energy_per_unit("meter:main", "counter:line1", window="1h")
        assert not result.empty
        assert "energy_per_unit" in result.columns
        assert "units_produced" in result.columns

    def test_energy_per_unit_empty_counter(self, energy_df):
        ec = EnergyConsumptionEvents(energy_df)
        result = ec.energy_per_unit("meter:main", "nonexistent:counter")
        assert result.empty


# ---------------------------------------------------------------------------
# EnergyEfficiencyEvents
# ---------------------------------------------------------------------------


class TestEnergyEfficiencyEvents:

    def test_efficiency_trend(self, energy_df):
        ee = EnergyEfficiencyEvents(energy_df)
        result = ee.efficiency_trend(
            "meter:main", "counter:line1", window="1h", trend_window=6
        )
        assert not result.empty
        assert "efficiency" in result.columns
        assert "rolling_avg_efficiency" in result.columns
        assert "trend_direction" in result.columns

    def test_efficiency_trend_empty(self, energy_df):
        ee = EnergyEfficiencyEvents(energy_df)
        result = ee.efficiency_trend("nonexistent:meter", "counter:line1")
        assert result.empty

    def test_idle_energy_waste(self, energy_df):
        ee = EnergyEfficiencyEvents(energy_df)
        result = ee.idle_energy_waste("meter:main", "state:machine1", window="1h")
        assert not result.empty
        assert "is_idle_waste" in result.columns
        assert "waste_energy" in result.columns
        # Night periods should show idle waste
        assert result["is_idle_waste"].any()

    def test_idle_energy_waste_empty(self, energy_df):
        ee = EnergyEfficiencyEvents(energy_df)
        result = ee.idle_energy_waste("nonexistent:meter", "state:machine1")
        assert result.empty

    def test_specific_energy_consumption(self, energy_df):
        ee = EnergyEfficiencyEvents(energy_df)
        result = ee.specific_energy_consumption(
            "meter:main", "counter:line1", window="1D"
        )
        assert not result.empty
        assert "sec" in result.columns
        assert "sec_trend" in result.columns

    def test_efficiency_comparison(self, energy_df):
        ee = EnergyEfficiencyEvents(energy_df)
        result = ee.efficiency_comparison("meter:main", "counter:line1")
        assert not result.empty
        assert "shift" in result.columns
        assert "avg_efficiency" in result.columns

    def test_efficiency_comparison_custom_shifts(self, energy_df):
        ee = EnergyEfficiencyEvents(energy_df)
        result = ee.efficiency_comparison(
            "meter:main",
            "counter:line1",
            shift_definitions={
                "day": ("06:00", "18:00"),
                "night": ("18:00", "06:00"),
            },
        )
        assert not result.empty
        shifts = result["shift"].tolist()
        assert any(s in ["day", "night"] for s in shifts)
