"""Tests for new energy event classes and data-model agnosticism."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.energy.carbon_intensity import CarbonIntensityEvents
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents
from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents
from ts_shape.events.energy.energy_performance_indicator import (
    EnergyPerformanceIndicatorEvents,
)
from ts_shape.events.energy.idle_energy_detection import IdleEnergyDetectionEvents

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_df(hours: int = 48) -> pd.DataFrame:
    """48-hour standard-model DataFrame with energy, counter, and state signals."""
    idx = pd.date_range("2026-01-13", periods=hours * 4, freq="15min", tz="UTC")
    n = len(idx)
    hour_of_day = idx.hour
    is_production = (hour_of_day >= 6) & (hour_of_day < 22)

    energy_vals = np.where(
        is_production, 50.0 + np.random.default_rng(0).normal(0, 5, n), 5.0
    )
    counter_vals = np.where(is_production, 100, 0).cumsum()
    state_vals = is_production.astype(bool)

    rows = []
    for i, ts in enumerate(idx):
        rows.append(
            {
                "systime": ts,
                "uuid": "meter:main",
                "value_double": energy_vals[i],
                "is_delta": True,
            }
        )
        rows.append(
            {
                "systime": ts,
                "uuid": "counter:line1",
                "value_integer": int(counter_vals[i]),
                "is_delta": True,
            }
        )
        rows.append(
            {
                "systime": ts,
                "uuid": "state:machine1",
                "value_bool": state_vals[i],
                "is_delta": True,
            }
        )

    return pd.DataFrame(rows)


def _make_csv_df(hours: int = 48) -> pd.DataFrame:
    """Raw CSV model: time | id | value columns only."""
    idx = pd.date_range("2026-01-13", periods=hours * 4, freq="15min", tz="UTC")
    n = len(idx)
    hour_of_day = idx.hour
    is_production = (hour_of_day >= 6) & (hour_of_day < 22)
    energy_vals = np.where(
        is_production, 50.0 + np.random.default_rng(1).normal(0, 5, n), 5.0
    )

    rows = [
        {"time": ts, "id": "meter:main", "value": energy_vals[i]}
        for i, ts in enumerate(idx)
    ]
    return pd.DataFrame(rows)


# ── normalize() static method ────────────────────────────────────────────────


class TestNormalizeStatic:
    def test_two_column_format(self):
        raw = pd.DataFrame(
            {
                "time": ["2026-01-13T00:00:00Z", "2026-01-13T00:15:00Z"],
                "value": [107.6, 184.09],
            }
        )
        out = EnergyConsumptionEvents.normalize(raw, series_id="sensor_001")
        assert list(out.columns) == ["systime", "uuid", "value_double", "is_delta"]
        assert (out["uuid"] == "sensor_001").all()
        assert out["value_double"].iloc[0] == pytest.approx(107.6)

    def test_id_column_overrides_series_id(self):
        raw = pd.DataFrame(
            {
                "time": ["2026-01-13T00:00:00Z"],
                "id": ["meter_42"],
                "value": [55.0],
            }
        )
        out = EnergyConsumptionEvents.normalize(
            raw, series_id="ignored", id_column="id"
        )
        assert out["uuid"].iloc[0] == "meter_42"

    def test_bad_values_become_nan(self):
        raw = pd.DataFrame({"time": ["2026-01-13T00:00:00Z"], "value": ["bad"]})
        out = EnergyConsumptionEvents.normalize(raw, series_id="s1")
        assert pd.isna(out["value_double"].iloc[0])

    def test_efficiency_normalize_equivalent(self):
        raw = pd.DataFrame(
            {
                "time": ["2026-01-13T00:00:00Z"],
                "value": [100.0],
            }
        )
        out1 = EnergyConsumptionEvents.normalize(raw, series_id="m1")
        out2 = EnergyEfficiencyEvents.normalize(raw, series_id="m1")
        pd.testing.assert_frame_equal(out1, out2)


# ── uuid_column agnosticism (existing classes) ────────────────────────────────


class TestUuidColumnAgnosticism:
    def test_consumption_with_id_column(self):
        df = _make_csv_df(hours=4)
        events = EnergyConsumptionEvents(df, time_column="time", uuid_column="id")
        result = events.consumption_by_window(
            "meter:main", value_column="value", window="1h"
        )
        assert not result.empty
        assert "consumption" in result.columns

    def test_efficiency_with_id_column(self):
        # Build a small CSV-model df with energy + state
        idx = pd.date_range("2026-01-13", periods=16, freq="15min", tz="UTC")
        rows = []
        for i, ts in enumerate(idx):
            rows.append({"time": ts, "id": "meter:main", "value": 50.0})
            rows.append({"time": ts, "id": "counter:line1", "value": float(i * 100)})
        df = pd.DataFrame(rows)
        events = EnergyEfficiencyEvents(df, time_column="time", uuid_column="id")
        result = events.specific_energy_consumption(
            "meter:main",
            "counter:line1",
            value_column_energy="value",
            value_column_counter="value",
            window="1h",
        )
        assert not result.empty

    def test_standard_model_unchanged(self):
        df = _make_df(hours=4)
        events = EnergyConsumptionEvents(df)
        result = events.consumption_by_window("meter:main", window="1h")
        assert not result.empty


# ── IdleEnergyDetectionEvents ────────────────────────────────────────────────


class TestIdleEnergyDetectionEvents:
    def setup_method(self):
        self.df = _make_df(hours=48)
        self.events = IdleEnergyDetectionEvents(self.df)

    def test_by_window_returns_expected_columns(self):
        result = self.events.idle_energy_by_window(
            "meter:main", "state:machine1", window="1h"
        )
        expected = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "is_delta",
            "total_energy",
            "idle_energy",
            "running_energy",
            "machine_running_pct",
            "idle_fraction",
        ]
        assert list(result.columns) == expected

    def test_night_windows_have_idle_energy(self):
        result = self.events.idle_energy_by_window(
            "meter:main", "state:machine1", window="1h"
        )
        night = result[result["start"].dt.hour < 6]
        # Night windows should have non-zero idle energy
        assert (night["idle_energy"] > 0).any()

    def test_idle_fraction_between_zero_and_one(self):
        result = self.events.idle_energy_by_window(
            "meter:main", "state:machine1", window="1h"
        )
        assert (result["idle_fraction"] >= 0.0).all()
        assert (result["idle_fraction"] <= 1.0).all()

    def test_by_shift_returns_three_shifts(self):
        result = self.events.idle_energy_by_shift("meter:main", "state:machine1")
        assert len(result) == 3
        assert set(result["shift"]) == {"shift_1", "shift_2", "shift_3"}

    def test_trend_direction_values(self):
        result = self.events.idle_energy_trend(
            "meter:main", "state:machine1", window="1h", trend_window=3
        )
        _valid = {"improving", "stable", "worsening", None}
        assert set(result["trend_direction"].dropna().unique()).issubset(
            {"improving", "stable", "worsening"}
        )

    def test_empty_on_unknown_uuid(self):
        result = self.events.idle_energy_by_window("unknown", "state:machine1")
        assert result.empty

    def test_raw_csv_model_path(self):
        df = _make_csv_df(hours=4)
        # Add boolean state signal in raw CSV format
        idx = pd.date_range("2026-01-13", periods=16, freq="15min", tz="UTC")
        state_rows = [
            {
                "time": ts,
                "id": "state:machine1",
                "value": float(ts.hour >= 6 and ts.hour < 22),
            }
            for ts in idx
        ]
        df = pd.concat([df, pd.DataFrame(state_rows)], ignore_index=True)
        events = IdleEnergyDetectionEvents(df, time_column="time", uuid_column="id")
        result = events.idle_energy_by_window(
            "meter:main",
            "state:machine1",
            value_column_energy="value",
            value_column_state="value",
        )
        assert not result.empty


class TestIdleFractionScale:
    """machine_running_pct is a 0-1 fraction, so idle_fraction = 1 - pct."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        # One hour, 10 one-per-6min samples: 3 running, 7 idle; 1 kWh each.
        t = pd.date_range("2026-01-13 08:00", periods=10, freq="6min", tz="UTC")
        energy = pd.DataFrame({"systime": t, "uuid": "meter:main", "value_double": 1.0})
        state = pd.DataFrame(
            {"systime": t, "uuid": "state:m1", "value_bool": [True] * 3 + [False] * 7}
        )
        return pd.concat([energy, state], ignore_index=True)

    def test_by_window_uses_fraction(self):
        ev = IdleEnergyDetectionEvents(self._frame())
        row = ev.idle_energy_by_window("meter:main", "state:m1", window="1h").iloc[0]
        assert row["machine_running_pct"] == pytest.approx(0.3)
        assert row["idle_fraction"] == pytest.approx(0.7)
        assert row["idle_energy"] == pytest.approx(7.0)
        assert row["running_energy"] == pytest.approx(3.0)
        assert row["duration_seconds"] == pytest.approx(3600.0)

    def test_by_shift_uses_fraction(self):
        ev = IdleEnergyDetectionEvents(self._frame())
        res = ev.idle_energy_by_shift("meter:main", "state:m1").set_index("shift")
        assert res.loc["shift_1", "total_energy"] == pytest.approx(10.0)
        assert res.loc["shift_1", "idle_energy"] == pytest.approx(7.0)
        assert res.loc["shift_1", "waste_fraction"] == pytest.approx(0.7)


# ── EnergyPerformanceIndicatorEvents ─────────────────────────────────────────


class TestEnergyPerformanceIndicatorEvents:
    def setup_method(self):
        self.df = _make_df(hours=48)
        self.events = EnergyPerformanceIndicatorEvents(self.df)

    def test_enpi_by_window_columns(self):
        result = self.events.enpi_by_window("meter:main", "counter:line1", window="1D")
        assert "enpi" in result.columns
        assert "energy_kwh" in result.columns
        assert "units_produced" in result.columns

    def test_enpi_is_positive(self):
        result = self.events.enpi_by_window("meter:main", "counter:line1", window="1D")
        valid = result.dropna(subset=["enpi"])
        assert (valid["enpi"] > 0).all()

    def test_enpi_vs_baseline_columns(self):
        result = self.events.enpi_vs_baseline(
            "meter:main", "counter:line1", window="1h", baseline_window=5
        )
        assert "baseline_enpi" in result.columns
        assert "is_anomaly" in result.columns
        assert "trend" in result.columns

    def test_trend_values_valid(self):
        result = self.events.enpi_vs_baseline(
            "meter:main", "counter:line1", window="1h", baseline_window=5
        )
        valid_trends = {"improving", "stable", "degrading"}
        assert set(result["trend"].unique()).issubset(valid_trends)

    def test_enpi_by_hierarchy_two_meters(self):
        # Duplicate meter:main as meter:secondary
        df2 = self.df.copy()
        df2.loc[df2["uuid"] == "meter:main", "uuid"] = "meter:secondary"
        combined = pd.concat([self.df, df2], ignore_index=True)
        events = EnergyPerformanceIndicatorEvents(combined)
        result = events.enpi_by_hierarchy(
            ["meter:main", "meter:secondary"], "counter:line1", window="1D"
        )
        assert set(result["meter_uuid"].unique()) == {"meter:main", "meter:secondary"}

    def test_zero_production_no_div_error(self):
        # Create a df where counter never changes (zero production)
        df_flat = self.df.copy()
        df_flat.loc[df_flat["uuid"] == "counter:line1", "value_integer"] = 0
        events = EnergyPerformanceIndicatorEvents(df_flat)
        result = events.enpi_by_window("meter:main", "counter:line1", window="1D")
        # enpi should be NaN where units_produced == 0, not raise
        assert not result.empty

    def test_raw_csv_model_path(self):
        idx = pd.date_range("2026-01-13", periods=16, freq="15min", tz="UTC")
        rows = []
        for i, ts in enumerate(idx):
            rows.append({"time": ts, "id": "meter:main", "value": 50.0})
            rows.append({"time": ts, "id": "counter:line1", "value": float(i * 100)})
        df = pd.DataFrame(rows)
        events = EnergyPerformanceIndicatorEvents(
            df, time_column="time", uuid_column="id"
        )
        result = events.enpi_by_window(
            "meter:main",
            "counter:line1",
            value_column_energy="value",
            value_column_counter="value",
            window="1h",
        )
        assert not result.empty


# ── CarbonIntensityEvents ─────────────────────────────────────────────────────

_FACTORS = {
    "meter:main": 0.233,  # Scope 2 — electricity
    "meter:gas": 2.034,  # Scope 1 — gas (we'll add gas signal below)
}
_SCOPE_MAP = {"meter:main": 2, "meter:gas": 1}


def _add_gas_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Add a gas meter signal to the test DataFrame."""
    timestamps = df[df["uuid"] == "meter:main"]["systime"].tolist()
    gas_rows = [
        {"systime": ts, "uuid": "meter:gas", "value_double": 10.0, "is_delta": True}
        for ts in timestamps
    ]
    return pd.concat([df, pd.DataFrame(gas_rows)], ignore_index=True)


class TestCarbonIntensityEvents:
    def setup_method(self):
        self.df = _add_gas_signal(_make_df(hours=48))

    def test_emissions_by_window_scope2_only(self):
        events = CarbonIntensityEvents(self.df, _FACTORS, scope_map=_SCOPE_MAP)
        result = events.emissions_by_window(scope=2, window="1D")
        assert (result["scope"] == 2).all()
        assert (result["kgco2e"] > 0).all()
        assert "emission_factor" in result.columns

    def test_emissions_by_window_scope1_only(self):
        events = CarbonIntensityEvents(self.df, _FACTORS, scope_map=_SCOPE_MAP)
        result = events.emissions_by_window(scope=1, window="1D")
        assert (result["scope"] == 1).all()
        assert (result["source_uuid"] == "meter:gas").all()

    def test_total_emissions_sums_scopes(self):
        events = CarbonIntensityEvents(self.df, _FACTORS, scope_map=_SCOPE_MAP)
        total = events.total_emissions_by_window(window="1D")
        assert "scope1_kgco2e" in total.columns
        assert "scope2_kgco2e" in total.columns
        assert "total_kgco2e" in total.columns
        assert (
            total["total_kgco2e"] == total["scope1_kgco2e"] + total["scope2_kgco2e"]
        ).all()

    def test_carbon_intensity_per_unit(self):
        events = CarbonIntensityEvents(
            self.df, {"meter:main": 0.233}, scope_map={"meter:main": 2}
        )
        result = events.carbon_intensity_per_unit("counter:line1", window="1D")
        assert "carbon_intensity" in result.columns
        assert "units_produced" in result.columns
        valid = result.dropna(subset=["carbon_intensity"])
        assert (valid["carbon_intensity"] > 0).all()

    def test_emission_factor_audit(self):
        events = CarbonIntensityEvents(self.df, _FACTORS, scope_map=_SCOPE_MAP)
        audit = events.emission_factor_audit()
        assert set(audit.columns) == {
            "source_uuid",
            "scope",
            "emission_factor_kgco2e_per_unit",
        }
        assert set(audit["source_uuid"]) == set(_FACTORS.keys())

    def test_unknown_uuid_ignored_gracefully(self):
        events = CarbonIntensityEvents(self.df, {"unknown_meter": 0.5}, scope_map={})
        result = events.emissions_by_window(window="1D")
        assert result.empty

    def test_raw_csv_model_path(self):
        idx = pd.date_range("2026-01-13", periods=16, freq="15min", tz="UTC")
        rows = [{"time": ts, "id": "meter:main", "value": 50.0} for ts in idx]
        df = pd.DataFrame(rows)
        events = CarbonIntensityEvents(
            df,
            {"meter:main": 0.233},
            time_column="time",
            uuid_column="id",
        )
        result = events.emissions_by_window(value_column="value", window="1h")
        assert not result.empty
        assert (result["kgco2e"] > 0).all()


class TestEnPIRegressionBaseline:
    """ISO 50006 regression baseline: energy = a + b * units."""

    @staticmethod
    def _frame(days=40, improve_from=30, base_load=200.0, slope=2.0):
        rng = np.random.default_rng(0)
        rows, count = [], 0
        start = pd.Timestamp("2024-01-01", tz="UTC")
        for d in range(days):
            units = int(rng.integers(50, 300))
            b = slope * (0.9 if d >= improve_from else 1.0)
            day = start + pd.Timedelta(days=d)
            # one interval-energy reading and two counter readings per day
            rows.append(
                (day + pd.Timedelta(hours=12), "meter", base_load + b * units, None)
            )
            rows.append((day, "counter", None, count))
            count += units
            rows.append(
                (day + pd.Timedelta(hours=23, minutes=59), "counter", None, count)
            )
        return pd.DataFrame(
            rows, columns=["systime", "uuid", "value_double", "value_integer"]
        )

    def test_volume_normalised_and_detects_lasting_improvement(self):
        ev = EnergyPerformanceIndicatorEvents(self._frame())
        out = ev.enpi_vs_baseline(
            "meter", "counter", baseline="regression", baseline_window=20
        )
        assert {"expected_energy_kwh", "energy_savings_kwh"} <= set(out.columns)
        before, after = out.iloc[:30], out.iloc[30:]
        # Baseline period: the model explains the load-dependent EnPI exactly.
        assert before["deviation_pct"].abs().max() < 1e-6
        assert not before["is_anomaly"].any()
        # After the 10 % slope improvement: savings on every day, kept.
        assert (after["energy_savings_kwh"] > 0).all()
        assert (after["deviation_pct"] < 0).all()

    def test_rolling_baseline_flags_low_output_days(self):
        # The default rolling baseline reads low-output days as anomalies
        # (base load per unit), the regression baseline does not.
        ev = EnergyPerformanceIndicatorEvents(self._frame(improve_from=99))
        rolling = ev.enpi_vs_baseline("meter", "counter", baseline_window=20)
        regression = ev.enpi_vs_baseline(
            "meter", "counter", baseline="regression", baseline_window=20
        )
        assert rolling["is_anomaly"].sum() > 0
        assert regression["is_anomaly"].sum() == 0

    def test_baseline_period_and_validation(self):
        ev = EnergyPerformanceIndicatorEvents(self._frame())
        out = ev.enpi_vs_baseline(
            "meter",
            "counter",
            baseline="regression",
            baseline_period=("2024-01-01", "2024-01-15"),
        )
        assert out.iloc[:30]["deviation_pct"].abs().max() < 1e-6
        with pytest.raises(ValueError, match="baseline must be"):
            ev.enpi_vs_baseline("meter", "counter", baseline="mean")
        with pytest.raises(ValueError, match="different production volumes"):
            ev.enpi_vs_baseline(
                "meter", "counter", baseline="regression", baseline_window=1
            )
