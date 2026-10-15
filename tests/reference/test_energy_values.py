"""Energy KPIs on small hand-computed examples (exact values, not shapes).

Shared hourly example (naive timestamps, 2024-01-01):

* meter ``m``: 15-min energy deltas (kWh) -> per hour 10, 20, 8, 40.
* meter ``m2``: 1 kWh per 15 min -> 4 per hour.
* counter ``c`` (value_integer) every 30 min: 0, 5 | 10, 20 | 30, 2 | 6, 26.
  The 30 -> 2 step at 02:30 is a reset. Reset-aware increments per reading
  are 0, 5 | 5, 10 | 10, 2 | 4, 20, so units per hour are 5, 15, 12, 24
  (an increment is booked in the hour of the reading that reports it).
"""

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

T0 = pd.Timestamp("2024-01-01 00:00")
HOURS = [T0 + pd.Timedelta(hours=h) for h in range(4)]
ENERGY = [10.0, 20.0, 8.0, 40.0]
UNITS = [5.0, 15.0, 12.0, 24.0]


def _frame():
    q = pd.Timedelta(minutes=15)
    m = [1, 2, 3, 4, 5, 5, 5, 5, 2, 2, 2, 2, 10, 10, 10, 10]
    rows = [
        {"systime": T0 + i * q, "uuid": "m", "value_double": float(v)}
        for i, v in enumerate(m)
    ]
    rows += [
        {"systime": T0 + i * q, "uuid": "m2", "value_double": 1.0} for i in range(16)
    ]
    c = [0, 5, 10, 20, 30, 2, 6, 26]
    rows += [
        {"systime": T0 + 2 * i * q, "uuid": "c", "value_integer": v}
        for i, v in enumerate(c)
    ]
    df = pd.DataFrame(rows)
    df["value_integer"] = df["value_integer"].astype("Int64")
    df["is_delta"] = True
    return df


def _bounds_ok(out, n=4, step=pd.Timedelta(hours=1)):
    assert list(out["start"]) == [T0 + i * step for i in range(n)]
    assert list(out["end"]) == [T0 + (i + 1) * step for i in range(n)]
    assert (out["duration_seconds"] == step.total_seconds()).all()


# ---------------------------------------------------------------------------
# EnergyConsumptionEvents
# ---------------------------------------------------------------------------


def test_energy_per_unit_reset_aware():
    out = EnergyConsumptionEvents(_frame()).energy_per_unit("m", "c", window="1h")
    _bounds_ok(out)
    assert list(out["energy"]) == ENERGY
    assert list(out["units_produced"]) == UNITS
    assert out["energy_per_unit"].tolist() == pytest.approx(
        [10 / 5, 20 / 15, 8 / 12, 40 / 24]
    )


def test_energy_per_unit_max_minus_min():
    out = EnergyConsumptionEvents(_frame()).energy_per_unit(
        "m", "c", window="1h", handle_resets=False
    )
    # Per-hour max - min: 5-0, 20-10, 30-2, 26-6.
    assert list(out["units_produced"]) == [5, 10, 28, 20]
    assert out["energy_per_unit"].tolist() == pytest.approx(
        [10 / 5, 20 / 10, 8 / 28, 40 / 20]
    )


def test_consumption_baseline_deviation_hand_computed():
    out = EnergyConsumptionEvents(_frame()).consumption_baseline_deviation(
        "m", window="1h", baseline_periods=2, deviation_threshold=0.2
    )
    _bounds_ok(out)
    assert list(out["consumption"]) == ENERGY
    # Rolling mean over the current and previous window (min_periods=1).
    assert out["baseline"].tolist() == pytest.approx([10, 15, 14, 24])
    assert out["deviation_pct"].tolist() == pytest.approx(
        [0.0, 5 / 15, -6 / 14, 16 / 24]
    )
    assert list(out["is_anomaly"]) == [False, True, True, True]


def test_consumption_baseline_deviation_needs_enough_windows():
    out = EnergyConsumptionEvents(_frame()).consumption_baseline_deviation(
        "m", window="1h", baseline_periods=5
    )
    assert out.empty


# ---------------------------------------------------------------------------
# EnergyEfficiencyEvents
# ---------------------------------------------------------------------------


def test_efficiency_trend_hand_computed():
    out = EnergyEfficiencyEvents(_frame()).efficiency_trend(
        "m", "c", window="1h", trend_window=2
    )
    _bounds_ok(out)
    assert list(out["energy"]) == ENERGY
    assert list(out["units"]) == UNITS
    # units / kWh: 0.5, 0.75, 1.5, 0.6. The rolling(2) efficiency is the
    # trailing pair as one period (ISO 50006: summed units / summed energy),
    # not the mean of the two ratios (formerly 0.5, 0.625, 1.125, 1.05).
    assert out["efficiency"].tolist() == pytest.approx([0.5, 0.75, 1.5, 0.6])
    assert out["rolling_avg_efficiency"].tolist() == pytest.approx(
        [5 / 10, 20 / 30, 27 / 28, 36 / 48]
    )
    # Slope +0.167, +0.298, -0.214 against the +-0.01 stable band.
    assert pd.isna(out["trend_direction"].iloc[0])
    assert list(out["trend_direction"].iloc[1:].astype(str)) == [
        "increasing",
        "increasing",
        "decreasing",
    ]


def test_specific_energy_consumption_hand_computed():
    out = EnergyEfficiencyEvents(_frame()).specific_energy_consumption(
        "m", "c", window="1h"
    )
    _bounds_ok(out)
    assert list(out["total_energy"]) == ENERGY
    assert list(out["total_output"]) == UNITS
    sec = [2.0, 20 / 15, 8 / 12, 40 / 24]
    assert out["sec"].tolist() == pytest.approx(sec)
    # Rolling(7) SEC as one period (summed energy / summed output): 2, 1.5,
    # 1.1875, 1.393. Worse than +10 % is worsening, better than -10 %
    # improving.
    assert list(out["sec_trend"]) == ["stable", "improving", "improving", "worsening"]


# ---------------------------------------------------------------------------
# EnergyPerformanceIndicatorEvents
# ---------------------------------------------------------------------------


def test_enpi_by_hierarchy_two_meters_hand_computed():
    out = EnergyPerformanceIndicatorEvents(_frame()).enpi_by_hierarchy(
        ["m2", "m"], "c", window="1h"
    )
    assert list(out["meter_uuid"]) == ["m", "m2"] * 4  # sorted by start, meter
    assert list(out["start"]) == [h for h in HOURS for _ in range(2)]
    assert (out["duration_seconds"] == 3600).all()
    m = out[out["meter_uuid"] == "m"]
    m2 = out[out["meter_uuid"] == "m2"]
    assert list(m["energy_kwh"]) == ENERGY
    assert list(m2["energy_kwh"]) == [4.0] * 4
    assert list(m["units_produced"]) == UNITS
    assert m["enpi"].tolist() == pytest.approx([e / u for e, u in zip(ENERGY, UNITS)])
    assert m2["enpi"].tolist() == pytest.approx([4 / u for u in UNITS])


# ---------------------------------------------------------------------------
# IdleEnergyDetectionEvents
# ---------------------------------------------------------------------------


def _idle_frame():
    """1 kWh every hour for three days; running 00-12 on day 1, all of day 2,
    idle all of day 3 (the state holds until the last energy reading)."""
    e = pd.DataFrame(
        {
            "systime": pd.date_range(T0, periods=72, freq="h"),
            "uuid": "m",
            "value_double": 1.0,
        }
    )
    s = pd.DataFrame(
        {
            "systime": [T0, T0 + pd.Timedelta(hours=12), T0 + pd.Timedelta(days=1)]
            + [T0 + pd.Timedelta(days=2)],
            "uuid": "run",
            "value_bool": [True, False, True, False],
        }
    )
    return pd.concat([e, s], ignore_index=True)


def test_idle_energy_trend_hand_computed():
    out = IdleEnergyDetectionEvents(_idle_frame()).idle_energy_trend(
        "m", "run", window="1D", trend_window=2
    )
    _bounds_ok(out, n=3, step=pd.Timedelta(days=1))
    # Day 1: running 50 % of 24 kWh -> 12 kWh idle; day 2: 0; day 3:
    # running 0 % is below idle_threshold 0.1 -> all 24 kWh are idle.
    assert out["idle_energy"].tolist() == pytest.approx([12.0, 0.0, 24.0])
    assert out["rolling_avg_idle_energy"].tolist() == pytest.approx([12.0, 6.0, 12.0])
    assert pd.isna(out["trend_direction"].iloc[0])
    assert list(out["trend_direction"].iloc[1:].astype(str)) == [
        "improving",
        "worsening",
    ]


# ---------------------------------------------------------------------------
# CarbonIntensityEvents
# ---------------------------------------------------------------------------


def _carbon():
    day = pd.Timedelta(days=1)
    rows = [
        # Electricity (scope 2, 0.4 kgCO2e/kWh): 60 + 40 on day 1, 50 on day 2.
        (T0, "elec", 60.0),
        (T0 + pd.Timedelta(hours=12), "elec", 40.0),
        (T0 + day, "elec", 50.0),
        # Gas (scope 1, 2.0 kgCO2e/m3): 10 on day 1, 5 on day 2.
        (T0, "gas", 10.0),
        (T0 + day, "gas", 5.0),
        # Second electricity feed, not in scope_map -> scope 2 by default.
        (T0, "elec2", 20.0),
    ]
    df = pd.DataFrame(rows, columns=["systime", "uuid", "value_double"])
    return CarbonIntensityEvents(
        df,
        {"elec": 0.4, "gas": 2.0, "elec2": 0.5},
        scope_map={"gas": 1, "elec": 2},
    )


def test_total_emissions_by_window_hand_computed():
    out = _carbon().total_emissions_by_window(window="1D")
    assert list(out["start"]) == [T0, T0 + pd.Timedelta(days=1)]
    assert list(out["end"]) == [T0 + pd.Timedelta(days=1), T0 + pd.Timedelta(days=2)]
    # Day 1: scope 1 = 10 * 2.0 = 20; scope 2 = 100 * 0.4 + 20 * 0.5 = 50.
    # Day 2: scope 1 = 5 * 2.0 = 10; scope 2 = 50 * 0.4 = 20.
    assert out["scope1_kgco2e"].tolist() == pytest.approx([20.0, 10.0])
    assert out["scope2_kgco2e"].tolist() == pytest.approx([50.0, 20.0])
    assert out["total_kgco2e"].tolist() == pytest.approx([70.0, 30.0])
    assert set(out["uuid"]) == {"energy:carbon"}


def test_emission_factor_audit_lists_configured_factors():
    audit = _carbon().emission_factor_audit()
    assert audit.to_dict("records") == [
        {"source_uuid": "elec", "scope": 2, "emission_factor_kgco2e_per_unit": 0.4},
        {"source_uuid": "gas", "scope": 1, "emission_factor_kgco2e_per_unit": 2.0},
        {"source_uuid": "elec2", "scope": 2, "emission_factor_kgco2e_per_unit": 0.5},
    ]


# ---------------------------------------------------------------------------
# Right-labeled windows ('W', 'ME'): methods that post-process an already
# bounded frame apply window_bounds a second time.
# ---------------------------------------------------------------------------


def _weekly_frame():
    """Three Monday-to-Sunday weeks: 1 kWh and 1 unit every hour."""
    t = pd.date_range("2024-01-01", periods=24 * 21, freq="h")  # a Monday
    m = pd.DataFrame({"systime": t, "uuid": "m", "value_double": 1.0})
    c = pd.DataFrame({"systime": t, "uuid": "c", "value_integer": np.arange(len(t))})
    s = pd.DataFrame({"systime": t[::24], "uuid": "run", "value_bool": False})
    return pd.concat([m, c, s], ignore_index=True)


WEEK_STARTS = list(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]))
WEEK = pd.Timedelta(days=7)


def _weekly_ok(out):
    assert list(out["start"]) == WEEK_STARTS
    assert list(out["end"] - out["start"]) == [WEEK] * 3
    assert (out["duration_seconds"] == WEEK.total_seconds()).all()


def test_weekly_bounds_of_single_pass_methods():
    # These apply window_bounds once: correct for 'W'.
    df = _weekly_frame()
    _weekly_ok(EnergyEfficiencyEvents(df).efficiency_trend("m", "c", window="W"))
    _weekly_ok(
        EnergyPerformanceIndicatorEvents(df).enpi_by_window("m", "c", window="W")
    )
    _weekly_ok(
        IdleEnergyDetectionEvents(df).idle_energy_by_window("m", "run", window="W")
    )


def test_total_emissions_weekly_windows():
    df = _weekly_frame()
    _weekly_ok(
        CarbonIntensityEvents(df, {"m": 1.0}).total_emissions_by_window(window="W")
    )


def test_consumption_baseline_deviation_weekly_windows():
    out = EnergyConsumptionEvents(_weekly_frame()).consumption_baseline_deviation(
        "m", window="W", baseline_periods=2
    )
    _weekly_ok(out)


def test_energy_per_unit_weekly_windows():
    out = EnergyConsumptionEvents(_weekly_frame()).energy_per_unit("m", "c", window="W")
    _weekly_ok(out)
    assert list(out["energy"]) == [168.0] * 3


def test_enpi_by_hierarchy_weekly_windows():
    out = EnergyPerformanceIndicatorEvents(_weekly_frame()).enpi_by_hierarchy(
        ["m"], "c", window="W"
    )
    _weekly_ok(out)


def test_idle_energy_trend_weekly_windows():
    out = IdleEnergyDetectionEvents(_weekly_frame()).idle_energy_trend(
        "m", "run", window="W"
    )
    _weekly_ok(out)


def test_carbon_intensity_per_unit_weekly_windows():
    out = CarbonIntensityEvents(_weekly_frame(), {"m": 2.0}).carbon_intensity_per_unit(
        "c", window="W"
    )
    _weekly_ok(out)
