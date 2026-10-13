"""Regression tests (Theme A): per-window counter quantities with resets."""

import pandas as pd

from ts_shape.events.energy.carbon_intensity import CarbonIntensityEvents
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents
from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents
from ts_shape.events.energy.energy_performance_indicator import (
    EnergyPerformanceIndicatorEvents,
)
from ts_shape.events.supplychain.inventory_monitoring import (
    InventoryMonitoringEvents,
)
from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics
from ts_shape.utils.counters import counter_per_window

# 0@08:00, 50@08:50, 70@09:10, 5@09:20 (reset), 25@09:40 -> 95 parts in total
COUNTER_T = pd.to_datetime(
    [
        "2024-01-01 08:00",
        "2024-01-01 08:50",
        "2024-01-01 09:10",
        "2024-01-01 09:20",
        "2024-01-01 09:40",
    ]
)
COUNTER_V = [0, 50, 70, 5, 25]


def _counter_series():
    return pd.Series(COUNTER_V, index=COUNTER_T)


def test_counter_per_window_default_is_legacy_max_minus_min():
    out = counter_per_window(_counter_series(), "1h")
    assert list(out) == [50, 65]


def test_counter_per_window_handle_resets():
    out = counter_per_window(_counter_series(), "1h", handle_resets=True)
    assert list(out) == [50.0, 45.0]
    assert out.sum() == 95.0


def _energy_frame():
    energy_t = pd.date_range("2024-01-01 08:00", periods=8, freq="15min")
    energy = pd.DataFrame({"systime": energy_t, "uuid": "m", "value_double": 1.0})
    counter = pd.DataFrame(
        {"systime": COUNTER_T, "uuid": "c", "value_integer": COUNTER_V}
    )
    return pd.concat([energy, counter], ignore_index=True)


def _total(df, col):
    return float(pd.to_numeric(df[col]).sum())


def test_energy_methods_thread_handle_resets():
    df = _energy_frame()
    cases = [
        (EnergyConsumptionEvents(df).energy_per_unit, "units_produced", {}),
        (EnergyEfficiencyEvents(df).efficiency_trend, "units", {}),
        (
            EnergyEfficiencyEvents(df).specific_energy_consumption,
            "total_output",
            {"window": "1h"},
        ),
        (
            EnergyPerformanceIndicatorEvents(df).enpi_by_window,
            "units_produced",
            {"window": "1h"},
        ),
    ]
    for method, col, kw in cases:
        legacy = method("m", "c", **kw)
        fixed = method("m", "c", handle_resets=True, **kw)
        assert _total(legacy, col) == 115.0, method.__name__
        assert _total(fixed, col) == 95.0, method.__name__


def test_carbon_intensity_per_unit_handle_resets():
    ev = CarbonIntensityEvents(_energy_frame(), emission_factors={"m": 0.4})
    legacy = ev.carbon_intensity_per_unit("c", window="1h")
    fixed = ev.carbon_intensity_per_unit("c", window="1h", handle_resets=True)
    assert _total(legacy, "units_produced") == 115.0
    assert _total(fixed, "units_produced") == 95.0


def test_time_stats_diff_handle_resets_and_uuid_grouping():
    df = pd.DataFrame({"systime": COUNTER_T, "uuid": "c", "value_double": COUNTER_V})
    other = df.assign(uuid="d", value_double=1000.0)
    both = pd.concat([df, other], ignore_index=True)
    out = TimeGroupedStatistics.calculate_statistic(
        both,
        "systime",
        "value_double",
        "1h",
        "diff",
        group_by_uuid=True,
        handle_resets=True,
    )
    assert out.loc["c", "difference"].sum() == 95.0
    assert out.loc["d", "difference"].sum() == 0.0


def test_inventory_consumption_rate_handle_refills():
    t = pd.date_range("2024-01-01 08:00", periods=4, freq="20min")
    df = pd.DataFrame(
        {"systime": t, "uuid": "lvl", "value_double": [100.0, 40, 100, 40]}
    )
    ev = InventoryMonitoringEvents(df, "lvl")
    assert ev.consumption_rate("1h")["consumption_rate"].iloc[0] == 0.0
    fixed = ev.consumption_rate("1h", handle_refills=True)
    assert fixed["consumption_rate"].sum() == 120.0
