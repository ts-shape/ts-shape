"""Regression test: efficiency_comparison treats the counter as cumulative."""

import pandas as pd

from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents


def test_total_output_is_counter_increase_not_sum_of_readings():
    t = pd.date_range("2024-01-01 07:00", periods=3, freq="h")
    energy = pd.DataFrame({"systime": t, "uuid": "m", "value_double": 10.0})
    counter = pd.DataFrame(
        {"systime": t, "uuid": "c", "value_integer": [100, 101, 102]}
    )
    df = pd.concat([energy, counter], ignore_index=True)
    out = EnergyEfficiencyEvents(df).efficiency_comparison("m", "c")
    row = out[out["shift"] == "shift_1"].iloc[0]
    assert row["total_output"] == 2
    assert row["total_energy"] == 30.0
