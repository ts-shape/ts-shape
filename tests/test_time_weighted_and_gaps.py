"""Regression tests (Themes C/D): time weighting and gap caps outside production."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents
from ts_shape.events.energy.idle_energy_detection import IdleEnergyDetectionEvents
from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents
from ts_shape.events.engineering.material_balance import MaterialBalanceEvents
from ts_shape.utils.time_weighted import time_weighted_resample


def test_time_weighted_resample_holds_values():
    s = pd.Series(
        [1.0, 0.0], index=pd.to_datetime(["2024-01-01 09:00", "2024-01-01 09:59"])
    )
    out = time_weighted_resample(s, "1h", data_end=pd.Timestamp("2024-01-01 11:00"))
    assert out.iloc[0] == pytest.approx(59 / 60)
    assert out.iloc[1] == 0.0


def test_time_weighted_resample_max_gap_and_tz():
    idx = pd.date_range("2024-03-31 00:00", periods=3, freq="2h", tz="Europe/Berlin")
    s = pd.Series([10.0, 20.0, 30.0], index=idx)
    assert time_weighted_resample(s, "1h", max_gap="1h").isna().all()
    assert str(time_weighted_resample(s, "1h").index.tz) == "Europe/Berlin"


def _idle_frame():
    # 12 x 1 kWh every 15 min, 08:00-10:45; one change-encoded state row.
    t = pd.date_range("2024-01-01 08:00", periods=12, freq="15min")
    energy = pd.DataFrame({"systime": t, "uuid": "m", "value_double": 1.0})
    state = pd.DataFrame(
        {
            "systime": [pd.Timestamp("2024-01-01 08:00")],
            "uuid": "s",
            "value_bool": [True],
        }
    )
    return pd.concat([energy, state], ignore_index=True)


def test_idle_energy_by_window_keeps_all_energy():
    ev = IdleEnergyDetectionEvents(_idle_frame())
    assert ev.idle_energy_by_window("m", "s")["total_energy"].sum() == 4.0
    out = ev.idle_energy_by_window("m", "s", time_weighted=True)
    assert out["total_energy"].sum() == 12.0
    assert (out["machine_running_pct"] == 1.0).all()


def test_idle_energy_waste_keeps_all_energy():
    ev = EnergyEfficiencyEvents(_idle_frame())
    out = ev.idle_energy_waste("m", "s", window="1h", time_weighted=True)
    assert out["energy_consumed"].sum() == 12.0


def test_error_integrals_max_gap():
    t = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:01", "2024-01-01 10:31"])
    sp = pd.DataFrame({"systime": [t[0]], "uuid": "sp", "value_double": [0.0]})
    pv = pd.DataFrame({"systime": t, "uuid": "pv", "value_double": [1.0, 1.0, 1.0]})
    ev = ControlLoopHealthEvents(
        pd.concat([sp, pv]), setpoint_uuid="sp", actual_uuid="pv"
    )
    # Default: the 30-minute comms gap is integrated as error.
    assert ev.error_integrals("1h")["iae"].iloc[0] == pytest.approx(1860.0)
    assert ev.error_integrals("1h", max_gap="5min")["iae"].iloc[0] == pytest.approx(
        60.0
    )


def _balance_frame(include_input=True):
    t = pd.date_range("2024-01-01 00:00", periods=2, freq="1h")
    rows = [pd.DataFrame({"systime": t, "uuid": "out", "value_double": 100.0})]
    if include_input:
        rows.append(
            pd.DataFrame({"systime": t[:1], "uuid": "in", "value_double": 100.0})
        )
    return pd.concat(rows, ignore_index=True)


def test_zero_input_with_output_is_not_balanced():
    out = MaterialBalanceEvents(_balance_frame(), ["in"], ["out"]).balance_check()
    second = out.iloc[1]
    assert second["total_input"] == 0.0
    assert np.isinf(second["imbalance_pct"])
    assert not second["balanced"]


def test_missing_input_window_is_unknown_not_imbalanced():
    ev = MaterialBalanceEvents(_balance_frame(), ["in"], ["out"], missing_as_zero=False)
    out = ev.balance_check()
    assert out.iloc[0]["balanced"]
    assert np.isnan(out.iloc[1]["total_input"])
    assert not out.iloc[1]["balanced"]


def test_material_balance_time_weighted():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:50", "2024-01-01 00:59"])
    flow_in = pd.DataFrame(
        {"systime": t, "uuid": "in", "value_double": [10.0, 50.0, 50.0]}
    )
    flow_out = pd.DataFrame({"systime": t[:1], "uuid": "out", "value_double": [10.0]})
    df = pd.concat([flow_in, flow_out])
    sample_mean = MaterialBalanceEvents(df, ["in"], ["out"]).balance_check()
    weighted = MaterialBalanceEvents(
        df, ["in"], ["out"], time_weighted=True
    ).balance_check()
    assert sample_mean["total_input"].iloc[0] == pytest.approx(110 / 3)
    # 10 held 50 min, 50 held 9 min (series ends at 00:59)
    assert weighted["total_input"].iloc[0] == pytest.approx((10 * 50 + 50 * 9) / 59)
