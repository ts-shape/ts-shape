"""Regression tests: PV is aligned with the setpoint in force (backward as-of)."""

import pandas as pd
import pytest

from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents
from ts_shape.events.engineering.disturbance_recovery import DisturbanceRecoveryEvents


def _change_encoded_sp_dense_pv():
    t0 = pd.Timestamp("2024-01-01 10:00")
    pv_t = pd.date_range(t0, periods=61, freq="min")
    sp = pd.DataFrame({"systime": [t0], "uuid": "sp", "value_double": [50.0]})
    pv = pd.DataFrame({"systime": pv_t, "uuid": "pv", "value_double": 60.0})
    return pd.concat([sp, pv], ignore_index=True)


def test_change_encoded_setpoint_keeps_every_pv_sample():
    ev = ControlLoopHealthEvents(
        _change_encoded_sp_dense_pv(), setpoint_uuid="sp", actual_uuid="pv"
    )
    out = ev.error_integrals(window="2h")
    assert len(out) == 1
    assert out["sample_count"].iloc[0] == 61
    # |error| = 10 for 3600 s
    assert out["iae"].iloc[0] == pytest.approx(36000.0)


def test_pv_before_setpoint_step_keeps_old_setpoint():
    t = pd.date_range("2024-01-01 10:00", periods=10, freq="min")
    sp = pd.DataFrame(
        {
            "systime": [t[0], t[5] + pd.Timedelta(seconds=10)],
            "uuid": "sp",
            "value_double": [50.0, 80.0],
        }
    )
    pv = pd.DataFrame({"systime": t, "uuid": "pv", "value_double": 50.0})
    df = pd.concat([sp, pv], ignore_index=True)
    ev = DisturbanceRecoveryEvents(df, signal_uuid="pv", setpoint_uuid="sp")
    base = ev._compute_baseline_and_deviation("5min")
    # 10:05:00 is 10 s before the step to 80; it must still compare to 50.
    assert base.loc[t[5], "baseline"] == 50.0
    assert base.loc[t[6], "baseline"] == 80.0
