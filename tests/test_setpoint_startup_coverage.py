"""Coverage for setpoint KPIs and startup extras (previously untested).

The setpoint fixtures are analytic step responses, so the expected KPI
values are known in closed form.
"""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents
from ts_shape.events.engineering.startup_events import StartupDetectionEvents

T0 = pd.Timestamp("2024-01-01")
TAU = 20.0  # first-order time constant [s]


def _step_response(pv_func):
    sp = pd.DataFrame(
        {
            "systime": [T0, T0 + pd.Timedelta("60s")],
            "uuid": "sp",
            "value_double": [0.0, 10.0],
        }
    )
    ts = np.arange(0, 600)
    pv_vals = np.where(ts < 60, 0.0, pv_func(ts - 60.0))
    pv = pd.DataFrame(
        {
            "systime": T0 + pd.to_timedelta(ts, unit="s"),
            "uuid": "pv",
            "value_double": pv_vals,
        }
    )
    return SetpointChangeEvents(pd.concat([sp, pv], ignore_index=True), "sp")


@pytest.fixture()
def first_order():
    return _step_response(lambda t: 10 * (1 - np.exp(-t / TAU)))


def test_rise_time_matches_tau_ln9(first_order):
    out = first_order.rise_time("pv")
    assert out["rise_time_seconds"].iloc[0] == pytest.approx(TAU * np.log(9), abs=1.0)
    assert out["reached_end"].iloc[0]


def test_decay_rate_matches_inverse_tau(first_order):
    out = first_order.decay_rate("pv")
    assert out["decay_rate_lambda"].iloc[0] == pytest.approx(1 / TAU, rel=0.02)
    assert out["fit_quality_r2"].iloc[0] > 0.99


def test_time_to_settle_derivative(first_order):
    out = first_order.time_to_settle_derivative("pv", rate_threshold=0.01)
    assert out["settled"].iloc[0]
    # |dPV/dt| = (10/tau) e^{-t/tau} < 0.01  ->  t > tau ln(10 / (tau * 0.01))
    assert out["final_rate"].iloc[0] < 0.01


def test_control_quality_metrics(first_order):
    row = first_order.control_quality_metrics("pv", tol=0.2).iloc[0]
    assert row["settled"]
    assert row["rise_time_seconds"] == pytest.approx(TAU * np.log(9), abs=1.0)
    assert row["decay_rate_lambda"] == pytest.approx(1 / TAU, rel=0.02)
    assert row["oscillation_count"] == 0


def test_oscillation_frequency_period():
    period = 40.0
    w = 2 * np.pi / period
    spe = _step_response(lambda t: 10 * (1 - np.exp(-t / 80) * np.cos(w * t)))
    out = spe.oscillation_frequency("pv", window="5m")
    # Zero crossings land exactly on samples here; each must count once.
    assert out["period_seconds"].iloc[0] == pytest.approx(period, abs=1.0)


def _speed_profile():
    v = (
        [0.0] * 120
        + list(np.linspace(0, 100, 25))
        + [100.0] * 120
        + [0.0] * 60
        + [30.0] * 6  # aborted attempt
        + [0.0] * 60
    )
    t = T0 + pd.to_timedelta(np.arange(len(v)) * 5, unit="s")
    speed = pd.DataFrame({"systime": t, "uuid": "speed", "value_double": v})
    current = speed.assign(uuid="current", value_double=np.array(v) * 0.5)
    return pd.concat([speed, current], ignore_index=True)


@pytest.fixture()
def startup():
    return StartupDetectionEvents(_speed_profile(), "speed")


def test_multi_signal_all_logic(startup):
    out = startup.detect_startup_multi_signal(
        {
            "speed": {"method": "threshold", "threshold": 10},
            "current": {"method": "threshold", "threshold": 5},
        },
        logic="all",
    )
    assert len(out) == 2
    assert all(set(x) == {"speed", "current"} for x in out["signals_triggered"])


def test_adaptive_ignores_flat_idle_baseline(startup):
    out = startup.detect_startup_adaptive(min_above="10s")
    # A flat idle signal (std 0) must not read as a startup at t0.
    assert out["start"].min() > T0
    assert list(out["start"].dt.strftime("%H:%M:%S")) == ["00:10:05", "00:27:05"]


def test_assess_startup_quality(startup):
    events = startup.detect_startup_by_threshold(threshold=10, min_above="10s")
    out = startup.assess_startup_quality(events)
    assert len(out) == len(events)
    assert {"smoothness_score", "stability_score", "max_value"} <= set(out.columns)


def test_track_startup_phases_non_overlapping(startup):
    phases = [
        {"name": "ramp", "condition": "range", "min_value": 10, "max_value": 90},
        {"name": "run", "condition": "threshold", "min_value": 90},
    ]
    out = startup.track_startup_phases(phases)
    assert list(out["phase_name"]) == ["ramp", "run"]
    assert out["completed"].all()
    assert out["next_phase"].iloc[0] == "run"


def test_detect_failed_startups(startup):
    out = startup.detect_failed_startups(
        threshold=10, completion_threshold=90, min_rise_duration="10s"
    )
    assert len(out) == 1
    assert out["max_value_reached"].iloc[0] == 30.0
    assert out["failure_reason"].iloc[0] == "dropped_below_threshold_before_completion"
