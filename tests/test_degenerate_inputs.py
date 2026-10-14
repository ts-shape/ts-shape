"""Regression tests: all-NaN signals and fits without spread."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.maintenance.reliability import ReliabilityEvents
from ts_shape.events.production.operating_state import OperatingStateEvents
from ts_shape.events.quality.capability_trending import CapabilityTrendingEvents
from ts_shape.events.supplychain.inventory_monitoring import (
    InventoryMonitoringEvents,
)


def _signal(times, values, uuid="sig", column="value_double"):
    return pd.DataFrame(
        {"systime": pd.to_datetime(times), "uuid": uuid, column: values}
    ).assign(is_delta=True)


def _minutes(n, start="2024-01-01 00:00"):
    return pd.date_range(start, periods=n, freq="1min")


# --- OperatingStateEvents ---------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("infer_states", {}),
        ("state_duration_statistics", {}),
        ("unusual_state_sequence", {}),
        ("dwell_time_violation", {"state": 1, "max_duration": "5min"}),
    ],
)
def test_operating_state_all_nan_signal_gives_empty_result(method, kwargs):
    df = _signal(_minutes(20), np.nan)
    ev = OperatingStateEvents(df, "sig")
    out = getattr(ev, method)(**kwargs)
    assert out.empty
    assert len(out.columns) > 0


def test_operating_state_ignores_missing_values():
    values = np.r_[np.zeros(10), np.full(10, 5.0)]
    holey = values.copy()
    holey[[3, 15]] = np.nan
    full = OperatingStateEvents(_signal(_minutes(20), values), "sig").infer_states()
    part = OperatingStateEvents(_signal(_minutes(20), holey), "sig").infer_states()
    # NaN samples carry no state: they do not form (high) states of their own.
    assert part["state"].tolist() == full["state"].tolist() == [0, 1]


# --- ReliabilityEvents --------------------------------------------------------


def _equal_failures():
    # Run 10 h, fail 1 h -- three times, then run 5 h: every time between
    # failures is exactly 10 h.
    times, states = [], []
    t = pd.Timestamp("2024-01-01")
    for _ in range(3):
        times += [t, t + pd.Timedelta("10h")]
        states += [False, True]
        t += pd.Timedelta("11h")
    times.append(t)
    states.append(False)
    times.append(t + pd.Timedelta("5h"))
    states.append(False)
    return _signal(times, states, uuid="fault", column="value_bool")


def test_weibull_fit_without_spread_is_nan_with_warning():
    ev = ReliabilityEvents(_equal_failures(), "fault")
    with pytest.warns(DataQualityWarning, match="no finite Weibull fit"):
        fit = ev.weibull_fit()
    assert len(fit) == 1
    assert fit[["beta", "eta_hours", "mttf_hours", "b10_hours"]].isna().all(axis=None)
    assert fit["failure_pattern"].iloc[0] == "unknown"


def test_reliability_at_degenerate_fit_is_nan_never_inf():
    ev = ReliabilityEvents(_equal_failures(), "fault")
    with pytest.warns(DataQualityWarning):
        curve = ev.reliability_at([1.0, 10.0, 100.0])
    assert len(curve) == 3
    numeric = curve[["reliability", "failure_probability", "hazard_per_hour"]]
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()
    assert numeric.isna().all(axis=None)


# --- InventoryMonitoringEvents ------------------------------------------------


def test_stockout_prediction_never_stocking_out_is_nan():
    levels = [100.0, 100.0, 110.0, 120.0, 90.0]
    out = InventoryMonitoringEvents(
        _signal(_minutes(5), levels), "sig"
    ).stockout_prediction(consumption_rate_window="2min")
    hours = out["estimated_stockout_time_hours"].to_numpy(dtype=float)
    assert not np.isinf(hours).any()
    assert np.isnan(hours[:4]).all()  # level not falling: no stockout projected
    assert hours[4] > 0


# --- CapabilityTrendingEvents -------------------------------------------------


def test_capability_trending_constant_signal_warns():
    df = _signal(_minutes(30), 50.0)
    ev = CapabilityTrendingEvents(df, "sig", upper_spec=60.0, lower_spec=40.0)
    with pytest.warns(DataQualityWarning, match="no spread"):
        out = ev.capability_over_time(window="10min")
    assert out[["cp", "cpk", "pp", "ppk"]].isna().all(axis=None)
