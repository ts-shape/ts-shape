"""Regression tests (Theme F): degenerate statistics outside production."""

import logging

import pandas as pd

from ts_shape.events.engineering.steady_state_detection import (
    SteadyStateDetectionEvents,
)
from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlRuleBased,
)
from ts_shape.events.supplychain.demand_pattern import DemandPatternEvents


def _spc(tol_values):
    t_tol = pd.date_range("2024-01-01", periods=len(tol_values), freq="1h")
    t_act = pd.date_range("2024-01-02", periods=10, freq="1h")
    tol = pd.DataFrame({"systime": t_tol, "uuid": "tol", "value_double": tol_values})
    act = pd.DataFrame(
        {"systime": t_act, "uuid": "act", "value_double": [10.0, 11.0] * 5}
    )
    return StatisticalProcessControlRuleBased(
        pd.concat([tol, act]), "value_double", "tol", "act", "evt"
    )


def test_cusum_with_constant_tolerance_reports_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        out = _spc([10.0, 10.0, 10.0]).detect_cusum_shifts()
    assert out.empty
    assert "CUSUM skipped" in caplog.text


def test_control_limits_single_row_warns(caplog):
    with caplog.at_level(logging.WARNING):
        limits = _spc([10.0]).calculate_control_limits()
    assert limits["3sigma_upper"].isna().all()
    assert "degenerate" in caplog.text


def test_steady_state_sparse_samples_are_not_steady():
    t = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:20", "2024-01-01 10:40"])
    df = pd.DataFrame({"systime": t, "uuid": "s", "value_double": [50.0, 90.0, 20.0]})
    ev = SteadyStateDetectionEvents(df, "s")
    assert ev.detect_steady_state(
        window="5min", std_threshold=1.0, min_duration="1min"
    ).empty


def test_demand_empty_periods_are_not_zero_demand():
    t = pd.to_datetime(["2024-01-01", "2024-01-05"])
    df = pd.DataFrame({"systime": t, "uuid": "d", "value_double": [5.0, 7.0]})
    out = DemandPatternEvents(df, "d").demand_by_period(period="1D")
    assert len(out) == 2


def test_seasonality_accepts_calendar_period():
    t = pd.date_range("2024-01-01", periods=90, freq="D")
    df = pd.DataFrame({"systime": t, "uuid": "d", "value_double": 1.0})
    out = DemandPatternEvents(df, "d").seasonality_summary(period="MS")
    assert not out.empty
