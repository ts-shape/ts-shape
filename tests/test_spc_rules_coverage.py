"""Coverage for the SPC rules, dynamic limits, CUSUM and interpretation."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlEvents,
)


def _spc(actual):
    # Tolerance 9, 10, 11 -> mean 10, sigma 1.
    tol = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=3, freq="h"),
            "uuid": "tol",
            "value_double": [9.0, 10.0, 11.0],
        }
    )
    act = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-02", periods=len(actual), freq="min"),
            "uuid": "act",
            "value_double": actual,
        }
    )
    return StatisticalProcessControlEvents(
        pd.concat([tol, act]), "value_double", "tol", "act", "evt"
    )


def _act(values):
    return pd.DataFrame({"value_double": values})


@pytest.fixture()
def spc():
    return _spc([10.0] * 5)


@pytest.fixture()
def limits(spc):
    return spc.control_limits()


def test_control_limits(limits):
    row = limits.iloc[0]
    assert row["mean"] == 10.0
    assert row["3sigma_upper"] == 13.0 and row["3sigma_lower"] == 7.0


def test_rule_2_nine_on_one_side(spc):
    assert len(spc.rule_2(_act([1.0] * 3 + [5.0] * 9))) == 1
    assert spc.rule_2(_act([1.0, 5.0] * 6)).empty


def test_rule_3_six_steady_increase(spc):
    assert len(spc.rule_3(_act([1.0, 2, 3, 4, 5, 6, 7]))) == 1
    assert spc.rule_3(_act([1.0, 2, 3, 2, 3, 4, 5])).empty


def test_rule_5_two_of_three_beyond_2sigma(spc, limits):
    assert len(spc.rule_5(_act([10.0, 12.5, 12.5]), limits)) == 1
    assert spc.rule_5(_act([10.0, 12.5, 10.0]), limits).empty


def test_rule_6_four_of_five_beyond_1sigma(spc, limits):
    assert len(spc.rule_6(_act([11.5] * 4 + [10.0]), limits)) == 1
    assert spc.rule_6(_act([11.5] * 3 + [10.0] * 2), limits).empty


def test_rule_7_fifteen_within_1sigma(spc, limits):
    assert len(spc.rule_7(_act([10.1] * 15), limits)) == 1
    assert spc.rule_7(_act([10.1] * 14), limits).empty


def test_rule_8_eight_within_1sigma(spc, limits):
    assert len(spc.rule_8(_act([10.1] * 9), limits)) == 2


def test_apply_rules_vectorized_and_interpret():
    spc = _spc([10.0] * 5 + [15.0] + [10.0] * 5)
    v = spc.detect_rule_violations(selected_rules=["rule_1"])
    assert list(v["rule"].unique()) == ["rule_1"]
    assert v["value"].iloc[0] == 15.0
    out = spc.interpret_violations(v)
    assert {"interpretation", "meaning", "recommendation"} <= set(out.columns)
    assert out["interpretation"].iloc[0].startswith("One or more points beyond 3-sigma")


@pytest.mark.parametrize("method", ["moving_range", "ewma"])
def test_dynamic_control_limits(method):
    vals = list(10 + np.sin(np.arange(30)))
    out = _spc(vals).dynamic_control_limits(method=method, window_size=5)
    assert len(out) == 30
    finite = out.dropna().iloc[1:]  # the first row has no spread yet
    assert (finite["3sigma_upper"] > finite["mean"]).all()
    assert (finite["3sigma_lower"] < finite["mean"]).all()


def test_cusum_detects_upward_shift():
    out = _spc([10.0] * 10 + [13.0] * 10).detect_cusum_shifts(k=0.5, h=4.0)
    assert not out.empty
    assert set(out["shift_direction"]) == {"upward"}
    assert (out["uuid"] == "evt").all()
