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
    # Nelson test 3 counts points: six points = five increases.
    assert len(spc.rule_3(_act([1.0, 2, 3, 4, 5, 6]))) == 1
    assert len(spc.rule_3(_act([1.0, 2, 3, 4, 5, 6, 7]))) == 2
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


def test_rule_8_eight_with_none_within_1sigma(spc, limits):
    # Nelson test 8: eight in a row, none within 1 sigma, either side.
    flagged = spc.rule_8(_act([11.5, 8.5] * 4 + [12.5]), limits)
    assert len(flagged) == 2
    assert spc.rule_8(_act([10.1] * 9), limits).empty
    assert spc.rule_8(_act([11.5, 8.5] * 3 + [10.5, 11.5]), limits).empty


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


# --- Nelson (1984) definitions, one sequence that must trigger and one that
# --- must not per rule. Center 10, sigma 1.

NELSON_CASES = {
    "rule_1": ([10.0, 13.5], [10.0, 12.9, 7.1]),
    "rule_2": ([10.5] * 9, [10.5] * 8 + [10.0] + [10.5] * 8),
    "rule_3": ([1.0, 2, 3, 4, 5, 6], [1.0, 2, 3, 4, 5, 5, 6]),
    "rule_4": ([10.0, 10.5] * 7, [10.0, 10.5] * 6 + [10.0]),
    "rule_5": ([10.0, 12.5, 13.5], [12.5, 7.5, 10.0]),
    "rule_6": ([12.5, 11.5, 13.5, 11.5, 10.0], [11.5, 8.5, 11.5, 8.5, 11.5]),
    "rule_7": ([10.5, 9.5] * 7 + [10.0], [10.5] * 14 + [11.5]),
    "rule_8": ([11.5, 8.5] * 4, [11.5, 8.5] * 3 + [10.5, 11.5]),
}


def _call(spc, rule, df, limits):
    if rule in ("rule_3", "rule_4"):
        return getattr(spc, rule)(df)
    if rule == "rule_2":
        return spc.rule_2(df, limits=limits)
    return getattr(spc, rule)(df, limits)


@pytest.mark.parametrize("rule", sorted(NELSON_CASES))
def test_nelson_rule_triggers_and_not(spc, limits, rule):
    hit, miss = NELSON_CASES[rule]
    assert len(_call(spc, rule, _act(hit), limits)) >= 1
    assert _call(spc, rule, _act(miss), limits).empty


@pytest.mark.parametrize("rule", sorted(NELSON_CASES))
def test_rules_do_not_mutate_input_and_return_bool(spc, limits, rule):
    df = _act(NELSON_CASES[rule][0])
    before = df.copy()
    out = _call(spc, rule, df, limits)
    pd.testing.assert_frame_equal(df, before)
    assert out[rule].dtype == bool


def test_rule_5_counts_points_beyond_3sigma_same_side_only(spc, limits):
    assert len(spc.rule_5(_act([10.0, 13.5, 13.5]), limits)) == 1
    assert spc.rule_5(_act([12.5, 7.5, 10.0]), limits).empty


def test_rule_6_counts_points_beyond_2sigma_same_side_only(spc, limits):
    assert len(spc.rule_6(_act([12.5] * 4 + [10.0]), limits)) == 1
    assert spc.rule_6(_act([11.5, 8.5, 11.5, 8.5, 10.0]), limits).empty


def test_rule_2_uses_chart_center_line(spc, limits):
    # All points sit above the chart center (10) although they straddle
    # their own mean; the chart center line decides.
    vals = [10.5, 11.5] * 5
    assert len(spc.rule_2(_act(vals), limits=limits)) == 2
    assert spc.rule_2(_act(vals)).empty  # own-mean fallback without limits


def test_vectorized_and_per_rule_paths_agree():
    rng = np.random.default_rng(0)
    vals = list(10 + rng.normal(0, 1.3, 300))
    for seq in NELSON_CASES.values():
        vals += list(seq[0]) + [10.0, 13.0, 7.0]
    spc = _spc(vals)
    vec = spc.detect_rule_violations()
    per = spc.detect_violations(include_severity=True)
    limits = spc.control_limits()
    act = spc.dataframe[spc.dataframe["uuid"] == "act"].sort_values("systime")
    for rule in NELSON_CASES:
        a = set(vec.loc[vec["rule"] == rule, "systime"])
        b = set(_call(spc, rule, act, limits)["systime"])
        assert a == b, rule
        assert a, rule  # every rule fires somewhere in this sequence
    assert set(vec["systime"]) == set(per["systime"])
