"""Nelson / Western Electric run rules at known points.

Source: L. S. Nelson (1984), "The Shewhart Control Chart -- Tests for
Special Causes", Journal of Quality Technology 16(4), 237-239; the zone
rules 1, 5 and 6 are the Western Electric rules (Western Electric
Statistical Quality Control Handbook, 1956; Montgomery, *Introduction to
Statistical Quality Control*, Table 5.1). The AIAG SPC manual (2nd ed.,
2005) lists the same tests.

Every chart has center line 10 and sigma 1: the tolerance rows are
``9, 10, 11`` (mean 10, sample standard deviation exactly 1), so the zone
edges are 9/11 (1 sigma), 8/12 (2 sigma) and 7/13 (3 sigma).

Each chart is a *neutral* pattern that violates no rule, with one
violation pattern spliced in. The expected flags (the point that completes
each qualifying window) are derived by hand in the comments, and a naive
loop implementation of the eight tests cross-checks every rule.
"""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlEvents,
)

T0 = pd.Timestamp("2024-01-01 00:00")

# Neutral pattern: no 9 on one side (it alternates sides), at most 2 steady
# moves, short alternation runs, never beyond 2 sigma, at most one point per
# five beyond 1 sigma on a side, and 11.5/8.5 break every zone-C run.
N = [10.5, 9.5, 10.5, 11.5, 9.5, 8.5]


def _spc(values):
    tol = pd.DataFrame(
        {
            "systime": [T0 - pd.Timedelta(hours=h) for h in (3, 2, 1)],
            "uuid": "tol",
            "value_double": [9.0, 10.0, 11.0],
        }
    )
    act = pd.DataFrame(
        {
            "systime": T0 + pd.to_timedelta(np.arange(len(values)), unit="min"),
            "uuid": "act",
            "value_double": [float(v) for v in values],
        }
    )
    df = pd.concat([tol, act], ignore_index=True)
    df["is_delta"] = True
    return StatisticalProcessControlEvents(
        df,
        value_column="value_double",
        tolerance_uuid="tol",
        actual_uuid="act",
        event_uuid="evt",
    )


def _actual(spc):
    df = spc.dataframe
    return df[df["uuid"] == "act"].sort_values("systime").reset_index(drop=True)


def _flagged(spc, rule):
    act = _actual(spc)
    limits = spc.control_limits()
    method = getattr(spc, rule)
    if rule in ("rule_3", "rule_4"):
        out = method(act)
    elif rule == "rule_2":
        out = method(act, limits=limits)
    else:
        out = method(act, limits)
    return sorted(out.index.tolist())


def nelson_reference(x, mu, s):
    """Naive loop implementation of Nelson's eight tests (last point of each window)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    flags = {f"rule_{i}": set() for i in range(1, 9)}
    for i in range(n):
        if abs(x[i] - mu) > 3 * s:
            flags["rule_1"].add(i)
        if i >= 8:
            w = x[i - 8 : i + 1]
            if all(w > mu) or all(w < mu):
                flags["rule_2"].add(i)
        if i >= 5:
            d = np.diff(x[i - 5 : i + 1])
            if all(d > 0) or all(d < 0):
                flags["rule_3"].add(i)
        if i >= 13:
            d = np.diff(x[i - 13 : i + 1])
            if all(d != 0) and all(d[k] * d[k + 1] < 0 for k in range(12)):
                flags["rule_4"].add(i)
        if i >= 2:
            w = x[i - 2 : i + 1]
            if sum(w > mu + 2 * s) >= 2 or sum(w < mu - 2 * s) >= 2:
                flags["rule_5"].add(i)
        if i >= 4:
            w = x[i - 4 : i + 1]
            if sum(w > mu + s) >= 4 or sum(w < mu - s) >= 4:
                flags["rule_6"].add(i)
        if i >= 14:
            w = x[i - 14 : i + 1]
            if all(abs(w - mu) < s):
                flags["rule_7"].add(i)
        if i >= 7:
            w = x[i - 7 : i + 1]
            if all(abs(w - mu) > s):
                flags["rule_8"].add(i)
    return {k: sorted(v) for k, v in flags.items()}


CHARTS = {
    # 13.5 (> 13) at 6 and 6.0 (< 7) at 13.
    "rule_1": (N + [13.5] + N + [6.0] + N, [6, 13]),
    # Ten points above (8.5 before, 9.5 after the spliced 9 and the next
    # N[0]=10.5): runs of nine end at 14 and 15.
    "rule_2": (
        N + [10.6, 11.4, 10.8, 11.2, 10.7, 11.3, 10.9, 11.1, 10.6] + N,
        [14, 15],
    ),
    # 8.5 (index 5) then six rising points: seven rising points, so the
    # six-point windows end at 10 and 11.
    "rule_3": (N + [9.0, 9.3, 9.7, 10.2, 10.6, 10.9] + N, [10, 11]),
    # 8.5 -> 10.2 -> 9.8 -> ... -> 9.8 -> 10.5 -> 9.5 -> 10.5 alternates from
    # index 4 to 22 (19 points): 14-point windows end at 17..22.
    "rule_4": (N + [10.2, 9.8] * 7 + N, [17, 18, 19, 20, 21, 22]),
    # 12.5, 10.5, 12.2: two of three above 12 -> 8. 8.5, 7.5, 7.8 and
    # 7.5, 7.8, 10.5: two of three below 8 -> 16 and 17.
    "rule_5": (N + [12.5, 10.5, 12.2] + N + [7.5, 7.8] + N, [8, 16, 17]),
    # 11.5, 11.2, 10.5, 11.8, 11.3: four of five above 11 -> 10.
    "rule_6": (N + [11.5, 11.2, 10.5, 11.8, 11.3] + N, [10]),
    # 15 zone-C points (6..20) followed by 10.5, 9.5, 10.5 of the next N:
    # an 18-point zone-C run, 15-point windows end at 20..23.
    "rule_7": (
        N
        + [10.2, 10.4, 9.8, 9.6, 10.1, 10.3, 9.9, 9.7]
        + [10.2, 10.6, 9.4, 9.8, 10.5, 10.1, 9.9]
        + N,
        [20, 21, 22, 23],
    ),
    # 8.5 (index 5) plus eight points outside zone C, alternating sides:
    # nine in a row, 8-point windows end at 12 and 13.
    "rule_8": (N + [11.5, 8.5, 11.8, 8.2, 12.2, 7.9, 11.4, 8.6] + N, [12, 13]),
}


@pytest.mark.parametrize("rule", sorted(CHARTS))
def test_rule_flags_exactly_the_hand_derived_points(rule):
    values, expected = CHARTS[rule]
    spc = _spc(values)
    assert _flagged(spc, rule) == expected
    # The naive reference agrees with the hand derivation.
    assert nelson_reference(values, 10.0, 1.0)[rule] == expected


def test_neutral_pattern_violates_no_rule():
    spc = _spc(N * 5)
    for rule in CHARTS:
        assert _flagged(spc, rule) == [], rule
    assert spc.detect_rule_violations().empty


@pytest.mark.parametrize("rule", sorted(CHARTS))
def test_detect_rule_violations_reports_the_same_points(rule):
    values, expected = CHARTS[rule]
    spc = _spc(values)
    out = spc.detect_rule_violations(selected_rules=[rule])
    times = list(T0 + pd.to_timedelta(expected, unit="min"))
    assert list(out["systime"]) == times
    assert list(out["value"]) == [float(values[i]) for i in expected]
    assert set(out["rule"]) == {rule}
    assert set(out["uuid"]) == {"evt"} and set(out["source_uuid"]) == {"act"}


@pytest.mark.parametrize("rule", sorted(CHARTS))
def test_detect_violations_reports_the_same_points(rule):
    values, expected = CHARTS[rule]
    out = _spc(values).detect_violations(selected_rules=[rule])
    assert list(out["systime"]) == list(T0 + pd.to_timedelta(expected, unit="min"))
    assert list(out["value_double"]) == [float(values[i]) for i in expected]
    assert set(out["triggered_rule"]) == {rule}


def test_all_rules_on_one_chart_match_reference():
    values = CHARTS["rule_1"][0]
    out = _spc(values).detect_rule_violations()
    ref = nelson_reference(values, 10.0, 1.0)
    got = {
        r: sorted(((g["systime"] - T0) // pd.Timedelta(minutes=1)).tolist())
        for r, g in out.groupby("rule")
    }
    assert got == {r: v for r, v in ref.items() if v}
    sev = dict(zip(out["rule"], out["severity"]))
    assert sev["rule_1"] == "critical"


def test_control_limits_are_mean_plus_k_sigma():
    lim = _spc(N).control_limits().iloc[0]
    assert lim["mean"] == pytest.approx(10.0)
    for k in (1, 2, 3):
        assert lim[f"{k}sigma_upper"] == pytest.approx(10.0 + k)
        assert lim[f"{k}sigma_lower"] == pytest.approx(10.0 - k)


def test_boundaries_are_strict():
    # "Beyond 3 sigma" is strict: 13.0 / 7.0 sit on the limit.
    spc = _spc(N + [13.0] + N + [7.0] + N)
    assert _flagged(spc, "rule_1") == []
    # A point exactly on the center line breaks a same-side run.
    run = [10.5] * 4 + [10.0] + [10.5] * 4
    assert _flagged(_spc(N + run + N), "rule_2") == []
    # An unchanged value breaks a trend.
    trend = [9.0, 9.3, 9.3, 9.7, 10.2, 10.6]
    assert _flagged(_spc([10.5, *trend, 9.5]), "rule_3") == []


def test_rule_5_needs_the_same_side():
    # 12.5 and 7.5 are both beyond 2 sigma, but on opposite sides.
    assert _flagged(_spc(N + [12.5, 7.5, 10.5] + N), "rule_5") == []


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("noise", [1.2, 0.35])
def test_random_charts_match_reference(seed, noise):
    rng = np.random.default_rng(seed)
    # A shifted process (noise 1.2) and a too-quiet one (noise 0.35, which
    # exercises rule 7): together every rule fires somewhere.
    values = np.round(
        10 + rng.normal(0, noise, 300) + np.repeat(rng.normal(0, 1.0, 10), 30), 1
    )
    spc = _spc(values)
    ref = nelson_reference(values, 10.0, 1.0)
    for rule in CHARTS:
        assert _flagged(spc, rule) == ref[rule], rule
