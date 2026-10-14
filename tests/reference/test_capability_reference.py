"""Process capability indices against textbook results.

Sources:

* D. C. Montgomery, *Introduction to Statistical Quality Control* (6th ed.),
  Section 8.3: a process with mean 53, sigma 2 and specifications
  LSL = 38, USL = 62 has Cp = 24 / 12 = 2.0, CPU = (62 - 53) / 6 = 1.5,
  CPL = (53 - 38) / 6 = 2.5, so Cpk = 1.5; with target T = 50 the Taguchi
  index is Cpm = (USL - LSL) / (6 sqrt(sigma^2 + (mu - T)^2)) = 24 / (6
  sqrt(13)) = 1.109 (Chan, Cheng & Spiring 1988, JQT 20(3)).
* The Six Sigma convention (Motorola; Harry 1988): a process whose nearer
  specification is 4.5 sigma from the mean has 3.4 defects per million
  opportunities, reported as a 6-sigma level with the 1.5 sigma shift.

Data are built so the sample statistics (ddof = 1) are exactly the
textbook mu and sigma; the hand-computed small-sample references below use
the same textbook formulas with numpy.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ts_shape.events.quality.capability_trending import CapabilityTrendingEvents
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

T0 = pd.Timestamp("2024-01-01 00:00")


def _with_moments(n, mean, sd, seed=0):
    """``n`` values whose sample mean and sample std (ddof=1) are exact."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + sd * z


def _tolerance_frame(values, usl=None, lsl=None):
    rows = []
    if usl is not None:
        rows.append({"systime": T0, "uuid": "usl", "value_double": usl})
    if lsl is not None:
        rows.append({"systime": T0, "uuid": "lsl", "value_double": lsl})
    rows += [
        {
            "systime": T0 + pd.Timedelta(minutes=i + 1),
            "uuid": "x",
            "value_double": float(v),
        }
        for i, v in enumerate(values)
    ]
    df = pd.DataFrame(rows)
    df["is_delta"] = True
    return df


def _tde(values, usl=None, lsl=None):
    return ToleranceDeviationEvents(
        _tolerance_frame(values, usl, lsl),
        tolerance_column="value_double",
        actual_column="value_double",
        actual_uuid="x",
        event_uuid="dev",
        upper_tolerance_uuid="usl" if usl is not None else None,
        lower_tolerance_uuid="lsl" if lsl is not None else None,
    )


# ---------------------------------------------------------------------------
# ToleranceDeviationEvents.capability_indices
# ---------------------------------------------------------------------------


def test_montgomery_cp_cpk_cpm_example():
    """Montgomery, Introduction to SQC, Sec. 8.3: mu 53, sigma 2, specs 38-62, T 50."""
    x = _with_moments(50, 53.0, 2.0)
    res = _tde(x, usl=62.0, lsl=38.0).capability_indices(target_value=50.0)
    assert res["mean"] == pytest.approx(53.0)
    assert res["std"] == pytest.approx(2.0)
    assert res["Cp"] == pytest.approx(2.0)
    assert res["Cpk"] == pytest.approx(1.5)
    # No subgroups: Pp/Ppk equal Cp/Cpk (documented).
    assert res["Pp"] == pytest.approx(2.0)
    assert res["Ppk"] == pytest.approx(1.5)
    assert res["Cpm"] == pytest.approx(24 / (6 * np.sqrt(13)))
    assert round(res["Cpm"], 2) == 1.11
    assert (res["usl"], res["lsl"], res["target"]) == (62.0, 38.0, 50.0)


def test_montgomery_one_sided_cpu():
    """One-sided upper spec: Cpk is CPU = (USL - mu) / 3 sigma = 1.5; Cp undefined."""
    x = _with_moments(50, 53.0, 2.0)
    res = _tde(x, usl=62.0).capability_indices()
    assert np.isnan(res["Cp"]) and np.isnan(res["Pp"])
    assert res["Cpk"] == pytest.approx(1.5)
    assert res["Ppk"] == pytest.approx(1.5)
    assert np.isnan(res["lsl"])
    assert res["target"] == 62.0
    assert "Cpm" not in res


def test_one_sided_lower_cpl():
    x = _with_moments(50, 53.0, 2.0)
    res = _tde(x, lsl=38.0).capability_indices()
    assert res["Cpk"] == pytest.approx(2.5)  # CPL = (53 - 38) / 6


def test_hand_computed_small_dataset():
    """Ten readings, specs 9.0-11.0; textbook formulas with numpy (ddof=1)."""
    x = np.array([10.12, 9.87, 10.31, 10.05, 9.94, 10.21, 10.08, 9.79, 10.17, 10.02])
    usl, lsl, target = 11.0, 9.0, 10.0
    mu, s = x.mean(), x.std(ddof=1)
    res = _tde(x, usl=usl, lsl=lsl).capability_indices(target_value=target)
    assert res["Cp"] == pytest.approx((usl - lsl) / (6 * s))
    assert res["Cpk"] == pytest.approx(min(usl - mu, mu - lsl) / (3 * s))
    assert res["Cpm"] == pytest.approx(
        (usl - lsl) / (6 * np.sqrt(s**2 + (mu - target) ** 2))
    )
    # By hand: mean 10.056, s = 0.158478 -> Cp = 2 / 0.950868, Cpk = 0.944 / 0.475434.
    assert res["Cp"] == pytest.approx(2.1033, abs=1e-4)
    assert res["Cpk"] == pytest.approx(1.9855, abs=1e-4)


def test_latest_spec_is_used():
    """The spec in force (the latest setting) is used."""
    df = _tolerance_frame(_with_moments(20, 53.0, 2.0), usl=70.0, lsl=30.0)
    later = pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(hours=5)] * 2,
            "uuid": ["usl", "lsl"],
            "value_double": [62.0, 38.0],
            "is_delta": True,
        }
    )
    tde = ToleranceDeviationEvents(
        pd.concat([df, later], ignore_index=True),
        tolerance_column="value_double",
        actual_column="value_double",
        actual_uuid="x",
        upper_tolerance_uuid="usl",
        lower_tolerance_uuid="lsl",
    )
    assert tde.capability_indices()["Cpk"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# CapabilityTrendingEvents
# ---------------------------------------------------------------------------

# Three 1 h windows of five readings each (specs 9.0 - 11.0).
WINDOWS = [
    [10.1, 9.9, 10.0, 10.2, 9.8],
    [10.3, 10.1, 10.4, 10.2, 10.5],
    [9.6, 10.4, 9.5, 10.6, 9.9],
]


def _trend_frame(groups, extra=()):
    rows = []
    for w, vals in enumerate(groups):
        for i, v in enumerate(vals):
            rows.append(
                {
                    "systime": T0 + pd.Timedelta(hours=w, minutes=10 * i),
                    "uuid": "x",
                    "value_double": float(v),
                }
            )
    rows += list(extra)
    return pd.DataFrame(rows)


def test_capability_over_time_hand_computed():
    """Per window: Cp/Cpk from the window's sample std (the within-window,
    short-term sigma); Pp/Ppk from the sample std of *all* readings (the
    overall, long-term sigma) with the window mean. Montgomery Sec. 8.3 /
    AIAG SPC manual definitions, with the time window as the subgroup.
    """
    usl, lsl = 11.0, 9.0
    cte = CapabilityTrendingEvents(
        _trend_frame(WINDOWS), "x", upper_spec=usl, lower_spec=lsl
    )
    out = cte.capability_over_time(window="1h")
    overall = np.concatenate(WINDOWS).std(ddof=1)
    assert len(out) == 3
    for w, vals in enumerate(WINDOWS):
        row = out.iloc[w]
        mu, s = np.mean(vals), np.std(vals, ddof=1)
        assert row["start"] == T0 + pd.Timedelta(hours=w)
        assert row["end"] == T0 + pd.Timedelta(hours=w + 1)
        assert row["n_samples"] == 5
        assert row["mean"] == pytest.approx(mu)
        assert row["std"] == pytest.approx(s)
        assert row["cp"] == pytest.approx((usl - lsl) / (6 * s))
        assert row["cpk"] == pytest.approx(min(usl - mu, mu - lsl) / (3 * s))
        assert row["pp"] == pytest.approx((usl - lsl) / (6 * overall))
        assert row["ppk"] == pytest.approx(min(usl - mu, mu - lsl) / (3 * overall))
    # Window 1 by hand: s = sqrt(0.1 / 4) = 0.158114, Cp = 2 / 0.948683.
    assert out.iloc[0]["cp"] == pytest.approx(2.108185, abs=1e-6)
    # Window 2 by hand: mean 10.3, Cpk = 0.7 / (3 * 0.158114) = 1.475730.
    assert out.iloc[1]["cpk"] == pytest.approx(1.475730, abs=1e-6)


def test_capability_over_time_montgomery_window():
    x = _with_moments(40, 53.0, 2.0)
    rows = [
        {"systime": T0 + pd.Timedelta(minutes=i), "uuid": "x", "value_double": v}
        for i, v in enumerate(x)
    ]
    cte = CapabilityTrendingEvents(
        pd.DataFrame(rows), "x", upper_spec=62.0, lower_spec=38.0
    )
    row = cte.capability_over_time(window="1h").iloc[0]
    assert row["cp"] == pytest.approx(2.0)
    assert row["cpk"] == pytest.approx(1.5)
    # One window: overall sigma == within sigma.
    assert row["pp"] == pytest.approx(2.0)
    assert row["ppk"] == pytest.approx(1.5)


def test_yield_estimate_six_sigma_3_4_dpmo():
    """Nearer spec 4.5 sigma from the mean -> 3.4 DPMO, sigma level 6.0."""
    x = _with_moments(40, 53.0, 2.0)
    rows = [
        {"systime": T0 + pd.Timedelta(minutes=i), "uuid": "x", "value_double": v}
        for i, v in enumerate(x)
    ]
    cte = CapabilityTrendingEvents(
        pd.DataFrame(rows), "x", upper_spec=62.0, lower_spec=38.0
    )
    row = cte.yield_estimate(window="1h").iloc[0]
    dpmo = (stats.norm.sf(4.5) + stats.norm.cdf(-7.5)) * 1e6
    assert row["dpmo"] == pytest.approx(dpmo, rel=1e-6)
    assert round(row["dpmo"], 1) == 3.4
    assert row["sigma_level"] == pytest.approx(6.0, abs=1e-6)
    assert row["estimated_yield_pct"] == pytest.approx(100 - dpmo / 1e4)


def test_spec_uuid_limits_hand_computed():
    spec = [
        {"systime": T0, "uuid": "usl", "value_double": 11.0},
        {"systime": T0, "uuid": "lsl", "value_double": 9.0},
    ]
    cte = CapabilityTrendingEvents(
        _trend_frame(WINDOWS, spec),
        "x",
        upper_spec_uuid="usl",
        lower_spec_uuid="lsl",
    )
    fixed = CapabilityTrendingEvents(
        _trend_frame(WINDOWS), "x", upper_spec=11.0, lower_spec=9.0
    )
    cols = ["cp", "cpk", "pp", "ppk"]
    pd.testing.assert_frame_equal(
        cte.capability_over_time(window="1h")[cols],
        fixed.capability_over_time(window="1h")[cols],
    )


def test_spec_uuid_limits_follow_the_spec_in_force():
    # Specs 9-11 during windows 0-1, tightened to 9.5-10.5 from window 2.
    spec = [
        {"systime": T0, "uuid": "usl", "value_double": 11.0},
        {"systime": T0, "uuid": "lsl", "value_double": 9.0},
        {"systime": T0 + pd.Timedelta(hours=2), "uuid": "usl", "value_double": 10.5},
        {"systime": T0 + pd.Timedelta(hours=2), "uuid": "lsl", "value_double": 9.5},
    ]
    cte = CapabilityTrendingEvents(
        _trend_frame(WINDOWS, spec),
        "x",
        upper_spec_uuid="usl",
        lower_spec_uuid="lsl",
    )
    out = cte.capability_over_time(window="1h")
    s0 = np.std(WINDOWS[0], ddof=1)
    assert out.iloc[0]["cp"] == pytest.approx(2.0 / (6 * s0))
    # Window 2 starts when the tightened spec takes effect.
    s2 = np.std(WINDOWS[2], ddof=1)
    assert out.iloc[2]["cp"] == pytest.approx(1.0 / (6 * s2))
    ys = cte.yield_estimate(window="1h")
    assert ys.iloc[0]["estimated_yield_pct"] > ys.iloc[2]["estimated_yield_pct"]
