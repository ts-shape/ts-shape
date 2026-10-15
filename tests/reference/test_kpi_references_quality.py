"""Hand-computed reference values for the quality, reliability and energy KPIs.

Each test pins one KPI to an independent computation (a hand example, the
textbook formula implemented directly in numpy, or a published worksheet)
and cites its source. They guard the KPIs that were checked and found
correct, and the SPC / capability definitions fixed to the textbook ones.

Main source: D. C. Montgomery, *Introduction to Statistical Quality
Control* (6th ed.): ch. 6 (individuals / moving-range chart), ch. 8
(process capability), ch. 9 (CUSUM, EWMA), ch. 11 (Hotelling T²).
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ts_shape.events.maintenance.reliability import ReliabilityEvents
from ts_shape.events.quality.gauge_repeatability import GaugeRepeatabilityEvents
from ts_shape.events.quality.multivariate_spc import MultivariateSPCEvents
from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlEvents,
)
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents
from ts_shape.transform.calculator.power_integration import PowerIntegrator

T0 = pd.Timestamp("2024-01-01 00:00")
D2 = 1.128  # d2 for n = 2 (Montgomery, Appendix VI)


def _spc(reference, actual, **kwargs):
    ref = pd.DataFrame(
        {
            "systime": T0 + pd.to_timedelta(np.arange(len(reference)), unit="min"),
            "uuid": "ref",
            "value_double": np.asarray(reference, dtype=float),
        }
    )
    act = pd.DataFrame(
        {
            "systime": T0
            + pd.Timedelta(days=1)
            + pd.to_timedelta(np.arange(len(actual)), unit="min"),
            "uuid": "act",
            "value_double": np.asarray(actual, dtype=float),
        }
    )
    df = pd.concat([ref, act], ignore_index=True)
    return StatisticalProcessControlEvents(
        df, "value_double", "ref", "act", "quality:spc", **kwargs
    )


def _mr_sigma(values):
    """Individuals-chart sigma: mean moving range / d2 (Montgomery Sec. 6.4)."""
    return np.mean(np.abs(np.diff(np.asarray(values, dtype=float)))) / D2


# ---------------------------------------------------------------------------
# Shewhart individuals chart (Montgomery Sec. 6.4)
# ---------------------------------------------------------------------------


def test_individuals_limits_use_moving_range_sigma():
    # Montgomery Sec. 6.4: UCL/LCL = x-bar +/- 3 MR-bar / d2. A reference
    # trending 1..20 has mean 10.5 and every moving range 1, so
    # sigma = 1 / 1.128 = 0.886525 -> UCL 13.1596, LCL 7.8404. The sample
    # std (5.9161, the pre-fix estimate) gave 28.2482 / -7.2482.
    ref = np.arange(1, 21, dtype=float)
    lim = _spc(ref, [10.5]).control_limits().iloc[0]
    sigma = 1 / D2
    assert lim["mean"] == pytest.approx(10.5)
    assert lim["3sigma_upper"] == pytest.approx(13.1596, abs=1e-4)
    assert lim["3sigma_lower"] == pytest.approx(7.8404, abs=1e-4)
    for k in (1, 2, 3):
        assert lim[f"{k}sigma_upper"] == pytest.approx(10.5 + k * sigma)
        assert lim[f"{k}sigma_lower"] == pytest.approx(10.5 - k * sigma)
    old = _spc(ref, [10.5], sigma_method="std").control_limits().iloc[0]
    assert old["3sigma_upper"] == pytest.approx(10.5 + 3 * ref.std(ddof=1))
    assert old["3sigma_upper"] == pytest.approx(28.2482, abs=1e-4)
    assert old["3sigma_lower"] == pytest.approx(-7.2482, abs=1e-4)


def test_rule_zones_use_the_moving_range_sigma():
    # Same reference: 13.0 is inside the textbook 3-sigma limit 13.16, 13.2
    # is beyond it. Under the sample std (limit 28.25) both are inside.
    spc = _spc(np.arange(1, 21, dtype=float), [10.5, 13.0, 13.2, 10.5])
    out = spc.detect_rule_violations(selected_rules=["rule_1"])
    assert out["value"].tolist() == [13.2]
    old = _spc(
        np.arange(1, 21, dtype=float), [10.5, 13.0, 13.2, 10.5], sigma_method="std"
    )
    assert old.detect_rule_violations(selected_rules=["rule_1"]).empty


# Nelson (1984), JQT 16(4): the run length of each test. Reference 9, 10, 11:
# center 10, sigma = 1 / 1.128 = 0.8865 (zones 10.89 / 11.77 / 12.66).
NELSON_WINDOWS = {
    # rule: (pattern that completes at its last point, same pattern one short)
    "rule_1": ([13.0], [12.5]),  # 1 point beyond 3 sigma (12.66)
    "rule_2": ([10.3] * 9, [10.3] * 8),  # 9 points on one side
    "rule_3": ([10.0, 10.1, 10.2, 10.3, 10.4, 10.5], [10.0, 10.1, 10.2, 10.3, 10.4]),
    "rule_4": ([10.2, 9.8] * 7, ([10.2, 9.8] * 7)[:13]),  # 14 alternating
    "rule_5": ([12.0, 10.0, 12.0], [12.0, 10.0, 10.0, 12.0]),  # 2 of 3 > 2 sigma
    "rule_6": (
        [11.2, 11.2, 10.0, 11.2, 11.2],  # 4 of 5 > 1 sigma
        [11.2, 11.2, 10.0, 10.0, 11.2, 11.2],
    ),
    "rule_7": ([10.3, 9.7] * 7 + [10.3], [10.3, 9.7] * 7),  # 15 within 1 sigma
    "rule_8": ([11.2, 8.8] * 4, ([11.2, 8.8] * 4)[:7]),  # 8 outside 1 sigma
}


@pytest.mark.parametrize("rule", sorted(NELSON_WINDOWS))
def test_nelson_rule_window_lengths(rule):
    hit, miss = NELSON_WINDOWS[rule]
    out = _spc([9.0, 10.0, 11.0], hit).detect_rule_violations(selected_rules=[rule])
    first = (out["systime"] - T0 - pd.Timedelta(days=1)) // pd.Timedelta(minutes=1)
    assert first.tolist() == [len(hit) - 1]
    miss_out = _spc([9.0, 10.0, 11.0], miss).detect_rule_violations(
        selected_rules=[rule]
    )
    assert miss_out.empty


# ---------------------------------------------------------------------------
# Rolling individuals chart and EWMA chart (Montgomery Sec. 6.4, 9.2)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
REFERENCE = np.round(10 + RNG.normal(0, 1, 40), 3)
ACTUAL = np.round(10 + RNG.normal(0, 1, 60) + np.r_[np.zeros(30), np.full(30, 1.0)], 3)


def test_rolling_individuals_chart_matches_direct_formula():
    w = 8
    out = _spc(REFERENCE, ACTUAL).dynamic_control_limits(
        method="moving_range", window_size=w, n_sigma=3.0
    )
    for i, row in out.iterrows():
        if i < w:
            # Not enough history: NaN limits.
            assert np.isnan(row["mean"]) and np.isnan(row["ucl"])
            continue
        prev = ACTUAL[i - w : i]
        center, sigma = prev.mean(), _mr_sigma(prev)
        assert row["value"] == ACTUAL[i]
        assert row["mean"] == pytest.approx(center)
        assert row["ucl"] == pytest.approx(center + 3 * sigma)
        assert row["lcl"] == pytest.approx(center - 3 * sigma)
        assert row["2sigma_upper"] == pytest.approx(center + 2 * sigma)
        assert row["1sigma_lower"] == pytest.approx(center - sigma)


@pytest.mark.parametrize("lam, span", [(0.2, None), (None, 9)])
def test_ewma_chart_matches_direct_formula(lam, span):
    # Montgomery Sec. 9.2: z_i = lambda x_i + (1 - lambda) z_{i-1},
    # z_0 = mu_0, limits mu_0 +/- L sigma sqrt(lambda / (2 - lambda)
    # [1 - (1 - lambda)^(2i)]).
    kwargs = {"ewma_lambda": lam} if lam is not None else {"window_size": span}
    out = _spc(REFERENCE, ACTUAL).dynamic_control_limits(
        method="ewma", n_sigma=2.7, **kwargs
    )
    lam = lam if lam is not None else 2 / (span + 1)
    mu0, sigma = REFERENCE.mean(), _mr_sigma(REFERENCE)
    z = np.empty(len(ACTUAL))
    prev = mu0
    for i, x in enumerate(ACTUAL):
        prev = lam * x + (1 - lam) * prev
        z[i] = prev
    i = np.arange(1, len(ACTUAL) + 1)
    width = sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    np.testing.assert_allclose(out["ewma"], z, rtol=1e-12)
    np.testing.assert_allclose(out["mean"], mu0, rtol=1e-12)
    np.testing.assert_allclose(out["ucl"], mu0 + 2.7 * width, rtol=1e-12)
    np.testing.assert_allclose(out["lcl"], mu0 - 2.7 * width, rtol=1e-12)
    np.testing.assert_allclose(out["3sigma_upper"], mu0 + 3 * width, rtol=1e-12)
    np.testing.assert_allclose(out["1sigma_lower"], mu0 - width, rtol=1e-12)
    # The asymptotic width is L sigma sqrt(lambda / (2 - lambda)).
    assert width[-1] == pytest.approx(sigma * np.sqrt(lam / (2 - lam)), rel=1e-3)


# ---------------------------------------------------------------------------
# Tabular CUSUM (Montgomery Sec. 9.1)
# ---------------------------------------------------------------------------


def test_cusum_detects_a_mean_shift_at_the_eleventh_point():
    # k = 0.5, h = 5 (Montgomery's recommended design). Reference 9, 10, 11:
    # mu_0 = 10, sigma = 1 / 1.128. A sustained shift of 0.98 sigma adds
    # 0.98 - 0.5 = 0.48 sigma per point: C+_10 = 4.8 sigma < H = 5 sigma,
    # C+_11 = 5.28 sigma > H -> first signal at the 11th shifted point.
    sigma = 1 / D2
    actual = [10.0] * 5 + [10.0 + 0.98 * sigma] * 15
    out = _spc([9.0, 10.0, 11.0], actual).detect_cusum_shifts(k=0.5, h=5.0)
    idx = (
        (out["systime"] - T0 - pd.Timedelta(days=1)) // pd.Timedelta(minutes=1)
    ).tolist()
    assert idx[0] == 5 + 10  # the 11th point of the shift
    assert idx == list(range(15, 20))
    assert set(out["shift_direction"]) == {"upward"}
    assert out["cusum_high"].iloc[0] == pytest.approx(11 * 0.48 * sigma)
    # With the sample std (sigma 1) the shift is only 0.869 sigma, 0.369
    # per point: C+_13 = 4.79 < 5, C+_14 = 5.16 -> the 14th point.
    old = _spc([9.0, 10.0, 11.0], actual, sigma_method="std")
    late = old.detect_cusum_shifts(k=0.5, h=5.0)
    assert (
        (late["systime"].iloc[0] - T0 - pd.Timedelta(days=1)) // pd.Timedelta(minutes=1)
    ) == 5 + 13


# ---------------------------------------------------------------------------
# Hotelling T² (Montgomery Sec. 11.3)
# ---------------------------------------------------------------------------


def test_hotelling_t2_phase_one_and_two_limits():
    rng = np.random.default_rng(5)
    n, m, p, alpha = 40, 25, 2, 0.0027
    cov = [[1.0, 0.8], [0.8, 1.0]]
    x = rng.multivariate_normal([50.0, 20.0], cov, size=n)
    t = T0 + pd.to_timedelta(np.arange(n), unit="min")
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": u, "value_double": x[:, j]})
            for j, u in enumerate(["a", "b"])
        ],
        ignore_index=True,
    )
    out = MultivariateSPCEvents(df, ["a", "b"]).hotelling_t2(
        reference_period=(t[0], t[m]), alpha=alpha
    )
    ref = x[:m]
    d = x - ref.mean(axis=0)
    t2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(np.cov(ref, rowvar=False)), d)
    np.testing.assert_allclose(out["t2"], t2, rtol=1e-10)
    # Phase I (Montgomery Sec. 11.3, individual observations):
    # UCL = (m - 1)^2 / m * Beta(1 - alpha; p/2, (m - p - 1)/2).
    ucl1 = (m - 1) ** 2 / m * stats.beta.ppf(1 - alpha, p / 2, (m - p - 1) / 2)
    # Phase II (Montgomery Sec. 11.3): p (m + 1)(m - 1) / (m^2 - m p) F(p, m - p).
    ucl2 = p * (m + 1) * (m - 1) / (m**2 - m * p) * stats.f.ppf(1 - alpha, p, m - p)
    assert out["phase"].tolist() == ["I"] * m + ["II"] * (n - m)
    np.testing.assert_allclose(out["ucl"].iloc[:m], ucl1)
    np.testing.assert_allclose(out["ucl"].iloc[m:], ucl2)


# ---------------------------------------------------------------------------
# Gauge R&R, AIAG MSA (4th ed.) average-and-range worksheet example
# ---------------------------------------------------------------------------


def test_gauge_rr_aiag_worksheet_numbers():
    # AIAG MSA 4th ed., Gage R&R data sheet / report example: 10 parts,
    # 3 appraisers, 3 trials with R-bar = 0.3417, X-bar_diff = 0.4446,
    # R_p = 3.511 -> EV 0.20188, AV 0.22963, GRR 0.30575, PV 1.10456,
    # TV 1.14610, %GRR 26.68, %PV 96.38, ndc 5. Data are built so the cell
    # ranges, appraiser means and part means reproduce those statistics.
    r_bar, x_diff, r_p = 0.3417, 0.4446, 3.511
    part_effect = np.linspace(-r_p / 2, r_p / 2, 10)
    op_effect = {"A": 0.0, "B": x_diff, "C": x_diff / 2}
    rows, ts = [], T0
    for p, pe in enumerate(part_effect):
        for op, oe in op_effect.items():
            for dev in (-r_bar / 2, 0.0, r_bar / 2):
                rows.append(
                    {
                        "systime": ts,
                        "uuid": "gauge",
                        "value_double": 5.0 + pe + oe + dev,
                        "value_string": f"part_{p}",
                        "operator": op,
                    }
                )
                ts += pd.Timedelta(seconds=1)
    row = (
        GaugeRepeatabilityEvents(
            pd.DataFrame(rows), "gauge", operator_column="operator"
        )
        .gauge_rr_summary()
        .iloc[0]
    )
    # K1 = 0.5908, K2 = 0.5231, K3 = 0.3146 are rounded in the manual.
    assert row["EV"] == pytest.approx(0.20188, rel=1e-3)
    assert row["AV"] == pytest.approx(0.22963, rel=1e-3)
    assert row["GRR"] == pytest.approx(0.30575, rel=1e-3)
    assert row["PV"] == pytest.approx(1.10456, rel=1e-3)
    assert row["TV"] == pytest.approx(1.14610, rel=1e-3)
    assert row["pct_GRR"] == pytest.approx(26.68, abs=0.05)
    assert row["pct_PV"] == pytest.approx(96.38, abs=0.05)
    assert row["ndc"] == 5


# ---------------------------------------------------------------------------
# Cp / Cpk / Pp / Ppk / Cpm on a hand example (AIAG SPC manual; Montgomery 8.3)
# ---------------------------------------------------------------------------


def test_capability_hand_example():
    # x = 10.0, 10.4, 10.2, 10.6, 9.8; LSL 9, USL 11, T 10.
    # mean 10.2; s^2 = (.04 + .04 + 0 + .16 + .16) / 4 = 0.1, s = 0.316228.
    # Pp = 2 / (6 s) = 1.054093; Ppk = (11 - 10.2) / (3 s) = 0.843274.
    # Cpm = 2 / (6 sqrt(0.1 + 0.2^2)) = 0.890871.
    # Moving ranges .4 .2 .4 .8: MR-bar 0.45, sigma_within = 0.398936:
    # Cp = 2 / (6 * 0.398936) = 0.835556; Cpk = 0.8 / (3 * 0.398936) = 0.668444.
    x = [10.0, 10.4, 10.2, 10.6, 9.8]
    rows = [
        {"systime": T0, "uuid": "usl", "value_double": 11.0},
        {"systime": T0, "uuid": "lsl", "value_double": 9.0},
    ] + [
        {"systime": T0 + pd.Timedelta(minutes=i + 1), "uuid": "x", "value_double": v}
        for i, v in enumerate(x)
    ]
    tde = ToleranceDeviationEvents(
        pd.DataFrame(rows),
        "value_double",
        "value_double",
        actual_uuid="x",
        upper_tolerance_uuid="usl",
        lower_tolerance_uuid="lsl",
    )
    res = tde.capability_indices(target_value=10.0).iloc[0]
    assert res["mean"] == pytest.approx(10.2)
    assert res["std"] == pytest.approx(0.316228, abs=1e-6)
    assert res["std_within"] == pytest.approx(0.398936, abs=1e-6)
    assert res["Pp"] == pytest.approx(1.054093, abs=1e-6)
    assert res["Ppk"] == pytest.approx(0.843274, abs=1e-6)
    assert res["Cp"] == pytest.approx(0.835556, abs=1e-6)
    assert res["Cpk"] == pytest.approx(0.668444, abs=1e-6)
    assert res["Cpm"] == pytest.approx(0.890871, abs=1e-6)


# ---------------------------------------------------------------------------
# Reliability and energy
# ---------------------------------------------------------------------------


def test_mtbf_mttr_docstring_example():
    # ReliabilityEvents docstring: hourly samples, faults at 02:00 and 06:00,
    # each lasting one sample (1 h). Observation 00:00-07:00 = 7 h: 5 h
    # operating, 2 h repair, 2 failures -> MTBF = 5 / 2 = 2.5 h,
    # MTTR = 2 / 2 = 1.0 h, availability = 5 / 7 = 71.4 % (MTBF / (MTBF + MTTR)).
    t = pd.date_range("2025-01-01", periods=8, freq="1h")
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": "fault",
            "value_bool": [False, False, True, False, False, False, True, False],
        }
    )
    kpi = ReliabilityEvents(df, "fault").mtbf_mttr().iloc[0]
    assert kpi["failures"] == 2
    assert kpi["mtbf_hours"] == pytest.approx(2.5)
    assert kpi["mttr_hours"] == pytest.approx(1.0)
    assert kpi["availability_pct"] == pytest.approx(100 * 5 / 7)
    assert round(kpi["availability_pct"], 1) == 71.4


def test_weibull_mle_matches_scipy_on_fixed_sample():
    # Two-parameter Weibull MLE (location 0) of complete data equals
    # scipy.stats.weibull_min.fit(floc=0) (Lawless, Statistical Models and
    # Methods for Lifetime Data, Sec. 5.2).
    tbf = [12.0, 30.5, 44.1, 51.0, 63.2, 70.8, 88.4, 95.0, 120.3, 150.7]
    rows = [(0.0, True)]
    h = 1.0
    for d in tbf:
        rows.append((h, False))
        h += d
        rows.append((h, True))
        h += 1.0
    rows.append((h, False))
    df = pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(hours=x) for x, _ in rows],
            "uuid": "fault",
            "value_bool": [v for _, v in rows],
        }
    )
    fit = ReliabilityEvents(df, "fault").weibull_fit().iloc[0]
    beta, _, eta = stats.weibull_min.fit(tbf, floc=0)
    assert fit["n_failures"] == len(tbf)
    assert fit["n_censored"] == 0
    assert fit["beta"] == pytest.approx(beta, rel=1e-4)
    assert fit["eta_hours"] == pytest.approx(eta, rel=1e-4)
    assert fit["mttf_hours"] == pytest.approx(eta * math.gamma(1 + 1 / beta), rel=1e-4)


def test_power_integration_step_hold_and_trapezoid():
    # 10 kW at 00:00, 5 kW at 01:00, 7.5 kW at 02:00 (the last sample closes
    # the data). Step hold (LOCF): 10 * 1 + 5 * 1 = 15.0 kWh. Trapezoid:
    # (10 + 5) / 2 + (5 + 7.5) / 2 = 7.5 + 6.25 = 13.75 kWh.
    df = pd.DataFrame(
        {
            "systime": pd.date_range(T0, periods=3, freq="1h"),
            "uuid": "meter",
            "value_double": [10.0, 5.0, 7.5],
        }
    )
    locf = PowerIntegrator.energy_by_window(df, window="1D", method="locf")
    linear = PowerIntegrator.energy_by_window(df, window="1D", method="linear")
    assert locf["kwh"].sum() == pytest.approx(15.0)
    assert linear["kwh"].sum() == pytest.approx(13.75)
