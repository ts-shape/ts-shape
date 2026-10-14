"""Weibull maximum-likelihood fit against a published dataset.

Source: Lieblein & Zelen (1956), "Statistical investigation of the fatigue
life of deep-groove ball bearings", J. Res. NBS 57(5): 23 bearing lives in
millions of revolutions. The two-parameter Weibull MLE is beta = 2.102,
eta = 81.88 (Lawless, *Statistical Models and Methods for Lifetime Data*,
2nd ed., Example 5.2.2, reports u = log(eta) = 4.405, b = 1/beta = 0.476).

The lives are used as hours between failures. Independently of scipy's
fitter, the MLE is also obtained by solving the Weibull likelihood equation
for beta (Lawless eq. 5.2.4; Abernethy, *The New Weibull Handbook*), which
with right-censored times t_j reads

    sum_all t^b ln t / sum_all t^b - 1/b - mean_failures(ln t) = 0,
    eta = (sum_all t^b / r) ** (1 / b),   r = number of failures.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from ts_shape.events.maintenance.reliability import ReliabilityEvents

BEARINGS = [
    17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.48, 51.84, 51.96, 54.12,
    55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64, 105.12, 105.84,
    127.92, 128.04, 173.40,
]  # fmt: skip

T0 = pd.Timestamp("2024-01-01 00:00")


def _fault_frame(tbf_hours, trailing_hours=0.0, repair_hours=1.0):
    """Repair, then (operate tbf, repair) for each tbf, then optional running."""
    rows = [(0.0, True)]
    t = repair_hours
    for tbf in tbf_hours:
        rows.append((t, False))
        t += tbf
        rows.append((t, True))
        t += repair_hours
    rows.append((t, False))
    if trailing_hours:
        rows.append((t + trailing_hours, False))
    return pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(hours=h) for h, _ in rows],
            "uuid": "fault",
            "value_bool": [v for _, v in rows],
        }
    )


def mle_reference(failures, censored=()):
    """Solve the Weibull likelihood equation (complete or right-censored data)."""
    f = np.asarray(failures, dtype=float)
    t = np.concatenate([f, np.asarray(censored, dtype=float)])
    lt, mean_lf = np.log(t), np.log(f).mean()

    def score(b):
        w = t**b
        return (w * lt).sum() / w.sum() - 1 / b - mean_lf

    beta = optimize.brentq(score, 0.05, 50, xtol=1e-14)
    eta = ((t**beta).sum() / len(f)) ** (1 / beta)
    return beta, eta


def test_ball_bearings_published_mle():
    fit = ReliabilityEvents(_fault_frame(BEARINGS), "fault").weibull_fit().iloc[0]
    assert fit["n_failures"] == 23
    assert fit["n_censored"] == 0
    # Published (Lawless Example 5.2.2).
    assert fit["beta"] == pytest.approx(2.102, abs=5e-4)
    assert fit["eta_hours"] == pytest.approx(81.88, abs=5e-3)
    # Independent solution of the likelihood equation.
    beta, eta = mle_reference(BEARINGS)
    assert fit["beta"] == pytest.approx(beta, rel=1e-4)
    assert fit["eta_hours"] == pytest.approx(eta, rel=1e-4)
    # Derived quantities from the closed forms.
    b, e = fit["beta"], fit["eta_hours"]
    assert fit["mttf_hours"] == pytest.approx(e * math.gamma(1 + 1 / b))
    assert fit["b10_hours"] == pytest.approx(e * (-math.log(0.9)) ** (1 / b))
    assert fit["failure_pattern"] == "wear_out"


def test_right_censored_running_period_matches_likelihood_equation():
    # The asset has been running 150 h since the last repair: censored.
    fit = (
        ReliabilityEvents(_fault_frame(BEARINGS, trailing_hours=150.0), "fault")
        .weibull_fit()
        .iloc[0]
    )
    assert fit["n_failures"] == 23
    assert fit["n_censored"] == 1
    beta, eta = mle_reference(BEARINGS, censored=[150.0])
    assert fit["beta"] == pytest.approx(beta, rel=1e-4)
    assert fit["eta_hours"] == pytest.approx(eta, rel=1e-4)
    # Censoring a long survivor raises the characteristic life.
    assert fit["eta_hours"] > 81.88


def test_reliability_at_uses_the_fitted_model():
    rel = (
        ReliabilityEvents(_fault_frame(BEARINGS), "fault")
        .reliability_at([81.88, 50.0], age_hours=0.0)
        .set_index("hours")
    )
    # R(eta) = exp(-1) by definition of the scale parameter.
    assert rel.loc[81.88, "reliability"] == pytest.approx(math.exp(-1), abs=1e-4)
    beta, eta = mle_reference(BEARINGS)
    assert rel.loc[50.0, "reliability"] == pytest.approx(
        math.exp(-((50.0 / eta) ** beta)), rel=1e-4
    )
