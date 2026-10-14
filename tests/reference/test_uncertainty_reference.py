"""GUM measurement uncertainty against the GUM's own worked numbers.

Source: JCGM 100:2008 (GUM), Section 4.4.3 and Figure 1: twenty repeated
observations of a temperature t have arithmetic mean 100.145 degC, sample
standard deviation s(t_k) = 1.489 degC and standard uncertainty of the
mean u(t) = s(t_k) / sqrt(20) = 0.333 degC (Type A evaluation, GUM 4.2).

The Type B budget follows GUM 4.3.7 (rectangular a / sqrt(3)), 4.3.9
(triangular a / sqrt(6)), the U-shaped (arcsine) a / sqrt(2) of EA-4/02, and
4.3.3 (a calibration certificate quoting U at k = 2 gives U / 2). The
combined uncertainty is the root-sum-of-squares of GUM 5.1.2 with unit
sensitivity coefficients, and U = k u_c (GUM 6.2.1). Every expected number is
computed by hand in the test.
"""

import math

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.measurement_uncertainty import (
    MeasurementUncertaintyEvents,
)

# GUM 4.4.3, the twenty observations of Figure 1 (degC).
GUM_443 = [
    96.90, 98.18, 98.25, 98.61, 99.03, 99.49, 99.56, 99.74, 99.89, 100.07,
    100.33, 100.42, 100.68, 100.95, 101.11, 101.20, 101.57, 101.84, 102.36, 102.72,
]  # fmt: skip

SOURCES = [
    # Calibration certificate: U = 0.04 degC at k = 2 -> u = 0.02.
    {"source": "calibration", "half_width": 0.04, "distribution": "normal"},
    # Display resolution 0.01 degC -> a = 0.005, rectangular.
    {"source": "resolution", "half_width": 0.005, "distribution": "rectangular"},
    # Reference drift within +-0.06, most likely near 0 -> triangular.
    {"source": "drift", "half_width": 0.06, "distribution": "triangular"},
    # Bath cycling between its extremes +-0.1 -> U-shaped.
    {"source": "bath", "half_width": 0.1, "distribution": "u-shaped"},
]
U_B = {
    "calibration": 0.02,
    "resolution": 0.005 / math.sqrt(3),
    "drift": 0.06 / math.sqrt(6),
    "bath": 0.1 / math.sqrt(2),
}


def _mue():
    t = pd.Timestamp("2024-01-01 08:00") + pd.to_timedelta(range(20), unit="min")
    df = pd.DataFrame({"systime": t, "uuid": "temp", "value_double": GUM_443})
    return MeasurementUncertaintyEvents(df, "temp")


def test_type_a_matches_gum_4_4_3():
    row = _mue().type_a_uncertainty(window="8h").iloc[0]
    assert row["n_samples"] == 20
    assert row["mean"] == pytest.approx(100.145)
    assert round(row["std"], 3) == 1.489
    assert round(row["type_a_uncertainty"], 3) == 0.333
    assert row["type_a_uncertainty"] == pytest.approx(
        np.std(GUM_443, ddof=1) / math.sqrt(20)
    )


def test_type_b_divisors_hand_computed():
    out = MeasurementUncertaintyEvents.type_b_uncertainty(SOURCES)
    got = dict(zip(out["source"], out["type_b_uncertainty"]))
    for name, u in U_B.items():
        assert got[name] == pytest.approx(u)
    divisors = dict(zip(out["source"], out["divisor"]))
    assert divisors["calibration"] == pytest.approx(2.0)
    assert divisors["resolution"] == pytest.approx(math.sqrt(3))
    assert divisors["drift"] == pytest.approx(math.sqrt(6))
    assert divisors["bath"] == pytest.approx(math.sqrt(2))


def test_combined_and_expanded_uncertainty_hand_computed():
    u_a = np.std(GUM_443, ddof=1) / math.sqrt(20)
    u_b = math.sqrt(sum(u**2 for u in U_B.values()))
    u_c = math.sqrt(u_a**2 + u_b**2)
    row = (
        _mue()
        .combined_uncertainty(window="8h", type_b_sources=SOURCES, coverage_factor=2.0)
        .iloc[0]
    )
    assert row["type_a"] == pytest.approx(u_a)
    assert row["type_b_combined"] == pytest.approx(u_b)
    assert row["combined_uncertainty"] == pytest.approx(u_c)
    assert row["expanded_uncertainty"] == pytest.approx(2 * u_c)
    # By hand: u_A^2 = 0.110833, u_B^2 = 0.0004 + 0.0000083 + 0.0006 + 0.005
    # = 0.0060083, so u_c = sqrt(0.116841) = 0.34182 and U = 0.68364.
    assert row["combined_uncertainty"] == pytest.approx(0.34182, abs=5e-5)
    assert row["expanded_uncertainty"] == pytest.approx(0.68364, abs=1e-4)


def test_uncertainty_budget_contributions_hand_computed():
    u_a = np.std(GUM_443, ddof=1) / math.sqrt(20)
    budget = _mue().uncertainty_budget(window="8h", type_b_sources=SOURCES)
    variances = {"repeatability (Type A)": u_a**2}
    variances.update({f"{k} (Type B)": u**2 for k, u in U_B.items()})
    total = sum(variances.values())
    got = dict(zip(budget["source"], budget["pct_contribution"]))
    assert set(got) == set(variances)
    for name, var in variances.items():
        assert got[name] == pytest.approx(100 * var / total)
    assert budget["pct_contribution"].sum() == pytest.approx(100.0)
    # Repeatability dominates: 0.110833 / 0.116841 = 94.86 % of the variance.
    assert got["repeatability (Type A)"] == pytest.approx(94.86, abs=0.01)
