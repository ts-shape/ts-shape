import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ts_shape.events.quality.multivariate_spc import (
    MultivariateSPCEvents,
    phase1_ucl,
    phase2_ucl,
)


def _long(t, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": u, "value_double": v})
            for u, v in cols.items()
        ],
        ignore_index=True,
    )


def _correlated(n, rng, rho=0.95, shift=None, tz=None):
    cov = [[1.0, rho], [rho, 1.0]]
    x = rng.multivariate_normal([100.0, 50.0], cov, size=n)
    if shift is not None:
        x[shift[0] :] += shift[1]
    t = pd.date_range("2024-01-01", periods=n, freq="10s", tz=tz)
    return _long(t, {"temp": x[:, 0], "press": x[:, 1]}), t


def test_control_limits_match_textbook_formulas():
    # Montgomery, Introduction to SQC, eq. 11.19 / 11.20
    p, m, a = 2, 30, 0.0027
    assert phase2_ucl(p, m, a) == pytest.approx(
        p * (m + 1) * (m - 1) / (m * (m - p)) * stats.f.ppf(1 - a, p, m - p)
    )
    assert phase1_ucl(p, m, a) == pytest.approx(
        (m - 1) ** 2 / m * stats.beta.ppf(1 - a, p / 2, (m - p - 1) / 2)
    )
    # Large m: the Phase II limit tends to the chi-square limit.
    assert phase2_ucl(3, 100_000, 0.01) == pytest.approx(
        stats.chi2.ppf(0.99, 3), rel=1e-3
    )
    with pytest.raises(ValueError):
        phase2_ucl(3, 3, 0.01)


def test_in_control_false_alarm_rate_matches_alpha():
    rng = np.random.default_rng(0)
    df, t = _correlated(20_000, rng)
    ref = (t[0], t[2_000])
    out = MultivariateSPCEvents(df, ["temp", "press"]).hotelling_t2(
        reference_period=ref, alpha=0.01
    )
    phase2 = out[out["phase"] == "II"]
    assert len(phase2) == 18_000
    assert phase2["out_of_control"].mean() == pytest.approx(0.01, abs=0.004)


def test_t2_catches_correlation_break_that_univariate_3_sigma_misses():
    rng = np.random.default_rng(1)
    # From sample 600 temperature rises 1 sigma while pressure falls 1
    # sigma: each stays inside its own 3-sigma band, the relation breaks
    # (expected T² = 2 / (1 - rho) = 40 against a UCL of about 12).
    df, t = _correlated(800, rng, shift=(600, np.array([1.0, -1.0])))
    ref = (t[0], t[500])
    wide = df.pivot(index="systime", columns="uuid", values="value_double")
    base = wide.iloc[:500]
    z = (wide.iloc[600:] - base.mean()) / base.std()
    assert (z.abs() > 3).any(axis=1).mean() < 0.15  # univariate: mostly silent

    ev = MultivariateSPCEvents(df, ["temp", "press"])
    out = ev.hotelling_t2(reference_period=ref)
    assert out["out_of_control"].iloc[600:].mean() > 0.9
    assert out["out_of_control"].iloc[500:600].mean() < 0.02

    runs = ev.detect_out_of_control(reference_period=ref, min_samples=5)
    assert len(runs) >= 1
    assert runs["sample_count"].sum() > 150
    assert set(runs.columns) >= {
        "start",
        "end",
        "duration_seconds",
        "max_t2",
        "top_contributor",
    }


def test_contributions_sum_to_t2_and_point_at_the_shifted_variable():
    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(0, 1, (n, 3))
    x[500:, 2] += 6.0  # only the third variable shifts
    t = pd.date_range("2024-01-01", periods=n, freq="1min")
    df = _long(t, {"a": x[:, 0], "b": x[:, 1], "c": x[:, 2]})
    ev = MultivariateSPCEvents(df, ["a", "b", "c"])
    ref = (t[0], t[400])
    c = ev.contributions(reference_period=ref)
    sums = c.groupby("systime")["contribution"].sum()
    t2 = c.groupby("systime")["t2"].first()
    assert np.allclose(sums, t2)
    late = c[c["systime"] >= t[500]]
    top = late.loc[late.groupby("systime")["contribution"].idxmax(), "source_uuid"]
    assert (top == "c").mean() > 0.95
    assert (
        ev.detect_out_of_control(reference_period=ref)["top_contributor"].mode()[0]
        == "c"
    )
    allrows = ev.contributions(reference_period=ref, only_out_of_control=False)
    assert len(allrows) == 3 * n


def test_alignment_holds_backward_with_max_gap_and_keeps_tz():
    t = pd.date_range("2024-03-31 00:00", periods=6, freq="30min", tz="Europe/Berlin")
    a = pd.DataFrame({"systime": t, "uuid": "a", "value_double": np.arange(6.0)})
    # b only at t0 and t4: held for 45 minutes at most
    b = pd.DataFrame({"systime": t[[0, 4]], "uuid": "b", "value_double": [1.0, 2.0]})
    ev = MultivariateSPCEvents(pd.concat([a, b]), ["a", "b"], max_gap="45min")
    assert ev.matrix["systime"].tolist() == [t[0], t[1], t[4], t[5]]
    assert str(ev.matrix["systime"].dt.tz) == "Europe/Berlin"


def test_empty_short_and_validation():
    empty = pd.DataFrame(columns=["systime", "uuid", "value_double"])
    ev = MultivariateSPCEvents(empty, ["a", "b"])
    assert ev.hotelling_t2().empty
    assert ev.contributions().empty
    assert ev.detect_out_of_control().empty
    with pytest.raises(ValueError, match="at least two"):
        MultivariateSPCEvents(empty, ["a"])
    t = pd.date_range("2024-01-01", periods=3, freq="1min")
    tiny = _long(t, {"a": np.arange(3.0), "b": np.arange(3.0) ** 2})
    with pytest.raises(ValueError, match="reference period"):
        MultivariateSPCEvents(tiny, ["a", "b"]).hotelling_t2()
