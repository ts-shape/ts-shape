"""Regression tests: outlier detection per uuid, with NaN and MAD == 0."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents


def _signal(values, uuid, start="2024-01-01"):
    t = pd.date_range(start, periods=len(values), freq="1h")
    return pd.DataFrame({"systime": t, "uuid": uuid, "value_double": values})


def test_zscore_ignores_nan():
    vals = [10.0] * 30 + [100.0] + [np.nan]
    out = OutlierDetectionEvents(
        _signal(vals, "a"), "value_double"
    ).detect_outliers_zscore()
    assert (out["value_double"] == 100.0).any()


def test_mad_zero_does_not_flag_every_deviation():
    # Quantised sensor: more than half the readings identical -> MAD == 0.
    vals = [5.0] * 12 + [5.1] * 5 + [4.9] * 5
    ev = OutlierDetectionEvents(_signal(vals, "a"), "value_double")
    assert ev.detect_outliers_mad().empty
    spiked = OutlierDetectionEvents(_signal(vals + [50.0], "a"), "value_double")
    assert (spiked.detect_outliers_mad()["value_double"] == 50.0).any()


def test_group_by_uuid_separates_signals():
    rng = np.random.default_rng(0)
    temp = 80 + rng.normal(0, 1, 50)
    pres = 2 + rng.normal(0, 0.05, 50)
    pres[25] = 2.6  # a real pressure spike, invisible next to the temperatures
    df = pd.concat([_signal(temp, "temp"), _signal(pres, "pres", "2024-01-01 00:30")])
    pooled = OutlierDetectionEvents(
        df, "value_double", group_by_uuid=False
    ).detect_outliers_zscore()
    assert not (pooled["value_double"] == 2.6).any()
    # Per-uuid scoring is the default.
    per_uuid = OutlierDetectionEvents(df, "value_double").detect_outliers_zscore()
    assert (per_uuid["value_double"] == 2.6).any()
    assert set(per_uuid["source_uuid"]) == {"pres"}


def test_isolation_forest_skips_nan():
    pytest.importorskip("sklearn")
    vals = [1.0] * 20 + [np.nan, 50.0]
    ev = OutlierDetectionEvents(_signal(vals, "a"), "value_double")
    out = ev.detect_outliers_isolation_forest(contamination=0.05)
    assert (out["value_double"] == 50.0).any()


def _zscore_cases():
    rng = np.random.default_rng(7)
    cases = [
        [1.0],
        [np.nan],
        [np.nan, np.nan],
        [5.0, 5.0, 5.0],
        [5.0, 5.0, np.nan],
        [0.0, -0.0],
        [1.0, 2.0],
        [np.inf, 1.0, 2.0],
    ]
    for n in (3, 100, 5000):
        for scale in (1e-9, 1.0, 1e9):
            a = rng.normal(loc=rng.choice([0.0, 1e6]), scale=scale, size=n)
            cases.append(a.copy())
            a[rng.random(n) < 0.2] = np.nan
            cases.append(a)
    return cases


@pytest.mark.parametrize("values", _zscore_cases())
def test_zscore_scores_match_scipy(values):
    # The z-scores are computed in numpy so that importing the module does not
    # load scipy.stats; they match scipy.stats.zscore(nan_policy="omit").
    import warnings

    from scipy import __version__ as scipy_version
    from scipy.stats import zscore

    from ts_shape.events.quality.outlier_detection import _abs_zscore

    series = pd.Series(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = np.abs(np.asarray(zscore(series, nan_policy="omit")))
    got = _abs_zscore(series)
    if tuple(int(p) for p in scipy_version.split(".")[:2]) >= (1, 18):
        np.testing.assert_array_equal(got, expected)  # bit for bit
    else:  # older scipy sums in a different order
        np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)


def test_zscore_ignores_nullable_missing_values():
    vals = pd.array([10.0] * 30 + [100.0, None], dtype="Float64")
    out = OutlierDetectionEvents(
        _signal(vals, "a"), "value_double"
    ).detect_outliers_zscore()
    assert out["value_double"].tolist() == [100.0]
