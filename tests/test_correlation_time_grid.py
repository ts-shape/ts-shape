"""Regression tests: correlation lags and windows are measured in time, not rows."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.correlation.signal_correlation import SignalCorrelationEvents


def _pair(times, a, b):
    return pd.concat(
        [
            pd.DataFrame({"systime": times, "uuid": "a", "value_double": a}),
            pd.DataFrame({"systime": times, "uuid": "b", "value_double": b}),
        ]
    ).dropna()


def test_lag_correlation_exact_across_gaps():
    rng = np.random.default_rng(1)
    t = pd.date_range("2024-01-01", periods=600, freq="1min")
    a = pd.Series(rng.normal(size=600).cumsum(), index=t)
    b = a.shift(3)  # b lags a by 3 minutes
    keep = (np.arange(600) % 20) < 10  # 10-minute logging gap every 20 minutes
    df = _pair(t[keep], a.to_numpy()[keep], b.to_numpy()[keep])
    out = SignalCorrelationEvents(df).lag_correlation("a", "b", max_lag=5)
    best = out[out["is_best_lag"]].iloc[0]
    assert best["lag_periods"] == -3
    assert best["correlation"] == pytest.approx(1.0)


def test_rolling_window_does_not_span_a_gap():
    rng = np.random.default_rng(2)
    t1 = pd.date_range("2024-01-01 00:00", periods=30, freq="1min")
    t2 = pd.date_range("2024-01-03 00:00", periods=30, freq="1min")  # 2-day gap
    t = t1.append(t2)
    a = rng.normal(size=60)
    df = _pair(t, a, a + rng.normal(scale=0.1, size=60))
    out = SignalCorrelationEvents(df).rolling_correlation("a", "b", window=10)
    after_gap = out[(out["systime"] >= t2[0]) & (out["systime"] < t2[4])]
    # min_periods = 5: the first 4 post-gap minutes have too few samples in
    # their 10-minute window (pre-gap rows must not count).
    assert after_gap.empty
    assert (out["systime"] >= t2[4]).any()
