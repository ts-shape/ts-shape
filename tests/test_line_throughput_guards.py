"""Guard: throughput_oee must not divide by a zero target rate."""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.events.production.line_throughput import LineThroughputEvents


def test_throughput_oee_zero_target_rate_no_nan():
    # Flat counter -> quantile(0.95) target rate is 0; performance must be 0,
    # not NaN/inf.
    t = pd.date_range("2024-01-01 00:00:00", periods=6, freq="1h")
    df = pd.DataFrame(
        {
            "uuid": ["counter"] * len(t),
            "systime": t,
            "value_integer": [100] * len(t),  # never increments
            "is_delta": [True] * len(t),
        }
    )
    res = LineThroughputEvents(df).throughput_oee("counter")
    assert not res["performance"].isna().any()
    assert not np.isinf(res["performance"]).any()
    assert (res["oee_score"] == 0.0).all()


def test_throughput_oee_explicit_zero_target_rate():
    t = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h")
    df = pd.DataFrame(
        {
            "uuid": ["counter"] * len(t),
            "systime": t,
            "value_integer": [0, 10, 20, 30],
            "is_delta": [True] * len(t),
        }
    )
    res = LineThroughputEvents(df).throughput_oee("counter", target_rate=0)
    assert (res["performance"] == 0.0).all()
