"""Guard: throughput_oee must not divide by a zero target rate."""

import numpy as np
import pandas as pd

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


def test_takt_adherence_cycle_equal_to_takt_is_not_violation():
    t = pd.to_datetime(
        ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:02:01"]
    )
    df = pd.DataFrame({"systime": t, "uuid": "c", "value_bool": [True] * 3})
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "systime": t + pd.Timedelta(seconds=1),
                    "uuid": "c",
                    "value_bool": [False] * 3,
                }
            ),
        ]
    )
    out = LineThroughputEvents(df).takt_adherence("c", takt_time="60s")
    assert out["cycle_time_seconds"].tolist() == [60.0, 61.0]
    assert out["violation"].tolist() == [False, True]
    strict = LineThroughputEvents(df).takt_adherence(
        "c", takt_time="60s", min_violation="1s"
    )
    assert strict["violation"].tolist() == [False, False]


def test_takt_adherence_numeric_counter_reset_is_not_a_cycle():
    t = pd.date_range("2024-01-01", periods=5, freq="30s")
    # 1, 2, reset to 0, 1, 2: the reset is not a cycle boundary.
    df = pd.DataFrame({"systime": t, "uuid": "cnt", "value_integer": [1, 2, 0, 1, 2]})
    out = LineThroughputEvents(df).takt_adherence(
        "cnt", value_column="value_integer", takt_time="30s"
    )
    # boundaries at 0 s, 30 s, 90 s, 120 s
    assert out["cycle_time_seconds"].tolist() == [30.0, 60.0, 30.0]
    assert out["violation"].tolist() == [False, True, False]


def test_count_parts_warns_on_non_numeric_counter_values():
    import pytest

    from ts_shape.errors import DataQualityWarning

    t = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1min")
    df = pd.DataFrame(
        {
            "uuid": ["counter"] * len(t),
            "systime": t,
            "value_integer": [0, 10, "bad", 30],
            "is_delta": [True] * len(t),
        }
    )
    with pytest.warns(DataQualityWarning, match="1 non-null value"):
        res = LineThroughputEvents(df).count_parts("counter", window="1min")
    assert res["part_count"].sum() == 30
