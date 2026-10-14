"""Regression tests: tolerances apply forward in time; capability limits."""

import operator

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents


def _df():
    t = pd.to_datetime(
        [
            "2024-01-01 09:00",
            "2024-01-01 09:30",
            "2024-01-01 09:31",
            "2024-01-01 10:00",
            "2024-01-01 10:30",
            "2024-01-01 10:31",
        ]
    )
    return pd.DataFrame(
        {
            "systime": t,
            "uuid": ["tol", "act", "act", "tol", "act", "act"],
            "value_double": [10.0, 12.0, 12.0, 15.0, 16.0, 16.0],
            "is_delta": True,
        }
    )


def _events(df, **kw):
    return ToleranceDeviationEvents(
        dataframe=df,
        tolerance_column="value_double",
        actual_column="value_double",
        tolerance_uuid="tol",
        actual_uuid="act",
        event_uuid="evt",
        compare_func=operator.ge,
        time_threshold="5min",
        **kw,
    )


def test_measurements_use_the_tolerance_set_before_them():
    out = _events(_df()).detect_deviations()
    # 12 >= 10 (09:30 group) and 16 >= 15 (10:30 group): two events, 2 rows each.
    assert len(out) == 4
    assert sorted(out["systime"].dt.strftime("%H:%M")) == [
        "09:30",
        "09:31",
        "10:30",
        "10:31",
    ]


def test_capability_single_tolerance_is_one_sided():
    df = _df()
    df.loc[df["uuid"] == "act", "value_double"] = [11.0, 12.0, 13.0, 12.0]
    res = _events(df).capability_indices()
    assert res["usl"] == 15.0  # the latest tolerance, not the first
    assert np.isnan(res["Cp"]) and np.isnan(res["lsl"])
    assert res["Cpk"] == pytest.approx(
        (15.0 - 12.0) / (3 * df.loc[df["uuid"] == "act", "value_double"].std()),
        abs=1e-4,
    )


def test_capability_single_measurement_raises():
    df = _df().iloc[:2]
    with pytest.raises(ValueError, match="undefined"):
        _events(df).capability_indices()
