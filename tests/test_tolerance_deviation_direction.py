"""Regression tests: tolerances apply forward in time; capability limits."""

import operator

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
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
    """The deprecated single ``tolerance_uuid`` path."""
    with pytest.warns(DeprecationWarning, match="tolerance_uuid is deprecated"):
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


def _upper_only(df):
    return ToleranceDeviationEvents(
        df,
        "value_double",
        "value_double",
        actual_uuid="act",
        event_uuid="evt",
        upper_tolerance_uuid="tol",
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
    res = _events(df).capability_indices().iloc[0]
    assert res["usl"] == 15.0  # the latest tolerance, not the first
    assert np.isnan(res["Cp"]) and np.isnan(res["lsl"])
    # Ppk uses the overall sample std; Cpk the within sigma MR-bar / 1.128
    # (moving ranges 1, 1, 1 -> 1 / 1.128).
    assert res["Ppk"] == pytest.approx(
        (15.0 - 12.0) / (3 * df.loc[df["uuid"] == "act", "value_double"].std()),
        abs=1e-4,
    )
    assert res["Cpk"] == pytest.approx((15.0 - 12.0) / (3 / 1.128))


@pytest.mark.parametrize(
    "values",
    [[12.0], [12.0, 12.0, 12.0, 12.0], [np.nan, np.nan, np.nan, np.nan]],
    ids=["single", "constant", "all_nan"],
)
def test_capability_without_spread_is_nan_with_warning(values):
    # Aligned with CapabilityTrendingEvents: NaN indices, not an error.
    df = _df()
    act = df.index[df["uuid"] == "act"]
    df = df.drop(index=act[len(values) :])
    df.loc[df["uuid"] == "act", "value_double"] = values
    with pytest.warns(DataQualityWarning, match="without spread"):
        res = _events(df).capability_indices().iloc[0]
    assert np.isnan(res["Cpk"]) and np.isnan(res["Ppk"])
    assert res["usl"] == 15.0
    assert pd.notna(res["start"])


def test_detect_deviations_keeps_value_and_single_sample_events():
    t = pd.to_datetime(
        ["2024-01-01 09:00", "2024-01-01 09:30", "2024-01-01 10:00", "2024-01-01 10:01"]
    )
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": ["tol", "act", "act", "act"],
            "value_double": [10.0, 12.0, 13.0, 14.0],
            "is_delta": True,
        }
    )
    out = _events(df).detect_deviations()
    # 09:30 is a one-sample event (one row); 10:00-10:01 a two-sample event.
    assert list(out["systime"].dt.strftime("%H:%M")) == ["09:30", "10:00", "10:01"]
    assert list(out["value_double"]) == [12.0, 13.0, 14.0]


def test_separate_tolerances_keep_value_and_single_sample_events():
    t = pd.to_datetime(
        ["2024-01-01 09:00", "2024-01-01 09:00", "2024-01-01 09:30", "2024-01-01 09:40"]
    )
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": ["usl", "lsl", "act", "act"],
            "value_double": [12.0, 8.0, 13.0, 10.0],
            "is_delta": True,
        }
    )
    out = ToleranceDeviationEvents(
        df,
        "value_double",
        "value_double",
        actual_uuid="act",
        event_uuid="evt",
        upper_tolerance_uuid="usl",
        lower_tolerance_uuid="lsl",
    ).detect_deviations()
    assert len(out) == 1
    assert out["value_double"].iloc[0] == 13.0


def test_missing_actual_uuid_raises():
    with pytest.raises(ValueError, match="actual_uuid is required"):
        ToleranceDeviationEvents(
            _df(), "value_double", "value_double", tolerance_uuid="tol"
        )


def test_capability_cpm_only_with_target():
    t = pd.date_range("2024-01-01", periods=6, freq="min")
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": ["usl", "lsl", "act", "act", "act", "act"],
            "value_double": [16.0, 4.0, 11.0, 12.0, 13.0, 12.0],
            "is_delta": True,
        }
    )
    ev = ToleranceDeviationEvents(
        df,
        "value_double",
        "value_double",
        actual_uuid="act",
        upper_tolerance_uuid="usl",
        lower_tolerance_uuid="lsl",
    )
    assert np.isnan(ev.capability_indices().iloc[0]["Cpm"])
    res = ev.capability_indices(target_value=10.0).iloc[0]
    sigma = pd.Series([11.0, 12.0, 13.0, 12.0]).std()
    pp = 12.0 / (6 * sigma)
    assert res["Pp"] == pytest.approx(pp)
    assert res["Cpm"] == pytest.approx(
        pp / np.sqrt(1 + ((12.0 - 10.0) / sigma) ** 2), abs=1e-4
    )
    # sigma_method="std" (the earlier behavior): Cp/Cpk from the overall
    # sigma, equal to Pp/Ppk.
    old = ev.capability_indices(target_value=10.0, sigma_method="std").iloc[0]
    assert old["Pp"] == old["Cp"] and old["Ppk"] == old["Cpk"]
    assert old["Cpm"] == res["Cpm"]
    # On target, Cpm equals Pp (both use the overall sigma).
    assert ev.capability_indices(target_value=12.0).iloc[0]["Cpm"] == res["Pp"]


def test_upper_tolerance_alone_is_one_sided():
    df = _df()
    out = _upper_only(df).detect_deviations()
    # 12 > 10 and 16 > 15: the same four rows as the single-tolerance path.
    assert sorted(out["systime"].dt.strftime("%H:%M")) == [
        "09:30",
        "09:31",
        "10:30",
        "10:31",
    ]
    assert list(out["deviation_abs"]) == pytest.approx([2.0, 2.0, 1.0, 1.0])
    # Percentages relative to the magnitude of the single limit.
    assert list(out["deviation_pct"]) == pytest.approx([20.0, 20.0, 100 / 15, 100 / 15])


def test_lower_tolerance_alone_is_one_sided():
    t = pd.date_range("2024-01-01", periods=4, freq="min")
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": ["lsl", "act", "act", "act"],
            "value_double": [10.0, 9.0, 11.0, 12.0],
            "is_delta": True,
        }
    )
    ev = ToleranceDeviationEvents(
        df,
        "value_double",
        "value_double",
        actual_uuid="act",
        lower_tolerance_uuid="lsl",
    )
    out = ev.detect_deviations()
    assert list(out["value_double"]) == [9.0]
    res = ev.capability_indices().iloc[0]
    assert np.isnan(res["usl"]) and np.isnan(res["Cp"])
    sigma = pd.Series([9.0, 11.0, 12.0]).std()
    assert res["Ppk"] == pytest.approx((32.0 / 3 - 10.0) / (3 * sigma))
    # Within sigma: moving ranges 2, 1 -> 1.5 / 1.128.
    assert res["Cpk"] == pytest.approx((32.0 / 3 - 10.0) / (3 * 1.5 / 1.128))


def test_capability_upper_only_matches_single_tolerance():
    df = _df()
    single = _events(df).capability_indices().iloc[0]
    upper = _upper_only(df).capability_indices().iloc[0]
    assert upper["Cpk"] == pytest.approx(single["Cpk"])
    assert np.isnan(upper["Cp"]) and np.isnan(upper["lsl"])


def test_no_tolerance_raises():
    with pytest.raises(ValueError, match="upper_tolerance_uuid"):
        ToleranceDeviationEvents(
            _df(), "value_double", "value_double", actual_uuid="act"
        )
