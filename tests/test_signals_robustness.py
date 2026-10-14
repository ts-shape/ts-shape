"""Regression tests for signals primitives, spec validation, inference and profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.signals import SignalSpec, SignalSpecs, infer_specs
from ts_shape.signals.primitives import (
    align,
    increments,
    intervals,
    per_window,
    shift_windows,
    windows,
)


def _frame(uuid, values, times, column="value_double"):
    return pd.DataFrame({"systime": times, "uuid": uuid, column: values})


# ---------- align ----------------------------------------------------------


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_align_mixes_time_units(unit):
    t = pd.date_range("2024-01-01", periods=3, freq="1min")
    left = _frame("pv", [1.0, 2.0, 3.0], t.as_unit("ns"))
    right = _frame("sp", [5.0], t[:1].as_unit(unit))
    out = align(left, SignalSpec("pv"), right, SignalSpec("sp"))
    assert out["sp"].tolist() == [5.0, 5.0, 5.0]


def test_align_with_pyarrow_timestamps():
    t = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
    left = _frame("pv", [1.0, 2.0], t)
    right = _frame("sp", [5.0, 6.0], pd.Series(t).astype("timestamp[us, UTC][pyarrow]"))
    out = align(left, SignalSpec("pv"), right, SignalSpec("sp"))
    assert out["sp"].tolist() == [5.0, 6.0]
    assert str(out["systime"].dt.tz) == "UTC"


def test_align_aware_vs_naive_raises():
    t = pd.date_range("2024-01-01", periods=2, freq="1min")
    left = _frame("pv", [1.0, 2.0], t)
    right = _frame("sp", [5.0, 6.0], t.tz_localize("UTC"))
    with pytest.raises(ValueError, match="timezone-aware"):
        align(left, SignalSpec("pv"), right, SignalSpec("sp"))


def test_align_same_uuid_gets_suffixes_and_time_column_name():
    t = pd.date_range("2024-01-01", periods=2, freq="1min")
    df = pd.DataFrame({"ts": t, "uuid": "x", "value_double": [1.0, 2.0]})
    out = align(df, SignalSpec("x"), df, SignalSpec("x"), time_column="ts")
    assert list(out.columns) == ["ts", "x_left", "x_right"]


# ---------- per_window -----------------------------------------------------


def test_per_window_boolean_with_na_is_fraction_of_known_time():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:20", "2024-01-01 00:40"])
    df = _frame(
        "r", pd.array([True, pd.NA, False], dtype="boolean"), t, column="value_bool"
    )
    spec = SignalSpec("r", kind="state", value_column="value_bool")
    out = per_window(df, spec, "1h", data_end=pd.Timestamp("2024-01-01 01:00"))
    # 20 min True, 20 min unknown, 20 min False -> 0.5 of the known time.
    assert out.iloc[0] == pytest.approx(0.5)
    assert out.name == "r"


def test_per_window_names_series_after_uuid_for_every_kind():
    t = pd.date_range("2024-01-01", periods=4, freq="20min")
    for kind in ("continuous", "counter", "event"):
        df = _frame("s", [1.0, 2.0, 3.0, 4.0], t)
        assert per_window(df, SignalSpec("s", kind=kind), "1h").name == "s"


def test_per_window_data_end_naive_on_aware_data():
    t = pd.date_range("2024-01-01", periods=2, freq="30min", tz="Europe/Berlin")
    df = _frame("s", [1.0, 3.0], t)
    out = per_window(df, SignalSpec("s"), "1h", data_end="2024-01-01 01:00")
    assert out.iloc[0] == pytest.approx(2.0)


def test_per_window_with_pyarrow_time_column():
    t = pd.Series(pd.date_range("2024-01-01", periods=3, freq="20min"))
    df = _frame("s", [1.0, 2.0, 3.0], t.astype("timestamp[ns][pyarrow]"))
    out = per_window(df, SignalSpec("s"), "1h")
    assert out.iloc[0] == pytest.approx(1.5)


@pytest.mark.parametrize("func", [intervals, increments])
def test_primitives_missing_columns_raise(func):
    df = pd.DataFrame({"systime": pd.date_range("2024", periods=2), "uuid": "c"})
    with pytest.raises(ColumnNotFoundError, match="value_integer"):
        func(df, SignalSpec("c", kind="counter", value_column="value_integer"))
    with pytest.raises(ColumnNotFoundError):
        per_window(df, SignalSpec("c"), "1h")


# ---------- windows --------------------------------------------------------


def test_windows_with_mixed_aware_and_naive_bounds():
    out = windows(pd.Timestamp("2024-01-01", tz="UTC"), "2024-01-03", "1D")
    assert len(out) == 2
    assert str(out["start"].dt.tz) == "UTC"


def test_shift_windows_with_mixed_bounds_and_empty_range():
    out = shift_windows(pd.Timestamp("2024-01-01 06:00", tz="UTC"), "2024-01-01 14:00")
    assert out["shift"].tolist() == ["shift_1"]
    empty = shift_windows("2024-01-01 06:00", "2024-01-01 06:00")
    assert empty.empty
    assert list(empty.columns) == [
        "production_day",
        "shift",
        "start",
        "end",
        "duration_seconds",
    ]


# ---------- spec validation ------------------------------------------------


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"reset_policy": "sometimes"}, "reset_policy"),
        ({"max_gap": "-5min"}, "max_gap"),
        ({"tz": "Mars/Olympus"}, "tz"),
    ],
)
def test_signal_spec_validates_fields(fields, match):
    with pytest.raises(ValueError, match=match):
        SignalSpec("x", **fields)


def test_override_with_unknown_field_raises_value_error():
    with pytest.raises(ValueError, match="bogus"):
        SignalSpecs({"x": SignalSpec("x")}).override({"x": {"bogus": 1}})
    df = ts_shape.make_timeseries(["x"], n_points=10)
    with pytest.raises(ValueError, match="bogus"):
        ts_shape.profile(df, specs={"x": {"bogus": 1}})


# ---------- inference / profile --------------------------------------------


def test_infer_specs_warns_on_uuid_type_clash_and_nan_uuid():
    t = pd.date_range("2024-01-01", periods=6, freq="1min")
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": pd.Series([1, "1", 1, "1", None, 1], dtype=object),
            "value_double": np.arange(6.0),
        }
    )
    with pytest.warns(DataQualityWarning) as record:
        specs = infer_specs(df)
    messages = " ".join(str(w.message) for w in record)
    assert "different types" in messages and "no uuid" in messages
    assert list(specs) == ["1"]


def test_profile_of_id_signal_and_pyarrow_values():
    report = ts_shape.profile(ts_shape.make_id_signal())
    assert report.table["kind"].iloc[0] == "state"
    df = ts_shape.make_timeseries(["a"], n_points=20)
    df["value_double"] = df["value_double"].astype("double[pyarrow]")
    assert len(ts_shape.profile(df).table) == 1


def test_profile_with_duplicate_index_labels():
    df = ts_shape.make_timeseries(["a", "b"], n_points=5)
    df.index = [0] * len(df)
    assert len(ts_shape.profile(df).table) == 2


def test_profile_mixed_offset_times_are_utc_with_warning():
    df = pd.DataFrame(
        {
            "systime": ["2024-03-30T12:00+01:00", "2024-03-31T12:00+02:00"],
            "uuid": "a",
            "value_double": [1.0, 2.0],
        }
    )
    with pytest.warns(DataQualityWarning, match="UTC"):
        report = ts_shape.profile(df)
    assert str(report.table["first"].iloc[0].tz) == "UTC"


def test_profile_missing_columns_raise():
    with pytest.raises(ColumnNotFoundError):
        ts_shape.profile(pd.DataFrame())
    with pytest.raises(ColumnNotFoundError, match="uuid"):
        ts_shape.profile(pd.DataFrame({"systime": [], "value_double": []}))


def test_profile_does_not_mutate_input():
    df = ts_shape.make_id_signal()
    before = df.copy()
    ts_shape.profile(df)
    pd.testing.assert_frame_equal(df, before)
