"""Regression tests for Pipeline, the dataset generators and the catalog."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape import Pipeline
from ts_shape.catalog import list_detectors
from ts_shape.datasets import make_id_signal, make_timeseries
from ts_shape.errors import DataQualityWarning

# ---------- Pipeline -------------------------------------------------------


def _keep(df: pd.DataFrame, **_: object) -> pd.DataFrame:
    return df


@pytest.mark.parametrize(
    "value",
    [np.array([1, 2]), pd.Series([1, 2]), pd.DataFrame({"a": [1]}), [1, 2]],
)
def test_non_string_kwargs_are_not_compared_with_sentinels(value):
    seen = {}

    def _capture(df, payload):
        seen["payload"] = payload
        return df

    Pipeline().transform(_capture, name="cap", payload=value).run(
        make_timeseries(n_points=3)
    )
    assert seen["payload"] is value


def test_double_dollar_escapes_a_literal():
    seen = {}

    def _capture(df, label):
        seen["label"] = label
        return df

    Pipeline().transform(_capture, name="cap", label="$$input").run(
        make_timeseries(n_points=3)
    )
    assert seen["label"] == "$input"


def test_duplicate_explicit_transform_name_raises():
    pipe = Pipeline().transform(_keep, name="clean")
    with pytest.raises(ValueError, match="duplicate step name"):
        pipe.transform(_keep, name="clean")


def test_default_transform_names_get_suffixes():
    pipe = Pipeline().transform(_keep).transform(_keep)
    assert [name for _, name in pipe.steps] == ["_keep", "_keep_2"]


@pytest.mark.parametrize("kind", ["transform", "detect"])
def test_step_named_input_raises(kind):
    with pytest.raises(ValueError, match="reserved"):
        getattr(Pipeline(), kind)(_keep, name="input")


def test_in_place_transform_does_not_mutate_caller_frame():
    df = make_timeseries(n_points=5)
    before = df.copy()

    def _scale_in_place(frame):
        frame["value_double"] *= 10
        return frame

    result = Pipeline().transform(_scale_in_place, name="scale").run(df)
    pd.testing.assert_frame_equal(df, before)
    assert result.data["value_double"].iloc[0] == pytest.approx(
        before["value_double"].iloc[0] * 10
    )


# ---------- datasets -------------------------------------------------------


def test_make_timeseries_accepts_a_single_uuid_string():
    df = make_timeseries("sensor:temp", n_points=4)
    assert set(df["uuid"]) == {"sensor:temp"}
    assert len(df) == 4


def test_make_timeseries_with_no_uuids_is_typed_and_empty():
    df = make_timeseries([])
    assert df.empty
    assert list(df.columns) == list(make_timeseries(n_points=1).columns)
    assert pd.api.types.is_datetime64_any_dtype(df["systime"])


def test_make_timeseries_outliers_without_noise_warn():
    with pytest.warns(DataQualityWarning, match="noise=0"):
        make_timeseries(n_points=10, noise=0, n_outliers=2)


def test_unused_value_columns_use_nullable_dtypes():
    for df in (make_timeseries(n_points=3), make_id_signal()):
        assert str(df["value_bool"].dtype) == "boolean"
        assert str(df["value_integer"].dtype) in ("Int64", "int64")
    assert str(make_timeseries(n_points=3)["value_integer"].dtype) == "Int64"


def test_make_id_signal_integer_ids_must_be_numbers():
    with pytest.raises(ValueError, match="whole-number"):
        make_id_signal(values=("A", "B"), value_column="value_integer")
    df = make_id_signal(values=(1, 2), value_column="value_integer", hold=2)
    assert df["value_integer"].tolist() == [1, 1, 2, 2]


# ---------- catalog --------------------------------------------------------


def test_list_detectors_lists_current_classes_only():
    df = list_detectors()
    names = set(df["name"])
    assert {"profile", "to_event_log", "make_timeseries"}.isdisjoint(names)
    assert "CycleTimeTracking" not in names  # deprecated alias
    for name in df["name"]:
        assert inspect.isclass(getattr(ts_shape, name)), name
