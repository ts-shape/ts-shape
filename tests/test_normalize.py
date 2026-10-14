"""Detectors give the same result whatever dtype backend the data uses."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.production.oee_calculator import OEEEvents
from ts_shape.utils.counters import counter_quantity
from ts_shape.utils.normalize import normalize_signal_frame

T = pd.date_range("2024-01-01 06:00", periods=5, freq="1h")


def _frame():
    run = pd.DataFrame({"systime": T, "uuid": "run", "value_bool": [True] * 5})
    count = pd.DataFrame(
        {"systime": T, "uuid": "count", "value_integer": [100, 300, 500, 0, 150]}
    )
    return pd.concat([run, count], ignore_index=True)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]", "float64"])
def test_counter_first_reading_is_never_production(dtype):
    assert counter_quantity(pd.Series([100, 300, 500, 0, 150], dtype=dtype)) == 550


@pytest.mark.parametrize("backend", ["numpy_nullable", "pyarrow"])
def test_detector_results_do_not_depend_on_the_dtype_backend(backend):
    plain = _frame()
    other = plain.convert_dtypes(dtype_backend=backend)
    a = OEEEvents(plain).performance("count", 3600.0, run_state_uuid="run")
    b = OEEEvents(other).performance("count", 3600.0, run_state_uuid="run")
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_text_numbers_are_parsed_and_unparseable_ones_reported():
    df = _frame()
    df["value_integer"] = [
        None if pd.isna(v) else str(int(v)) for v in df["value_integer"]
    ]
    df.loc[df.index[-1], "value_integer"] = "n/a"
    with pytest.warns(DataQualityWarning, match="1 non-null value") as rec:
        out = normalize_signal_frame(df.copy())
    assert rec[0].filename == __file__
    assert out["value_integer"].dtype == np.float64
    assert out["value_integer"].iloc[5] == 100


def test_nullable_booleans_become_plain_values_with_nan_for_missing():
    df = pd.DataFrame(
        {
            "systime": T[:3],
            "uuid": pd.array(["a", "a", None], dtype="string"),
            "value_bool": pd.array([True, None, False], dtype="boolean"),
            "value_double": pd.array([1.5, None, 2.0], dtype="Float64"),
            "other": pd.array([1, 2, 3], dtype="Int64"),
        }
    )
    out = normalize_signal_frame(df.copy())
    assert out["value_bool"].iloc[0] is True and out["value_bool"].iloc[2] is False
    assert pd.isna(out["value_bool"].iloc[1])
    assert out["value_double"].dtype == np.float64
    assert out["uuid"].dtype == object and pd.isna(out["uuid"].iloc[2])
    assert str(out["other"].dtype) == "Int64"  # not a signal column: untouched


def test_rows_with_the_same_timestamp_have_a_fixed_order():
    df = pd.DataFrame(
        {
            "systime": [T[0]] * 3,
            "uuid": ["c", "a", "b"],
            "value_double": [3.0, 1.0, 2.0],
        }
    )
    a = OEEEvents(df).dataframe["uuid"].tolist()
    b = OEEEvents(df.iloc[::-1]).dataframe["uuid"].tolist()
    assert a == b == ["a", "b", "c"]
