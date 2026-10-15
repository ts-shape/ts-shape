"""Shared input helpers in ts_shape.utils.frames."""

import pandas as pd
import pytest

from ts_shape.errors import ColumnNotFoundError
from ts_shape.utils.frames import (
    as_datetime,
    localize_like,
    require_columns,
    same_time_axis,
    signal_frame_input,
)


def test_require_columns_names_function_and_missing():
    df = pd.DataFrame({"systime": [], "uuid": []})
    with pytest.raises(ColumnNotFoundError, match="f: column 'value_double'"):
        require_columns(df, ["systime", "value_double", None], where="f")
    require_columns(df, ["systime", None], where="f")


def test_as_datetime_parses_text_and_types_empty():
    s = as_datetime(pd.Series(["2024-01-01", "2024-01-02"], name="t"), where="f")
    assert pd.api.types.is_datetime64_any_dtype(s)
    empty = as_datetime(pd.Series([], dtype=object, name="t"), where="f")
    assert str(empty.dtype) == "datetime64[ns]"
    with pytest.raises(ValueError, match="f: column 't'"):
        as_datetime(pd.Series(["x", "y"], name="t"), where="f")


def test_as_datetime_pyarrow_keeps_tz():
    pytest.importorskip("pyarrow")
    s = pd.Series(pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))
    s = s.astype("timestamp[ns, tz=UTC][pyarrow]")
    out = as_datetime(s, where="f")
    assert str(out.dt.tz) == "UTC"
    assert not isinstance(out.dtype, pd.ArrowDtype)


def test_same_time_axis_units_and_tz():
    a = pd.Series(
        pd.date_range("2024-01-01", periods=2, freq="h").astype("datetime64[us]")
    )
    b = pd.Series(
        pd.date_range("2024-01-01", periods=2, freq="h").astype("datetime64[s]")
    )
    x, y = same_time_axis(a, b, where="f")
    assert x.dtype == y.dtype
    aware = pd.Series(pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="naive"):
        same_time_axis(aware, b, where="f")
    berlin = aware.dt.tz_convert("Europe/Berlin")
    x, y = same_time_axis(aware, berlin, where="f")
    assert x.dtype == y.dtype and (x == y).all()


def test_localize_like():
    naive = pd.Series(pd.date_range("2024-01-01", periods=2, freq="h"))
    berlin = naive.dt.tz_localize("Europe/Berlin")
    assert localize_like("2024-01-01 05:00", berlin).tz is not None
    assert localize_like("2024-01-01 05:00+01:00", naive) == pd.Timestamp(
        "2024-01-01 04:00"
    )
    assert localize_like("2024-01-01 05:00", naive) == pd.Timestamp("2024-01-01 05:00")


def test_signal_frame_input_normalizes_copy():
    seen = {}

    class C:
        @classmethod
        @signal_frame_input()
        def f(cls, dataframe, time_column="systime"):
            seen["dtype"] = dataframe["value_double"].dtype
            dataframe["value_double"] = 0.0
            return dataframe

    df = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=2),
            "value_double": pd.array([1.0, None], dtype="Float64"),
        }
    )
    C.f(df)
    assert seen["dtype"] == "float64"
    assert str(df["value_double"].dtype) == "Float64"


def test_as_datetime_unknown_layout_parses_without_warning():
    """Text pandas cannot infer a layout for is parsed value by value, quietly."""
    import warnings

    s = pd.Series(["5/1/24 10:00", "6/1/24 11:00"], name="t")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = as_datetime(s, where="f")
    assert list(out) == [
        pd.Timestamp("2024-05-01 10:00"),
        pd.Timestamp("2024-06-01 11:00"),
    ]
    # mixed layouts are read too
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mixed = as_datetime(pd.Series(["2024-01-01 10:00", "01/02/2024"]), where="f")
    assert list(mixed) == [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-02")]


def test_to_ns_naive_aware_and_units():
    import numpy as np

    from ts_shape.utils.frames import to_ns

    naive = pd.Series(pd.to_datetime(["2024-01-01 01:00", None]).as_unit("us"))
    ns = to_ns(naive)
    assert ns.dtype == np.int64
    assert ns[0] == pd.Timestamp("2024-01-01 01:00").value
    assert ns[1] == np.iinfo(np.int64).min
    berlin = naive.dt.tz_localize("Europe/Berlin")
    # an aware value gives its UTC instant
    assert to_ns(berlin)[0] == pd.Timestamp("2024-01-01 00:00").value
    assert list(to_ns(pd.DatetimeIndex(berlin))) == list(to_ns(berlin))
