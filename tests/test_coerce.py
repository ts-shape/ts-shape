"""Tests for ts_shape.utils.coerce.to_numeric_checked."""

import warnings

import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.coerce import to_numeric_checked


def test_numeric_values_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        out = to_numeric_checked(pd.Series([1, None, "2.5"]), what="x")
    assert out.tolist()[0] == 1.0
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == 2.5


def test_non_numeric_values_are_counted():
    values = pd.Series(["1", "n/a", None, "oops"])
    with pytest.warns(DataQualityWarning, match="2 non-null value.*in column 'v'"):
        out = to_numeric_checked(values, what="column 'v'")
    assert out.isna().tolist() == [False, True, True, True]
