"""Regression tests for the legacy-code cleanup (downcasting, UTC folders, ...)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.events._compat import state_mask
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.frames import ffill_values

# ---- no silent downcasting -------------------------------------------------


@pytest.fixture()
def no_future_warnings():
    """Turn pandas' FutureWarning (deprecated silent downcasting) into errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        yield


def _object_bools() -> pd.Series:
    return pd.Series([True, None, False, np.nan, True], dtype=object)


def test_bool_or_false_matches_fillna_astype(no_future_warnings):
    for values in (
        _object_bools(),
        pd.Series([True, pd.NA, False], dtype="boolean"),
        pd.Series([1.0, np.nan, 0.0]),
        pd.Series([True, False]),
    ):
        out = bool_or_false(values)
        assert out.dtype == bool
        assert out.tolist() == [bool(v) if pd.notna(v) else False for v in values]
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    keep = bool_or_false(pd.Series([True, None, False], index=idx, name="s"))
    assert keep.index.equals(idx) and keep.name == "s"


def test_ffill_values_keeps_object_dtype(no_future_warnings):
    filled = ffill_values(pd.Series([None, True, None, False, None], dtype=object))
    assert filled.dtype == object
    assert filled.tolist()[1:] == [True, True, False, False]
    assert pd.isna(filled.iloc[0])
    idx = pd.Index([10, 11, 12])
    same = ffill_values(pd.Series(["a", None, "b"], index=idx))
    assert same.index.equals(idx) and same.tolist() == ["a", "a", "b"]


def test_state_mask_object_bool_column(no_future_warnings):
    data = pd.DataFrame({"value_bool": _object_bools(), "value_string": None})
    running = state_mask(data, "value_string", True, bool_value=True)
    assert running.tolist() == [True, False, False, False, True]
    stopped = state_mask(data, "value_string", False, bool_value=False)
    assert stopped.tolist() == [False, False, True, False, False]


def test_cycle_extractors_on_object_bools(no_future_warnings):
    from ts_shape.features.cycles.cycles_extractor import CycleExtractor

    times = pd.date_range("2024-01-01", periods=6, freq="min")
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": "s",
            "value_bool": pd.Series(
                [False, True, None, False, True, False], dtype=object
            ),
        }
    )
    cycles = CycleExtractor(df, start_uuid="s").process_persistent_cycle()
    assert len(cycles) == 2
    changes = CycleExtractor(df, start_uuid="s").process_value_change_cycle()
    assert len(changes) >= 1


def test_future_no_silent_downcasting_path():
    """The helpers give the same answer with pandas' opt-in future behavior."""
    if not hasattr(pd.options, "future") or not hasattr(
        pd.options.future, "no_silent_downcasting"
    ):
        pytest.skip("option removed in this pandas version")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.option_context("future.no_silent_downcasting", True):
            out = bool_or_false(_object_bools())
            data = pd.DataFrame({"value_bool": _object_bools(), "value_string": None})
            mask = state_mask(data, "value_string", True, bool_value=True)
    assert out.tolist() == [True, False, False, False, True]
    assert mask.tolist() == [True, False, False, False, True]
