"""Tests for the shared reset-aware counter helpers."""

import pandas as pd  # type: ignore
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.counters import (
    count_resets,
    counter_increments,
    counter_quantity,
)


def test_counter_quantity_monotonic():
    values = pd.Series([100, 150, 220, 300])
    assert counter_quantity(values) == 200.0
    assert counter_quantity(values, handle_resets=True) == 200.0


def test_counter_quantity_handles_reset_by_default():
    # Counter resets mid-window: 400 produced, then 150 after the reset.
    values = pd.Series([100, 300, 500, 0, 150])
    assert counter_quantity(values) == 550.0  # 400 + 150
    assert counter_quantity(values, handle_resets=False) == 50.0  # last - first


def test_counter_quantity_reset_below_first_clamps_to_zero_without_resets():
    # Ends below where it started -> last - first clamps a productive window to 0.
    values = pd.Series([500, 700, 0, 100])
    assert counter_quantity(values) == 300.0  # 200 + 100
    assert counter_quantity(values, handle_resets=False) == 0.0


def test_counter_non_numeric_values_warn():
    values = pd.Series([100, "n/a", 300, "oops"], dtype=object)
    with pytest.warns(DataQualityWarning, match="2 non-null value"):
        assert counter_quantity(values) == 200.0


def test_counter_quantity_empty():
    assert counter_quantity(pd.Series([], dtype=float)) == 0.0
    assert counter_quantity(pd.Series([], dtype=float), handle_resets=True) == 0.0


def test_counter_increments_first_is_baseline():
    values = pd.Series([10, 12, 15])
    inc = counter_increments(values)
    assert inc.iloc[0] == 0.0  # first reading is baseline, not production
    assert inc.iloc[1] == 2.0
    assert inc.iloc[2] == 3.0
    assert inc.sum() == 5.0


def test_counter_increments_reset_counts_post_reset_value():
    values = pd.Series([40, 45, 0, 5])
    inc = counter_increments(values)
    # 0 (baseline), +5, reset -> post-reset value 0, +5
    assert list(inc) == [0.0, 5.0, 0.0, 5.0]
    assert inc.sum() == 10.0


def test_count_resets():
    assert count_resets(pd.Series([1, 2, 3])) == 0
    assert count_resets(pd.Series([1, 2, 0, 3, 1])) == 2


def test_counter_quantity_handles_non_numeric_gaps():
    values = pd.Series([100, None, 300])
    # NaN is dropped; 100 -> 300 = 200
    assert counter_quantity(values) == 200.0
