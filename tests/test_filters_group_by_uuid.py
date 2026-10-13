"""Regression tests: edge/change filters on multi-uuid frames."""

import pandas as pd

from ts_shape.transform.filter.boolean_filter import BooleanFilter
from ts_shape.transform.filter.string_filter import StringFilter


def _frame():
    t = pd.date_range("2024-01-01", periods=2, freq="1min")
    a = pd.DataFrame(
        {
            "systime": t,
            "uuid": "a",
            "value_bool": [False, True],
            "value_string": ["x", "y"],
        }
    )
    b = pd.DataFrame(
        {
            "systime": t,
            "uuid": "b",
            "value_bool": [False, False],
            "value_string": ["y", "y"],
        }
    )
    return pd.concat([a, b], ignore_index=True)


def test_falling_edge_not_across_uuids():
    df = _frame()
    assert len(BooleanFilter.filter_falling_value_bool(df)) == 1  # a:True -> b:False
    assert BooleanFilter.filter_falling_value_bool(df, group_by_uuid=True).empty


def test_raising_edge_per_uuid():
    out = BooleanFilter.filter_raising_value_bool(_frame(), group_by_uuid=True)
    assert list(out["uuid"]) == ["a"]


def test_string_changes_per_uuid():
    df = _frame()
    out = StringFilter.detect_changes_in_string(df, group_by_uuid=True)
    # First row of each uuid has no predecessor and counts as a change.
    assert list(zip(out["uuid"], out["value_string"])) == [
        ("a", "x"),
        ("a", "y"),
        ("b", "y"),
    ]
