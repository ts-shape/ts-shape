import pandas as pd  # type: ignore
from ts_shape.transform.filter.boolean_filter import IsDeltaFilter, BooleanFilter


def test_is_delta_true():
    data = {
        "is_delta": [True, False, True, False, True],
        "systime": pd.date_range(start="2022-01-01", periods=5, freq="min"),
    }
    df = pd.DataFrame(data)

    result = IsDeltaFilter.filter_is_delta_true(df)

    assert len(result) == 3
    assert all(result["is_delta"])


def test_is_delta_false():
    data = {
        "is_delta": [True, False, True, False, True],
        "systime": pd.date_range(start="2022-01-01", periods=5, freq="min"),
    }
    df = pd.DataFrame(data)

    result = IsDeltaFilter.filter_is_delta_false(df)

    assert len(result) == 2
    assert not any(result["is_delta"])


def test_boolean_filter_falling():
    data = {
        "value_bool": [True, True, False, True, False],
        "systime": pd.date_range(start="2022-01-01", periods=5, freq="min"),
    }
    df = pd.DataFrame(data)

    result = BooleanFilter.filter_falling_value_bool(df)

    assert len(result) == 2
    assert not result["value_bool"].iloc[0]
    assert not result["value_bool"].iloc[1]


def test_boolean_filter_raising():
    data = {
        "value_bool": [False, True, False, True, False],
        "systime": pd.date_range(start="2022-01-01", periods=5, freq="min"),
    }
    df = pd.DataFrame(data)

    result = BooleanFilter.filter_raising_value_bool(df)

    assert len(result) == 2
    assert result["value_bool"].iloc[0]
    assert result["value_bool"].iloc[1]
