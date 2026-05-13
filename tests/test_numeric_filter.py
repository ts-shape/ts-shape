import pandas as pd  # type: ignore
from ts_shape.transform.filter.numeric_filter import IntegerFilter, DoubleFilter


def test_integer_filter_match():
    df = pd.DataFrame(
        {
            "value_integer": [10, 20, 30, 40, 50],
            "systime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
        }
    )
    result = IntegerFilter.filter_value_integer_match(df, integer_value=30)
    expected = pd.DataFrame(
        {"value_integer": [30], "systime": pd.to_datetime(["2023-01-03"])}, index=[2]
    )
    pd.testing.assert_frame_equal(result, expected)


def test_integer_filter_not_match():
    df = pd.DataFrame(
        {
            "value_integer": [10, 20, 30, 40, 50],
            "systime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
        }
    )
    result = IntegerFilter.filter_value_integer_not_match(df, integer_value=30)
    expected = pd.DataFrame(
        {
            "value_integer": [10, 20, 40, 50],
            "systime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-04", "2023-01-05"]
            ),
        },
        index=[0, 1, 3, 4],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_integer_filter_between():
    df = pd.DataFrame(
        {
            "value_integer": [10, 20, 30, 40, 50],
            "systime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
        }
    )
    result = IntegerFilter.filter_value_integer_between(df, min_value=20, max_value=40)
    expected = pd.DataFrame(
        {
            "value_integer": [20, 30, 40],
            "systime": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
        },
        index=[1, 2, 3],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_double_filter_nan_and_between():
    df = pd.DataFrame(
        {
            "value_double": [0.5, 1.5, float("nan"), 2.5, 3.5],
            "systime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
        }
    )
    result_non_nan = DoubleFilter.filter_nan_value_double(df)
    expected_non_nan = pd.DataFrame(
        {
            "value_double": [0.5, 1.5, 2.5, 3.5],
            "systime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-04", "2023-01-05"]
            ),
        },
        index=[0, 1, 3, 4],
    )
    pd.testing.assert_frame_equal(result_non_nan, expected_non_nan)

    result_between = DoubleFilter.filter_value_double_between(
        df, min_value=1.0, max_value=3.0
    )
    expected_between = pd.DataFrame(
        {
            "value_double": [1.5, 2.5],
            "systime": pd.to_datetime(["2023-01-02", "2023-01-04"]),
        },
        index=[1, 3],
    )
    pd.testing.assert_frame_equal(result_between, expected_between)
