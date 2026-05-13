import pandas as pd  # type: ignore
from ts_shape.transform.filter.custom_filter import CustomFilter


def test_custom_filter_conditions_query():
    df = pd.DataFrame(
        {
            "value_integer": [1, 10, 5, 20],
            "value_double": [0.1, 0.9, 0.3, 0.2],
            "systime": pd.date_range("2023-01-01", periods=4, freq="h"),
        }
    )

    out = CustomFilter.filter_custom_conditions(
        df, "value_integer > 5 and value_double < 0.5"
    )
    # rows index 1: (10, 0.9) filtered out by value_double; index 3: (20,0.2) keeps
    assert out.index.tolist() == [3]
