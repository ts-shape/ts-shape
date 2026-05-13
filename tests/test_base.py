import pandas as pd  # type: ignore
from ts_shape.utils.base import Base


def test_base_sorts_by_given_time_column():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
            "value_integer": [3, 1, 2],
        }
    )
    base = Base(df, column_name="systime")
    out = base.get_dataframe()
    assert list(out["systime"]) == sorted(df["systime"].tolist())


def test_base_detects_time_column_when_not_provided():
    df = pd.DataFrame(
        {
            "created_time": ["2023-01-02", "2023-01-01", "2023-01-03"],
            "value_integer": [2, 1, 3],
        }
    )
    base = Base(df)
    out = base.get_dataframe()
    # created_time should be converted to datetime and sorted asc
    assert pd.api.types.is_datetime64_any_dtype(out["created_time"])
    assert list(out["created_time"]) == sorted(
        pd.to_datetime(df["created_time"]).tolist()
    )
