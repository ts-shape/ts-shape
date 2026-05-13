import pandas as pd  # type: ignore
from ts_shape.transform.filter.string_filter import StringFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter


def test_string_filter_basic_operations():
    df = pd.DataFrame(
        {
            "value_string": ["alpha", "beta", None, "alphabet", "BETA"],
            "systime": pd.date_range("2023-01-01", periods=5, freq="D"),
        }
    )

    not_na = StringFilter.filter_na_value_string(df)
    assert len(not_na) == 4

    match = StringFilter.filter_value_string_match(df, "beta")
    assert match["value_string"].tolist() == ["beta"]

    not_match = StringFilter.filter_value_string_not_match(df, "beta")
    assert "beta" not in not_match["value_string"].tolist()

    contains = StringFilter.filter_string_contains(df, "alpha")
    assert contains["value_string"].tolist() == ["alpha", "alphabet"]

    cleaned = StringFilter.regex_clean_value_string(
        df.copy(), column_name="value_string", regex_pattern=r"\\d+", replacement=""
    )
    # unchanged strings because no digits
    assert cleaned["value_string"].fillna("").str.contains(r"\d").sum() == 0

    # change detection
    df2 = pd.DataFrame({"value_string": ["A", "A", "B", "B", "B", "C"]})
    changes = StringFilter.detect_changes_in_string(df2, "value_string")
    # Includes first row and when value changes (A->B, B->C)
    assert changes["value_string"].tolist() == ["A", "B", "C"]


def test_datetime_filter_range_and_points():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
            "value_integer": [1, 2, 3, 4, 5],
        }
    )

    after = DateTimeFilter.filter_after_date(df, "systime", "2023-01-02")
    assert after["systime"].min() > pd.Timestamp("2023-01-02")

    before = DateTimeFilter.filter_before_date(df, "systime", "2023-01-04")
    assert before["systime"].max() < pd.Timestamp("2023-01-04")

    between = DateTimeFilter.filter_between_dates(
        df, "systime", "2023-01-02", "2023-01-05"
    )
    assert between["systime"].min() > pd.Timestamp("2023-01-02")
    assert between["systime"].max() < pd.Timestamp("2023-01-05")

    after_dt = DateTimeFilter.filter_after_datetime(
        df, "systime", "2023-01-02 00:00:00"
    )
    assert after_dt["systime"].min() > pd.Timestamp("2023-01-02")

    before_dt = DateTimeFilter.filter_before_datetime(
        df, "systime", "2023-01-04 00:00:00"
    )
    assert before_dt["systime"].max() < pd.Timestamp("2023-01-04")

    between_dt = DateTimeFilter.filter_between_datetimes(
        df, "systime", "2023-01-02 01:00:00", "2023-01-04 12:00:00"
    )
    assert between_dt["systime"].min() > pd.Timestamp("2023-01-02 01:00:00")
    assert between_dt["systime"].max() < pd.Timestamp("2023-01-04 12:00:00")
