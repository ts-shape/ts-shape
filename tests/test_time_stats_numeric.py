import pandas as pd  # type: ignore
from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics


def make_df():
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:30:00",
                    "2023-01-01 01:00:00",
                    "2023-01-01 01:30:00",
                ]
            ),
            "value_double": [1.0, 2.0, 3.0, 5.0],
        }
    )


def test_calculate_statistic_and_range_diff():
    df = make_df()
    mean_df = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "mean"
    )
    assert "mean" in mean_df.columns

    sum_df = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "sum"
    )
    assert sum_df["sum"].iloc[0] == 3.0

    diff_df = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "diff"
    )
    assert diff_df["difference"].iloc[1] == 2.0  # 5 - 3

    range_df = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "range"
    )
    assert range_df["range"].iloc[0] == 1.0


def test_calculate_multiple_and_custom():
    df = make_df()
    combo = TimeGroupedStatistics.calculate_statistics(
        df, "systime", "value_double", "h", ["mean", "sum", "range"]
    )
    assert set(["mean", "sum", "range"]).issubset(combo.columns)

    custom = TimeGroupedStatistics.calculate_custom_func(
        df, "systime", "value_double", "h", lambda s: (s.max() - s.min())
    )
    assert "custom" in custom.columns
