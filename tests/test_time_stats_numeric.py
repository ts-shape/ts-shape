import pandas as pd  # type: ignore
import pytest

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

    # Default: a cumulative counter; every increment is booked in the window
    # of the reading that reports it (01:00 adds 1, 01:30 adds 2).
    diff_df = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "diff"
    )
    assert diff_df["difference"].iloc[1] == 3.0
    plain = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "diff", handle_resets=False
    )
    assert plain["difference"].iloc[1] == 2.0  # 5 - 3

    range_df = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "range"
    )
    assert range_df["range"].iloc[0] == 1.0


def test_diff_handles_counter_reset_by_default():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2023-01-01", periods=4, freq="15min"),
            "value_double": [100.0, 300.0, 0.0, 150.0],
        }
    )
    out = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "diff"
    )
    assert out["difference"].iloc[0] == 350.0  # 200 + 150 after the reset
    plain = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "diff", handle_resets=False
    )
    assert plain["difference"].iloc[0] == 50.0


def test_statistic_is_per_uuid_by_default():
    df = pd.concat(
        [make_df().assign(uuid="a"), make_df().assign(value_double=100.0, uuid="b")],
        ignore_index=True,
    )
    out = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "mean"
    )
    assert out.loc[("a", pd.Timestamp("2023-01-01 00:00")), "mean"] == 1.5
    assert out.loc[("b", pd.Timestamp("2023-01-01 00:00")), "mean"] == 100.0
    pooled = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "h", "mean", group_by_uuid=False
    )
    assert pooled["mean"].iloc[0] == pytest.approx(50.75)


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


def test_calculate_statistics_per_uuid_by_default():
    import ts_shape
    from ts_shape.features.time_stats.time_stats_numeric import (
        TimeGroupedStatistics,
    )

    df = ts_shape.make_timeseries(["a", "b"], n_points=50)
    out = TimeGroupedStatistics.calculate_statistics(
        df, "systime", "value_double", "10min", ["mean", "max"]
    )
    assert list(out.columns) == ["mean", "max"]
    assert set(out.index.get_level_values("uuid")) == {"a", "b"}
    one = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "10min", "mean"
    )
    assert out["mean"].equals(one["mean"])
