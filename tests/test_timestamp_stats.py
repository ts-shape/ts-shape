import pandas as pd  # type: ignore
from ts_shape.features.stats.timestamp_stats import TimestampStatistics


def test_timestamp_statistics_distributions_and_ranges():
    ts = pd.to_datetime(
        [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-02 02:00:00",
            "2023-02-01 03:00:00",
            "2023-02-01 04:00:00",
        ]
    )
    df = pd.DataFrame({"systime": ts})

    assert TimestampStatistics.count_null(df) == 0
    assert TimestampStatistics.count_not_null(df) == 5
    assert TimestampStatistics.earliest_timestamp(df) == ts.min()
    assert TimestampStatistics.latest_timestamp(df) == ts.max()
    assert TimestampStatistics.timestamp_range(df) == ts.max() - ts.min()
    assert TimestampStatistics.most_frequent_day(df) in range(7)
    assert TimestampStatistics.most_frequent_hour(df) in range(24)
    assert not TimestampStatistics.year_distribution(df).empty
    assert not TimestampStatistics.month_distribution(df).empty
    assert not TimestampStatistics.weekday_distribution(df).empty
    assert not TimestampStatistics.hour_distribution(df).empty
    assert TimestampStatistics.average_time_gap(df).total_seconds() >= 0
    assert TimestampStatistics.median_timestamp(df) in ts.values
    assert not TimestampStatistics.timestamp_quartiles(df).empty
    assert not TimestampStatistics.days_with_most_activity(df).empty
