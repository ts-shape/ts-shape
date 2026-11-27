import pandas as pd

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


def test_average_time_gap_multi_uuid_is_per_signal():
    """On a merged two-signal frame the inter-signal transition must not
    count as a sample gap. Signal A fires every minute, signal B every
    hour; the pooled per-signal mean is the mean of those two cadences,
    not polluted by the February-to-April jump from the last A row to the
    first B row.
    """
    a_times = pd.date_range("2024-02-01", periods=5, freq="1min")
    b_times = pd.date_range("2024-04-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "systime": a_times.append(b_times),
            "uuid": ["a"] * 5 + ["b"] * 5,
        }
    )

    # Per-signal (default): gaps of 1min on A, 1h on B, pooled mean = 30.5min.
    per_signal = TimestampStatistics.average_time_gap(df)
    assert per_signal == pd.Timedelta(seconds=(60 + 3600) / 2)

    # Opting out (single-stream) reproduces the old behaviour: the
    # February-to-April jump dominates and the mean is tens of days.
    single = TimestampStatistics.average_time_gap(df, uuid_column=None)
    assert single > pd.Timedelta(days=1)


def test_standard_deviation_timestamps_multi_uuid_is_per_signal():
    """The same per-signal grouping applies to the std-of-gaps reducer."""
    a_times = pd.date_range("2024-02-01", periods=4, freq="1min")
    b_times = pd.date_range("2024-02-01 12:00", periods=4, freq="1min")
    df = pd.DataFrame(
        {
            "systime": a_times.append(b_times),
            "uuid": ["a"] * 4 + ["b"] * 4,
        }
    )

    # Every per-signal gap is exactly 1 minute, so the pooled std is 0.
    per_signal = TimestampStatistics.standard_deviation_timestamps(df)
    assert per_signal == pd.Timedelta(0)

    # Without the grouping the 12-hour inter-signal jump balloons the std.
    single = TimestampStatistics.standard_deviation_timestamps(df, uuid_column=None)
    assert single > pd.Timedelta(hours=1)


def test_time_gap_tolerates_missing_uuid_column():
    """A plain frame without a uuid column falls back to single-stream."""
    df = pd.DataFrame({"systime": pd.date_range("2024-01-01", periods=3, freq="1min")})
    assert TimestampStatistics.average_time_gap(df) == pd.Timedelta(minutes=1)
    assert TimestampStatistics.standard_deviation_timestamps(df) == pd.Timedelta(0)
