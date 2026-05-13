import pandas as pd  # type: ignore
from ts_shape.features.stats.string_stats import StringStatistics


def test_string_statistics():
    df = pd.DataFrame({"value_string": ["a", "bb", "A1", "bb", None, "CCC"]})
    assert StringStatistics.count_unique(df) == 4  # None excluded by nunique
    assert StringStatistics.most_frequent(df) == "bb"
    assert StringStatistics.count_most_frequent(df) == 2
    assert StringStatistics.count_null(df) == 1
    assert (
        StringStatistics.average_string_length(df)
        == pd.Series(["a", "bb", "A1", "bb", "CCC"]).str.len().mean()
    )
    assert StringStatistics.longest_string(df) == "CCC"
    assert StringStatistics.shortest_string(df) == "a"

    summary_df = StringStatistics.summary_as_dataframe(df, "value_string")
    assert summary_df.shape[0] == 1
    assert "most_frequent" in summary_df.columns

    # substring/startswith/endswith
    assert StringStatistics.contains_substring_count(df, "b") == 2  # 'bb' appears twice
    assert StringStatistics.starts_with_count(df, "b") == 2
    assert StringStatistics.ends_with_count(df, "b") == 2
    assert StringStatistics.uppercase_percentage(df) >= 0.0
    assert StringStatistics.lowercase_percentage(df) >= 0.0
    assert StringStatistics.contains_digit_count(df) == 1
