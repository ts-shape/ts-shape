import pandas as pd  # type: ignore
from ts_shape.features.stats.boolean_stats import BooleanStatistics


def test_boolean_statistics():
    df = pd.DataFrame({"value_bool": [True, False, True, None, False]})
    assert BooleanStatistics.count_true(df) == 2
    assert BooleanStatistics.count_false(df) == 2
    assert BooleanStatistics.count_null(df) == 1
    assert BooleanStatistics.count_not_null(df) == 4

    # 50% True out of non-nulls -> not exactly 0.5 due to ints? mean() on bool gives fraction
    df2 = pd.DataFrame({"value_bool": [True, False]})
    assert BooleanStatistics.is_balanced(df2, "value_bool")

    summary = BooleanStatistics.summary_as_dict(df2, "value_bool")
    assert set(
        [
            "true_count",
            "false_count",
            "true_percentage",
            "false_percentage",
            "mode",
            "is_balanced",
        ]
    ).issubset(summary.keys())
