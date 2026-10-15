import pandas as pd

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


def test_counts_return_python_int():
    df = pd.DataFrame({"value_bool": [True, False, None, True]})
    for fn in (
        BooleanStatistics.count_true,
        BooleanStatistics.count_false,
        BooleanStatistics.count_null,
        BooleanStatistics.count_not_null,
    ):
        assert type(fn(df)) is int
    assert BooleanStatistics.count_true(df) == 2
    assert BooleanStatistics.count_false(df) == 1
    assert BooleanStatistics.count_null(df) == 1
    assert BooleanStatistics.count_not_null(df) == 3


def test_is_balanced_is_exact_on_counts():
    balanced = pd.DataFrame({"b": [True, False, None, True, False]})
    assert BooleanStatistics.is_balanced(balanced, "b") is True
    unbalanced = pd.DataFrame({"b": [True, True, False]})
    assert BooleanStatistics.is_balanced(unbalanced, "b") is False
    empty = pd.DataFrame({"b": pd.Series([], dtype=bool)})
    assert BooleanStatistics.is_balanced(empty, "b") is False
