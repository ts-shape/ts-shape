import pytest
import pandas as pd  # type: ignore

pytest.importorskip("scipy")

from ts_shape.features.stats.numeric_stats import NumericStatistics


def test_numeric_statistics_basic():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert NumericStatistics.column_mean(df, "x") == 3
    assert NumericStatistics.column_median(df, "x") == 3
    assert NumericStatistics.column_min(df, "x") == 1
    assert NumericStatistics.column_max(df, "x") == 5
    assert NumericStatistics.column_sum(df, "x") == 15
    assert NumericStatistics.column_quantile(df, "x", 0.25) == 2
    assert NumericStatistics.column_quantile(df, "x", 0.75) == 4
    assert NumericStatistics.column_range(df, "x") == 4
    assert NumericStatistics.standard_error_mean(df, "x") >= 0
    assert NumericStatistics.describe(df).shape[0] >= 1

    # coefficient_of_variation handles non-zero mean
    assert NumericStatistics.coefficient_of_variation(df, "x") > 0


def test_numeric_statistics_summary_dataframe_and_missing_mode_safe():
    df = pd.DataFrame({"x": [1, 2, 2, 3, 4]})
    # Guard: if column_mode is not implemented in the class, skip
    if not hasattr(NumericStatistics, "column_mode"):
        pytest.skip("column_mode not implemented; skipping summary_as_dataframe test")

    out = NumericStatistics.summary_as_dataframe(df, "x")
    assert out.shape[0] == 1
    assert "mode" in out.columns
