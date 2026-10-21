"""Round 17 (PR J, part b): hand-computed references for features and filters.

F7  TimeWindowedFeatureTable: true [start, end) for right-labeled windows.
F8  CycleExtractor: a cycle running at the first non-missing sample is partial.
F9  CrossSignalAnalytics.granger_causality: Bonferroni over the lags.
F10 not_match filters keep missing values of nullable dtypes.
F11 NumericStatistics.column_sum / column_range do not wrap int64.
D2  TimeWindowedFeatureTable.compute: every window appears.
D3  TimezoneShift.revert_to_original_timezone: clear error on naive input.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.features.cross_signal import CrossSignalAnalytics
from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.features.segment_analysis.time_windowed_features import (
    TimeWindowedFeatureTable,
)
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.transform.filter.numeric_filter import IntegerFilter
from ts_shape.transform.filter.string_filter import StringFilter
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift

T = pd.Timestamp


# ---------------------------------------------------------------------------
# F7: window bounds of right-labeled frequencies
# ---------------------------------------------------------------------------


def _daily(first: str, last: str) -> pd.DataFrame:
    times = pd.date_range(first, last, freq="D")
    return pd.DataFrame({"systime": times, "uuid": "a", "value_double": 1.0})


def test_f7_month_end_window_bounds():
    # 2024-01-10..01-31 is 22 days, 02-01..02-20 is 20 days.
    out = TimeWindowedFeatureTable.compute_long(
        _daily("2024-01-10", "2024-02-20"),
        window="ME",
        segment_column=None,
        metrics=["sum"],
    )
    assert out["time_window"].tolist() == [T("2024-01-31"), T("2024-02-29")]
    assert out["start"].tolist() == [T("2024-01-01"), T("2024-02-01")]
    assert out["end"].tolist() == [T("2024-02-01"), T("2024-03-01")]
    assert out["sample_count"].tolist() == [22, 20]
    assert out["sum"].tolist() == [22.0, 20.0]
    assert out["duration_seconds"].tolist() == [31 * 86400.0, 29 * 86400.0]


def test_f7_weekly_window_bounds():
    # 'W' = weeks ending Sunday. 2024-01-03 (Wed) .. 2024-01-09 (Tue):
    # Wed..Sun 01-07 is 5 days in the week [01-01, 01-08), Mon 01-08 and
    # Tue 01-09 are 2 days in [01-08, 01-15).
    out = TimeWindowedFeatureTable.compute_long(
        _daily("2024-01-03", "2024-01-09"),
        window="W",
        segment_column=None,
        metrics=["sum"],
    )
    assert out["time_window"].tolist() == [T("2024-01-07"), T("2024-01-14")]
    assert out["start"].tolist() == [T("2024-01-01"), T("2024-01-08")]
    assert out["end"].tolist() == [T("2024-01-08"), T("2024-01-15")]
    assert out["sample_count"].tolist() == [5, 2]


def test_f7_left_labeled_window_unchanged():
    out = TimeWindowedFeatureTable.compute_long(
        _daily("2024-01-30", "2024-02-02"),
        window="MS",
        segment_column=None,
        metrics=["sum"],
    )
    assert out["start"].tolist() == out["time_window"].tolist()
    assert out["start"].tolist() == [T("2024-01-01"), T("2024-02-01")]
    assert out["end"].tolist() == [T("2024-02-01"), T("2024-03-01")]


# ---------------------------------------------------------------------------
# D2: compute() has one row per window
# ---------------------------------------------------------------------------


def test_d2_compute_keeps_a_window_without_enough_data():
    # Window 00:01 holds one sample (3.0): no group of 2, but the row stays.
    times = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:30",
            "2024-01-01 00:01:10",
            "2024-01-01 00:02:00",
            "2024-01-01 00:02:30",
        ]
    )
    df = pd.DataFrame(
        {"systime": times, "uuid": "a", "value_double": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    out = TimeWindowedFeatureTable.compute(
        df, window="1min", segment_column=None, metrics=["sum"]
    )
    assert out["time_window"].tolist() == [
        T("2024-01-01 00:00"),
        T("2024-01-01 00:01"),
        T("2024-01-01 00:02"),
    ]
    assert out["a__sum"].tolist()[0] == 3.0
    assert np.isnan(out["a__sum"].tolist()[1])
    assert out["a__sum"].tolist()[2] == 9.0


def test_d2_compute_without_any_group_lists_the_windows():
    times = pd.to_datetime(["2024-01-01 00:00:10", "2024-01-01 00:01:10"])
    df = pd.DataFrame({"systime": times, "uuid": "a", "value_double": [1.0, 2.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = TimeWindowedFeatureTable.compute(
            df, window="1min", segment_column=None, metrics=["sum"]
        )
    assert list(out.columns) == ["time_window"]
    assert out["time_window"].tolist() == [
        T("2024-01-01 00:00"),
        T("2024-01-01 00:01"),
    ]


# ---------------------------------------------------------------------------
# F8: a cycle running at the first non-missing sample is partial
# ---------------------------------------------------------------------------


def _signal(column: str, values: list) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=len(values), freq="min")
    return pd.DataFrame({"systime": times, "uuid": "c", column: values})


def test_f8_persistent_leading_missing_is_partial():
    # First known sample (00:01) is True: the cycle 00:01..00:03 began before.
    df = _signal("value_bool", [None, True, True, False, False])
    out = CycleExtractor(df, "c").process_persistent_cycle()
    assert out["cycle_start"].tolist() == [T("2024-01-01 00:01")]
    assert out["cycle_end"].tolist() == [T("2024-01-01 00:03")]
    assert out["is_complete"].tolist() == [False]


def test_f8_persistent_leading_missing_then_false_is_complete():
    df = _signal("value_bool", [None, False, True, False])
    out = CycleExtractor(df, "c").process_persistent_cycle()
    assert out["cycle_start"].tolist() == [T("2024-01-01 00:02")]
    assert out["is_complete"].tolist() == [True]


def test_f8_trigger_leading_missing_is_partial():
    # Triggers at 00:01 (already True when first seen) and 00:03.
    df = _signal("value_bool", [None, True, False, True, False])
    out = CycleExtractor(df, "c").process_trigger_cycle()
    assert out["cycle_start"].tolist() == [T("2024-01-01 00:01"), T("2024-01-01 00:03")]
    assert out["is_complete"].tolist() == [False, False]


def test_f8_separate_start_end_leading_missing_is_partial():
    df = _signal("value_bool", [None, True, False, True, False])
    out = CycleExtractor(df, "c").process_separate_start_end_cycle()
    assert out["cycle_start"].tolist() == [T("2024-01-01 00:01"), T("2024-01-01 00:03")]
    assert out["cycle_end"].tolist()[0] == T("2024-01-01 00:03")
    assert out["is_complete"].tolist() == [False, False]


def test_f8_step_sequence_leading_missing_is_partial():
    df = _signal("value_integer", [np.nan, 1, 2, 3, 1])
    out = CycleExtractor(df, "c").process_step_sequence(1, 3)
    assert out["cycle_start"].tolist() == [T("2024-01-01 00:01"), T("2024-01-01 00:04")]
    assert out["cycle_end"].tolist()[0] == T("2024-01-01 00:03")
    assert out["is_complete"].tolist() == [False, False]


# ---------------------------------------------------------------------------
# F9: Granger causality with Bonferroni over the lags
# ---------------------------------------------------------------------------


def test_f9_granger_reports_adjusted_p_value():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(120)
    y = rng.standard_normal(120)
    res = CrossSignalAnalytics(pd.DataFrame({"x": x, "y": y})).granger_causality(
        "x", "y", max_lag=4
    )
    assert len(res["results_by_lag"]) == 4
    assert res["p_value"] == min(r["p_value"] for r in res["results_by_lag"])
    assert res["p_value_adjusted"] == pytest.approx(min(1.0, res["p_value"] * 4))
    assert res["is_causal"] == (res["p_value_adjusted"] < 0.05)


def test_f9_granger_detects_a_real_lagged_cause():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(300)
    y = np.r_[0.0, 0.0, 0.8 * x[:-2]] + 0.3 * rng.standard_normal(300)
    res = CrossSignalAnalytics(pd.DataFrame({"x": x, "y": y})).granger_causality(
        "x", "y", max_lag=5
    )
    assert res["is_causal"] is True
    assert res["p_value_adjusted"] < 1e-6


def test_f9_granger_false_positive_rate_on_independent_noise():
    # Unadjusted best-of-10-lags: about 15% false positives; Bonferroni
    # keeps it at or below the 5% level.
    rng = np.random.default_rng(0)
    pairs = 200
    hits = 0
    for _ in range(pairs):
        frame = pd.DataFrame(
            {"x": rng.standard_normal(200), "y": rng.standard_normal(200)}
        )
        res = CrossSignalAnalytics(frame).granger_causality("x", "y", max_lag=10)
        hits += bool(res["is_causal"])
    assert hits / pairs <= 0.07


def test_f9_granger_too_few_samples_has_nan_adjusted():
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 1.0]})
    res = CrossSignalAnalytics(frame).granger_causality("x", "y")
    assert np.isnan(res["p_value_adjusted"])
    assert res["is_causal"] is False


# ---------------------------------------------------------------------------
# F10: not_match keeps missing values of nullable dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["Int64", "float64", "object"])
def test_f10_integer_not_match_keeps_missing(dtype):
    values = [0, 1, None] if dtype != "float64" else [0.0, 1.0, np.nan]
    df = pd.DataFrame({"value_integer": pd.array(values, dtype=dtype)})
    out = IntegerFilter.filter_value_integer_not_match(df, integer_value=0)
    assert out.index.tolist() == [1, 2]


def test_f10_integer_match_and_between_still_drop_missing():
    df = pd.DataFrame({"value_integer": pd.array([0, 1, None], dtype="Int64")})
    assert IntegerFilter.filter_value_integer_match(
        df, integer_value=0
    ).index.tolist() == [0]
    out = IntegerFilter.filter_value_integer_between(df, min_value=0, max_value=5)
    assert out.index.tolist() == [0, 1]


@pytest.mark.parametrize("dtype", ["string", "object"])
def test_f10_string_not_match_keeps_missing(dtype):
    df = pd.DataFrame({"value_string": pd.array(["a", "b", None], dtype=dtype)})
    out = StringFilter.filter_value_string_not_match(df, "a")
    assert out.index.tolist() == [1, 2]
    assert StringFilter.filter_value_string_match(df, "a").index.tolist() == [0]


# ---------------------------------------------------------------------------
# F11: integer sum and range do not wrap around
# ---------------------------------------------------------------------------


def test_f11_sum_beyond_int64_is_exact():
    df = pd.DataFrame({"x": [2**62, 2**62]})
    assert NumericStatistics.column_sum(df, "x") == 2**63


def test_f11_sum_exact_beyond_float_precision():
    # 2**62 + 1 cannot be represented in float64; the Python-int path keeps it.
    df = pd.DataFrame({"x": [2**62 + 1, 2**62]})
    assert NumericStatistics.column_sum(df, "x") == 2**63 + 1


def test_f11_small_sum_unchanged():
    df = pd.DataFrame({"x": [1, 2, 3]})
    total = NumericStatistics.column_sum(df, "x")
    assert total == 6
    assert isinstance(total, np.integer)


def test_f11_range_beyond_int64_is_exact():
    df = pd.DataFrame({"x": [2**62 + 1, -(2**62)]})
    assert NumericStatistics.column_range(df, "x") == 2**63 + 1


def test_f11_small_range_unchanged():
    df = pd.DataFrame({"x": [5, -3, 2]})
    assert NumericStatistics.column_range(df, "x") == 8
    empty = pd.DataFrame({"x": pd.Series([], dtype="int64")})
    assert np.isnan(NumericStatistics.column_range(empty, "x"))


# ---------------------------------------------------------------------------
# D3: revert_to_original_timezone on naive input
# ---------------------------------------------------------------------------


def test_d3_revert_naive_column_raises_clear_error():
    df = pd.DataFrame({"systime": pd.date_range("2024-01-01", periods=2, freq="h")})
    with pytest.raises(ValueError, match="shift_timezone"):
        TimezoneShift.revert_to_original_timezone(df, "systime", "Europe/Berlin")


def test_d3_revert_aware_column_converts():
    times = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    out = TimezoneShift.revert_to_original_timezone(
        pd.DataFrame({"systime": times}), "systime", "Europe/Berlin"
    )
    assert out["systime"].tolist() == [
        T("2024-01-01 01:00", tz="Europe/Berlin"),
        T("2024-01-01 02:00", tz="Europe/Berlin"),
    ]
