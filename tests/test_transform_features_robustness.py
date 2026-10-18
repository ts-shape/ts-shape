"""Regression tests: robustness and consistency of transform/** and features/**.

One test (or a few) per fixed defect: silent wrong results first, then
crashes on empty, untyped, nullable or pyarrow input, then missing columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.features.cross_signal import CrossSignalAnalytics
from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.features.export import FeatureMatrixExporter
from ts_shape.features.pattern_recognition import PatternRecognition
from ts_shape.features.segment_analysis.profile_comparison import ProfileComparison
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor
from ts_shape.features.segment_analysis.time_windowed_features import (
    TimeWindowedFeatureTable,
)
from ts_shape.features.stats.boolean_stats import BooleanStatistics
from ts_shape.features.stats.feature_table import DescriptiveFeatures
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.features.stats.string_stats import StringStatistics
from ts_shape.features.stats.timestamp_stats import TimestampStatistics
from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.transform.calculator.power_integration import PowerIntegrator
from ts_shape.transform.filter.boolean_filter import BooleanFilter
from ts_shape.transform.filter.custom_filter import CustomFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.filter.numeric_filter import DoubleFilter, IntegerFilter
from ts_shape.transform.filter.string_filter import StringFilter
from ts_shape.transform.functions.lambda_func import LambdaProcessor
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.transform.time_functions.timestamp_converter import TimestampConverter
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift

ARROW_TS = pd.ArrowDtype(pa.timestamp("ns"))


def _untyped(*columns: str) -> pd.DataFrame:
    """An empty frame whose columns have no dtype (as read from nothing)."""
    return pd.DataFrame({c: pd.Series([], dtype=object) for c in columns})


# ---------------------------------------------------------------------------
# 1. CrossSignalAnalytics
# ---------------------------------------------------------------------------


def _lagged(shift: int = 5, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).cumsum()
    b = np.r_[np.zeros(shift), a[:-shift]]  # b repeats a `shift` samples later
    t = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame({"systime": t, "a": a, "b": b})


def test_lead_lag_names_the_signal_that_moves_first():
    result = CrossSignalAnalytics(_lagged()).lead_lag("a", "b", max_lag=20)
    assert result["leader"] == "a"
    assert result["follower"] == "b"
    assert result["optimal_lag"] == 5
    reverse = CrossSignalAnalytics(_lagged()).lead_lag("b", "a", max_lag=20)
    assert reverse["leader"] == "a"
    assert reverse["optimal_lag"] == -5


def test_lead_lag_matrix_positive_means_row_leads():
    matrix = CrossSignalAnalytics(_lagged()).lead_lag_matrix(max_lag=20)
    assert matrix.loc["a", "b"] == 5
    assert matrix.loc["b", "a"] == -5


def test_cross_signal_sorts_by_time():
    df = _lagged()
    shuffled = df.sample(frac=1.0, random_state=1)
    assert CrossSignalAnalytics(shuffled).lead_lag(
        "a", "b", max_lag=20
    ) == CrossSignalAnalytics(df).lead_lag("a", "b", max_lag=20)


def test_cross_signal_aligns_rows_pairwise():
    df = _lagged()
    gapped = df.copy()
    gapped.loc[10:19, "a"] = np.nan  # dropping per column would shift b vs a
    expected = CrossSignalAnalytics(df.drop(index=range(10, 20))).lead_lag(
        "a", "b", max_lag=20
    )
    assert CrossSignalAnalytics(gapped).lead_lag("a", "b", max_lag=20) == expected


@pytest.mark.parametrize("dtype", ["Float64", "double[pyarrow]", "Int64"])
def test_cross_signal_accepts_nullable_and_pyarrow(dtype):
    df = _lagged()
    df["a"] = (df["a"] * 100).round()
    df["b"] = (df["b"] * 100).round()
    typed = df.astype({"a": dtype, "b": dtype})
    analytics = CrossSignalAnalytics(typed)
    assert analytics.signals == ["a", "b"]
    assert analytics.lead_lag("a", "b", max_lag=20) == CrossSignalAnalytics(
        df
    ).lead_lag("a", "b", max_lag=20)


def test_cross_signal_all_nan_gives_nan_not_an_error():
    df = pd.DataFrame({"a": [np.nan] * 5, "b": [np.nan] * 5})
    analytics = CrossSignalAnalytics(df)
    assert np.isnan(analytics.lead_lag("a", "b")["correlation_at_lag"])
    assert np.isnan(analytics.granger_causality("a", "b")["p_value"])
    assert np.isnan(analytics.transfer_entropy("a", "b"))
    assert np.isnan(analytics.synchronization_index("a", "b"))


def test_cross_signal_missing_column():
    with pytest.raises(ColumnNotFoundError):
        CrossSignalAnalytics(_lagged()).lead_lag("a", "nope")


# ---------------------------------------------------------------------------
# 2. PatternRecognition
# ---------------------------------------------------------------------------


def _pattern_frame(tz: str | None = None, n: int = 200) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="1s", tz=tz)
    return pd.DataFrame(
        {"systime": t, "uuid": "s", "value_double": np.sin(np.linspace(0, 20, n))}
    )


def test_pattern_recognition_sorts_by_time():
    df = _pattern_frame()
    expected = PatternRecognition.discover_motifs(df, window_size=20, top_k=2)
    shuffled = df.sample(frac=1.0, random_state=3)
    got = PatternRecognition.discover_motifs(shuffled, window_size=20, top_k=2)
    pd.testing.assert_frame_equal(got, expected)


def test_pattern_recognition_needs_uuid_for_several_signals():
    df = pd.concat([_pattern_frame(), _pattern_frame().assign(uuid="t")])
    with pytest.raises(ValueError, match="uuid="):
        PatternRecognition.discover_discords(df, window_size=20)
    out = PatternRecognition.discover_discords(df, window_size=20, signal_uuid="t")
    assert len(out) > 0


def test_pattern_recognition_keeps_timezone():
    df = _pattern_frame(tz="Europe/Berlin")
    out = PatternRecognition.discover_motifs(df, window_size=20, top_k=1)
    assert str(out["time_a"].dt.tz) == "Europe/Berlin"
    out = PatternRecognition.template_match(df, df["value_double"].to_numpy()[:20])
    assert str(out["start_time"].dt.tz) == "Europe/Berlin"


def test_normalize_false_uses_raw_distance():
    df = _pattern_frame()
    template = df["value_double"].to_numpy()[:20]
    shifted = template + 10.0  # same shape, other level
    z = PatternRecognition.compute_distance_profile(df, shifted, normalize=True)
    raw = PatternRecognition.compute_distance_profile(df, shifted, normalize=False)
    assert z[0] < 1e-6
    assert raw[0] == pytest.approx(10.0 * np.sqrt(20))
    assert PatternRecognition.template_match(
        df, shifted, threshold=1.0, normalize=False
    ).empty


def test_pattern_recognition_skips_nan_windows():
    df = _pattern_frame()
    df.loc[100, "value_double"] = np.nan
    profile = PatternRecognition.compute_distance_profile(
        df, df["value_double"].to_numpy()[:20]
    )
    assert np.isnan(profile[81:101]).all()
    assert not np.isnan(profile[:81]).any()
    discords = PatternRecognition.discover_discords(df, window_size=20, top_k=3)
    assert not discords["start_index"].between(81, 100).any()
    assert np.isfinite(discords["distance"]).all()


def test_pattern_recognition_short_signal_gives_empty_typed_frame():
    df = _pattern_frame(n=5)
    motifs = PatternRecognition.discover_motifs(df, window_size=20)
    assert motifs.empty
    assert list(motifs.columns) == [
        "motif_rank",
        "index_a",
        "index_b",
        "distance",
        "time_a",
        "time_b",
    ]
    assert motifs["time_a"].dtype == df["systime"].dtype
    assert PatternRecognition.discover_discords(df, window_size=20).empty
    profile = PatternRecognition.compute_distance_profile(df, np.ones(20), metric="dtw")
    assert len(profile) == 0


def test_template_match_empty_result_has_columns():
    df = _pattern_frame()
    out = PatternRecognition.template_match(df, np.ones(10), threshold=-1.0)
    assert out.empty
    assert list(out.columns) == ["start_index", "distance", "start_time", "end_time"]


# ---------------------------------------------------------------------------
# 3./4. Edge and change filters
# ---------------------------------------------------------------------------


def test_bool_edges_hold_state_over_missing():
    t = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame(
        {"systime": t, "uuid": "a", "value_bool": pd.Series([False, None, True])}
    )
    rising = BooleanFilter.filter_raising_value_bool(df)
    assert list(rising.index) == [2]
    df_fall = df.assign(value_bool=pd.Series([True, None, False]))
    assert list(BooleanFilter.filter_falling_value_bool(df_fall).index) == [2]


def test_bool_edges_in_time_order_per_uuid():
    t = pd.date_range("2024-01-01", periods=3, freq="1min")
    a = pd.DataFrame({"systime": t, "uuid": "a", "value_bool": [False, True, True]})
    b = pd.DataFrame({"systime": t, "uuid": "b", "value_bool": [True, True, False]})
    df = pd.concat([a, b], ignore_index=True).iloc[[5, 1, 3, 0, 4, 2]]
    rising = BooleanFilter.filter_raising_value_bool(df)
    assert list(zip(rising["uuid"], rising["systime"])) == [("a", t[1])]
    falling = BooleanFilter.filter_falling_value_bool(df)
    assert list(zip(falling["uuid"], falling["systime"])) == [("b", t[2])]


@pytest.mark.parametrize("dtype", ["boolean", "bool[pyarrow]"])
def test_bool_edges_nullable(dtype):
    t = pd.date_range("2024-01-01", periods=4, freq="1min")
    df = pd.DataFrame(
        {"systime": t, "value_bool": pd.array([False, None, True, False], dtype=dtype)}
    )
    assert list(BooleanFilter.filter_raising_value_bool(df).index) == [2]
    assert list(BooleanFilter.filter_falling_value_bool(df).index) == [3]


@pytest.mark.parametrize("dtype", [object, "string", "string[pyarrow]"])
def test_string_changes_na_safe(dtype):
    t = pd.date_range("2024-01-01", periods=6, freq="1min")
    values = [None, None, "x", "x", None, "y"]
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": "a",
            "value_string": pd.Series(values, dtype=dtype),
        }
    )
    out = StringFilter.detect_changes_in_string(df)
    # first row always; NaN -> NaN is no change; x -> NaN and NaN -> y are.
    assert list(out.index) == [0, 2, 4, 5]


def test_string_changes_in_time_order():
    t = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame({"systime": t, "value_string": ["x", "x", "y"]}).iloc[::-1]
    assert sorted(StringFilter.detect_changes_in_string(df).index) == [0, 2]


# ---------------------------------------------------------------------------
# 5. DataHarmonizer
# ---------------------------------------------------------------------------


def _harmonize_frame() -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=6, freq="1s")
    return pd.DataFrame(
        {
            "systime": list(t) + list(t[[0, 5]]),
            "uuid": ["a"] * 6 + ["b"] * 2,
            "value_double": [0.0, np.nan, np.nan, np.nan, 4.0, 5.0, 1.0, 2.0],
        }
    )


def test_fill_gaps_measures_between_valid_samples():
    h = DataHarmonizer(_harmonize_frame())
    # The NaN run of 'a' sits between valid samples 4 s apart: not filled at 3 s.
    out = h.fill_gaps(strategy="interpolate", max_gap="3s")
    assert out["a"].isna().sum() == 3
    out = h.fill_gaps(strategy="interpolate", max_gap="4s")
    assert out["a"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_pivot_keeps_signals_without_values():
    df = _harmonize_frame()
    df.loc[df["uuid"] == "b", "value_double"] = np.nan
    wide = DataHarmonizer(df).pivot_to_wide()
    assert list(wide.columns) == ["a", "b"]
    assert wide["b"].isna().all()


def test_align_asof_warns_on_unknown_uuid():
    with pytest.warns(DataQualityWarning, match="nope"):
        out = DataHarmonizer(_harmonize_frame()).align_asof("a", "nope")
    assert out["value_right"].isna().all()


def test_harmonizer_on_empty_untyped_and_pyarrow():
    h = DataHarmonizer(_untyped("systime", "uuid", "value_double"))
    assert h.resample_to_uniform("1s").empty
    assert h.merge_multi_signals(window="1s").empty
    assert h.fill_gaps(max_gap="1s").empty
    assert h.align_asof("a", "b").empty
    gaps = h.detect_gaps()
    assert gaps.empty
    assert str(gaps["start"].dtype).startswith("datetime64")
    assert gaps["duration_seconds"].dtype == "float64"

    df = _harmonize_frame().astype({"systime": ARROW_TS})
    h = DataHarmonizer(df)
    assert isinstance(h.resample_to_uniform("1s").index, pd.DatetimeIndex)
    assert len(h.align_asof("a", "b")) == 6


def test_detect_gaps_canonical_columns():
    gaps = DataHarmonizer(_harmonize_frame()).detect_gaps(threshold="2s")
    row = gaps.iloc[0]
    assert row["uuid"] == "b"
    assert row["start"] == row["gap_start"]
    assert row["end"] == row["gap_end"]
    assert row["duration_seconds"] == 5.0


def test_harmonizer_missing_value_column():
    df = _harmonize_frame()
    with pytest.raises(ColumnNotFoundError):
        DataHarmonizer(df, value_column="value_integer")


# ---------------------------------------------------------------------------
# 6. No mutation of the caller's frame
# ---------------------------------------------------------------------------


def test_lambda_processor_does_not_mutate():
    df = pd.DataFrame({"x": [1, 2]})
    out = LambdaProcessor.apply_function(df, "x", lambda v: v * 10)
    assert df["x"].tolist() == [1, 2]
    assert out["x"].tolist() == [10, 20]


def test_revert_timezone_does_not_mutate():
    t = pd.date_range("2024-01-01", periods=2, freq="1h", tz="Europe/Berlin")
    df = pd.DataFrame({"systime": t})
    out = TimezoneShift.revert_to_original_timezone(df, "systime", "UTC")
    assert str(df["systime"].dt.tz) == "Europe/Berlin"
    assert str(out["systime"].dt.tz) == "UTC"


# ---------------------------------------------------------------------------
# 7. TimeGroupedStatistics.calculate_custom_func per uuid
# ---------------------------------------------------------------------------


def test_custom_func_per_uuid():
    t = pd.date_range("2024-01-01", periods=4, freq="15min")
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "a", "value_double": [1.0] * 4}),
            pd.DataFrame({"systime": t, "uuid": "b", "value_double": [9.0] * 4}),
        ]
    )
    out = TimeGroupedStatistics.calculate_custom_func(
        df, "systime", "value_double", "1h", lambda s: s.max()
    )
    assert out.loc[("a", t[0]), "custom"] == 1.0
    assert out.loc[("b", t[0]), "custom"] == 9.0
    pooled = TimeGroupedStatistics.calculate_custom_func(
        df, "systime", "value_double", "1h", lambda s: s.max(), group_by_uuid=False
    )
    assert pooled["custom"].iloc[0] == 9.0


# ---------------------------------------------------------------------------
# 8. DescriptiveFeatures
# ---------------------------------------------------------------------------


def _long(n: int = 4) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="1min")
    flag = pd.DataFrame(
        {
            "systime": t,
            "uuid": "flag",
            "value_bool": [True, False, True, True][:n],
            "value_double": np.nan,
            "value_string": None,
        }
    )
    temp = pd.DataFrame(
        {
            "systime": t,
            "uuid": "temp",
            "value_bool": np.nan,
            "value_double": [1.0, 2.0, 3.0, 4.0][:n],
            "value_string": None,
        }
    )
    return pd.concat([flag, temp], ignore_index=True)


def test_descriptive_features_empty_frame():
    assert DescriptiveFeatures(_long().iloc[:0]).compute() == {}
    assert DescriptiveFeatures(_long().iloc[:0]).compute("dataframe").empty


def test_descriptive_features_stable_keys_without_is_delta():
    out = DescriptiveFeatures(_long()).compute()
    assert out["flag"].keys() == out["temp"].keys()
    assert set(out["flag"]) == {"overall", "value_bool", "value_double", "value_string"}
    assert np.isnan(out["flag"]["overall"]["is_delta_sum"])
    assert out["flag"]["value_bool"]["boolean_stats"]["true_count"] == 3


def test_descriptive_features_pyarrow_bool():
    df = _long()
    df = df[df["uuid"] == "flag"].astype({"value_bool": "bool[pyarrow]"})
    out = DescriptiveFeatures(df).compute()
    assert out["flag"]["value_bool"]["boolean_stats"]["true_count"] == 3


# ---------------------------------------------------------------------------
# 9. FeatureMatrixExporter
# ---------------------------------------------------------------------------


def test_feature_matrix_sample_std_and_value_columns_only():
    df = pd.DataFrame(
        {
            "uuid": ["a", "a", "a"],
            "value_double": [1.0, 2.0, 3.0],
            "is_delta": [1, 1, 1],
            "cycle_number": [1, 2, 3],
        }
    )
    matrix = FeatureMatrixExporter.to_feature_matrix(df)
    assert list(matrix.columns) == [
        "a__value_double__max",
        "a__value_double__mean",
        "a__value_double__min",
        "a__value_double__std",
    ]
    assert matrix["a__value_double__std"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. TimestampStatistics parses text timestamps
# ---------------------------------------------------------------------------


def test_timestamp_statistics_compares_times_not_text():
    df = pd.DataFrame(
        {"systime": ["2024-01-02 09:00", "2024-01-10 08:00", "2023-12-31 00:00"]}
    )
    assert TimestampStatistics.earliest_timestamp(df) == pd.Timestamp("2023-12-31")
    assert TimestampStatistics.latest_timestamp(df) == pd.Timestamp("2024-01-10 08:00")


# ---------------------------------------------------------------------------
# 11.-15. Crashes on empty, untyped, text, nullable and pyarrow input
# ---------------------------------------------------------------------------


def test_datetime_filter_on_empty_and_text_time():
    out = DateTimeFilter.filter_after_date(_untyped("systime"), date="2024-01-01")
    assert out.empty
    df = pd.DataFrame({"systime": ["2024-01-01 10:00", "2024-01-03 10:00"]})
    out = DateTimeFilter.filter_between_dates(
        df, start_date="2024-01-02", end_date="2024-01-04"
    )
    assert list(out.index) == [1]


def test_timezone_shift_on_empty_and_lowercase_utc():
    empty = _untyped("systime", "end")
    assert TimezoneShift.detect_timezone_awareness(empty, "systime") is False
    assert TimezoneShift.calculate_time_difference(empty, "systime", "end").empty
    assert TimezoneShift.revert_to_original_timezone(empty, "systime", "utc").empty
    shifted = TimezoneShift.shift_timezone(empty, "systime", "utc", "Europe/Berlin")
    assert str(shifted["systime"].dt.tz) == "Europe/Berlin"
    added = TimezoneShift.add_timezone_column(empty, "systime", "UTC", "utc")
    assert "systime_UTC" in added.columns


def test_string_filters_on_all_nan_float_column():
    df = pd.DataFrame({"value_string": [np.nan, np.nan]})
    assert StringFilter.filter_string_contains(df, "a").empty
    cleaned = StringFilter.regex_clean_value_string(df)
    assert cleaned["value_string"].isna().all()


def test_numeric_filters_parse_text_numbers():
    df = pd.DataFrame(
        {"value_integer": ["1", "5", "10"], "value_double": ["1.5", "x", "3"]}
    )
    assert list(
        IntegerFilter.filter_value_integer_between(df, "value_integer", 2, 10).index
    ) == [1, 2]
    assert list(
        IntegerFilter.filter_value_integer_match(df, "value_integer", 5).index
    ) == [1]
    with pytest.warns(DataQualityWarning):
        out = DoubleFilter.filter_value_double_between(df, "value_double", 1.0, 2.0)
    assert list(out.index) == [0]


def test_integer_calc_pyarrow_mod_and_missing_column():
    df = pd.DataFrame({"value_integer": pd.array([5, 7, None], dtype="int64[pyarrow]")})
    out = IntegerCalc.mod_column(df, "value_integer", 3)
    assert out["value_integer"].tolist()[:2] == [2, 1]
    assert out["value_integer"].dtype == df["value_integer"].dtype
    for name in (
        "scale_column",
        "offset_column",
        "divide_column",
        "subtract_column",
        "calculate_with_fixed_factors",
        "mod_column",
        "power_column",
    ):
        with pytest.raises(ColumnNotFoundError):
            getattr(IntegerCalc, name)(df, "nope")


def test_integer_calc_text_numbers_are_numbers():
    df = pd.DataFrame({"value_integer": ["1", "2"]})
    assert IntegerCalc.scale_column(df, "value_integer", 2)[
        "value_integer"
    ].tolist() == [2, 4]


def test_custom_filter_missing_column():
    with pytest.raises(ColumnNotFoundError):
        CustomFilter.filter_custom_conditions(pd.DataFrame({"a": [1]}), "b > 0")


def test_timestamp_converter_single_column_string():
    df = pd.DataFrame({"ts": [0, 60]})
    out = TimestampConverter.convert_to_datetime(df, columns="ts", unit="s")
    assert out["ts"].iloc[1] == pd.Timestamp("1970-01-01 00:01", tz="UTC")


def test_unit_converter_non_numeric_text_warns():
    pytest.importorskip("pint")
    from ts_shape.transform.calculator.unit_conversion import UnitConverter

    df = pd.DataFrame({"value_double": ["1", "n/a"]})
    with pytest.warns(DataQualityWarning):
        out = UnitConverter.convert_column(df, "bar", "kPa")
    assert out["value_double"].iloc[0] == pytest.approx(100.0)
    assert np.isnan(out["value_double"].iloc[1])


def test_power_integrator_pyarrow_and_data_end_tz():
    t = pd.date_range("2024-01-01", periods=4, freq="30min", tz="Europe/Berlin")
    df = pd.DataFrame({"systime": t, "uuid": "m", "value_double": 10.0})
    arrow = df.astype(
        {"systime": pd.ArrowDtype(pa.timestamp("ns", tz="Europe/Berlin"))}
    )
    out = PowerIntegrator.energy_by_window(arrow, data_end="2024-01-01 02:00")
    assert out["kwh"].tolist() == [10.0, 10.0]
    assert str(out["start"].dt.tz) == "Europe/Berlin"


def test_power_integrator_typed_empty_outputs():
    empty = _untyped("systime", "uuid", "value_double")
    windows = PowerIntegrator.energy_by_window(empty)
    assert str(windows["start"].dtype).startswith("datetime64")
    assert windows["kwh"].dtype == "float64"
    deltas = PowerIntegrator.to_energy_deltas(empty)
    assert str(deltas["systime"].dtype).startswith("datetime64")
    segments = PowerIntegrator.segment_energy(empty)
    assert segments["kwh"].dtype == "float64"


def test_time_grouped_statistics_on_empty_and_text_time():
    empty = _untyped("systime", "uuid", "value_double")
    assert TimeGroupedStatistics.calculate_statistic(
        empty, "systime", "value_double", "1h", "mean"
    ).empty
    df = pd.DataFrame(
        {
            "systime": ["2024-01-01 00:10", "2024-01-01 00:20"],
            "value_double": [1.0, 3.0],
        }
    )
    out = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "1h", "mean"
    )
    assert out["mean"].iloc[0] == 2.0


def test_numeric_statistics_empty_untyped_and_pyarrow():
    empty = _untyped("value_double")
    assert np.isnan(NumericStatistics.column_iqr(empty, "value_double"))
    assert np.isnan(NumericStatistics.summary_as_dict(empty, "value_double")["iqr"])
    df = pd.DataFrame(
        {"v": pd.array([1.0, 2.0, 3.0, 9.0, None], dtype="double[pyarrow]")}
    )
    expected = pd.Series([1.0, 2.0, 3.0, 9.0]).kurt()
    assert NumericStatistics.column_kurtosis(df, "v") == pytest.approx(expected)


def test_time_windowed_features_pyarrow_and_typed_empty():
    t = pd.date_range("2024-01-01", periods=6, freq="20s", tz="UTC")
    df = pd.DataFrame({"systime": t, "uuid": "a", "value_double": np.arange(6.0)})
    arrow = df.astype({"systime": pd.ArrowDtype(pa.timestamp("ns", tz="UTC"))})
    out = TimeWindowedFeatureTable.compute_long(arrow, segment_column=None)
    assert len(out) == 2
    assert str(out["time_window"].dt.tz) == "UTC"
    empty = TimeWindowedFeatureTable.compute_long(
        _untyped("systime", "uuid", "value_double"), segment_column=None
    )
    assert str(empty["time_window"].dtype).startswith("datetime64")
    assert str(empty["start"].dtype).startswith("datetime64")
    assert empty["mean"].dtype == "float64"


def test_boolean_and_timestamp_statistics_on_empty():
    empty = _untyped("value_bool", "systime")
    assert BooleanStatistics.mode(empty, "value_bool") is None
    assert BooleanStatistics.summary_as_dict(empty, "value_bool")["mode"] is None
    assert np.isnan(BooleanStatistics.true_percentage(empty))
    assert np.isnan(BooleanStatistics.false_percentage(empty))
    assert TimestampStatistics.most_frequent_timestamp(empty) is None
    assert TimestampStatistics.most_frequent_day(empty) is None
    assert TimestampStatistics.most_frequent_hour(empty) is None
    assert TimestampStatistics.count_most_frequent_timestamp(empty) == 0
    assert TimestampStatistics.hour_distribution(empty).empty
    nat = pd.DataFrame({"systime": pd.Series([pd.NaT, pd.NaT])})
    assert TimestampStatistics.most_frequent_timestamp(nat) is None


def test_string_statistics_percentages_nan_without_values():
    df = pd.DataFrame({"value_string": [None, None]})
    assert np.isnan(StringStatistics.uppercase_percentage(df))
    assert np.isnan(StringStatistics.lowercase_percentage(df))


# ---------------------------------------------------------------------------
# 16. ProfileComparison group keys
# ---------------------------------------------------------------------------


def _profiles(group_column: str) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=40, freq="1min")
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": ["p"] * 20 + ["q"] * 20,
            "value_double": np.r_[np.arange(20.0), np.arange(20.0) * 2],
            "segment_value": (["A"] * 10 + ["B"] * 10) * 2,
            "segment_index": ([0] * 10 + [1] * 10) * 2,
        }
    )
    return SegmentProcessor.compute_metric_profiles(df, group_column=group_column)


def test_profile_comparison_detects_the_profile_group_key():
    by_value = _profiles("segment_value")
    changes = ProfileComparison.detect_changes(by_value)
    assert "segment_value" in changes.columns
    pairs = ProfileComparison.find_similar_pairs(by_value)
    assert set(pairs["group_a"]) <= {"A", "B"}
    by_index = _profiles("segment_index")
    assert "segment_index" in ProfileComparison.detect_changes(by_index).columns


def test_profile_comparison_without_segment_column():
    profiles = _profiles("segment_value").drop(columns="segment_value")
    with pytest.raises(ColumnNotFoundError, match="segment"):
        ProfileComparison.detect_changes(profiles)


def test_distance_matrix_on_empty_profiles():
    empty = _profiles("segment_value").iloc[:0]
    assert ProfileComparison.compute_distance_matrix(empty).empty


# ---------------------------------------------------------------------------
# 17. CycleExtractor columns, detect_gaps typed (above)
# ---------------------------------------------------------------------------


def test_cycle_extractor_requires_time_column():
    df = pd.DataFrame({"ts": pd.date_range("2024", periods=3), "value_bool": True})
    with pytest.raises(ColumnNotFoundError):
        CycleExtractor(df, "s")


def test_cycle_extractor_custom_column_names():
    t = pd.date_range("2024-01-01", periods=4, freq="1min")
    df = pd.DataFrame(
        {"ts": t, "signal": "s", "value_bool": [False, True, False, True]}
    )
    out = CycleExtractor(
        df, "s", time_column="ts", uuid_column="signal"
    ).process_persistent_cycle()
    assert out["cycle_start"].tolist() == [t[1], t[3]]
    assert out["cycle_end"].iloc[0] == t[2]


# ---------------------------------------------------------------------------
# 18. Missing columns raise ColumnNotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda df: NumericStatistics.column_mean(df, "nope"),
        lambda df: BooleanStatistics.count_true(df, "nope"),
        lambda df: StringStatistics.count_unique(df, "nope"),
        lambda df: TimestampStatistics.earliest_timestamp(df, "nope"),
        lambda df: StringFilter.filter_na_value_string(df, "nope"),
        lambda df: DateTimeFilter.filter_after_date(df, "2024", time_column="nope"),
        lambda df: TimeGroupedStatistics.calculate_statistic(
            df, "systime", "nope", "1h", "mean"
        ),
        lambda df: PatternRecognition.discover_motifs(df, value_column="nope"),
    ],
)
def test_missing_column_raises_column_not_found(call):
    df = pd.DataFrame(
        {"systime": pd.date_range("2024", periods=3), "value_double": [1.0, 2, 3]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ColumnNotFoundError):
            call(df)
