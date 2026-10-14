"""Renamed parameters in transform/** and features/**: old names still work.

Each old keyword is a deprecated alias of the canonical name: it warns
(``DeprecationWarning``) and gives the same result.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from ts_shape.features.cycles.cycle_processor import CycleDataProcessor
from ts_shape.features.export import FeatureMatrixExporter
from ts_shape.features.stats.boolean_stats import BooleanStatistics
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.features.stats.string_stats import StringStatistics
from ts_shape.features.stats.timestamp_stats import TimestampStatistics
from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.transform.calculator.power_integration import PowerIntegrator
from ts_shape.transform.filter.boolean_filter import BooleanFilter, IsDeltaFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.filter.numeric_filter import DoubleFilter, IntegerFilter
from ts_shape.transform.filter.string_filter import StringFilter
from ts_shape.transform.functions.lambda_func import LambdaProcessor
from ts_shape.transform.harmonization import DataHarmonizer


def _frame() -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=12, freq="10min")
    return pd.DataFrame(
        {
            "systime": t,
            "uuid": "s",
            "value_double": np.arange(12.0),
            "value_integer": np.arange(12) % 3,
            "value_bool": np.arange(12) % 2 == 0,
            "value_string": ["a", "B"] * 6,
            "is_delta": True,
            "cycle": ["c0"] * 6 + ["c1"] * 6,
        }
    )


CASES = [
    (IsDeltaFilter.filter_is_delta_true, {}, "column_name", "value_column", "is_delta"),
    (
        BooleanFilter.filter_raising_value_bool,
        {},
        "column_name",
        "value_column",
        "value_bool",
    ),
    (
        IntegerFilter.filter_value_integer_match,
        {"integer_value": 1},
        "column_name",
        "value_column",
        "value_integer",
    ),
    (
        DoubleFilter.filter_value_double_between,
        {"max_value": 5},
        "column_name",
        "value_column",
        "value_double",
    ),
    (
        StringFilter.filter_string_contains,
        {"substring": "a"},
        "column_name",
        "value_column",
        "value_string",
    ),
    (
        IntegerCalc.scale_column,
        {"factor": 2},
        "column_name",
        "value_column",
        "value_integer",
    ),
    (
        LambdaProcessor.apply_function,
        {"func": lambda v: v + 1},
        "column_name",
        "value_column",
        "value_double",
    ),
    (NumericStatistics.column_mean, {}, "column_name", "value_column", "value_double"),
    (
        BooleanStatistics.true_percentage,
        {},
        "column_name",
        "value_column",
        "value_bool",
    ),
    (
        StringStatistics.uppercase_percentage,
        {},
        "column_name",
        "value_column",
        "value_string",
    ),
    (TimestampStatistics.latest_timestamp, {}, "column_name", "time_column", "systime"),
    (
        DateTimeFilter.filter_after_date,
        {"date": "2024-01-01 01:00"},
        "column_name",
        "time_column",
        "systime",
    ),
    (
        TimeGroupedStatistics.calculate_statistic,
        {"stat_method": "max"},
        "freq",
        "window",
        "1h",
    ),
    (PowerIntegrator.energy_by_window, {}, "freq", "window", "30min"),
    (FeatureMatrixExporter.to_feature_matrix, {}, "uuid_col", "uuid_column", "uuid"),
    (
        FeatureMatrixExporter.to_feature_matrix,
        {},
        "value_cols",
        "value_columns",
        ["value_double"],
    ),
    (FeatureMatrixExporter.to_feature_matrix, {}, "group_col", "group_column", "cycle"),
]


def _same(a, b) -> None:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    else:
        assert a == b or (pd.isna(a) and pd.isna(b))


@pytest.mark.parametrize("func, kwargs, old, new, value", CASES)
def test_old_keyword_warns_and_matches(func, kwargs, old, new, value):
    df = _frame()
    expected = func(df, **{new: value}, **kwargs)
    with pytest.warns(DeprecationWarning, match=old):
        got = func(df, **{old: value}, **kwargs)
    _same(got, expected)
    params = inspect.signature(func).parameters
    assert new in params and old not in params


def test_dataframe_keyword_rename():
    df = _frame()
    with pytest.warns(DeprecationWarning, match="'df'"):
        got = FeatureMatrixExporter.to_feature_matrix(df=df)
    pd.testing.assert_frame_equal(got, FeatureMatrixExporter.to_feature_matrix(df))


def test_harmonizer_window_rename():
    h = DataHarmonizer(_frame())
    with pytest.warns(DeprecationWarning, match="freq"):
        got = h.resample_to_uniform(freq="30min")
    pd.testing.assert_frame_equal(got, h.resample_to_uniform(window="30min"))


def test_cycle_processor_renames():
    df = _frame()
    cycles = pd.DataFrame(
        {
            "cycle_start": df["systime"].iloc[[0, 6]].to_numpy(),
            "cycle_end": df["systime"].iloc[[5, 11]].to_numpy(),
            "id": ["c0", "c1"],
        }
    )
    with pytest.warns(DeprecationWarning):
        old = CycleDataProcessor(cycles, df, cycle_uuid_col="id", systime_col="systime")
    new = CycleDataProcessor(cycles, df, cycle_uuid_column="id", time_column="systime")
    with pytest.warns(DeprecationWarning, match="metric"):
        got = old.compare_cycles("c0", metric="value_double")
    pd.testing.assert_frame_equal(
        got, new.compare_cycles("c0", value_column="value_double")
    )
    with pytest.warns(DeprecationWarning, match="metric"):
        assert old.identify_golden_cycles(metric="value_double") == (
            new.identify_golden_cycles(value_column="value_double")
        )


def test_datetime_filter_old_argument_order():
    df = _frame()
    expected = DateTimeFilter.filter_between_dates(
        df, "2024-01-01 00:30", "2024-01-01 01:00"
    )
    with pytest.warns(DeprecationWarning, match="time column before the bounds"):
        got = DateTimeFilter.filter_between_dates(
            df, "systime", "2024-01-01 00:30", "2024-01-01 01:00"
        )
    pd.testing.assert_frame_equal(got, expected)
    with pytest.warns(DeprecationWarning):
        got = DateTimeFilter.filter_after_date(df, "systime", date="2024-01-01 01:00")
    pd.testing.assert_frame_equal(
        got, DateTimeFilter.filter_after_date(df, "2024-01-01 01:00")
    )


def test_datetime_filter_date_is_required():
    with pytest.raises(TypeError):
        DateTimeFilter.filter_after_date(_frame())


def test_time_grouped_statistics_defaults():
    df = _frame()
    out = TimeGroupedStatistics.calculate_statistic(df)
    assert list(out.columns) == ["mean"]
    expected = TimeGroupedStatistics.calculate_statistic(
        df, "systime", "value_double", "1h", "mean"
    )
    pd.testing.assert_frame_equal(out, expected)
    assert list(TimeGroupedStatistics.calculate_statistics(df).columns) == [
        "mean",
        "min",
        "max",
    ]
    with pytest.raises(TypeError, match="func"):
        TimeGroupedStatistics.calculate_custom_func(df)
