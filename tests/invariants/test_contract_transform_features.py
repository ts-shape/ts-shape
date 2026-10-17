"""Contract test for every public callable in ``ts_shape.transform`` and
``ts_shape.features``.

Each callable has an entry in :data:`REGISTRY` with a minimal call on a
small long-format frame (one signal ``"s"``). Every entry is checked for:

``empty``
    The standard frame with zero rows does not raise, and time / numeric
    result columns keep their dtype (``start`` is datetime, ``kwh`` is
    float, ...).
``missing``
    Dropping a column the call needs raises
    :class:`~ts_shape.errors.ColumnNotFoundError`.
``tz``
    With tz-aware (Europe/Berlin) input, every datetime in the result is
    tz-aware.
``no_mutation``
    The input frame is unchanged after the call.
``nullable``
    ``Float64`` / ``Int64`` / ``boolean`` value columns give the same result
    as NumPy columns.

:func:`test_registry_covers_every_public_callable` fails when a public
callable of the two packages has no entry (or no listed exemption).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ts_shape.features
import ts_shape.transform
from ts_shape.errors import ColumnNotFoundError
from ts_shape.features.cross_signal import CrossSignalAnalytics
from ts_shape.features.cycles.cycle_processor import CycleDataProcessor
from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.features.export import FeatureMatrixExporter
from ts_shape.features.pattern_recognition import PatternRecognition
from ts_shape.features.segment_analysis.profile_comparison import ProfileComparison
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
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
from ts_shape.transform.calculator.unit_conversion import (
    PINT_AVAILABLE,
    UnitConverter,
)
from ts_shape.transform.filter.boolean_filter import BooleanFilter, IsDeltaFilter
from ts_shape.transform.filter.custom_filter import CustomFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.filter.numeric_filter import DoubleFilter, IntegerFilter
from ts_shape.transform.filter.string_filter import StringFilter
from ts_shape.transform.functions.lambda_func import LambdaProcessor
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.transform.time_functions.timestamp_converter import TimestampConverter
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift
from ts_shape.utils.frames import localize_like

BERLIN = "Europe/Berlin"
N = 48

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------


def base_frame(tz: str | None = None) -> pd.DataFrame:
    """One signal, every value column filled, one missing ``value_double``."""
    i = np.arange(N)
    double = np.sin(i / 3.0) + (i % 5) * 0.1
    double[7] = np.nan
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=N, freq="1min", tz=tz),
            "uuid": "s",
            "value_double": double,
            "value_integer": ((i // 4) % 3).astype("int64"),
            "value_bool": ((i // 3) % 2 == 0),
            "value_string": np.where(i < N // 2, "order1", "order2").astype(object),
            "is_delta": True,
        }
    )


def nullable(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {"value_double": "Float64", "value_integer": "Int64", "value_bool": "boolean"}
    )


def two_signals(df: pd.DataFrame) -> pd.DataFrame:
    other = df.assign(uuid="t", value_double=df["value_double"] * 2 + 1)
    return pd.concat([df, other], ignore_index=True)


def three_signals(df: pd.DataFrame) -> pd.DataFrame:
    third = df.assign(uuid="u", value_double=df["value_double"] * -1)
    return pd.concat([two_signals(df), third], ignore_index=True)


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """Wide frame for CrossSignalAnalytics: three numeric signals."""
    return pd.DataFrame(
        {
            "systime": df["systime"],
            "a": df["value_double"],
            "b": df["value_integer"],
            "c": df["value_double"] * 2,
        }
    )


def with_segments(df: pd.DataFrame) -> pd.DataFrame:
    seg = df["value_string"].astype(object)
    return df.assign(
        segment_value=seg,
        segment_index=(seg != seg.shift()).cumsum().astype("int64") - 1,
    )


def profiles(df: pd.DataFrame) -> pd.DataFrame:
    return SegmentProcessor.compute_metric_profiles(
        with_segments(three_signals(df)), group_column="segment_index"
    )


def distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return ProfileComparison.compute_distance_matrix(profiles(df), group_column="uuid")


def aware(df: pd.DataFrame) -> pd.DataFrame:
    times = df["systime"]
    if times.dt.tz is None:
        times = times.dt.tz_localize("UTC")
    return df.assign(systime=times)


def cycles_for(df: pd.DataFrame) -> pd.DataFrame:
    times = df.get("systime", pd.Series(dtype="datetime64[ns]"))
    starts = [
        localize_like(t, times)
        for t in ("2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30")
    ]
    return pd.DataFrame(
        {
            "cycle_start": starts,
            "cycle_end": [s + pd.Timedelta("15min") for s in starts],
            "cycle_uuid": ["c0", "c1", "c2"],
        }
    )


QUERY = np.sin(np.linspace(0, 3, 6))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """How to call one public callable.

    ``make`` turns the long frame into the call's input; ``required`` names
    columns of that input whose absence must raise ColumnNotFoundError;
    ``skip`` maps check name -> reason; ``typed=False`` skips the dtype check
    of an empty result whose columns are not time/number outputs.
    """

    call: Callable[[Any], Any]
    make: Callable[[pd.DataFrame], Any] = lambda df: df
    required: tuple[str, ...] = ()
    skip: dict[str, str] = field(default_factory=dict)
    typed: bool = True


def _int(method: str, **kwargs: Any) -> Case:
    return Case(
        lambda df: getattr(IntegerCalc, method)(df, "value_integer", **kwargs),
        required=("value_integer",),
    )


def _stat(cls: type, column: str, method: str, *args: Any, **kwargs: Any) -> Case:
    return Case(
        lambda df: (
            getattr(cls, method)(df, *args, column, **kwargs)
            if args
            else getattr(cls, method)(df, column, **kwargs)
        ),
        required=(column,),
    )


def _harm(method: str, *args: Any, **kwargs: Any) -> Case:
    return Case(
        lambda df: getattr(DataHarmonizer(df), method)(*args, **kwargs),
        make=two_signals,
        required=("systime", "uuid", "value_double"),
    )


def _cross(method: str, *args: Any, pair: bool = True, **kwargs: Any) -> Case:
    return Case(
        lambda df: getattr(CrossSignalAnalytics(df), method)(*args, **kwargs),
        make=wide,
        required=("b",) if pair else (),
    )


def _cycles(method: str, *args: Any, **kwargs: Any) -> Case:
    return Case(
        lambda df: getattr(CycleDataProcessor(cycles_for(df), df), method)(
            *args, **kwargs
        ),
        required=("systime",),
    )


def _extract(method: str, *args: Any, **kwargs: Any) -> Case:
    return Case(
        lambda df: getattr(CycleExtractor(df, "s"), method)(*args, **kwargs),
        required=("systime",),
    )


def _on_cycles(method: str, **kwargs: Any) -> Case:
    def call(df: pd.DataFrame) -> Any:
        ex = CycleExtractor(df, "s")
        return getattr(ex, method)(ex.process_persistent_cycle(), **kwargs)

    return Case(call, required=("systime",))


TIME = ("systime",)
SIGNAL = ("systime", "uuid", "value_double")

REGISTRY: dict[str, Case] = {
    # -- transform.calculator -------------------------------------------------
    "IntegerCalc.scale_column": _int("scale_column", factor=2),
    "IntegerCalc.offset_column": _int("offset_column", offset_value=1),
    "IntegerCalc.divide_column": _int("divide_column", divisor=2),
    "IntegerCalc.subtract_column": _int("subtract_column", subtract_value=1),
    "IntegerCalc.calculate_with_fixed_factors": _int(
        "calculate_with_fixed_factors", multiply_factor=2, add_factor=1
    ),
    "IntegerCalc.mod_column": _int("mod_column", mod_value=2),
    "IntegerCalc.power_column": _int("power_column", power_value=2),
    "PowerIntegrator.segment_energy": Case(
        lambda df: PowerIntegrator.segment_energy(df, window="10min"),
        required=SIGNAL,
    ),
    "PowerIntegrator.energy_by_window": Case(
        lambda df: PowerIntegrator.energy_by_window(df, window="10min", max_gap="5min"),
        required=SIGNAL,
    ),
    "PowerIntegrator.to_energy_deltas": Case(
        lambda df: PowerIntegrator.to_energy_deltas(df, window="10min"),
        required=SIGNAL,
    ),
    "UnitConverter.convert_column": Case(
        lambda df: UnitConverter.convert_column(df, "bar", "kPa"),
        required=("value_double",),
        skip={} if PINT_AVAILABLE else {"*": "pint is not installed"},
    ),
    # -- transform.filter -----------------------------------------------------
    "IsDeltaFilter.filter_is_delta_true": Case(
        IsDeltaFilter.filter_is_delta_true, required=("is_delta",)
    ),
    "IsDeltaFilter.filter_is_delta_false": Case(
        IsDeltaFilter.filter_is_delta_false, required=("is_delta",)
    ),
    "BooleanFilter.filter_falling_value_bool": Case(
        BooleanFilter.filter_falling_value_bool, required=("value_bool",)
    ),
    "BooleanFilter.filter_raising_value_bool": Case(
        BooleanFilter.filter_raising_value_bool, required=("value_bool",)
    ),
    "CustomFilter.filter_custom_conditions": Case(
        lambda df: CustomFilter.filter_custom_conditions(df, "value_double > 0"),
        required=("value_double",),
    ),
    "DateTimeFilter.filter_after_date": Case(
        lambda df: DateTimeFilter.filter_after_date(df, "2024-01-01"), required=TIME
    ),
    "DateTimeFilter.filter_before_date": Case(
        lambda df: DateTimeFilter.filter_before_date(df, "2024-01-02"),
        required=TIME,
    ),
    "DateTimeFilter.filter_between_dates": Case(
        lambda df: DateTimeFilter.filter_between_dates(df, "2023-12-31", "2024-01-02"),
        required=TIME,
    ),
    "DateTimeFilter.filter_after_datetime": Case(
        lambda df: DateTimeFilter.filter_after_datetime(df, "2024-01-01 00:20"),
        required=TIME,
    ),
    "DateTimeFilter.filter_before_datetime": Case(
        lambda df: DateTimeFilter.filter_before_datetime(df, "2024-01-01 00:20"),
        required=TIME,
    ),
    "DateTimeFilter.filter_between_datetimes": Case(
        lambda df: DateTimeFilter.filter_between_datetimes(
            df, "2024-01-01 00:10", "2024-01-01 00:30"
        ),
        required=TIME,
    ),
    "IntegerFilter.filter_value_integer_match": Case(
        lambda df: IntegerFilter.filter_value_integer_match(df, integer_value=1),
        required=("value_integer",),
    ),
    "IntegerFilter.filter_value_integer_not_match": Case(
        lambda df: IntegerFilter.filter_value_integer_not_match(df, integer_value=1),
        required=("value_integer",),
    ),
    "IntegerFilter.filter_value_integer_between": Case(
        lambda df: IntegerFilter.filter_value_integer_between(
            df, min_value=1, max_value=2
        ),
        required=("value_integer",),
    ),
    "DoubleFilter.filter_nan_value_double": Case(
        DoubleFilter.filter_nan_value_double, required=("value_double",)
    ),
    "DoubleFilter.filter_value_double_between": Case(
        lambda df: DoubleFilter.filter_value_double_between(
            df, min_value=-0.5, max_value=0.5
        ),
        required=("value_double",),
    ),
    "StringFilter.filter_na_value_string": Case(
        StringFilter.filter_na_value_string, required=("value_string",)
    ),
    "StringFilter.filter_value_string_match": Case(
        lambda df: StringFilter.filter_value_string_match(df, "order1"),
        required=("value_string",),
    ),
    "StringFilter.filter_value_string_not_match": Case(
        lambda df: StringFilter.filter_value_string_not_match(df, "order1"),
        required=("value_string",),
    ),
    "StringFilter.filter_string_contains": Case(
        lambda df: StringFilter.filter_string_contains(df, "2"),
        required=("value_string",),
    ),
    "StringFilter.regex_clean_value_string": Case(
        StringFilter.regex_clean_value_string, required=("value_string",)
    ),
    "StringFilter.detect_changes_in_string": Case(
        StringFilter.detect_changes_in_string, required=("value_string",)
    ),
    # -- transform.functions / time_functions ---------------------------------
    "LambdaProcessor.apply_function": Case(
        lambda df: LambdaProcessor.apply_function(df, "value_double", lambda v: v * 2),
        required=("value_double",),
    ),
    "TimestampConverter.convert_to_datetime": Case(
        lambda df: TimestampConverter.convert_to_datetime(df, "epoch", unit="s"),
        make=lambda df: df.assign(epoch=np.arange(len(df), dtype="int64") * 60),
        required=("epoch",),
    ),
    "TimezoneShift.shift_timezone": Case(
        lambda df: TimezoneShift.shift_timezone(df, "systime", "UTC", BERLIN),
        required=TIME,
    ),
    "TimezoneShift.add_timezone_column": Case(
        lambda df: TimezoneShift.add_timezone_column(df, "systime", "UTC", BERLIN),
        required=TIME,
    ),
    "TimezoneShift.detect_timezone_awareness": Case(
        lambda df: TimezoneShift.detect_timezone_awareness(df, "systime"),
        required=TIME,
    ),
    "TimezoneShift.revert_to_original_timezone": Case(
        lambda df: TimezoneShift.revert_to_original_timezone(df, "systime", "UTC"),
        make=aware,
        required=TIME,
    ),
    "TimezoneShift.calculate_time_difference": Case(
        lambda df: TimezoneShift.calculate_time_difference(df, "systime", "end"),
        make=lambda df: df.assign(end=df["systime"] + pd.Timedelta("1h")),
        required=("end",),
    ),
    # -- transform.harmonization ----------------------------------------------
    "DataHarmonizer.pivot_to_wide": _harm("pivot_to_wide"),
    "DataHarmonizer.resample_to_uniform": _harm("resample_to_uniform", "5min"),
    "DataHarmonizer.detect_gaps": _harm("detect_gaps", "30s"),
    "DataHarmonizer.fill_gaps": _harm("fill_gaps", max_gap="5min"),
    "DataHarmonizer.align_asof": _harm("align_asof", "s", "t", tolerance="1min"),
    "DataHarmonizer.merge_multi_signals": _harm("merge_multi_signals", window="5min"),
    # -- features.cross_signal ------------------------------------------------
    "CrossSignalAnalytics.granger_causality": _cross(
        "granger_causality", "a", "b", max_lag=2
    ),
    "CrossSignalAnalytics.transfer_entropy": _cross("transfer_entropy", "a", "b"),
    "CrossSignalAnalytics.pairwise_transfer_entropy": _cross(
        "pairwise_transfer_entropy", bins=4, pair=False
    ),
    "CrossSignalAnalytics.synchronization_index": _cross(
        "synchronization_index", "a", "b"
    ),
    "CrossSignalAnalytics.pairwise_synchronization": _cross(
        "pairwise_synchronization", pair=False
    ),
    "CrossSignalAnalytics.lead_lag": _cross("lead_lag", "a", "b", max_lag=5),
    "CrossSignalAnalytics.lead_lag_matrix": _cross(
        "lead_lag_matrix", max_lag=5, pair=False
    ),
    # -- features.cycles ------------------------------------------------------
    "CycleDataProcessor.split_by_cycle": _cycles("split_by_cycle"),
    "CycleDataProcessor.merge_dataframes_by_cycle": _cycles(
        "merge_dataframes_by_cycle"
    ),
    "CycleDataProcessor.group_by_cycle_uuid": _cycles("group_by_cycle_uuid"),
    "CycleDataProcessor.split_dataframes_by_group": Case(
        lambda df: CycleDataProcessor(cycles_for(df), df).split_dataframes_by_group(
            [df], "uuid"
        ),
        required=TIME,
    ),
    "CycleDataProcessor.compute_cycle_statistics": _cycles("compute_cycle_statistics"),
    "CycleDataProcessor.compare_cycles": _cycles("compare_cycles", "c0"),
    "CycleDataProcessor.identify_golden_cycles": _cycles("identify_golden_cycles"),
    "CycleExtractor.process_persistent_cycle": _extract("process_persistent_cycle"),
    "CycleExtractor.process_trigger_cycle": _extract("process_trigger_cycle"),
    "CycleExtractor.process_separate_start_end_cycle": _extract(
        "process_separate_start_end_cycle"
    ),
    "CycleExtractor.process_step_sequence": _extract("process_step_sequence", 1, 2),
    "CycleExtractor.process_state_change_cycle": _extract("process_state_change_cycle"),
    "CycleExtractor.process_value_change_cycle": _extract("process_value_change_cycle"),
    "CycleExtractor.validate_cycles": _on_cycles("validate_cycles"),
    "CycleExtractor.detect_overlapping_cycles": _on_cycles("detect_overlapping_cycles"),
    "CycleExtractor.suggest_method": _extract("suggest_method"),
    "CycleExtractor.get_extraction_stats": _extract("get_extraction_stats"),
    "CycleExtractor.reset_stats": _extract("reset_stats"),
    # -- features.export ------------------------------------------------------
    "FeatureMatrixExporter.to_feature_matrix": Case(
        FeatureMatrixExporter.to_feature_matrix, required=("uuid",)
    ),
    # -- features.pattern_recognition -----------------------------------------
    "PatternRecognition.discover_motifs": Case(
        lambda df: PatternRecognition.discover_motifs(df, window_size=8, top_k=2),
        required=("value_double",),
    ),
    "PatternRecognition.discover_discords": Case(
        lambda df: PatternRecognition.discover_discords(df, window_size=8, top_k=2),
        required=("value_double",),
    ),
    "PatternRecognition.similarity_search": Case(
        lambda df: PatternRecognition.similarity_search(df, QUERY, top_k=2),
        required=("value_double",),
    ),
    "PatternRecognition.template_match": Case(
        lambda df: PatternRecognition.template_match(df, QUERY, threshold=5.0),
        required=("value_double",),
    ),
    "PatternRecognition.compute_distance_profile": Case(
        lambda df: PatternRecognition.compute_distance_profile(df, QUERY),
        required=("value_double",),
    ),
    # -- features.segment_analysis --------------------------------------------
    "ProfileComparison.compute_distance_matrix": Case(
        lambda p: ProfileComparison.compute_distance_matrix(p, group_column="uuid"),
        make=profiles,
        required=("uuid",),
    ),
    "ProfileComparison.cluster": Case(
        lambda m: ProfileComparison.cluster(m, n_clusters=2), make=distance_matrix
    ),
    "ProfileComparison.find_similar": Case(
        lambda m: ProfileComparison.find_similar(m, "s", top_k=2),
        make=distance_matrix,
        skip={"empty": "needs the target label, which an empty matrix lacks"},
    ),
    "ProfileComparison.detect_anomalous": Case(
        ProfileComparison.detect_anomalous, make=distance_matrix
    ),
    "ProfileComparison.detect_changes": Case(
        ProfileComparison.detect_changes,
        make=profiles,
        required=("uuid", "segment_index"),
    ),
    "ProfileComparison.find_similar_pairs": Case(
        lambda p: ProfileComparison.find_similar_pairs(p, top_k=3),
        make=profiles,
        required=("uuid", "segment_index"),
    ),
    "SegmentExtractor.extract_time_ranges": Case(
        lambda df: SegmentExtractor.extract_time_ranges(df, "s"),
        required=("systime", "uuid", "value_string"),
    ),
    "SegmentProcessor.apply_ranges": Case(
        lambda df: SegmentProcessor.apply_ranges(
            df, SegmentExtractor.extract_time_ranges(df, "s")
        ),
        required=("systime",),
    ),
    "SegmentProcessor.compute_metric_profiles": Case(
        SegmentProcessor.compute_metric_profiles,
        make=with_segments,
        required=("uuid", "value_double", "segment_value"),
    ),
    "TimeWindowedFeatureTable.compute_long": Case(
        lambda df: TimeWindowedFeatureTable.compute_long(df, window="10min"),
        make=with_segments,
        required=("systime", "uuid", "value_double", "segment_value"),
    ),
    "TimeWindowedFeatureTable.compute": Case(
        lambda df: TimeWindowedFeatureTable.compute(df, window="10min"),
        make=with_segments,
        required=("systime", "uuid", "value_double", "segment_value"),
    ),
    # -- features.stats -------------------------------------------------------
    # describe's columns are the input's columns, holding statistics.
    "NumericStatistics.describe": Case(NumericStatistics.describe, typed=False),
    "NumericStatistics.column_quantile": Case(
        lambda df: NumericStatistics.column_quantile(df, "value_double", 0.9),
        required=("value_double",),
    ),
    "StringStatistics.most_common_n_strings": _stat(
        StringStatistics, "value_string", "most_common_n_strings", 2
    ),
    "StringStatistics.contains_substring_count": _stat(
        StringStatistics, "value_string", "contains_substring_count", "1"
    ),
    "StringStatistics.starts_with_count": _stat(
        StringStatistics, "value_string", "starts_with_count", "order"
    ),
    "StringStatistics.ends_with_count": _stat(
        StringStatistics, "value_string", "ends_with_count", "1"
    ),
    "TimestampStatistics.days_with_most_activity": Case(
        lambda df: TimestampStatistics.days_with_most_activity(df, n=2),
        required=TIME,
    ),
    "DescriptiveFeatures.overall_stats": Case(
        lambda df: DescriptiveFeatures(df).overall_stats(df)
    ),
    "DescriptiveFeatures.compute_per_group": Case(
        lambda df: DescriptiveFeatures(df).compute_per_group(df)
    ),
    "DescriptiveFeatures.compute": Case(
        lambda df: DescriptiveFeatures(df).compute(), required=("uuid",)
    ),
    "DescriptiveFeatures.compute_windowed": Case(
        lambda df: DescriptiveFeatures(df).compute_windowed("10min"),
        required=("uuid", "systime"),
    ),
    # -- features.time_stats --------------------------------------------------
    "TimeGroupedStatistics.calculate_statistic": Case(
        lambda df: TimeGroupedStatistics.calculate_statistic(df, window="10min"),
        required=("systime", "value_double"),
    ),
    "TimeGroupedStatistics.calculate_statistics": Case(
        lambda df: TimeGroupedStatistics.calculate_statistics(df, window="10min"),
        required=("systime", "value_double"),
    ),
    "TimeGroupedStatistics.calculate_custom_func": Case(
        lambda df: TimeGroupedStatistics.calculate_custom_func(
            df, window="10min", func=lambda s: s.max()
        ),
        required=("systime", "value_double"),
    ),
}

# Single-column statistics: one entry per method, same call shape.
for _cls, _column in (
    (NumericStatistics, "value_double"),
    (BooleanStatistics, "value_bool"),
    (StringStatistics, "value_string"),
    (TimestampStatistics, "systime"),
):
    for _name, _member in vars(_cls).items():
        _key = f"{_cls.__name__}.{_name}"
        if _name.startswith("_") or _key in REGISTRY:
            continue
        REGISTRY[_key] = _stat(_cls, _column, _name)

#: Public callables that take no frame, so no frame contract applies.
EXEMPT: dict[str, str] = {
    "TimezoneShift.list_available_timezones": "takes no frame",
    "UnitConverter.convert_value": "converts a scalar, takes no frame",
    "UnitConverter.conversion_factor": "takes two unit names, no frame",
}

KEYS = sorted(REGISTRY)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIME_COLUMNS = {
    "systime",
    "start",
    "end",
    "time_window",
    "window_start",
    "gap_start",
    "gap_end",
    "cycle_start",
    "cycle_end",
    "segment_start",
    "segment_end",
    "start_time",
    "end_time",
    "time_a",
    "time_b",
}
NUMERIC_COLUMNS = {
    "duration_seconds",
    "kwh",
    "avg_kw",
    "coverage",
    "sample_count",
    "distance",
    "dtw_distance",
    "change_score",
    "anomaly_score",
    "z_score",
    "start_index",
    "value_count",
}


def _run(case: Case, df: pd.DataFrame) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.call(case.make(df))


def _skip(case: Case, check: str) -> None:
    reason = case.skip.get(check) or case.skip.get("*")
    if reason:
        pytest.skip(reason)


def _frames(result: Any) -> list[pd.DataFrame | pd.Series]:
    """Every DataFrame / Series inside a result."""
    if isinstance(result, pd.DataFrame | pd.Series):
        return [result]
    if isinstance(result, dict):
        return [f for v in result.values() for f in _frames(v)]
    if isinstance(result, list | tuple):
        return [f for v in result for f in _frames(v)]
    return []


def _datetimes(result: Any) -> list[tuple[str, Any]]:
    """(label, tz) of every datetime value or column inside a result."""
    found: list[tuple[str, Any]] = []
    if isinstance(result, pd.Timestamp):
        found.append(("scalar", result.tzinfo))
    elif isinstance(result, dict):
        for k, v in result.items():
            found += [(f"{k}.{label}", tz) for label, tz in _datetimes(v)]
    elif isinstance(result, list | tuple):
        for v in result:
            found += _datetimes(v)
    elif isinstance(result, pd.DataFrame | pd.Series):
        frame = result.to_frame() if isinstance(result, pd.Series) else result
        for i, col in enumerate(frame.columns):
            values = frame.iloc[:, i]
            if pd.api.types.is_datetime64_any_dtype(values.dtype):
                found.append((str(col), values.dt.tz))
        indexes = (
            [frame.index.get_level_values(i) for i in range(frame.index.nlevels)]
            if isinstance(frame.index, pd.MultiIndex)
            else [frame.index]
        )
        for index in indexes:
            if isinstance(index, pd.DatetimeIndex):
                found.append((f"index {index.name}", index.tz))
    return found


def _canon(x: Any) -> Any:
    """A comparable form: NaN/NA -> None, numbers rounded, dtypes ignored."""
    if isinstance(x, pd.DataFrame):
        return (
            [str(c) for c in x.columns],
            _canon(x.index),
            [_canon(x.iloc[:, i]) for i in range(x.shape[1])],
        )
    if isinstance(x, pd.Series):
        return (_canon(x.index), [_canon(v) for v in x.astype(object).tolist()])
    if isinstance(x, pd.Index):
        return [_canon(v) for v in x.tolist()]
    if isinstance(x, np.ndarray):
        return [_canon(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_canon(v) for v in x]
    if x is None or (
        not isinstance(x, str) and pd.api.types.is_scalar(x) and pd.isna(x)
    ):
        return None
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | float | np.number):
        return round(float(x), 9)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_empty_input(key: str) -> None:
    case = REGISTRY[key]
    _skip(case, "empty")
    result = _run(case, base_frame().iloc[:0])
    if not case.typed:
        return
    for frame in _frames(result):
        if not isinstance(frame, pd.DataFrame):
            continue
        for i, col in enumerate(frame.columns):
            dtype = frame.iloc[:, i].dtype
            if col in TIME_COLUMNS:
                assert pd.api.types.is_datetime64_any_dtype(
                    dtype
                ), f"{key}: empty result column {col!r} is {dtype}, not datetime"
            if col in NUMERIC_COLUMNS:
                assert pd.api.types.is_numeric_dtype(
                    dtype
                ), f"{key}: empty result column {col!r} is {dtype}, not numeric"


@pytest.mark.parametrize(
    "key, column",
    [(key, column) for key in KEYS for column in REGISTRY[key].required],
)
def test_missing_column(key: str, column: str) -> None:
    case = REGISTRY[key]
    _skip(case, "missing")
    data = case.make(base_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ColumnNotFoundError):
            case.call(data.drop(columns=[column]))


@pytest.mark.parametrize("key", KEYS)
def test_tz_preserved(key: str) -> None:
    case = REGISTRY[key]
    _skip(case, "tz")
    result = _run(case, base_frame(tz=BERLIN))
    naive = [label for label, tz in _datetimes(result) if tz is None]
    assert not naive, f"{key}: tz-naive datetimes from tz-aware input: {naive}"


@pytest.mark.parametrize("key", KEYS)
def test_input_not_mutated(key: str) -> None:
    case = REGISTRY[key]
    _skip(case, "no_mutation")
    data = case.make(base_frame())
    before = data.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case.call(data)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("key", KEYS)
def test_nullable_input_same_result(key: str) -> None:
    case = REGISTRY[key]
    _skip(case, "nullable")
    expected = _canon(_run(case, base_frame()))
    got = _canon(_run(case, nullable(base_frame())))
    assert got == expected


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _public_callables() -> set[str]:
    names: set[str] = set()
    for package in (ts_shape.transform, ts_shape.features):
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if any(part.startswith("_") for part in info.name.split(".")):
                continue
            module = importlib.import_module(info.name)
            for name, obj in vars(module).items():
                if name.startswith("_") or getattr(obj, "__module__", None) != (
                    module.__name__
                ):
                    continue
                if inspect.isclass(obj):
                    for member in vars(obj):
                        if not member.startswith("_") and callable(
                            getattr(obj, member)
                        ):
                            names.add(f"{name}.{member}")
                elif callable(obj):
                    names.add(name)
    return names


def test_registry_covers_every_public_callable() -> None:
    public = _public_callables()
    missing = sorted(public - set(REGISTRY) - set(EXEMPT))
    assert not missing, f"add a REGISTRY entry (or exemption) for: {missing}"
    stale = sorted((set(REGISTRY) | set(EXEMPT)) - public)
    assert not stale, f"REGISTRY entries for callables that do not exist: {stale}"
