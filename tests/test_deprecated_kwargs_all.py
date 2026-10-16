"""Every renamed keyword argument in ts_shape still works under its old name.

The shims come from :func:`ts_shape.utils.deprecation.renamed_kwargs`. This
module finds all of them by walking the package, then checks each
``(callable, old, new)`` triple:

* the reported signature shows the new name and not the old one;
* passing both names raises :class:`TypeError`;
* passing the old name warns and forwards the value under the new name.

The forwarding check swaps the wrapped function for a recorder, so it runs
in microseconds and needs no domain data. A handful of real calls, one per
module family, then checks that the old and the new name give identical
results.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.utils.deprecation import renamed_kwargs

_SHIM_CODE = renamed_kwargs(old="new")(lambda **kw: kw).__code__


def _modules():
    for info in pkgutil.walk_packages(ts_shape.__path__, "ts_shape."):
        try:
            yield importlib.import_module(info.name)
        except ImportError:  # loader backends with optional dependencies
            continue


def _shim(obj):
    """The ``renamed_kwargs`` wrapper in ``obj``'s ``__wrapped__`` chain."""
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        if getattr(obj, "__code__", None) is _SHIM_CODE:
            return obj
        obj = getattr(obj, "__wrapped__", None)
    return None


def _collect():
    found = {}
    for mod in _modules():
        for name, obj in vars(mod).items():
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isfunction(obj):
                members = [(name, obj, obj)]
            elif inspect.isclass(obj):
                members = []
                for attr, raw in vars(obj).items():
                    func = (
                        raw.__func__
                        if isinstance(raw, staticmethod | classmethod)
                        else raw
                    )
                    if callable(func):
                        members.append((f"{name}.{attr}", func, getattr(obj, attr)))
            else:
                continue
            for qualname, func, public in members:
                shim = _shim(func)
                if shim is not None:
                    found[f"{mod.__name__}:{qualname}"] = (public, shim)
    return found


SHIMS = _collect()
CASES = sorted(
    (key, old, new)
    for key, (_, shim) in SHIMS.items()
    for old, new in shim.__deprecated_kwargs__.items()
)


def test_shims_were_found():
    # Guards the walker itself: there are well over a hundred renamed kwargs.
    assert len(CASES) > 100
    assert "ts_shape.utils.base:Base.__init__" in SHIMS


def _recording_shim(shim, calls):
    """A copy of ``shim`` whose wrapped function records its keyword arguments."""

    def record(*args, **kwargs):
        calls.append(kwargs)
        return "sentinel-result"

    cells = []
    for var, cell in zip(shim.__code__.co_freevars, shim.__closure__):
        cells.append(types.CellType(record) if var == "func" else cell)
    return types.FunctionType(
        shim.__code__, shim.__globals__, shim.__name__, shim.__defaults__, tuple(cells)
    )


@pytest.mark.parametrize("key,old,new", CASES, ids=[f"{k}:{o}" for k, o, _ in CASES])
def test_renamed_kwarg(key, old, new):
    public, shim = SHIMS[key]
    assert old != new

    params = inspect.signature(public).parameters
    assert new in params, f"{key}: new name {new!r} missing from signature"
    assert old not in params, f"{key}: old name {old!r} still in signature"

    sentinel = object()
    with pytest.raises(TypeError, match=rf"{old!r}.*{new!r}"):
        shim(**{old: sentinel, new: sentinel})

    calls = []
    recorder = _recording_shim(shim, calls)
    with pytest.warns(DeprecationWarning, match=rf"{old!r} is deprecated, use {new!r}"):
        assert recorder(**{old: sentinel}) == "sentinel-result"
    assert calls == [{new: sentinel}]

    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recorder(**{new: sentinel})
    assert calls == [{new: sentinel}]


# ---------------------------------------------------------------------------
# Real calls: old and new name give the same result
# ---------------------------------------------------------------------------


def _frame():
    rng = np.random.default_rng(0)
    n = 48
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "uuid": "sig:1",
            "value_double": rng.normal(50.0, 5.0, n),
            "value_integer": rng.integers(0, 10, n),
            "value_bool": rng.integers(0, 2, n).astype(bool),
            "value_string": rng.choice(["A", "B"], n),
            "is_delta": True,
        }
    )


def _numeric_stats(**kw):
    from ts_shape.features.stats.numeric_stats import NumericStatistics

    return NumericStatistics.column_mean(_frame(), **kw)


def _timestamp_stats(**kw):
    from ts_shape.features.stats.timestamp_stats import TimestampStatistics

    return TimestampStatistics.earliest_timestamp(_frame(), **kw)


def _numeric_filter(**kw):
    from ts_shape.transform.filter.numeric_filter import DoubleFilter

    return DoubleFilter.filter_value_double_between(
        _frame(), min_value=45.0, max_value=55.0, **kw
    )


def _time_grouped(**kw):
    from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics

    return TimeGroupedStatistics.calculate_statistic(
        _frame(), "systime", "value_double", stat_method="mean", **kw
    )


def _harmonizer(**kw):
    from ts_shape.transform.harmonization import DataHarmonizer

    return DataHarmonizer(_frame()).resample_to_uniform(**kw)


def _data_gaps(**kw):
    from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents

    return DataGapAnalysisEvents(_frame(), "sig:1").coverage_by_period(**kw)


def _operating_range(**kw):
    from ts_shape.events.engineering.operating_range import OperatingRangeEvents

    return OperatingRangeEvents(_frame(), "sig:1").time_in_range(window="6h", **kw)


def _feature_matrix(**kw):
    from ts_shape.features.export import FeatureMatrixExporter

    return FeatureMatrixExporter.to_feature_matrix(**kw)


def _base_init(**kw):
    from ts_shape.utils.base import Base

    return Base(_frame(), **kw).dataframe


REAL_CALLS = [
    (
        "numeric_stats",
        _numeric_stats,
        {"column_name": "value_double"},
        {"value_column": "value_double"},
    ),
    (
        "timestamp_stats",
        _timestamp_stats,
        {"column_name": "systime"},
        {"time_column": "systime"},
    ),
    (
        "numeric_filter",
        _numeric_filter,
        {"column_name": "value_double"},
        {"value_column": "value_double"},
    ),
    ("time_grouped", _time_grouped, {"freq": "2h"}, {"window": "2h"}),
    ("harmonizer", _harmonizer, {"freq": "30min"}, {"window": "30min"}),
    ("data_gaps", _data_gaps, {"freq": "6h"}, {"window": "6h"}),
    (
        "operating_range",
        _operating_range,
        {"lower": 45.0, "upper": 55.0},
        {"lower_limit": 45.0, "upper_limit": 55.0},
    ),
    ("feature_matrix", _feature_matrix, {"df": _frame()}, {"dataframe": _frame()}),
    ("base_init", _base_init, {"column_name": "systime"}, {"time_column": "systime"}),
]


def _assert_same(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize(
    "call,old_kw,new_kw",
    [c[1:] for c in REAL_CALLS],
    ids=[c[0] for c in REAL_CALLS],
)
def test_old_name_gives_same_result(call, old_kw, new_kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        expected = call(**new_kw)
    with pytest.warns(DeprecationWarning, match="is deprecated, use"):
        got = call(**old_kw)
    _assert_same(got, expected)


# ---------------------------------------------------------------------------
# Round 11 (naming consistency): every renamed keyword is shimmed
# ---------------------------------------------------------------------------

_C = "ts_shape.events.correlation.signal_correlation:SignalCorrelationEvents"
_DT = "ts_shape.events.production.downtime_tracking:DowntimeEvents"
_IE = "ts_shape.events.energy.idle_energy_detection:IdleEnergyDetectionEvents"
_ST = "ts_shape.events.production.setup_time_tracking:SetupTimeEvents"
_SH = "ts_shape.events.production.shift_handover:ShiftHandoverEvents"
_SPC = (
    "ts_shape.events.quality.statistical_process_control"
    ":StatisticalProcessControlEvents"
)
_SQ = "ts_shape.events.quality.signal_quality:SignalQualityEvents"
_MPT = (
    "ts_shape.events.production.multi_process_traceability"
    ":MultiProcessTraceabilityEvents.__init__"
)
_RT = "ts_shape.events.production.routing_traceability:RoutingTraceabilityEvents"
_CP = "ts_shape.features.cycles.cycle_processor:CycleDataProcessor"
_TS = "ts_shape.loader.timeseries"

#: ``(callable, old, new)`` of every keyword renamed in round 11.
ROUND11 = [
    *[
        (f"ts_shape.eventlog.{fn}", "df", "dataframe")
        for fn in (
            "lambda_rules.backtest:run_backtest",
            "lambda_rules.detector:LambdaDetector.evaluate",
            "lambda_rules.detector:LambdaDetector.to_event_log",
            "lambda_rules.evaluator:evaluate",
            "normalizer:to_event_log",
            "objects:attach_objects",
            "objects:detect_objects",
            "objects:object_intervals",
        )
    ],
    *[
        (f"{_C}.{m}", old, f"signal_{old}")
        for m in ("rolling_correlation", "correlation_breakdown", "lag_correlation")
        for old in ("uuid_a", "uuid_b")
    ],
    *[
        (f"ts_shape.events.energy.{cls}.normalize", old, new)
        for cls in (
            "consumption_analysis:EnergyConsumptionEvents",
            "efficiency_tracking:EnergyEfficiencyEvents",
        )
        for old, new in (("df", "dataframe"), ("id_column", "uuid_column"))
    ],
    *[
        (key, "state_uuid", "run_state_uuid")
        for key in (
            "ts_shape.events.energy.efficiency_tracking"
            ":EnergyEfficiencyEvents.idle_energy_waste",
            f"{_IE}.idle_energy_by_window",
            f"{_IE}.idle_energy_by_shift",
            f"{_IE}.idle_energy_trend",
            "ts_shape.events.maintenance.reliability:ReliabilityEvents.__init__",
            f"{_DT}.downtime_by_shift",
            f"{_DT}.downtime_by_reason",
            f"{_DT}.top_downtime_reasons",
            f"{_DT}.availability_trend",
            "ts_shape.events.production.long_downtime_events"
            ":LongDowntimeEvents.__init__",
            f"{_ST}.setup_durations",
            f"{_ST}.setup_by_product",
            f"{_ST}.setup_statistics",
            f"{_ST}.setup_trend",
            f"{_SH}.generate_report",
            f"{_SH}.highlight_issues",
        )
    ],
    (
        "ts_shape.events.production.continuous_process_alignment"
        ":ContinuousProcessAlignmentEvents.segment_by_cut",
        "cut_value_column",
        "value_column_cut",
    ),
    (_MPT, "id_value_column", "value_column_id"),
    (_MPT, "handover_value_column", "value_column_handover"),
    (
        "ts_shape.events.production.oee_calculator:OEEEvents.performance",
        "run_value_column",
        "value_column_state",
    ),
    (f"{_RT}.__init__", "id_value_column", "value_column_id"),
    (f"{_RT}.__init__", "routing_value_column", "value_column_routing"),
    *[
        (
            "ts_shape.events.production.period_summary"
            ":PeriodSummaryEvents.compare_periods",
            old,
            new,
        )
        for old, new in (("period1", "period_a"), ("period2", "period_b"))
    ],
    (
        "ts_shape.events.quality.sampling_resolution"
        ":SamplingResolutionEvents.detect_aliasing",
        "expected_frequency",
        "expected_frequency_hz",
    ),
    (f"{_SQ}.detect_missing_data", "expected_freq", "expected_interval"),
    (f"{_SQ}.data_completeness", "expected_freq", "expected_interval"),
    *[(f"{_SPC}.rule_{i}", "df", "dataframe") for i in range(1, 9)],
    (f"{_CP}.group_by_cycle_uuid", "data", "dataframe"),
    (f"{_CP}.split_dataframes_by_group", "column", "group_column"),
    (
        f"{_TS}.azure_blob_loader:AzureBlobParquetLoader.list_structure",
        "parquet_only",
        "only_parquet",
    ),
    (
        f"{_TS}.databricks_unity_parquet_loader"
        ":DatabricksUnityParquetLoader.list_structure",
        "parquet_only",
        "only_parquet",
    ),
]


@pytest.mark.parametrize(
    "key,old,new", ROUND11, ids=[f"{k.rsplit(':', 1)[1]}:{o}" for k, o, _ in ROUND11]
)
def test_round11_rename_is_shimmed(key, old, new):
    # The shim exists, so test_renamed_kwarg checks it like every other one.
    if key.startswith(_TS) and key not in SHIMS:
        pytest.skip("loader backend with an optional dependency")
    assert (key, old, new) in CASES


def _two_signals():
    frame = _frame()
    other = frame.assign(uuid="sig:2", value_double=frame["value_double"] * 2 + 1)
    return pd.concat([frame, other], ignore_index=True)


def _downtime_by_shift(**kw):
    from ts_shape.events.production.downtime_tracking import DowntimeEvents

    return DowntimeEvents(_frame()).downtime_by_shift(
        running_value=True, value_column="value_bool", **kw
    )


def _missing_data(**kw):
    from ts_shape.events.quality.signal_quality import SignalQualityEvents

    return SignalQualityEvents(_frame(), "sig:1").detect_missing_data(**kw)


def _compare_periods(**kw):
    from ts_shape.events.production.period_summary import PeriodSummaryEvents

    day = ("2024-01-01", "2024-01-01")
    return PeriodSummaryEvents(_frame()).compare_periods(
        "sig:1", **{k: day for k in kw}
    )


def _oee_performance(**kw):
    from ts_shape.events.production.oee_calculator import OEEEvents

    return OEEEvents(_frame()).performance("sig:1", 60.0, run_state_uuid="sig:1", **kw)


def _spc_rule(**kw):
    from ts_shape.events.quality.statistical_process_control import (
        StatisticalProcessControlEvents,
    )

    spc = StatisticalProcessControlEvents(
        _frame(), "value_double", "sig:1", "sig:1", "quality:spc"
    )
    return spc.rule_3(**{k: _frame() for k in kw})


def _aliasing(**kw):
    from ts_shape.events.quality.sampling_resolution import SamplingResolutionEvents

    return SamplingResolutionEvents(_frame(), "sig:1").detect_aliasing(
        window="6h", **kw
    )


def _long_downtime(**kw):
    from ts_shape.events.production.long_downtime_events import LongDowntimeEvents

    return LongDowntimeEvents(_frame(), **kw).detect_long_downtime(min_gap="30min")


def _rolling_correlation(**kw):
    from ts_shape.events.correlation.signal_correlation import (
        SignalCorrelationEvents,
    )

    return SignalCorrelationEvents(_two_signals()).rolling_correlation(
        window="15min", window_size=4, **kw
    )


def _normalize(**kw):
    from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents

    raw = pd.DataFrame(
        {"time": ["2026-01-13T00:00:00Z"], "id": ["meter_42"], "value": [55.0]}
    )
    frame_key = "df" if "id_column" in kw else "dataframe"
    return EnergyConsumptionEvents.normalize(
        series_id="ignored", **{frame_key: raw}, **kw
    )


ROUND11_CALLS = [
    (
        "downtime_by_shift",
        _downtime_by_shift,
        {"state_uuid": "sig:1"},
        {"run_state_uuid": "sig:1"},
    ),
    (
        "missing_data",
        _missing_data,
        {"expected_freq": "15min"},
        {"expected_interval": "15min"},
    ),
    (
        "compare_periods",
        _compare_periods,
        {"period1": None, "period2": None},
        {"period_a": None, "period_b": None},
    ),
    (
        "oee_performance",
        _oee_performance,
        {"run_value_column": "value_bool"},
        {"value_column_state": "value_bool"},
    ),
    ("spc_rule", _spc_rule, {"df": None}, {"dataframe": None}),
    (
        "aliasing",
        _aliasing,
        {"expected_frequency": 0.001},
        {"expected_frequency_hz": 0.001},
    ),
    (
        "long_downtime",
        _long_downtime,
        {"state_uuid": "sig:1"},
        {"run_state_uuid": "sig:1"},
    ),
    (
        "rolling_correlation",
        _rolling_correlation,
        {"uuid_a": "sig:1", "uuid_b": "sig:2"},
        {"signal_uuid_a": "sig:1", "signal_uuid_b": "sig:2"},
    ),
    ("normalize", _normalize, {"id_column": "id"}, {"uuid_column": "id"}),
]


@pytest.mark.parametrize(
    "call,old_kw,new_kw",
    [c[1:] for c in ROUND11_CALLS],
    ids=[c[0] for c in ROUND11_CALLS],
)
def test_round11_old_name_gives_same_result(call, old_kw, new_kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        expected = call(**new_kw)
    with pytest.warns(DeprecationWarning, match="is deprecated, use"):
        got = call(**old_kw)
    _assert_same(got, expected)


def _renamed_attributes():
    from ts_shape.events.maintenance.reliability import ReliabilityEvents
    from ts_shape.events.production.long_downtime_events import LongDowntimeEvents
    from ts_shape.events.production.multi_process_traceability import (
        MultiProcessTraceabilityEvents,
    )
    from ts_shape.events.production.routing_traceability import (
        RoutingTraceabilityEvents,
    )

    frame = _frame()
    trace = MultiProcessTraceabilityEvents(
        frame, [{"id_uuid": "sig:1", "station": "S1"}]
    )
    routing = RoutingTraceabilityEvents(frame, "sig:1", "sig:1")
    return [
        (LongDowntimeEvents(frame, "sig:1"), "state_uuid", "run_state_uuid"),
        (ReliabilityEvents(frame, "sig:1"), "state_uuid", "run_state_uuid"),
        (trace, "id_value_column", "value_column_id"),
        (trace, "handover_value_column", "value_column_handover"),
        (routing, "id_value_column", "value_column_id"),
        (routing, "routing_value_column", "value_column_routing"),
    ]


def test_round11_renamed_attributes_still_read():
    for obj, old, new in _renamed_attributes():
        with pytest.warns(DeprecationWarning, match=f"{old} is deprecated"):
            assert getattr(obj, old) == getattr(obj, new)
