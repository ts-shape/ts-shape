"""Input contract of the public functions outside the detectors.

Every public callable of ``ts_shape.signals``, ``ts_shape.profiling``,
``ts_shape.pipeline``, ``ts_shape.eventlog`` (the offline parts), the context
helpers and ``ts_shape.datasets`` is registered in :data:`CASES` with a
minimal call on the standard long-format frame, or in :data:`EXEMPT` with the
reason it has no such call. For every case:

* an empty standard frame does not raise and gives a typed result;
* a frame without a required column raises ``ColumnNotFoundError``;
* a tz-aware time column stays in its timezone, or comes back in UTC where
  the function documents it (the OCEL event log);
* the input frame is not changed;
* nullable and pyarrow dtypes give the same result as NumPy dtypes.

``test_every_public_callable_is_registered`` fails for a new public callable
that is in neither table.
"""

from __future__ import annotations

import inspect
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import ts_shape.context.value_mapping as value_mapping
import ts_shape.datasets as datasets
import ts_shape.eventlog as eventlog
import ts_shape.loader.combine.integrator as integrator
import ts_shape.loader.context.context_enricher as context_enricher
import ts_shape.pipeline as pipeline
import ts_shape.profiling as profiling
import ts_shape.signals as signals
import ts_shape.signals.primitives as primitives
from invariants import harness as H
from ts_shape.errors import ColumnNotFoundError
from ts_shape.signals import SignalSpec

MODULES = {
    "signals": signals,
    "signals.primitives": primitives,
    "profiling": profiling,
    "pipeline": pipeline,
    "eventlog": eventlog,
    "context": value_mapping,
    "context_enricher": context_enricher,
    "integrator": integrator,
    "datasets": datasets,
}

CONT, STATE, COUNTER, TEXT = H.CONT, H.STATE, H.COUNTER, H.TEXT
POINT = "OutlierDetectionEvents.detect_outliers_zscore"


def standard_frame(tz: str | None = None) -> pd.DataFrame:
    """One hour of the invariant harness's plant frame (fast)."""
    df = H.plant_frame(tz)
    return df[df["systime"] < df["systime"].min() + pd.Timedelta("1h")].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Builders: one minimal call per public callable
# ---------------------------------------------------------------------------


def _point_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The outlier detector's output layout, built from the CONT signal."""
    rows = df[df["uuid"] == CONT]
    keep = [c for c in ("systime", "uuid", "value_double") if c in rows.columns]
    return rows[keep].assign(source_uuid=rows["uuid"])


def _event_log(df: pd.DataFrame) -> eventlog.EventLog:
    return eventlog.to_event_log(_point_frame(df), detector=POINT)


def _id_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["uuid"] == TEXT].copy()
    out["uuid"] = "sig:id"
    return out


_ID_SPEC = [eventlog.ObjectSpec("sig:id", "batch")]


def _intervals(df: pd.DataFrame) -> pd.DataFrame:
    return eventlog.object_intervals(_id_frame(df), _ID_SPEC)


def _lambda(df: pd.DataFrame) -> pd.DataFrame:
    spec = eventlog.RuleSpec(
        id="contract",
        class_name="LambdaContract",
        method_name="hot",
        pack="quality",
        shape="point",
        archetype="threshold",
        template="quality.contract.hot",
        trigger=eventlog.TriggerSpec(expression="value_double > 50"),
        standard_attrs={"ts_shape:method": "m", "ts_shape:direction": "above"},
    )
    return eventlog.LambdaDetector(spec).evaluate(df)


def _mapping_file() -> str:
    folder = Path(tempfile.mkdtemp())
    path = folder / "map.csv"
    pd.DataFrame({"key": ["Running", "Stopped"], "name": ["run", "stop"]}).to_csv(
        path, index=False
    )
    return str(path)


def _value_mapper(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unmapped codes of other signals
        return value_mapping.ValueMapper(
            df, _mapping_file(), "value_string", "key", "name"
        ).map_values()


def _enricher(df: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            "uuid": [CONT, STATE, COUNTER, TEXT, H.CONT2, H.SETPOINT],
            "unit": list("abcdef"),
        }
    )
    return context_enricher.ContextEnricher(df).enrich_with_metadata(meta)


def _integrator(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no timeseries data" on empty input
        return integrator.DataIntegratorHybrid.combine_data(
            timeseries_sources=[df], uuids=[CONT]
        )


@dataclass(frozen=True)
class Case:
    """A minimal call and the contract it follows."""

    run: Callable[[pd.DataFrame], Any]
    #: Column whose absence must raise ColumnNotFoundError (None: no check).
    required: str | None = "systime"
    #: Timezone of time columns in the result: "preserve" or "utc".
    tz: str = "preserve"
    #: Compare results across dtype backends.
    dtypes: bool = True
    #: Empty input gives the same columns as real input.
    same_empty_columns: bool = True


CASES: dict[str, Case] = {
    # -- signals ----------------------------------------------------------
    "signals.infer_specs": Case(lambda df: signals.infer_specs(df).to_frame()),
    "signals.infer_spec": Case(
        lambda df: signals.infer_spec(df[df["uuid"] == CONT], CONT),
        dtypes=False,  # notes differ in wording only
    ),
    "signals.primitives.signal_series": Case(
        lambda df: primitives.signal_series(df, SignalSpec(CONT))
    ),
    "signals.primitives.intervals": Case(
        lambda df: primitives.intervals(
            df, SignalSpec(STATE, kind="state", value_column="value_bool")
        )
    ),
    "signals.primitives.increments": Case(
        lambda df: primitives.increments(
            df, SignalSpec(COUNTER, kind="counter", value_column="value_integer")
        )
    ),
    "signals.primitives.per_window": Case(
        lambda df: primitives.per_window(df, SignalSpec(CONT), "15min")
    ),
    "signals.primitives.align": Case(
        lambda df: primitives.align(df, SignalSpec(CONT), df, SignalSpec(H.CONT2))
    ),
    # -- profiling / pipeline ---------------------------------------------
    "profiling.profile": Case(lambda df: profiling.profile(df).table),
    "signals.identify_objects": Case(
        lambda df: signals.identify_objects(df, min_samples=1).table
    ),
    "pipeline.Pipeline": Case(
        lambda df: pipeline.Pipeline()
        .transform(lambda d: d.assign(value_double=d["value_double"] * 2), name="x2")
        .run(df)
        .data,
        required=None,  # passes frames through, needs no column
        dtypes=False,  # returns the input rows with their own dtypes
    ),
    # -- eventlog -----------------------------------------------------------
    "eventlog.to_event_log": Case(
        lambda df: _event_log(df).events, tz="utc", same_empty_columns=False
    ),
    "eventlog.concat": Case(
        lambda df: eventlog.concat(_event_log(df), _event_log(df)).events,
        tz="utc",
        same_empty_columns=False,
    ),
    "eventlog.align_columns": Case(
        lambda df: eventlog.align_columns(_event_log(df))[0].events,
        tz="utc",
        same_empty_columns=False,
    ),
    "eventlog.to_event_log_ocel": Case(
        lambda df: eventlog.to_event_log_ocel(_event_log(df)).events,
        tz="utc",
        same_empty_columns=False,
    ),
    "eventlog.object_intervals": Case(_intervals, tz="utc"),
    "eventlog.detect_objects": Case(
        lambda df: eventlog.detect_objects(_id_frame(df), _ID_SPEC).object_changes,
        tz="utc",
    ),
    "eventlog.attach_objects": Case(
        lambda df: eventlog.attach_objects(
            _event_log(df), _id_frame(df), _ID_SPEC
        ).relations,
        tz="utc",
        same_empty_columns=False,
    ),
    "eventlog.relate": Case(
        lambda df: eventlog.relate(_event_log(df), _intervals(df)).relations,
        tz="utc",
    ),
    "eventlog.link_events_to_objects": Case(
        lambda df: eventlog.link_events_to_objects(
            _event_log(df).events, _intervals(df)
        ),
        tz="utc",
    ),
    "eventlog.derive_o2o": Case(
        lambda df: eventlog.derive_o2o(_intervals(df)), tz="utc"
    ),
    "eventlog.presence_events": Case(
        lambda df: eventlog.presence_events(_intervals(df)), tz="utc"
    ),
    "eventlog.object_kpis": Case(
        lambda df: eventlog.object_kpis(
            _intervals(df), object_type="batch", window="15min"
        ).throughput,
        tz="utc",
    ),
    "eventlog.wip_over_time": Case(
        lambda df: eventlog.wip_over_time(
            _intervals(df), object_type="batch", window="15min"
        ),
        tz="utc",
    ),
    "eventlog.carrier_loops": Case(
        lambda df: eventlog.carrier_loops(_intervals(df), object_type="batch").loops,
        tz="utc",
    ),
    "eventlog.attach_counts": Case(
        lambda df: eventlog.attach_counts(
            _intervals(df), df, COUNTER, object_type="batch", station="sig:id"
        ),
    ),
    "eventlog.compile_expression": Case(
        lambda df: eventlog.compile_expression("value_double > 50")(df),
        required="value_double",
    ),
    "eventlog.LambdaDetector": Case(
        _lambda,
        required="value_double",
        dtypes=False,  # a point rule returns the matching input rows as they are
    ),
    # -- context ------------------------------------------------------------
    "context.ValueMapper": Case(_value_mapper, required="value_string", dtypes=False),
    "context_enricher.ContextEnricher": Case(_enricher, required="uuid", dtypes=False),
    "integrator.DataIntegratorHybrid": Case(
        _integrator,
        required=None,  # a missing join key is a warning by design
        dtypes=False,  # returns the input rows with their own dtypes
        same_empty_columns=False,  # empty input gives the standard columns
    ),
}

#: Public callables without a frame-in / frame-out call, and why.
EXEMPT: dict[str, str] = {
    "signals.SignalSpec": "value object, validated in test_signals_robustness",
    "signals.SignalSpecs": "value object, validated in test_signals_robustness",
    "signals.primitives.windows": "takes time bounds, not a frame; see test_bounds",
    "signals.primitives.shift_windows": "takes time bounds, not a frame; see test_bounds",
    "profiling.ProfileReport": "result container of profile()",
    "signals.ObjectReport": "result container of identify_objects()",
    "signals.register_identifier_type": "registry; see test_identify_objects",
    "signals.register_identifier_pattern": "registry; see test_identify_objects",
    "pipeline.PipelineResult": "result container of Pipeline.run()",
    "eventlog.EventLog": "result container",
    "eventlog.OCEL2Tables": "result container",
    "eventlog.BacktestResult": "result container",
    "eventlog.ObjectKPIs": "result container of object_kpis()",
    "eventlog.CarrierLoops": "result container of carrier_loops()",
    "eventlog.ObjectSpec": "value object",
    "eventlog.LabelRule": "value object",
    "eventlog.RuleSpec": "value object",
    "eventlog.TriggerSpec": "value object",
    "eventlog.UnsafeExpression": "exception type",
    "eventlog.to_event_log_xes": "an empty log has no case objects: raises by design",
    "eventlog.object_specs_from_metadata": "reads a metadata table, not signals",
    "eventlog.describe": "introspection of a detector name",
    "eventlog.validate": "checks an EventLog",
    "eventlog.where_used": "queries an EventLog's o2o graph; see eventlog/test_query",
    "eventlog.where_from": "queries an EventLog's o2o graph; see eventlog/test_query",
    "eventlog.lineage": "queries an EventLog's o2o graph; see eventlog/test_query",
    "eventlog.empty_o2o": "constructor of an empty table",
    "eventlog.empty_object_changes": "constructor of an empty table",
    "eventlog.register_object_type": "mutates the global object-type registry",
    "eventlog.register_adapter": "mutates the global adapter registry",
    "eventlog.clear_overrides": "mutates the global adapter registry",
    "eventlog.has_override": "registry lookup",
    "eventlog.register_lambda_rule": "mutates the global REGISTRY",
    "eventlog.unregister_lambda_rule": "mutates the global REGISTRY",
    "eventlog.load_dicts": "mutates the global REGISTRY",
    "eventlog.load_yaml": "reads a file and mutates the global REGISTRY",
    "eventlog.run_backtest": "needs a registered rule; covered by LambdaDetector",
    "datasets.make_timeseries": "generator without a frame input",
    "datasets.make_id_signal": "generator without a frame input",
    "datasets.make_object_signals": "generator without a frame input",
}


def _public_callables() -> set[str]:
    names: set[str] = set()
    for prefix, module in MODULES.items():
        public = getattr(module, "__all__", None) or [
            n for n in vars(module) if not n.startswith("_")
        ]
        for name in public:
            obj = getattr(module, name)
            if not callable(obj):
                continue
            if getattr(obj, "__module__", "") and not str(obj.__module__).startswith(
                "ts_shape"
            ):
                continue  # re-exported third-party names (pd, np, ...)
            if module.__name__ not in ("ts_shape.eventlog", "ts_shape.signals") and (
                getattr(obj, "__module__", module.__name__) != module.__name__
            ):
                continue  # imported helper, registered where it is defined
            names.add(f"{prefix}.{name}")
    return names


def test_every_public_callable_is_registered():
    names = _public_callables()
    missing = sorted(names - set(CASES) - set(EXEMPT))
    assert not missing, (
        "register these public callables in CASES (or EXEMPT with a reason): "
        f"{missing}"
    )
    stale = sorted((set(CASES) | set(EXEMPT)) - names)
    assert not stale, f"registered names that are no longer public: {stale}"


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def _run(case: Case, df: pd.DataFrame) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.run(df)


def _columns(result: Any) -> list[str]:
    frames = H._frames(result)
    return [str(c) for f in frames for c in f.columns]


@pytest.mark.parametrize("name", sorted(CASES))
def test_empty_frame_gives_typed_result(name):
    case = CASES[name]
    full = _run(case, standard_frame())
    empty = _run(case, standard_frame().iloc[0:0])
    assert type(empty) is type(full)
    if isinstance(full, pd.DataFrame | pd.Series):
        assert len(empty) == 0
        if case.same_empty_columns:
            assert _columns(empty) == _columns(full)
        elif isinstance(full, pd.DataFrame):
            assert len(empty.columns) > 0


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c.required))
def test_missing_column_raises_column_not_found(name):
    case = CASES[name]
    df = standard_frame().drop(columns=[case.required])
    with pytest.raises(ColumnNotFoundError):
        _run(case, df)


@pytest.mark.parametrize("name", sorted(CASES))
def test_timezone_is_kept_or_utc(name):
    case = CASES[name]
    result = _run(case, standard_frame(H.BERLIN))
    assert not H._naive_datetime_columns(result), "tz-aware input lost its tz"
    expected = H.BERLIN if case.tz == "preserve" else "UTC"
    for frame in H._frames(result):
        for col in frame.columns:
            s = frame[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                assert str(s.dt.tz) == expected, col
        if isinstance(frame.index, pd.DatetimeIndex):
            assert str(frame.index.tz) == expected


@pytest.mark.parametrize("name", sorted(CASES))
def test_input_is_not_changed(name):
    df = standard_frame(H.BERLIN)
    before = df.copy()
    _run(CASES[name], df)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("backend", ["nullable", "pyarrow"])
@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c.dtypes))
def test_dtype_backends_give_the_same_result(name, backend):
    case = CASES[name]
    base = standard_frame()
    variant = H.dtype_variants(base)[backend]
    assert H._same(_run(case, base), _run(case, variant))


def test_bounds():
    start = pd.Timestamp("2024-03-30 12:00", tz=H.BERLIN)
    for out in (
        primitives.windows(start, "2024-04-01", "1D"),
        primitives.shift_windows(start, "2024-03-31 12:00"),
    ):
        assert str(out["start"].dt.tz) == H.BERLIN
        assert len(out)
    for func in (
        lambda: primitives.windows(start, start, "1D"),
        lambda: primitives.shift_windows(start, start),
    ):
        empty = func()
        assert empty.empty and "start" in empty.columns


def test_generators_give_standard_frames():
    for df in (datasets.make_timeseries(n_points=3), datasets.make_id_signal()):
        assert list(df.columns)[:7] == list(H.plant_frame().columns)[:2] + [
            "value_bool",
            "value_integer",
            "value_double",
            "value_string",
            "is_delta",
        ]
    assert inspect.isfunction(datasets.make_timeseries)
