"""Regression tests for the event-log robustness round.

Content-based event ids, object bindings, activity templates with missing
values, backwards intervals, boolean severity, static-shape timestamps,
user-supplied ``o2o`` / ``object_changes`` tables, object intervals and the
lambda-rule compiler / loader.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.eventlog import (
    OCEL_ACTIVITY,
    OCEL_EID,
    OCEL_OID,
    OCEL_TIMESTAMP,
    OCEL_TYPE,
    TS_SEVERITY,
    EventLog,
    ObjectSpec,
    RuleSpec,
    TriggerSpec,
    UnsafeExpression,
    clear_overrides,
    compile_expression,
    concat,
    has_override,
    link_events_to_objects,
    load_dicts,
    object_intervals,
    object_specs_from_metadata,
    register_adapter,
    register_lambda_rule,
    to_event_log,
    to_event_log_ocel,
    unregister_lambda_rule,
)
from ts_shape.eventlog.archetypes import ARCHETYPE_BY_METHOD
from ts_shape.eventlog.taxonomy import REGISTRY

POINT = "OutlierDetectionEvents.detect_outliers_zscore"
INTERVAL = "MachineStateEvents.detect_run_idle"
STATIC = "GaugeRepeatabilityEvents.repeatability"


def _points(source: str = "asset-A", n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=n, freq="1min", tz="UTC"),
            "uuid": ["m"] * n,
            "value_double": [1.0, 50.0, -30.0][:n],
            "is_delta": [False] * n,
            "source_uuid": [source] * n,
            "severity_score": [1.0, 4.0, 5.0][:n],
        }
    )


def _intervals() -> pd.DataFrame:
    start = pd.date_range("2026-05-07", periods=3, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "start": start,
            "end": start + pd.Timedelta("5min"),
            "uuid": ["m"] * 3,
            "source_uuid": ["asset-A"] * 3,
            "state": ["run", "idle", "run"],
            "duration_seconds": [300.0] * 3,
        }
    )


# ---------- 1. content-based event ids -------------------------------------


def test_eids_do_not_depend_on_row_position():
    df = _points()
    log_a = to_event_log(df, detector=POINT)
    log_b = to_event_log(df.iloc[::-1], detector=POINT)
    assert set(log_a.events[OCEL_EID]) == set(log_b.events[OCEL_EID])
    merged = concat(log_a, log_b)
    assert len(merged.events) == len(df)


def test_eids_are_deterministic_across_runs():
    df = _points()
    first = to_event_log(df, detector=POINT).events[OCEL_EID].tolist()
    second = to_event_log(df.copy(), detector=POINT).events[OCEL_EID].tolist()
    assert first == second


def test_same_detector_on_two_machines_does_not_collide():
    log_a = to_event_log(_points("machine-1"), detector=POINT)
    log_b = to_event_log(_points("machine-2"), detector=POINT)
    assert not set(log_a.events[OCEL_EID]) & set(log_b.events[OCEL_EID])
    assert len(concat(log_a, log_b).events) == 6


def test_eid_depends_on_values_not_dtype():
    df = _points()
    other = df.copy()
    other["value_double"] = other["value_double"].astype("Float64")
    a = to_event_log(df, detector=POINT).events[OCEL_EID].tolist()
    b = to_event_log(other, detector=POINT).events[OCEL_EID].tolist()
    assert a == b
    changed = df.copy()
    changed.loc[0, "value_double"] = 2.0
    c = to_event_log(changed, detector=POINT).events[OCEL_EID].tolist()
    assert c[0] != a[0] and c[1:] == a[1:]


def test_identical_rows_get_distinct_eids():
    df = pd.concat([_points(n=1)] * 2, ignore_index=True)
    log = to_event_log(df, detector=POINT)
    assert log.events[OCEL_EID].is_unique


# ---------- 2. object bindings ---------------------------------------------


def test_scalar_none_binds_no_object():
    log = to_event_log(_points(), detector=POINT, objects={"batch": None})
    assert "None" not in set(log.objects[OCEL_OID])
    assert "batch" not in set(log.objects[OCEL_TYPE])


def test_series_binding_aligns_by_index_label():
    df = _points().set_axis([10, 11, 12])
    binding = pd.Series(["c", "a", "b"], index=[12, 10, 11])
    log = to_event_log(df, detector=POINT, objects={"shift": binding})
    rel = log.relations[log.relations[OCEL_TYPE] == "shift"]
    by_eid = dict(zip(rel[OCEL_EID], rel[OCEL_OID]))
    ordered = [by_eid[e] for e in log.events.sort_values(OCEL_TIMESTAMP)[OCEL_EID]]
    assert ordered == ["a", "b", "c"]


@pytest.mark.parametrize(
    "binding",
    [pd.Series(["a", "b"]), pd.Series(["a", "b", "c"], index=[5, 6, 7])],
)
def test_misaligned_series_binding_raises(binding):
    with pytest.raises(ValueError, match="objects\\['shift'\\]"):
        to_event_log(_points(), detector=POINT, objects={"shift": binding})


# ---------- 3. templates, backwards intervals, severity, static ------------


def test_nan_template_field_renders_unknown_with_warning():
    df = _intervals()
    df.loc[1, "state"] = np.nan
    with pytest.warns(DataQualityWarning, match="unknown"):
        log = to_event_log(df, detector=INTERVAL)
    activities = log.events[OCEL_ACTIVITY].tolist()
    assert not any("nan" in a for a in activities)
    assert any("unknown" in a for a in activities)


def test_interval_ending_before_start_is_dropped():
    df = _intervals()
    df.loc[1, "end"] = df.loc[1, "start"] - pd.Timedelta("1min")
    with pytest.warns(DataQualityWarning, match="end before they start"):
        log = to_event_log(df, detector=INTERVAL)
    assert len(log.events) == 2
    assert set(log.relations[OCEL_EID]) == set(log.events[OCEL_EID])


def test_boolean_severity_maps_to_warn_and_info():
    df = _points()
    df["severity"] = [True, False, True]
    log = to_event_log(df, detector=POINT)
    by_time = log.events.sort_values(OCEL_TIMESTAMP)
    assert by_time[TS_SEVERITY].tolist() == ["warn", "info", "warn"]


def test_unknown_severity_warns_with_data_quality_warning():
    df = _points()
    df["severity"] = ["bogus", "info", "warn"]
    with pytest.warns(DataQualityWarning, match="severity"):
        to_event_log(df, detector=POINT)


def _static_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "part": ["A", "B"],
            "mean": [1.0, 1.1],
            "range": [0.1, 0.1],
            "repeatability_std": [0.05, 0.05],
            "EV": [0.15, 0.15],
        }
    )


def test_static_shape_uses_explicit_timestamp():
    ts = pd.Timestamp("2026-01-02 03:04", tz="Europe/Berlin")
    log = to_event_log(_static_frame(), detector=STATIC, timestamp=ts)
    assert (log.events[OCEL_TIMESTAMP] == ts.tz_convert("UTC")).all()


def test_static_shape_default_timestamp_is_fixed():
    log = to_event_log(_static_frame(), detector=STATIC)
    assert (log.events[OCEL_TIMESTAMP] == pd.Timestamp(0, tz="UTC")).all()


def test_naive_timestamps_are_read_as_utc():
    df = _points()
    df["systime"] = df["systime"].dt.tz_localize(None)
    log = to_event_log(df, detector=POINT)
    assert str(log.events[OCEL_TIMESTAMP].dt.tz) == "UTC"
    assert log.events[OCEL_TIMESTAMP].min() == pd.Timestamp("2026-05-07", tz="UTC")


# ---------- 4. user tables, pm4py export, override aliases -----------------


def test_object_changes_with_naive_timestamps_are_localized():
    changes = pd.DataFrame(
        {
            "ocel:oid": ["asset-A"],
            "ocel:type": ["asset"],
            "ocel:field": ["recipe"],
            "ocel:value": ["R1"],
            "ocel:timestamp": [pd.Timestamp("2026-05-07 00:00")],
        }
    )
    log = to_event_log(_points(), detector=POINT, object_changes=changes)
    assert log.object_changes[OCEL_TIMESTAMP].iloc[0] == pd.Timestamp(
        "2026-05-07", tz="UTC"
    )


def test_o2o_with_wrong_keys_raises_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        to_event_log(
            _points(),
            detector=POINT,
            o2o=[{"oid": "asset-A", "oid2": "line-1"}],
        )


def test_pm4py_object_changes_keep_field_named_like_base_column():
    changes = pd.DataFrame(
        {
            "ocel:oid": ["asset-A"],
            "ocel:type": ["asset"],
            "ocel:field": ["ocel:type"],
            "ocel:value": ["press"],
            "ocel:timestamp": [pd.Timestamp("2026-05-07", tz="UTC")],
        }
    )
    log = to_event_log(_points(), detector=POINT, object_changes=changes)
    with pytest.warns(DataQualityWarning, match="base column"):
        tables = to_event_log_ocel(log, pm4py=True)
    out = tables.object_changes
    assert out["ocel:type"].iloc[0] == "asset"
    assert out["ocel:type_field"].iloc[0] == "press"


def test_clear_overrides_resolves_class_aliases():
    @register_adapter("CycleTimeEvents", "cycle_time_statistics")
    def _custom(frame, **_):  # pragma: no cover - never called
        return EventLog()

    assert has_override("CycleTimeTracking", "cycle_time_statistics")
    clear_overrides("CycleTimeTracking", "cycle_time_statistics")
    assert not has_override("CycleTimeEvents", "cycle_time_statistics")

    register_adapter("CycleTimeEvents", "cycle_time_statistics")(_custom)
    clear_overrides("CycleTimeTracking")
    assert not has_override("CycleTimeEvents", "cycle_time_statistics")


# ---------- 5. objects ----------------------------------------------------


def _id_frame(values, minutes, column="value_string"):
    times = pd.Timestamp("2026-01-01") + pd.to_timedelta(minutes, unit="min")
    return pd.DataFrame({"systime": times, "uuid": "id", column: values})


def test_min_duration_merges_neighbours_before_filtering():
    df = _id_frame(["A", "X", "A", "B", "B"], [0, 20, 21, 90, 120])
    iv = object_intervals(df, [ObjectSpec("id", "batch", min_duration="30min")])
    assert iv["oid"].tolist() == ["A", "B"]
    assert iv["start"].iloc[0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert iv["end"].iloc[0] == pd.Timestamp("2026-01-01 01:30", tz="UTC")


def test_float_ids_render_as_integers_per_value():
    df = _id_frame([1.0, 1.5, np.nan, 2.0], [0, 10, 20, 30], column="value_double")
    iv = object_intervals(df, [ObjectSpec("id", "serial", value_column="value_double")])
    assert iv["oid"].tolist() == ["1", "1.5", "2"]


def test_link_events_by_float_key_column():
    intervals = pd.DataFrame({"oid": ["1", "2"], "type": ["batch", "batch"]})
    events = pd.DataFrame(
        {
            OCEL_EID: ["e1", "e2", "e3"],
            OCEL_TIMESTAMP: pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "batch": [1.0, np.nan, 2.0],
        }
    )
    rel = link_events_to_objects(
        events, intervals, key_columns=["batch"], contain=False
    )
    assert dict(zip(rel[OCEL_EID], rel[OCEL_OID])) == {"e1": "1", "e3": "2"}


def test_object_intervals_of_empty_untyped_frame_raises():
    with pytest.raises(ColumnNotFoundError):
        object_intervals(pd.DataFrame(), [ObjectSpec("id", "batch")])


def test_link_events_without_eid_raises():
    with pytest.raises(ColumnNotFoundError, match="ocel:eid"):
        link_events_to_objects(
            pd.DataFrame({OCEL_TIMESTAMP: []}),
            pd.DataFrame({"oid": [], "type": []}),
        )


def test_object_specs_from_metadata_validates_value_column():
    md = pd.DataFrame(
        {"uuid": ["s1"], "object_type": ["batch"], "object_value_column": ["vals"]}
    )
    with pytest.raises(ValueError, match="value column"):
        object_specs_from_metadata(md)


# ---------- 6. lambda rules -----------------------------------------------


@pytest.mark.parametrize(
    "expression, hint",
    [
        ("(x > 1) and (y < 0)", "&"),
        ("(x > 1) or (y < 0)", "|"),
        ("not (x > 1)", "~"),
        ("state in ['run']", "isin"),
        ("state not in ['run']", "isin"),
        ("max(x, y) > 1", "element-wise"),
        ("x > 1 & y == 2", "parentheses"),
        ("x * 2", "condition"),
    ],
)
def test_compile_expression_rejects_python_logic(expression, hint):
    with pytest.raises(UnsafeExpression, match=hint):
        compile_expression(expression)


def test_isin_is_supported():
    fn = compile_expression("state.isin(['run', 'setup']) & ~(x > 1)")
    df = pd.DataFrame({"state": ["run", "idle", "setup"], "x": [0, 0, 5]})
    assert fn(df).tolist() == [True, False, False]


def test_unknown_column_raises_column_not_found():
    fn = compile_expression("torque > 1")
    with pytest.raises(ColumnNotFoundError, match="torque"):
        fn(pd.DataFrame({"x": [1]}))


def test_non_boolean_result_raises_clear_error():
    fn = compile_expression("x")
    with pytest.raises(ValueError, match="True / False"):
        fn(pd.DataFrame({"x": [1.0, 2.0]}))


def _rule(method: str, **trigger) -> dict:
    return {
        "id": method,
        "class_name": "LambdaRobust",
        "method_name": method,
        "pack": "maintenance",
        "shape": "interval",
        "archetype": "interval",
        "template": "maintenance.robust.hot",
        "trigger": {"expression": "temp > 85", **trigger},
        "standard_attrs": {"ts_shape:lifecycle_state": "hot"},
    }


def test_load_dicts_accepts_group_by_string():
    try:
        (det,) = load_dicts([_rule("group_str", group_by="source_uuid")])
        assert det.spec.trigger.group_by == ("source_uuid",)
    finally:
        unregister_lambda_rule("LambdaRobust", "group_str")


def test_load_dicts_missing_expression_raises_value_error():
    rule = _rule("no_expr")
    rule["trigger"] = {"min_duration_s": 5}
    with pytest.raises(ValueError, match="expression"):
        load_dicts([rule])
    rule.pop("trigger")
    with pytest.raises(ValueError, match="trigger"):
        load_dicts([rule])


def test_load_dicts_is_all_or_nothing():
    good = _rule("batch_ok")
    bad = _rule("batch_bad")
    bad["trigger"] = {"expression": "temp > 85 and temp < 90"}
    with pytest.raises(UnsafeExpression):
        load_dicts([good, bad])
    assert ("LambdaRobust", "batch_ok") not in REGISTRY
    assert ("LambdaRobust", "batch_ok") not in ARCHETYPE_BY_METHOD


def test_load_dicts_rolls_back_on_duplicate_registration():
    first = _rule("dup_existing")
    try:
        load_dicts([first])
        with pytest.raises(ValueError, match="already registered"):
            load_dicts([_rule("dup_new"), first])
        assert ("LambdaRobust", "dup_new") not in REGISTRY
        assert ("LambdaRobust", "dup_existing") in REGISTRY
    finally:
        unregister_lambda_rule("LambdaRobust", "dup_existing")


def test_register_lambda_rule_compiles_expression():
    spec = RuleSpec(
        id="bad",
        class_name="LambdaRobust",
        method_name="bad_expr",
        pack="maintenance",
        shape="point",
        archetype="threshold",
        template="x.y.z",
        trigger=TriggerSpec(expression="(x > 1) and (y > 1)"),
        standard_attrs={"ts_shape:method": "m", "ts_shape:direction": "above"},
    )
    with pytest.raises(UnsafeExpression):
        register_lambda_rule(spec)
    assert ("LambdaRobust", "bad_expr") not in REGISTRY


def _hot_frame(temps, times=None) -> pd.DataFrame:
    if times is None:
        times = pd.date_range("2026-01-01", periods=len(temps), freq="10s", tz="UTC")
    return pd.DataFrame(
        {"systime": times, "temp": temps, "source_uuid": ["a"] * len(temps)}
    )


def test_interval_rule_is_half_open_and_stable_on_duplicate_times():
    try:
        (det,) = load_dicts([_rule("half_open")])
        out = det.evaluate(_hot_frame([80, 90, 91, 80, 80]))
        assert out["start"].iloc[0] == pd.Timestamp("2026-01-01 00:00:10", tz="UTC")
        assert out["end"].iloc[0] == pd.Timestamp("2026-01-01 00:00:30", tz="UTC")
        times = pd.to_datetime(
            ["2026-01-01 00:00:00", "2026-01-01 00:00:10", "2026-01-01 00:00:10"]
        ).tz_localize("UTC")
        a = det.evaluate(_hot_frame([90, 80, 90], times))
        b = det.evaluate(_hot_frame([90, 80, 90], times))
        pd.testing.assert_frame_equal(a, b)
    finally:
        unregister_lambda_rule("LambdaRobust", "half_open")


def test_interval_rule_with_no_hits_returns_typed_empty_frame():
    try:
        (det,) = load_dicts([_rule("no_hits")])
        out = det.evaluate(_hot_frame([10, 20, 30]))
        assert out.empty
        assert list(out.columns) == ["start", "end", "source_uuid", "sample_count"]
        assert str(out["start"].dt.tz) == "UTC"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(det.to_event_log(_hot_frame([10, 20]))) == 0
    finally:
        unregister_lambda_rule("LambdaRobust", "no_hits")


def test_unregister_lambda_rule_resolves_aliases():
    REGISTRY[("CycleTimeEvents", "_robust_probe")] = REGISTRY[next(iter(REGISTRY))]
    unregister_lambda_rule("CycleTimeTracking", "_robust_probe")
    assert ("CycleTimeEvents", "_robust_probe") not in REGISTRY
