"""Tests for the polish-batch additions:

* Static-shape eid idempotency.
* Severity literal pass-through warning + coercion.
* ``EventLog.filter_by_pack`` / ``filter_by_object`` after the merge refactor.
* ``concat`` dedup when the same eid appears in two inputs.
* ``pd.Series`` object bindings.
* ``describe()`` introspection helper.
* ``clear_overrides`` / ``has_override`` registry management.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from ts_shape.eventlog import (
    OCEL_ACTIVITY,
    OCEL_EID,
    OCEL_OID,
    OCEL_TYPE,
    TS_PACK,
    TS_SEVERITY,
    EventLog,
    LabelRule,
    clear_overrides,
    concat,
    describe,
    has_override,
    register_adapter,
    to_event_log,
)
from ts_shape.eventlog import schema as S
from ts_shape.eventlog.adapters import adapt
from ts_shape.eventlog.taxonomy import REGISTRY


# ---------- 1. Static-shape eid idempotency --------------------------------

def test_static_shape_eids_are_idempotent():
    """Re-running a static-shape detector on the same data must yield the
    same eids — the static shape should hash on a stable sentinel, not
    ``pd.Timestamp.now()``.
    """
    df = pd.DataFrame({
        "part": ["A", "B", "C"],
        "mean": [1.0, 1.1, 1.2],
        "range": [0.1, 0.1, 0.1],
        "repeatability_std": [0.05, 0.05, 0.05],
        "EV": [0.15, 0.15, 0.15],
    })
    log_a = to_event_log(df, detector="GaugeRepeatabilityEvents.repeatability")
    log_b = to_event_log(df, detector="GaugeRepeatabilityEvents.repeatability")
    assert list(log_a.events[OCEL_EID]) == list(log_b.events[OCEL_EID])


# ---------- 2. Severity literal pass-through -------------------------------

def test_severity_literal_outside_vocabulary_warns_and_coerces():
    """A literal severity column with values outside info/warn/critical
    must coerce to <NA> with a warning, not silently leak through."""
    df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=3, freq="1min", tz="UTC"),
        "value_double": [1.0, 1.0, 1.0],
        "uuid": ["m"]*3,
        "is_delta": [False]*3,
        "source_uuid": ["asset-A"]*3,
        # detect_outliers_zscore declares severity_field="severity_score";
        # we want to test the *literal* severity branch, so use a different
        # detector. tolerance.deviation reads severity_field="severity"
        # numerically — so use a custom rule instead.
    })
    rule = LabelRule(
        template="quality.test.literal_severity",
        pack="quality",
        shape="point",
    )
    df2 = df.assign(severity=["info", "extreme", "warn"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = adapt(df2, rule=rule, detector="X.y", objects=None, qualifiers=None)
    # 'extreme' is not in vocabulary → coerced to <NA>.
    assert log.events[TS_SEVERITY].tolist() == ["info", pd.NA, "warn"]
    assert any("severity column contains values outside" in str(w.message)
               for w in caught)


# ---------- 3. EventLog.filter_by_pack / filter_by_object ------------------

def _two_pack_log() -> EventLog:
    """Build a small log with one quality event and one production event
    sharing an asset object."""
    state_df = pd.DataFrame({
        "start": pd.to_datetime(["2026-05-07T10:00"], utc=True),
        "end":   pd.to_datetime(["2026-05-07T10:05"], utc=True),
        "uuid":         ["e"], "source_uuid": ["asset-A"], "is_delta": [False],
        "state":        ["run"], "duration_seconds": [300.0],
    })
    out_df = pd.DataFrame({
        "systime": pd.to_datetime(["2026-05-07T10:02"], utc=True),
        "value_double": [50.0], "uuid": ["sensor"],
        "is_delta": [False], "source_uuid": ["asset-A"],
        "severity_score": [4.0],
    })
    log_p = to_event_log(state_df, detector="MachineStateEvents.detect_run_idle")
    log_q = to_event_log(out_df, detector="OutlierDetectionEvents.detect_outliers_zscore")
    return concat(log_p, log_q)


def test_filter_by_pack_keeps_only_matching_events():
    log = _two_pack_log()
    only_q = log.filter_by_pack("quality")
    assert (only_q.events[TS_PACK] == "quality").all()
    assert len(only_q.events) == 1
    # Objects pruned to those reachable from filtered events.
    assert (only_q.objects[OCEL_OID] == "asset-A").all()


def test_filter_by_object_traces_all_events_for_an_asset():
    log = _two_pack_log()
    only_a = log.filter_by_object("asset-A", "asset")
    assert len(only_a.events) == 2
    assert {only_a.events[TS_PACK].iloc[0], only_a.events[TS_PACK].iloc[1]} == {
        "quality", "production",
    }


def test_filter_by_pack_with_no_matches_returns_empty():
    log = _two_pack_log()
    nothing = log.filter_by_pack("supplychain")
    assert len(nothing.events) == 0
    assert nothing.objects.empty


# ---------- 4. concat dedup -----------------------------------------------

def test_concat_deduplicates_identical_eids():
    """Concat'ing the same log twice must not double-count events."""
    df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=3, freq="1min", tz="UTC"),
        "value_double": [1.0, 50.0, 1.0],
        "uuid": ["m"]*3, "is_delta": [False]*3, "source_uuid": ["asset-A"]*3,
        "severity_score": [1.0, 4.0, 1.0],
    })
    log = to_event_log(df, detector="OutlierDetectionEvents.detect_outliers_zscore")
    doubled = concat(log, log)
    assert len(doubled.events) == len(log.events)


# ---------- 5. pd.Series object binding -----------------------------------

def test_series_object_binding_aligns_with_legacy_rows():
    """Passing a pd.Series as an object binding must align with the
    detector output one-for-one."""
    df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=3, freq="1min", tz="UTC"),
        "value_double": [1.0, 50.0, -30.0],
        "uuid": ["m"]*3, "is_delta": [False]*3, "source_uuid": ["asset-A"]*3,
        "severity_score": [1.0, 4.0, 5.0],
    })
    log = to_event_log(
        df,
        detector="OutlierDetectionEvents.detect_outliers_zscore",
        objects={"shift": pd.Series(["A", "B", "C"])},
    )
    shift_objs = log.objects[log.objects[OCEL_TYPE] == "shift"]
    assert set(shift_objs[OCEL_OID]) == {"A", "B", "C"}


# ---------- 6. describe() helper ------------------------------------------

def test_describe_returns_full_method_record():
    info = describe("OutlierDetectionEvents.detect_outliers_zscore")
    assert info["class_name"] == "OutlierDetectionEvents"
    assert info["method_name"] == "detect_outliers_zscore"
    assert info["pack"] == "quality"
    assert info["shape"] == "point"
    assert info["archetype"] == "threshold"
    assert info["activity_template"] == "quality.outlier.zscore"
    assert info["severity_field"] == "severity_score"
    assert info["produces_objects"] == ("asset",)
    assert "ts_shape:method" in info["standard_attrs"]
    assert info["has_override"] is False


def test_describe_unknown_detector_raises():
    with pytest.raises(KeyError):
        describe("DoesNotExist.do_thing")


def test_describe_bad_format_raises():
    with pytest.raises(ValueError):
        describe("missing_dot")


# ---------- 7. has_override / clear_overrides -----------------------------

def test_has_override_starts_false_for_real_detector():
    assert has_override("OutlierDetectionEvents", "detect_outliers_zscore") is False


def test_register_then_clear_specific():
    REGISTRY[("DummyDetector", "dummy")] = LabelRule(
        template="production.dummy", pack="production", shape="point",
        produces_objects=(),
        standard_attrs={"ts_shape:method": "dummy", "ts_shape:direction": "above"},
    )
    try:
        @register_adapter("DummyDetector", "dummy")
        def _stub(legacy_df, *, rule, detector, objects, qualifiers):
            return EventLog()

        assert has_override("DummyDetector", "dummy") is True
        clear_overrides("DummyDetector", "dummy")
        assert has_override("DummyDetector", "dummy") is False
    finally:
        REGISTRY.pop(("DummyDetector", "dummy"), None)


def test_clear_overrides_no_args_wipes_everything():
    REGISTRY[("DummyB", "m1")] = LabelRule(
        template="production.b1", pack="production", shape="point",
        produces_objects=(),
        standard_attrs={"ts_shape:method": "b1", "ts_shape:direction": "above"},
    )
    REGISTRY[("DummyB", "m2")] = LabelRule(
        template="production.b2", pack="production", shape="point",
        produces_objects=(),
        standard_attrs={"ts_shape:method": "b2", "ts_shape:direction": "above"},
    )
    try:
        @register_adapter("DummyB", "m1")
        def _a(legacy_df, *, rule, detector, objects, qualifiers):
            return EventLog()
        @register_adapter("DummyB", "m2")
        def _b(legacy_df, *, rule, detector, objects, qualifiers):
            return EventLog()
        assert has_override("DummyB", "m1") and has_override("DummyB", "m2")
        clear_overrides()
        assert not has_override("DummyB", "m1")
        assert not has_override("DummyB", "m2")
    finally:
        REGISTRY.pop(("DummyB", "m1"), None)
        REGISTRY.pop(("DummyB", "m2"), None)


def test_clear_overrides_class_only():
    REGISTRY[("DummyC", "m1")] = LabelRule(
        template="production.c1", pack="production", shape="point",
        produces_objects=(),
        standard_attrs={"ts_shape:method": "c1", "ts_shape:direction": "above"},
    )
    REGISTRY[("DummyC", "m2")] = LabelRule(
        template="production.c2", pack="production", shape="point",
        produces_objects=(),
        standard_attrs={"ts_shape:method": "c2", "ts_shape:direction": "above"},
    )
    try:
        @register_adapter("DummyC", "m1")
        def _a(legacy_df, *, rule, detector, objects, qualifiers):
            return EventLog()
        @register_adapter("DummyC", "m2")
        def _b(legacy_df, *, rule, detector, objects, qualifiers):
            return EventLog()
        clear_overrides("DummyC")
        assert not has_override("DummyC", "m1")
        assert not has_override("DummyC", "m2")
    finally:
        REGISTRY.pop(("DummyC", "m1"), None)
        REGISTRY.pop(("DummyC", "m2"), None)
