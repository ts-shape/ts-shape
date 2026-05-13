"""Tests for the lambda-rule subsystem.

The lambda subsystem mutates the global REGISTRY / ARCHETYPE_BY_METHOD
dicts at registration time. Every test that registers a rule uses a
distinct ``(class_name, method_name)`` pair and explicitly cleans up
via :func:`unregister_lambda_rule`, so the rest of the eventlog test
suite (notably ``test_adapter_coverage.py``) sees a pristine REGISTRY.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.eventlog import (
    OCEL_ACTIVITY,
    OCEL_OID,
    OCEL_TYPE,
    TS_DETECTOR,
    TS_DURATION_S,
    TS_PACK,
    TS_SEVERITY,
    LambdaDetector,
    RuleSpec,
    TriggerSpec,
    UnsafeExpression,
    compile_expression,
    concat,
    load_dicts,
    register_lambda_rule,
    run_backtest,
    to_event_log,
    to_flat_df,
    to_ocel_tables,
    unregister_lambda_rule,
)
from ts_shape.eventlog.archetypes import ARCHETYPE_BY_METHOD
from ts_shape.eventlog.taxonomy import REGISTRY


# ---------- fixtures -------------------------------------------------------

@pytest.fixture()
def torque_df() -> pd.DataFrame:
    """A 10-row torque/state frame with 2 obvious overload events."""
    return pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=10, freq="1min", tz="UTC"),
        "torque": [60.0, 62.0, 70.0, 78.0, 80.0, 72.0, 50.0, 90.0, 65.0, 60.0],
        "state": ["run"] * 8 + ["idle", "idle"],
        "severity_score": [1.0, 1.0, 2.0, 3.5, 4.7, 2.0, 1.0, 4.9, 1.0, 1.0],
        "source_uuid": ["asset-A"] * 10,
    })


@pytest.fixture()
def bearing_df() -> pd.DataFrame:
    """Two-asset bearing-temperature stream with one short and one long run."""
    # asset-A: 4 contiguous >85 samples spaced 15s → 45s window (> min_duration_s=30) → kept
    # asset-A: 1 isolated >85 sample later → dropped
    # asset-B: 0 >85 samples → no events
    ts = pd.date_range("2026-05-07", periods=24, freq="15s", tz="UTC")
    df_a = pd.DataFrame({
        "systime": ts,
        "bearing_temp_c": [
            82, 83, 86, 88, 87, 86, 80, 82,  # rows 0-7: long hot window rows 2-5
            83, 84, 90, 83, 83, 82, 81, 80,  # rows 8-15: short hot at row 10
            83, 84, 82, 81, 80, 82, 83, 81,
        ],
        "source_uuid": ["asset-A"] * 24,
    })
    df_b = pd.DataFrame({
        "systime": ts,
        "bearing_temp_c": [80] * 24,
        "source_uuid": ["asset-B"] * 24,
    })
    return pd.concat([df_a, df_b], ignore_index=True)


# ---------- expression compiler -------------------------------------------

def test_compile_expression_rejects_unsafe():
    for expr in [
        "__import__('os').system('rm')",
        "open('/etc/passwd').read()",
        "x.__class__",
        "[i for i in range(10)]",
        "lambda x: x",
    ]:
        with pytest.raises(UnsafeExpression):
            compile_expression(expr)


def test_compile_expression_basic():
    # Bitwise & has higher precedence than comparison in Python, so parens
    # are required around each comparison — same convention as pandas.eval.
    fn = compile_expression("(x > 1) & (y < 0)")
    df = pd.DataFrame({"x": [0, 2, 3, 0], "y": [-1, -1, 1, -1]})
    mask = fn(df)
    assert list(mask) == [False, True, False, False]


def test_compile_expression_only_allowed_calls():
    fn = compile_expression("abs(x) > 2")
    df = pd.DataFrame({"x": [-3, -1, 1, 3]})
    assert list(fn(df)) == [True, False, False, True]
    with pytest.raises(UnsafeExpression):
        compile_expression("sum(x) > 2")


def test_compile_expression_nan_treated_as_false():
    fn = compile_expression("x > 1")
    df = pd.DataFrame({"x": [2.0, float("nan"), 0.0]})
    assert list(fn(df)) == [True, False, False]


# ---------- spec validation ------------------------------------------------

def test_rule_spec_requires_lambda_prefix():
    with pytest.raises(ValueError, match="class_name must start with 'Lambda'"):
        RuleSpec(
            id="bad", class_name="MyDetector", method_name="foo",
            pack="maintenance", shape="point", archetype="threshold",
            template="x.y.z",
            trigger=TriggerSpec(expression="x > 1"),
        )


def test_rule_spec_validates_pack_shape_archetype():
    with pytest.raises(ValueError, match="pack must be one of"):
        RuleSpec(
            id="x", class_name="LambdaX", method_name="m",
            pack="bogus", shape="point", archetype="threshold",
            template="x.y.z", trigger=TriggerSpec(expression="x > 1"),
        )


def test_rule_spec_interval_requires_interval_archetype():
    with pytest.raises(ValueError, match="archetype='interval'"):
        RuleSpec(
            id="x", class_name="LambdaX", method_name="m",
            pack="maintenance", shape="interval", archetype="threshold",
            template="x.y.z",
            trigger=TriggerSpec(expression="x > 1", min_duration_s=10),
        )


# ---------- registration --------------------------------------------------

def _spec_high_torque(class_name: str = "LambdaToolWear") -> RuleSpec:
    return RuleSpec(
        id="high_torque",
        class_name=class_name,
        method_name="high_torque",
        pack="maintenance",
        shape="point",
        archetype="threshold",
        template="maintenance.tool.high_torque",
        trigger=TriggerSpec(expression="(torque > 75)"),
        produces_objects=("asset",),
        severity_field="severity_score",
        value_field="torque",
        standard_attrs={
            "ts_shape:method": "lambda_threshold",
            "ts_shape:direction": "above",
            "ts_shape:threshold_high": 75.0,
        },
    )


def _spec_bearing_hot(class_name: str = "LambdaBearing") -> RuleSpec:
    return RuleSpec(
        id="bearing_hot",
        class_name=class_name,
        method_name="hot_window",
        pack="maintenance",
        shape="interval",
        archetype="interval",
        template="maintenance.bearing.hot",
        trigger=TriggerSpec(
            expression="bearing_temp_c > 85",
            min_duration_s=30,
            group_by=("source_uuid",),
        ),
        produces_objects=("asset",),
        value_field="bearing_temp_c",
        standard_attrs={
            "ts_shape:lifecycle_state": "hot",
        },
    )


def test_register_lambda_rule_writes_registry():
    spec = _spec_high_torque("LambdaReg1")
    try:
        det = register_lambda_rule(spec)
        assert isinstance(det, LambdaDetector)
        assert (spec.class_name, spec.method_name) in REGISTRY
        assert ARCHETYPE_BY_METHOD[(spec.class_name, spec.method_name)] == "threshold"
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)
    assert (spec.class_name, spec.method_name) not in REGISTRY


def test_register_lambda_rule_validates_archetype_attrs():
    bad = RuleSpec(
        id="missing_attrs",
        class_name="LambdaReg2",
        method_name="missing_attrs",
        pack="maintenance",
        shape="point",
        archetype="threshold",
        template="maintenance.x.y",
        trigger=TriggerSpec(expression="torque > 1"),
        standard_attrs={"ts_shape:method": "lambda_threshold"},  # direction missing
    )
    with pytest.raises(ValueError, match="missing required standard_attrs keys"):
        register_lambda_rule(bad)


def test_register_lambda_rule_rejects_duplicate():
    spec = _spec_high_torque("LambdaReg3")
    try:
        register_lambda_rule(spec)
        with pytest.raises(ValueError, match="already registered"):
            register_lambda_rule(spec)
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


def test_register_lambda_rule_rejects_unknown_standard_attrs():
    bad = RuleSpec(
        id="bad_attr",
        class_name="LambdaReg4",
        method_name="bad_attr",
        pack="maintenance",
        shape="point",
        archetype="threshold",
        template="maintenance.x.y",
        trigger=TriggerSpec(expression="x > 1"),
        standard_attrs={
            "ts_shape:method": "x", "ts_shape:direction": "above",
            "ts_shape:unknown_key": 1,
        },
    )
    with pytest.raises(ValueError, match="unknown keys"):
        register_lambda_rule(bad)


# ---------- end-to-end: point ---------------------------------------------

def test_point_rule_end_to_end(torque_df):
    spec = _spec_high_torque("LambdaPoint1")
    try:
        det = register_lambda_rule(spec)
        log = det.to_event_log(torque_df)

        # Three torque > 75 rows: indices 3 (78), 4 (80), 7 (90).
        assert len(log.events) == 3
        assert set(log.events[OCEL_ACTIVITY]) == {"maintenance.tool.high_torque"}
        assert set(log.events[TS_DETECTOR]) == {"LambdaPoint1.high_torque"}
        assert set(log.events[TS_PACK]) == {"maintenance"}
        # severity_score 3.5/4.7/4.9 → warn/critical/critical
        sev = list(log.events[TS_SEVERITY])
        assert sev.count("critical") == 2
        assert sev.count("warn") == 1
        # standard attrs columns landed.
        assert "ts_shape:method" in log.events.columns
        assert (log.events["ts_shape:method"] == "lambda_threshold").all()
        assert (log.events["ts_shape:threshold_high"] == 75.0).all()
        # source_uuid auto-extracted as asset.
        assert (log.relations[OCEL_TYPE] == "asset").all()
        assert set(log.relations[OCEL_OID]) == {"asset-A"}
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


# ---------- end-to-end: interval ------------------------------------------

def test_interval_rule_coalesces(bearing_df):
    spec = _spec_bearing_hot("LambdaInterval1")
    try:
        det = register_lambda_rule(spec)
        log = det.to_event_log(bearing_df)

        # asset-A long window (4 samples × 15s = 45s gap → 45s) kept;
        # asset-A short window (1 sample, 0s) dropped;
        # asset-B has no hot samples → no events.
        assert len(log.events) == 1
        ev = log.events.iloc[0]
        assert ev[OCEL_ACTIVITY] == "maintenance.bearing.hot"
        # Duration is 45s (4 samples spaced 15s apart → first to last = 45s).
        assert ev[TS_DURATION_S] == pytest.approx(45.0)
        assert ev["ts_shape:lifecycle_state"] == "hot"
        # Asset binding via source_uuid.
        assert set(log.relations[OCEL_OID]) == {"asset-A"}
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


def test_interval_rule_drops_runs_shorter_than_min_duration():
    df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=4, freq="5s", tz="UTC"),
        "bearing_temp_c": [86, 87, 88, 80],  # 3 samples × 5s = 15s window
        "source_uuid": ["asset-A"] * 4,
    })
    spec = _spec_bearing_hot("LambdaInterval2")
    try:
        det = register_lambda_rule(spec)
        log = det.to_event_log(df)
        # 15s window < min_duration_s=30 → dropped.
        assert len(log.events) == 0
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


# ---------- canonical-pipeline round-trips ---------------------------------

def test_lambda_log_passes_schema_validation(torque_df):
    """to_event_log validates by default; reaching the assert means OK."""
    spec = _spec_high_torque("LambdaSchema1")
    try:
        det = register_lambda_rule(spec)
        log = det.to_event_log(torque_df)
        assert len(log.events) > 0
        # validate=True is the default; an invalid log would have raised.
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


def test_lambda_log_round_trips_to_flat_and_ocel(torque_df, bearing_df):
    p_spec = _spec_high_torque("LambdaRound1")
    i_spec = _spec_bearing_hot("LambdaRound2")
    try:
        p_det = register_lambda_rule(p_spec)
        i_det = register_lambda_rule(i_spec)
        log = concat(p_det.to_event_log(torque_df), i_det.to_event_log(bearing_df))

        xes = to_flat_df(log, case_object_type="asset")
        assert not xes.empty
        assert "concept:name" in xes.columns

        events_df, objects_df, relations_df = to_ocel_tables(log)
        assert len(events_df) == len(log.events)
        assert len(relations_df) == len(log.relations)
    finally:
        unregister_lambda_rule(p_spec.class_name, p_spec.method_name)
        unregister_lambda_rule(i_spec.class_name, i_spec.method_name)


# ---------- coverage-test compatibility ------------------------------------

def test_adapter_coverage_orphan_check_exempts_lambdas(torque_df):
    """Registering a lambda rule must not break test_no_orphan_registry_entries."""
    from tests.eventlog.test_adapter_coverage import test_no_orphan_registry_entries

    spec = _spec_high_torque("LambdaCoverage1")
    try:
        register_lambda_rule(spec)
        # Should not raise — the test exempts class names starting with "Lambda".
        test_no_orphan_registry_entries()
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


def test_adapter_coverage_archetype_completeness_with_lambdas():
    from tests.eventlog.test_adapter_coverage import (
        test_archetype_assignment_is_complete,
    )

    spec = _spec_bearing_hot("LambdaCoverage2")
    try:
        register_lambda_rule(spec)
        test_archetype_assignment_is_complete()
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


# ---------- load_dicts (YAML-equivalent in-process) -----------------------

def test_load_dicts_registers_multiple_rules(torque_df, bearing_df):
    entries = [
        {
            "id": "high_torque",
            "class_name": "LambdaLoadDict1",
            "method_name": "high_torque",
            "pack": "maintenance",
            "shape": "point",
            "archetype": "threshold",
            "template": "maintenance.tool.high_torque",
            "trigger": {"expression": "torque > 75"},
            "standard_attrs": {
                "ts_shape:method": "lambda_threshold",
                "ts_shape:direction": "above",
            },
        },
        {
            "id": "bearing_hot",
            "class_name": "LambdaLoadDict2",
            "method_name": "hot_window",
            "pack": "maintenance",
            "shape": "interval",
            "archetype": "interval",
            "template": "maintenance.bearing.hot",
            "trigger": {
                "expression": "bearing_temp_c > 85",
                "min_duration_s": 30,
                "group_by": ["source_uuid"],
            },
            "standard_attrs": {"ts_shape:lifecycle_state": "hot"},
        },
    ]
    detectors = load_dicts(entries)
    try:
        assert len(detectors) == 2
        p_log = detectors[0].to_event_log(torque_df)
        i_log = detectors[1].to_event_log(bearing_df)
        merged = concat(p_log, i_log)
        assert len(merged.events) >= 4  # 3 torque + 1 bearing
    finally:
        unregister_lambda_rule("LambdaLoadDict1", "high_torque")
        unregister_lambda_rule("LambdaLoadDict2", "hot_window")


# ---------- backtest -------------------------------------------------------

def test_backtest_hit_count(torque_df):
    spec = _spec_high_torque("LambdaBacktest1")
    try:
        det = register_lambda_rule(spec)
        result = run_backtest(det, torque_df)
        assert result.hit_count == 3
        assert result.by_severity.get("critical", 0) == 2
        assert result.by_severity.get("warn", 0) == 1
        assert result.by_asset == {"asset-A": 3}
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)


# ---------- to_event_log dispatch via taxonomy ----------------------------

def test_to_event_log_resolves_lambda_via_registry(torque_df):
    """Confirms LambdaDetector goes through the same dispatch as built-ins."""
    spec = _spec_high_torque("LambdaDispatch1")
    try:
        det = register_lambda_rule(spec)
        legacy = det.evaluate(torque_df)
        # Direct call to the public to_event_log entry point.
        log = to_event_log(legacy, detector="LambdaDispatch1.high_torque")
        assert len(log.events) == 3
    finally:
        unregister_lambda_rule(spec.class_name, spec.method_name)
