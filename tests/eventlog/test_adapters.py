"""Per-shape adapter tests using representative real detectors."""
from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.eventlog import (
    OCEL_ACTIVITY,
    OCEL_EID,
    OCEL_OID,
    OCEL_TIMESTAMP,
    OCEL_TYPE,
    TS_DETECTOR,
    TS_DURATION_S,
    TS_PACK,
    TS_SEVERITY,
    TS_START_TIMESTAMP,
    TS_VALUE,
    to_event_log,
)
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents


# ---------- fixtures --------------------------------------------------------

@pytest.fixture()
def run_idle_df() -> pd.DataFrame:
    ts = pd.date_range("2026-05-07", periods=20, freq="30s", tz="UTC")
    return pd.DataFrame({
        "systime": ts,
        "value_bool": [True]*5 + [False]*5 + [True]*5 + [False]*5,
        "uuid": ["m"]*20,
        "is_delta": [True]+[False]*4 + [True]+[False]*4
                  + [True]+[False]*4 + [True]+[False]*4,
    })


@pytest.fixture()
def outlier_df() -> pd.DataFrame:
    return pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=20, freq="1min", tz="UTC"),
        "value_double": [1.0]*8 + [50.0] + [1.0]*5 + [-30.0] + [1.0]*5,
        "uuid": ["m"]*20,
        "is_delta": [False]*20,
        "source_uuid": ["asset-A"]*20,
    })


# ---------- interval shape (machine_state.detect_run_idle) ------------------

def test_interval_adapter_basic_shape(run_idle_df):
    legacy = MachineStateEvents(run_idle_df, run_state_uuid="m").detect_run_idle()
    log = to_event_log(legacy, detector="MachineStateEvents.detect_run_idle")

    assert len(log.events) == len(legacy)
    # Activity templated on the `state` field.
    activities = set(log.events[OCEL_ACTIVITY])
    assert {"production.machine_state.run", "production.machine_state.idle"} <= activities

    # Interval columns populated.
    assert log.events[TS_START_TIMESTAMP].notna().all()
    assert (log.events[TS_DURATION_S] > 0).all()

    # Pack/detector tags.
    assert (log.events[TS_PACK] == "production").all()
    assert (log.events[TS_DETECTOR] == "MachineStateEvents.detect_run_idle").all()

    # Object: source_uuid → asset binding.
    assert log.has_objects
    assert (log.objects[OCEL_TYPE] == "asset").all()


def test_interval_adapter_eids_unique(run_idle_df):
    legacy = MachineStateEvents(run_idle_df, run_state_uuid="m").detect_run_idle()
    log = to_event_log(legacy, detector="MachineStateEvents.detect_run_idle")
    assert log.events[OCEL_EID].is_unique


# ---------- point shape (outlier_zscore) ------------------------------------

def test_point_adapter_basic_shape(outlier_df):
    legacy = OutlierDetectionEvents(
        outlier_df, value_column="value_double"
    ).detect_outliers_zscore()
    log = to_event_log(legacy, detector="OutlierDetectionEvents.detect_outliers_zscore")

    assert len(log.events) == len(legacy)
    assert (log.events[OCEL_ACTIVITY] == "quality.outlier.zscore").all()
    assert log.events[TS_START_TIMESTAMP].isna().all()
    # severity_score got mapped to severity bucket (warn/critical).
    assert log.events[TS_SEVERITY].notna().all()
    # value got pulled from value_double automatically.
    assert log.events[TS_VALUE].notna().all()


# ---------- empty-input handling --------------------------------------------

def test_empty_input_returns_empty_log():
    empty = pd.DataFrame()
    log = to_event_log(empty, detector="MachineStateEvents.detect_run_idle")
    assert len(log) == 0
    assert not log.has_objects


# ---------- unknown detector ------------------------------------------------

def test_unknown_detector_raises():
    with pytest.raises(KeyError, match="no taxonomy entry"):
        to_event_log(pd.DataFrame(), detector="DoesNotExist.do_thing")


# ---------- bad detector spec ----------------------------------------------

def test_bad_detector_spec_raises():
    with pytest.raises(ValueError, match="ClassName.method_name"):
        to_event_log(pd.DataFrame(), detector="missing_dot")


# ---------- caller-supplied objects= ----------------------------------------

def test_caller_can_supply_contextual_object_types(outlier_df):
    """The adapter only auto-extracts ``asset`` from ``source_uuid``, but a
    caller can attach contextual types like ``batch`` via ``objects=``."""
    legacy = OutlierDetectionEvents(
        outlier_df, value_column="value_double"
    ).detect_outliers_zscore()
    legacy = legacy.assign(batch_id="B-2026-117")
    log = to_event_log(
        legacy,
        detector="OutlierDetectionEvents.detect_outliers_zscore",
        objects={"batch": "batch_id"},
        qualifiers={"asset": "produced_on", "batch": "during_batch"},
    )
    types = set(log.objects[OCEL_TYPE])
    assert {"asset", "batch"} <= types
    # qualifier propagated to relations.
    assert "during_batch" in set(log.relations["ocel:qualifier"].dropna())


# ---------- severity bucket mapping -----------------------------------------

def test_severity_bucket_thresholds():
    df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=3, freq="1min", tz="UTC"),
        "value_double": [1.0, 1.0, 1.0],
        "uuid": ["m", "m", "m"],
        "is_delta": [False, False, False],
        "source_uuid": ["asset-A"]*3,
        "severity_score": [1.0, 3.5, 5.0],
    })
    log = to_event_log(df, detector="OutlierDetectionEvents.detect_outliers_zscore")
    sev = log.events[TS_SEVERITY].tolist()
    assert sev == ["info", "warn", "critical"]


# ---------- NaN handling in templated activity names ------------------------

def test_templated_activity_renders_nan_as_unknown():
    """Templated columns with NaN values should render as 'unknown',
    not the literal pandas 'nan' string.
    """
    import numpy as np
    from ts_shape.eventlog.taxonomy import REGISTRY

    df = pd.DataFrame({
        "start": pd.to_datetime(["2026-05-07T10:00", "2026-05-07T10:05"], utc=True),
        "end":   pd.to_datetime(["2026-05-07T10:04", "2026-05-07T10:09"], utc=True),
        "uuid":         ["e1", "e2"],
        "source_uuid":  ["asset-A", "asset-A"],
        "is_delta":     [False, False],
        "state":        ["run", np.nan],
        "duration_seconds": [240.0, 240.0],
    })
    log = to_event_log(df, detector="MachineStateEvents.detect_run_idle")
    activities = log.events[OCEL_ACTIVITY].tolist()
    assert "production.machine_state.unknown" in activities
    assert "production.machine_state.nan" not in activities


# ---------- standard_attrs typo handling -----------------------------------

def test_standard_attrs_typo_in_numeric_source_warns():
    """A numeric standard attr (e.g. baseline) given an identifier-like
    string that doesn't match any column should warn, not silently
    broadcast the typo as a literal."""
    import warnings
    from ts_shape.eventlog import EventLog
    from ts_shape.eventlog.taxonomy import LabelRule
    from ts_shape.eventlog.adapters import adapt

    df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=3, freq="1min", tz="UTC"),
        "uuid": ["x"]*3,
        "is_delta": [False]*3,
        "source_uuid": ["asset-A"]*3,
    })
    rule = LabelRule(
        template="quality.outlier.test",
        pack="quality",
        shape="point",
        produces_objects=("asset",),
        standard_attrs={"ts_shape:baseline": "rolling_mean_typo"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = adapt(df, rule=rule, detector="X.y", objects=None, qualifiers=None)
    assert any("looks like a column name" in str(w.message) for w in caught)
    assert isinstance(log, EventLog)


def test_standard_attrs_string_literal_does_not_warn(outlier_df):
    """Plain literal strings like 'zscore' for ts_shape:method must NOT warn —
    string-typed standard attrs are commonly used for enum-like literals."""
    import warnings
    legacy = OutlierDetectionEvents(
        outlier_df, value_column="value_double"
    ).detect_outliers_zscore()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_event_log(legacy,
                     detector="OutlierDetectionEvents.detect_outliers_zscore")
    typo_warnings = [w for w in caught if "looks like a column name" in str(w.message)]
    assert not typo_warnings


# ---------- interval-shape fallback to point shape -------------------------

def test_interval_rule_falls_back_to_point_when_no_start_end_columns():
    """A rule declared `interval` should fall back to point-shape behaviour
    when the legacy DataFrame lacks `start`/`end` columns. No crash."""
    df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=3, freq="1min", tz="UTC"),
        "uuid": ["x"]*3,
        "is_delta": [False]*3,
        "source_uuid": ["asset-A"]*3,
        "lifecycle_state": ["chattering"]*3,
    })
    # AlarmManagementEvents.chattering_detection is declared `interval`
    # in the registry; here we feed it a point-shaped DataFrame.
    log = to_event_log(df, detector="AlarmManagementEvents.chattering_detection")
    assert len(log.events) == 3
    # All falls-back events get NaT for start_timestamp.
    assert log.events[TS_START_TIMESTAMP].isna().all()


# ---------- static shape time generation -----------------------------------

def test_static_shape_all_rows_share_timestamp():
    """Static shape uses pd.Timestamp.now(tz='UTC') broadcast — every row
    must share the same timestamp."""
    df = pd.DataFrame({
        "part": ["A", "B", "C"],
        "mean": [1.0, 1.1, 1.2],
        "range": [0.1, 0.1, 0.1],
        "repeatability_std": [0.05, 0.05, 0.05],
        "EV": [0.15, 0.15, 0.15],
    })
    log = to_event_log(df, detector="GaugeRepeatabilityEvents.repeatability")
    timestamps = log.events[OCEL_TIMESTAMP].unique()
    assert len(timestamps) == 1


# ---------- scalar object bindings -----------------------------------------

def test_scalar_object_binding_broadcasts(outlier_df):
    """A scalar value passed to objects= should broadcast to every row."""
    legacy = OutlierDetectionEvents(
        outlier_df, value_column="value_double"
    ).detect_outliers_zscore()
    log = to_event_log(
        legacy,
        detector="OutlierDetectionEvents.detect_outliers_zscore",
        objects={"shift": "A"},  # scalar broadcast
    )
    shift_objects = log.objects[log.objects[OCEL_TYPE] == "shift"]
    assert len(shift_objects) == 1
    assert shift_objects[OCEL_OID].iloc[0] == "A"
