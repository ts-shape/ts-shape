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
    return pd.DataFrame(
        {
            "systime": ts,
            "value_bool": [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5,
            "uuid": ["m"] * 20,
            "is_delta": [True]
            + [False] * 4
            + [True]
            + [False] * 4
            + [True]
            + [False] * 4
            + [True]
            + [False] * 4,
        }
    )


@pytest.fixture()
def outlier_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=20, freq="1min", tz="UTC"),
            "value_double": [1.0] * 8 + [50.0] + [1.0] * 5 + [-30.0] + [1.0] * 5,
            "uuid": ["m"] * 20,
            "is_delta": [False] * 20,
            "source_uuid": ["asset-A"] * 20,
        }
    )


# ---------- interval shape (machine_state.detect_run_idle) ------------------


def test_interval_adapter_basic_shape(run_idle_df):
    legacy = MachineStateEvents(run_idle_df, run_state_uuid="m").detect_run_idle()
    log = to_event_log(legacy, detector="MachineStateEvents.detect_run_idle")

    assert len(log.events) == len(legacy)
    # Activity templated on the `state` field.
    activities = set(log.events[OCEL_ACTIVITY])
    assert {
        "production.machine_state.run",
        "production.machine_state.idle",
    } <= activities

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
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=3, freq="1min", tz="UTC"),
            "value_double": [1.0, 1.0, 1.0],
            "uuid": ["m", "m", "m"],
            "is_delta": [False, False, False],
            "source_uuid": ["asset-A"] * 3,
            "severity_score": [1.0, 3.5, 5.0],
        }
    )
    log = to_event_log(df, detector="OutlierDetectionEvents.detect_outliers_zscore")
    sev = log.events[TS_SEVERITY].tolist()
    assert sev == ["info", "warn", "critical"]
