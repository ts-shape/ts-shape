"""Tests for to_flat_df (XES-style) and to_ocel_tables exporters."""
from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.eventlog import (
    EventLog,
    XES_ACTIVITY,
    XES_CASE,
    XES_LIFECYCLE,
    XES_TIMESTAMP,
    concat,
    to_event_log,
    to_flat_df,
    to_ocel_tables,
)
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents


@pytest.fixture()
def small_log():
    state_df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=20, freq="30s", tz="UTC"),
        "value_bool": [True]*5 + [False]*5 + [True]*5 + [False]*5,
        "uuid": ["asset-A"]*20,
        "is_delta": [True]+[False]*4 + [True]+[False]*4
                  + [True]+[False]*4 + [True]+[False]*4,
    })
    legacy_state = MachineStateEvents(state_df, run_state_uuid="asset-A").detect_run_idle()
    legacy_state["source_uuid"] = "asset-A"
    state_log = to_event_log(legacy_state, detector="MachineStateEvents.detect_run_idle")

    out_df = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=20, freq="1min", tz="UTC"),
        "value_double": [1.0]*8 + [50.0] + [1.0]*5 + [-30.0] + [1.0]*5,
        "uuid": ["asset-A"]*20,
        "is_delta": [False]*20,
        "source_uuid": ["asset-A"]*20,
    })
    legacy_out = OutlierDetectionEvents(out_df, value_column="value_double").detect_outliers_zscore()
    out_log = to_event_log(legacy_out, detector="OutlierDetectionEvents.detect_outliers_zscore")
    return concat(state_log, out_log)


def test_to_flat_df_single_lifecycle(small_log):
    flat = to_flat_df(small_log, case_object_type="asset", lifecycle="single")
    assert {XES_CASE, XES_ACTIVITY, XES_TIMESTAMP, XES_LIFECYCLE}.issubset(flat.columns)
    assert (flat[XES_CASE] == "asset-A").all()
    assert (flat[XES_LIFECYCLE] == "complete").all()
    activities = set(flat[XES_ACTIVITY])
    assert "production.machine_state.run" in activities
    assert "quality.outlier.zscore" in activities


def test_to_flat_df_two_row_lifecycle(small_log):
    flat = to_flat_df(small_log, case_object_type="asset", lifecycle="two_row")
    # Interval rows expand to start+complete; point rows stay as one row.
    assert (flat[XES_LIFECYCLE].isin(["start", "complete"])).all()
    # At least one start row exists (for the run/idle intervals).
    assert (flat[XES_LIFECYCLE] == "start").sum() > 0


def test_to_flat_df_unknown_object_type_raises(small_log):
    with pytest.raises(ValueError, match="no objects of type"):
        to_flat_df(small_log, case_object_type="batch")


def test_to_flat_df_no_objects_raises():
    log = EventLog()  # empty, no objects
    with pytest.raises(ValueError, match="requires objects"):
        to_flat_df(log)


def test_to_flat_df_invalid_lifecycle(small_log):
    with pytest.raises(ValueError, match="invalid lifecycle"):
        to_flat_df(small_log, lifecycle="weird")


def test_to_ocel_tables_returns_three_frames(small_log):
    events, objects, relations = to_ocel_tables(small_log)
    assert isinstance(events, pd.DataFrame)
    assert isinstance(objects, pd.DataFrame)
    assert isinstance(relations, pd.DataFrame)
    assert len(events) == len(small_log.events)
    assert len(objects) == len(small_log.objects)
    assert len(relations) == len(small_log.relations)
