"""Tests for to_event_log_xes (XES-style) and to_event_log_ocel exporters."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.eventlog import (
    OCEL_FIELD,
    OCEL_OID,
    OCEL_OID2,
    OCEL_QUALIFIER,
    OCEL_TIMESTAMP,
    OCEL_TYPE,
    OCEL_VALUE,
    XES_ACTIVITY,
    XES_CASE,
    XES_LIFECYCLE,
    XES_RESOURCE,
    XES_TIMESTAMP,
    EventLog,
    OCEL2Tables,
    concat,
    to_event_log,
    to_event_log_ocel,
    to_event_log_xes,
    validate,
)
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents


@pytest.fixture()
def small_log():
    state_df = pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=20, freq="30s", tz="UTC"),
            "value_bool": [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5,
            "uuid": ["asset-A"] * 20,
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
    legacy_state = MachineStateEvents(
        state_df, run_state_uuid="asset-A"
    ).detect_run_idle()
    legacy_state["source_uuid"] = "asset-A"
    state_log = to_event_log(
        legacy_state, detector="MachineStateEvents.detect_run_idle"
    )

    out_df = pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=20, freq="1min", tz="UTC"),
            "value_double": [1.0] * 8 + [50.0] + [1.0] * 5 + [-30.0] + [1.0] * 5,
            "uuid": ["asset-A"] * 20,
            "is_delta": [False] * 20,
            "source_uuid": ["asset-A"] * 20,
        }
    )
    legacy_out = OutlierDetectionEvents(
        out_df, value_column="value_double"
    ).detect_outliers_zscore()
    out_log = to_event_log(
        legacy_out, detector="OutlierDetectionEvents.detect_outliers_zscore"
    )
    return concat(state_log, out_log)


def test_to_event_log_xes_single_lifecycle(small_log):
    flat = to_event_log_xes(small_log, case_object_type="asset", lifecycle="single")
    assert {XES_CASE, XES_ACTIVITY, XES_TIMESTAMP, XES_LIFECYCLE}.issubset(flat.columns)
    assert (flat[XES_CASE] == "asset-A").all()
    assert (flat[XES_LIFECYCLE] == "complete").all()
    activities = set(flat[XES_ACTIVITY])
    assert "production.machine_state.run" in activities
    assert "quality.outlier.zscore" in activities


def test_to_event_log_xes_single_is_time_ordered(small_log):
    flat = to_event_log_xes(small_log, case_object_type="asset", lifecycle="single")
    # Within each case the trace must be ordered by timestamp (XES requirement).
    for _, trace in flat.groupby(XES_CASE):
        ts = trace[XES_TIMESTAMP].tolist()
        assert ts == sorted(ts)


def test_to_event_log_xes_two_row_lifecycle(small_log):
    flat = to_event_log_xes(small_log, case_object_type="asset", lifecycle="two_row")
    # Interval rows expand to start+complete; point rows stay as one row.
    assert (flat[XES_LIFECYCLE].isin(["start", "complete"])).all()
    # At least one start row exists (for the run/idle intervals).
    assert (flat[XES_LIFECYCLE] == "start").sum() > 0


def test_org_resource_absent_without_operator(small_log):
    # The log has only an `asset` object — org:resource must NOT be fabricated
    # from the case id.
    flat = to_event_log_xes(small_log, case_object_type="asset")
    assert XES_RESOURCE not in flat.columns


def _spiky_outlier_df():
    """20 flat points with one extreme spike — reliably flags a z-score outlier."""
    return pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=20, freq="1min", tz="UTC"),
            "value_double": [1.0] * 10 + [500.0] + [1.0] * 9,
            "uuid": ["asset-A"] * 20,
            "source_uuid": ["asset-A"] * 20,
        }
    )


def test_org_resource_from_operator_relation():
    df = _spiky_outlier_df()
    df["op"] = "alice"
    legacy = OutlierDetectionEvents(
        df, value_column="value_double"
    ).detect_outliers_zscore()
    log = to_event_log(
        legacy,
        detector="OutlierDetectionEvents.detect_outliers_zscore",
        objects={"asset": "source_uuid", "operator": "op"},
        qualifiers={"operator": "operated_by"},
    )
    assert len(log) > 0
    flat = to_event_log_xes(log, case_object_type="asset")
    assert XES_RESOURCE in flat.columns
    # Resource is the operator, never equal to the case id.
    assert (flat[XES_RESOURCE] == "alice").all()
    assert (flat[XES_RESOURCE] != flat[XES_CASE]).all()


def test_to_event_log_xes_unknown_object_type_raises(small_log):
    with pytest.raises(ValueError, match="no objects of type"):
        to_event_log_xes(small_log, case_object_type="batch")


def test_to_event_log_xes_no_objects_raises():
    log = EventLog()  # empty, no objects
    with pytest.raises(ValueError, match="requires objects"):
        to_event_log_xes(log)


def test_to_event_log_xes_invalid_lifecycle(small_log):
    with pytest.raises(ValueError, match="invalid lifecycle"):
        to_event_log_xes(small_log, lifecycle="weird")


def test_to_event_log_ocel_returns_five_tables(small_log):
    tables = to_event_log_ocel(small_log)
    assert isinstance(tables, OCEL2Tables)
    assert len(tables.events) == len(small_log.events)
    assert len(tables.objects) == len(small_log.objects)
    assert len(tables.relations) == len(small_log.relations)
    # O2O and object_changes default to empty frames with canonical columns.
    assert OCEL_OID2 in tables.o2o.columns
    assert {OCEL_FIELD, OCEL_VALUE}.issubset(tables.object_changes.columns)


def test_to_event_log_ocel_is_a_copy(small_log):
    tables = to_event_log_ocel(small_log)
    original = small_log.events[OCEL_TIMESTAMP].copy()
    tables.events.loc[:, OCEL_TIMESTAMP] = pd.NaT
    # Mutating the export must not touch the source log.
    pd.testing.assert_series_equal(small_log.events[OCEL_TIMESTAMP], original)


def test_o2o_and_object_changes_round_trip():
    df = _spiky_outlier_df()
    legacy = OutlierDetectionEvents(
        df, value_column="value_double"
    ).detect_outliers_zscore()
    o2o = [{OCEL_OID: "asset-A", OCEL_OID2: "line-1", OCEL_QUALIFIER: "part_of"}]
    changes = [
        {
            OCEL_OID: "asset-A",
            OCEL_TYPE: "asset",
            OCEL_FIELD: "firmware",
            OCEL_VALUE: "v2",
            OCEL_TIMESTAMP: pd.Timestamp("2026-05-07", tz="UTC"),
        }
    ]
    # 'line-1' must exist as an object for o2o to validate.
    log = to_event_log(
        legacy,
        detector="OutlierDetectionEvents.detect_outliers_zscore",
        objects={"asset": "source_uuid", "station": lambda r: "line-1"},
        o2o=o2o,
        object_changes=changes,
    )
    tables = to_event_log_ocel(log)
    assert tables.o2o.iloc[0][OCEL_OID2] == "line-1"
    assert tables.object_changes.iloc[0][OCEL_FIELD] == "firmware"


def test_validate_rejects_o2o_unknown_object(small_log):
    bad = EventLog(
        events=small_log.events,
        objects=small_log.objects,
        relations=small_log.relations,
        o2o=pd.DataFrame(
            {OCEL_OID: ["asset-A"], OCEL_OID2: ["ghost"], OCEL_QUALIFIER: ["x"]}
        ).astype("string"),
    )
    with pytest.raises(ValueError, match="o2o references unknown object"):
        validate(bad)
