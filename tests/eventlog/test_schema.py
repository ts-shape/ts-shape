"""Schema-level tests: required columns, validation, empty frames."""

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
    TS_PACK,
    EventLog,
    validate,
)
from ts_shape.eventlog import schema as S


def test_empty_eventlog_validates():
    log = EventLog()
    validate(log)
    assert len(log) == 0
    assert not log.has_objects


def test_missing_required_column_raises():
    log = EventLog(events=pd.DataFrame({OCEL_EID: ["e-1"]}))
    with pytest.raises(ValueError, match="missing required columns"):
        validate(log)


def test_duplicate_eid_raises():
    events = S.empty_events()
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                {
                    OCEL_EID: ["e-1", "e-1"],
                    OCEL_ACTIVITY: ["a", "a"],
                    OCEL_TIMESTAMP: pd.to_datetime(
                        ["2026-01-01", "2026-01-02"], utc=True
                    ),
                    TS_DETECTOR: ["x", "x"],
                    TS_PACK: ["quality", "quality"],
                }
            ),
        ],
        ignore_index=True,
    )
    log = EventLog(events=events)
    with pytest.raises(ValueError, match="duplicate"):
        validate(log)


def test_relation_to_unknown_event_raises():
    relations = pd.DataFrame(
        {
            OCEL_EID: ["e-missing"],
            OCEL_OID: ["o-1"],
            OCEL_TYPE: ["asset"],
            S.OCEL_QUALIFIER: [pd.NA],
        }
    )
    log = EventLog(relations=relations)
    with pytest.raises(ValueError, match="unknown"):
        validate(log)


def test_register_object_type_rejects_invalid():
    with pytest.raises(ValueError):
        S.register_object_type("")
    with pytest.raises(ValueError):
        S.register_object_type("bad:name")


def test_register_object_type_adds_known():
    S.register_object_type("custom_type_xyz")
    assert S.is_known_object_type("custom_type_xyz")
