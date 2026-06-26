"""Unit tests for the EventLog query surface (ts_shape.eventlog.model).

The ``filter_by_*`` methods are how OCEL consumers slice a combined log into a
self-consistent sub-log (events + only the objects/relations/o2o/changes those
events use). Built here with ``relate`` so the sub-logs carry real relations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.eventlog import EventLog, relate, schema, validate


@pytest.fixture()
def intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oid": ["B1", "B2", "S1"],
            "type": ["batch", "batch", "serial"],
            "start": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 00:00"], utc=True
            ),
            "end": pd.to_datetime(
                ["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-01 00:30"], utc=True
            ),
        }
    )


@pytest.fixture()
def two_pack_log(intervals) -> EventLog:
    """A combined log: two quality events + one production event, with objects."""
    events = pd.DataFrame(
        {
            schema.OCEL_EID: ["q1", "q2", "p1"],
            schema.OCEL_ACTIVITY: ["outlier", "outlier", "downtime"],
            schema.OCEL_TIMESTAMP: pd.to_datetime(
                ["2026-01-01 00:15", "2026-01-01 00:20", "2026-01-01 01:30"], utc=True
            ),
            schema.TS_PACK: ["quality", "quality", "production"],
        }
    )
    return relate(EventLog(events=events), intervals, hierarchy={"serial": "batch"})


def test_len_repr_and_has_objects(two_pack_log):
    assert len(two_pack_log) == 3
    assert two_pack_log.has_objects
    r = repr(two_pack_log)
    assert r.startswith("EventLog(events=3, objects=")
    assert "relations=" in r and "o2o=" in r


def test_filter_by_pack_returns_consistent_sublog(two_pack_log):
    quality = two_pack_log.filter_by_pack("quality")
    assert set(quality.events[schema.OCEL_EID]) == {"q1", "q2"}
    # Sub-log keeps only relations for the retained events...
    assert set(quality.relations[schema.OCEL_EID]) <= {"q1", "q2"}
    # ...and only the objects those relations reference.
    used = set(quality.relations[schema.OCEL_OID])
    assert set(quality.objects[schema.OCEL_OID]) == used
    validate(quality)


def test_filter_by_pack_empty_for_absent_pack(two_pack_log):
    none = two_pack_log.filter_by_pack("energy")
    assert len(none) == 0
    assert not none.has_objects
    assert none.relations.empty and none.o2o.empty


def test_filter_by_object_narrows_events_and_relations(two_pack_log):
    # Only the production event p1 (01:30) falls in B2's window [01:00, 02:00].
    sub = two_pack_log.filter_by_object("B2")
    assert set(sub.events[schema.OCEL_EID]) == {"p1"}
    assert set(sub.objects[schema.OCEL_OID]) == {"B2"}


def test_filter_by_object_with_type_filter(two_pack_log):
    sub = two_pack_log.filter_by_object("B1", type_="batch")
    # q1 and q2 both fall inside B1's window.
    assert set(sub.events[schema.OCEL_EID]) == {"q1", "q2"}
    # A mismatched type yields no events.
    assert len(two_pack_log.filter_by_object("B1", type_="serial")) == 0


def test_empty_eventlog_defaults_are_valid():
    log = EventLog()
    assert len(log) == 0
    assert not log.has_objects
    validate(log)
