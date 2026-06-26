"""Automated event<->object relationship derivation.

Covers the public layer that turns an *event list* + an *object list* into the
full OCEL 2.0 relational tables, with no access to the raw signals:
``link_events_to_objects`` (E2O), ``derive_o2o`` (O2O) and ``relate`` (assemble
a validated, ingestion-ready :class:`EventLog`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.eventlog import (
    EventLog,
    derive_o2o,
    link_events_to_objects,
    relate,
    schema,
    to_event_log_ocel,
    validate,
)


def _utc(*stamps: str) -> pd.DatetimeIndex:
    return pd.to_datetime(list(stamps), utc=True)


@pytest.fixture()
def intervals() -> pd.DataFrame:
    """Two batches and two serials with explicit presence windows."""
    return pd.DataFrame(
        {
            "oid": ["B1", "B2", "S1", "S2"],
            "type": ["batch", "batch", "serial", "serial"],
            "start": _utc(
                "2026-01-01 00:00",
                "2026-01-01 01:00",
                "2026-01-01 00:00",
                "2026-01-01 00:30",
            ),
            "end": _utc(
                "2026-01-01 01:00",
                "2026-01-01 02:00",
                "2026-01-01 00:30",
                "2026-01-01 01:00",
            ),
        }
    )


@pytest.fixture()
def events() -> pd.DataFrame:
    """Three events landing in different object windows."""
    return pd.DataFrame(
        {
            schema.OCEL_EID: ["e1", "e2", "e3"],
            schema.OCEL_ACTIVITY: ["inspect", "inspect", "inspect"],
            schema.OCEL_TIMESTAMP: _utc(
                "2026-01-01 00:15",  # in B1 + S1
                "2026-01-01 00:45",  # in B1 + S2
                "2026-01-01 01:30",  # in B2 only
            ),
            schema.TS_PACK: ["quality", "quality", "quality"],
        }
    )


# ---------------------------------------------------------------------------
# link_events_to_objects (E2O)
# ---------------------------------------------------------------------------


def test_containment_links_all_overlapping_objects(events, intervals):
    rel = link_events_to_objects(events, intervals)
    # e1->{B1,S1}, e2->{B1,S2}, e3->{B2} = 5 relations.
    assert len(rel) == 5
    by_eid = rel.groupby(schema.OCEL_EID)[schema.OCEL_OID].agg(set).to_dict()
    assert by_eid["e1"] == {"B1", "S1"}
    assert by_eid["e2"] == {"B1", "S2"}
    assert by_eid["e3"] == {"B2"}


def test_qualifiers_are_stamped_by_type(events, intervals):
    rel = link_events_to_objects(
        events, intervals, qualifiers={"batch": "processed_in"}
    )
    batch_rows = rel[rel[schema.OCEL_TYPE] == "batch"]
    serial_rows = rel[rel[schema.OCEL_TYPE] == "serial"]
    assert set(batch_rows[schema.OCEL_QUALIFIER]) == {"processed_in"}
    # Unmapped type -> NA qualifier.
    assert serial_rows[schema.OCEL_QUALIFIER].isna().all()


def test_key_column_matching_without_time(events, intervals):
    ev = events.copy()
    ev["batch"] = ["B1", "B2", "B1"]
    rel = link_events_to_objects(ev, intervals, key_columns=["batch"], contain=False)
    assert len(rel) == 3
    assert set(rel[schema.OCEL_OID]) == {"B1", "B2"}
    assert set(rel[schema.OCEL_TYPE]) == {"batch"}


def test_containment_and_key_are_unioned_and_deduped(events, intervals):
    ev = events.copy()
    ev["batch"] = ["B1", "B1", "B2"]  # same links containment already finds
    rel = link_events_to_objects(ev, intervals, key_columns=["batch"])
    # e1->{B1,S1}, e2->{B1,S2}, e3->{B2}; key match adds nothing new -> still 5.
    assert len(rel) == 5
    assert not rel.duplicated(
        subset=[schema.OCEL_EID, schema.OCEL_OID, schema.OCEL_TYPE]
    ).any()


def test_link_empty_inputs_return_empty_relations(events, intervals):
    empty_ev = link_events_to_objects(events.iloc[0:0], intervals)
    empty_iv = link_events_to_objects(events, intervals.iloc[0:0])
    assert empty_ev.empty and list(empty_ev.columns) == list(
        schema.empty_relations().columns
    )
    assert empty_iv.empty


def test_link_missing_required_columns_raises(events):
    bad = pd.DataFrame({"oid": ["B1"]})  # no 'type'
    with pytest.raises(ValueError, match="missing required column"):
        link_events_to_objects(events, bad)


# ---------------------------------------------------------------------------
# derive_o2o (O2O)
# ---------------------------------------------------------------------------


def test_derive_o2o_hierarchy_part_of(intervals):
    o2o = derive_o2o(intervals, hierarchy={"serial": "batch"})
    # S1 & S2 both overlap B1 -> two part_of edges (child -> parent).
    assert len(o2o) == 2
    assert set(o2o[schema.OCEL_QUALIFIER]) == {"part_of"}
    assert set(o2o[schema.OCEL_OID]) == {"S1", "S2"}
    assert set(o2o[schema.OCEL_OID2]) == {"B1"}


def test_derive_o2o_cooccurrence_without_hierarchy(intervals):
    o2o = derive_o2o(intervals)
    assert set(o2o[schema.OCEL_QUALIFIER]) == {"co_occurs"}
    # batch<->serial overlaps only (B1/S1, B1/S2); B2 overlaps neither serial.
    assert len(o2o) == 2


def test_derive_o2o_without_time_bounds_is_empty():
    iv = pd.DataFrame({"oid": ["B1", "S1"], "type": ["batch", "serial"]})
    assert derive_o2o(iv).empty


# ---------------------------------------------------------------------------
# relate (full assembly)
# ---------------------------------------------------------------------------


def test_relate_builds_full_ingestion_ready_log(events, intervals):
    log = relate(
        EventLog(events=events),
        intervals,
        hierarchy={"serial": "batch"},
        qualifiers={"batch": "processed_in"},
    )
    assert len(log.events) == 3
    assert set(log.objects[schema.OCEL_OID]) == {"B1", "B2", "S1", "S2"}
    assert len(log.relations) == 5
    assert len(log.o2o) == 2
    # lifecycle active/released per object = 2 * 4 objects.
    assert len(log.object_changes) == 8
    validate(log)  # raises if not OCEL-valid

    tables = to_event_log_ocel(log)
    assert len(tables.relations) == 5
    assert len(tables.o2o) == 2


def test_relate_infer_o2o_false_skips_o2o(events, intervals):
    log = relate(EventLog(events=events), intervals, infer_o2o=False)
    assert log.o2o.empty
    assert len(log.relations) == 5


def test_relate_merges_with_preexisting_objects(events, intervals):
    # An event log that already carries an object + relation.
    base = EventLog(
        events=events,
        objects=pd.DataFrame(
            {schema.OCEL_OID: ["W1"], schema.OCEL_TYPE: ["work_order"]}
        ).astype("string"),
    )
    log = relate(base, intervals)
    # Pre-existing object is preserved alongside the newly related ones.
    assert "W1" in set(log.objects[schema.OCEL_OID])
    assert {"B1", "B2", "S1", "S2"} <= set(log.objects[schema.OCEL_OID])
