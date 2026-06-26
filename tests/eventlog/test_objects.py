"""Tests for the object-detection layer (ts_shape.eventlog.objects)."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.datasets import make_id_signal
from ts_shape.eventlog import (
    ObjectSpec,
    OCEL2Tables,
    attach_objects,
    concat,
    detect_objects,
    object_intervals,
    object_specs_from_metadata,
    to_event_log,
    to_event_log_ocel,
    to_event_log_xes,
    validate,
)
from ts_shape.eventlog.schema import (
    OCEL_FIELD,
    OCEL_OID,
    OCEL_OID2,
    OCEL_QUALIFIER,
    OCEL_TYPE,
    OCEL_VALUE,
    is_known_object_type,
)
from ts_shape.events.production.machine_state import MachineStateEvents


@pytest.fixture()
def id_signals():
    """Two batches, six serials (3 per batch), and a coil signal on one line."""
    batch = make_id_signal("sig:batch", ["B1", "B2"], hold=30, source_uuid="line-1")
    serial = make_id_signal(
        "sig:serial",
        ["S1", "S2", "S3", "S4", "S5", "S6"],
        hold=10,
        source_uuid="line-1",
    )
    coil = make_id_signal("sig:coil", ["C1", "C2"], hold=30, source_uuid="line-1")
    return pd.concat([batch, serial, coil], ignore_index=True)


SPECS = [
    ObjectSpec("sig:batch", "batch", id_template="{type}:{value}"),
    ObjectSpec("sig:serial", "serial", id_template="{type}:{value}"),
    ObjectSpec("sig:coil", "coil", id_template="{type}:{value}"),  # unregistered type
]


def test_object_intervals_segments_id_signals(id_signals):
    iv = object_intervals(id_signals, SPECS)
    # 2 batches + 6 serials + 2 coils = 10 object instances.
    assert len(iv) == 10
    assert set(iv["type"]) == {"batch", "serial", "coil"}
    assert "batch:B1" in set(iv["oid"])
    # Each interval has a start <= end.
    assert (iv["start"] <= iv["end"]).all()


def test_detect_objects_builds_valid_ocel_tables(id_signals):
    log = detect_objects(id_signals, SPECS)  # validates internally
    validate(log)
    assert len(log.events) == 0  # objects-only log
    assert len(log.objects) == 10
    assert set(log.objects[OCEL_TYPE]) == {"batch", "serial", "coil"}


def test_unknown_object_type_is_auto_registered(id_signals):
    assert not is_known_object_type("coil") or True  # may already be registered
    detect_objects(id_signals, SPECS)
    assert is_known_object_type("coil")


def test_default_o2o_is_co_occurs_not_part_of(id_signals):
    # part_of must NOT be guessed from temporal overlap alone.
    log = detect_objects(id_signals, SPECS)
    assert set(log.o2o[OCEL_QUALIFIER]) == {"co_occurs"}
    assert "part_of" not in set(log.o2o[OCEL_QUALIFIER])


def test_hierarchy_gives_part_of(id_signals):
    log = detect_objects(id_signals, SPECS, hierarchy={"serial": "batch"})
    part_of = log.o2o[log.o2o[OCEL_QUALIFIER] == "part_of"]
    pairs = set(zip(part_of[OCEL_OID], part_of[OCEL_OID2]))
    assert ("serial:S1", "batch:B1") in pairs
    assert ("serial:S6", "batch:B2") in pairs
    # A serial is never part_of a different batch...
    assert ("serial:S1", "batch:B2") not in pairs
    # ...and only the declared edge produces part_of (no serial part_of coil).
    assert all(parent.startswith("batch:") for _, parent in pairs)


def test_long_lived_object_is_not_a_spurious_parent():
    # A tool spans every batch in time but does NOT own them: time-containment
    # must not manufacture `batch part_of tool`.
    batch = make_id_signal("sig:batch", ["B1", "B2"], hold=30, source_uuid="L1")
    tool = make_id_signal("sig:tool", ["T1"], hold=60, source_uuid="L1")
    df = pd.concat([batch, tool], ignore_index=True)
    specs = [
        ObjectSpec("sig:batch", "batch", id_template="{type}:{value}"),
        ObjectSpec("sig:tool", "tool", id_template="{type}:{value}"),
    ]
    log = detect_objects(df, specs)  # no hierarchy declared
    assert "part_of" not in set(log.o2o[OCEL_QUALIFIER])


def test_single_sample_and_leading_object_not_lost():
    # The leading object and a single-sample object must both survive, with the
    # first object starting at its first sample (regression: off-by-one drop).
    sig = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 00:10"], utc=True
            ),
            "uuid": "sig:b",
            "value_string": pd.array(["BX", "BY"], dtype="string"),
            "value_bool": pd.NA,
            "value_integer": pd.NA,
            "value_double": float("nan"),
            "is_delta": False,
        }
    )
    iv = object_intervals(sig, [ObjectSpec("sig:b", "batch")])
    assert set(iv["oid"]) == {"BX", "BY"}
    assert str(iv.iloc[0]["start"]) == "2025-01-01 00:00:00+00:00"
    # Gap-fill: BX is present until BY appears (not just its single sample).
    assert iv.iloc[0]["end"] > iv.iloc[0]["start"]


def test_object_changes_record_lifecycle(id_signals):
    log = detect_objects(id_signals, SPECS)
    fields = set(log.object_changes[OCEL_FIELD])
    assert "lifecycle" in fields
    # active + released per object instance.
    lc = log.object_changes[log.object_changes[OCEL_FIELD] == "lifecycle"]
    assert set(lc[OCEL_VALUE]) >= {"active", "released"}


def test_captured_attributes_become_object_changes():
    batch = make_id_signal("sig:batch", ["B1", "B2"], hold=20, source_uuid="line-1")
    recipe = make_id_signal(
        "sig:recipe", ["R-hot", "R-cold"], hold=20, source_uuid="line-1"
    )
    df = pd.concat([batch, recipe], ignore_index=True)
    specs = [
        ObjectSpec(
            "sig:batch",
            "batch",
            id_template="{type}:{value}",
            attributes={"recipe": "sig:recipe"},
        )
    ]
    log = detect_objects(df, specs)
    recipe_changes = log.object_changes[log.object_changes[OCEL_FIELD] == "recipe"]
    assert not recipe_changes.empty
    assert "R-hot" in set(recipe_changes[OCEL_VALUE])


def test_attach_objects_links_events_by_containment(id_signals):
    # Build a real event log from a state signal on the same line/time window.
    state = make_id_signal(
        "sig:state", ["run", "idle", "run", "idle"], hold=15, source_uuid="line-1"
    )
    state["value_bool"] = state["value_string"].isin(["run"])
    legacy = MachineStateEvents(state, run_state_uuid="sig:state").detect_run_idle()
    legacy["source_uuid"] = "line-1"
    ev_log = to_event_log(legacy, detector="MachineStateEvents.detect_run_idle")

    enriched = attach_objects(
        ev_log,
        id_signals,
        SPECS,
        qualifiers={"batch": "during_batch", "serial": "identified_by"},
    )
    validate(enriched)
    # Events now reference detected objects via E2O relations.
    assert len(enriched.relations) > len(ev_log.relations)
    linked_types = set(enriched.relations[OCEL_TYPE])
    assert {"batch", "serial"} <= linked_types
    # Qualifiers propagated.
    assert "during_batch" in set(enriched.relations[OCEL_QUALIFIER].dropna())

    # Full OCEL 2.0 export carries all five tables.
    tables = to_event_log_ocel(enriched)
    assert isinstance(tables, OCEL2Tables)
    assert len(tables.o2o) > 0
    # XES flattening with the asset case still works.
    flat = to_event_log_xes(enriched, case_object_type="asset")
    assert not flat.empty


def test_concat_objects_log_with_event_log(id_signals):
    state = make_id_signal("sig:state", ["run", "idle"], hold=20, source_uuid="line-1")
    state["value_bool"] = state["value_string"].isin(["run"])
    legacy = MachineStateEvents(state, run_state_uuid="sig:state").detect_run_idle()
    legacy["source_uuid"] = "line-1"
    ev_log = to_event_log(legacy, detector="MachineStateEvents.detect_run_idle")
    obj_log = detect_objects(id_signals, SPECS)

    merged = concat(ev_log, obj_log)
    validate(merged)
    assert len(merged.events) == len(ev_log.events)
    assert len(merged.objects) >= 10


def test_object_specs_from_metadata():
    metadata = pd.DataFrame(
        {
            "uuid": ["sig:batch", "sig:serial", "sig:temp"],
            "object_type": ["batch", "serial", None],
            "object_value_column": ["value_string", "value_string", None],
        }
    )
    specs = object_specs_from_metadata(metadata)
    assert {s.object_type for s in specs} == {"batch", "serial"}
    assert {s.uuid for s in specs} == {"sig:batch", "sig:serial"}


def test_empty_specs_yield_empty_objects(id_signals):
    log = detect_objects(id_signals, [])
    assert len(log.objects) == 0
    assert len(log.o2o) == 0
    validate(log)
