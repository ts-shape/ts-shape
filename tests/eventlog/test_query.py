"""Genealogy queries (where_used / where_from / lineage) and coherent sub-logs."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from ts_shape.datasets import make_object_signals
from ts_shape.errors import ObjectNotFoundError
from ts_shape.eventlog import (
    EventLog,
    ObjectSpec,
    detect_objects,
    lineage,
    object_intervals,
    schema,
    where_from,
    where_used,
)

T0 = pd.Timestamp("2025-01-01 06:00", tz="UTC")


def _h(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


TYPES = {
    "M1": "material",
    "B1": "batch",
    "B2": "batch",
    "S1": "serial",
    "S2": "serial",
    "S3": "serial",
    "S4": "serial",
    "HU1": "handling_unit",
    "HU2": "handling_unit",
    "A1": "asset",
}
#: child -> parent; S4 was reworked from B1 into B2 (two batch parents).
EDGES = [
    ("B1", "M1"),
    ("B2", "M1"),
    ("S1", "B1"),
    ("S2", "B1"),
    ("S3", "B2"),
    ("S4", "B1"),
    ("S4", "B2"),
    ("S1", "HU1"),
    ("S2", "HU1"),
    ("S3", "HU2"),
    ("S4", "HU2"),
]
#: Presence in hours after T0; M1 and A1 have no lifecycle (presence unknown).
LIFE = {
    "B1": (0, 2),
    "B2": (2, 4),
    "S1": (0, 1),
    "S2": (1, 2),
    "S3": (2, 3),
    "S4": (1.5, 3.5),
    "HU1": (1, 2.5),
    "HU2": (3, 5),
}
EVENTS = {
    "e1": (0.5, ["S1", "B1", "A1"]),
    "e2": (1.75, ["S4", "B1", "HU1", "A1"]),
    "e3": (2.5, ["S3", "B2", "A1"]),
    "e4": (3.25, ["S4", "HU2", "A1"]),
}


def _genealogy() -> EventLog:
    objects = pd.DataFrame(
        {"ocel:oid": list(TYPES), "ocel:type": list(TYPES.values())}
    ).astype("string")
    o2o = pd.DataFrame(EDGES, columns=["ocel:oid", "ocel:oid_2"]).assign(
        **{"ocel:qualifier": "part_of"}
    )
    # A co_occurs edge is not genealogy and is never followed by default.
    o2o.loc[len(o2o)] = ["A1", "B1", "co_occurs"]
    changes = [
        (oid, TYPES[oid], "lifecycle", value, _h(at))
        for oid, (a, b) in LIFE.items()
        for value, at in (("active", a), ("released", b))
    ]
    changes += [
        ("B1", "batch", "recipe", "R1", _h(0)),
        ("B1", "batch", "recipe", "R2", _h(1)),
    ]
    events = pd.DataFrame(
        {
            "ocel:eid": list(EVENTS),
            "ocel:activity": "production.step",
            "ts_shape:detector": "Test.step",
            "ts_shape:pack": "production",
            "ocel:timestamp": [_h(t) for t, _ in EVENTS.values()],
        }
    )
    relations = pd.DataFrame(
        [
            (eid, oid, TYPES[oid], "involves")
            for eid, (_, oids) in EVENTS.items()
            for oid in oids
        ],
        columns=["ocel:eid", "ocel:oid", "ocel:type", "ocel:qualifier"],
    )
    log = EventLog(
        events=events,
        objects=objects,
        relations=relations,
        o2o=o2o.astype("string"),
        object_changes=pd.DataFrame(
            changes,
            columns=[
                "ocel:oid",
                "ocel:type",
                "ocel:field",
                "ocel:value",
                "ocel:timestamp",
            ],
        ),
    )
    schema.validate(log)
    return log


@pytest.fixture
def log() -> EventLog:
    return _genealogy()


def _expect(
    rows: list[tuple[str, str | None, str | None, int, str | None]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["oid", "object_type", "qualifier", "depth", "via"]
    )
    return frame.astype(
        {
            "oid": "string",
            "object_type": "string",
            "qualifier": "string",
            "depth": "int64",
            "via": "string",
        }
    )


def _check(got: pd.DataFrame, rows: list) -> None:
    pd.testing.assert_frame_equal(got, _expect(rows))


# ---------------------------------------------------------------------------
# where_used / where_from / lineage
# ---------------------------------------------------------------------------


def test_where_used_material_reaches_every_serial(log):
    _check(
        where_used(log, "M1"),
        [
            ("B1", "batch", "part_of", 1, "M1"),
            ("B2", "batch", "part_of", 1, "M1"),
            ("S1", "serial", "part_of", 2, "B1"),
            ("S2", "serial", "part_of", 2, "B1"),
            ("S3", "serial", "part_of", 2, "B2"),
            # Two parents at the same depth: one row per edge.
            ("S4", "serial", "part_of", 2, "B1"),
            ("S4", "serial", "part_of", 2, "B2"),
        ],
    )


def test_where_used_handling_unit_and_method(log):
    _check(
        log.where_used("HU2"),
        [
            ("S3", "serial", "part_of", 1, "HU2"),
            ("S4", "serial", "part_of", 1, "HU2"),
        ],
    )
    pd.testing.assert_frame_equal(log.where_used("M1"), where_used(log, "M1"))


def test_where_from_reworked_serial_has_two_batches(log):
    _check(
        where_from(log, "S4"),
        [
            ("B1", "batch", "part_of", 1, "S4"),
            ("B2", "batch", "part_of", 1, "S4"),
            ("HU2", "handling_unit", "part_of", 1, "S4"),
            ("M1", "material", "part_of", 2, "B1"),
            ("M1", "material", "part_of", 2, "B2"),
        ],
    )
    pd.testing.assert_frame_equal(log.where_from("S4"), where_from(log, "S4"))


def test_depth_limits(log):
    _check(
        where_used(log, "M1", depth=1),
        [
            ("B1", "batch", "part_of", 1, "M1"),
            ("B2", "batch", "part_of", 1, "M1"),
        ],
    )
    _check(where_from(log, "S1", depth=0), [])
    assert where_used(log, "M1", depth=5).equals(where_used(log, "M1"))
    with pytest.raises(ValueError, match="depth"):
        where_used(log, "M1", depth=-1)


def test_lineage_signed_depth(log):
    _check(
        lineage(log, "B1"),
        [
            ("M1", "material", "part_of", -1, "B1"),
            ("B1", "batch", None, 0, None),
            ("S1", "serial", "part_of", 1, "B1"),
            ("S2", "serial", "part_of", 1, "B1"),
            ("S4", "serial", "part_of", 1, "B1"),
        ],
    )
    _check(
        log.lineage("S1", depth=1),
        [
            ("B1", "batch", "part_of", -1, "S1"),
            ("HU1", "handling_unit", "part_of", -1, "S1"),
            ("S1", "serial", None, 0, None),
        ],
    )


def test_time_window_uses_object_presence(log):
    # B2 is present from 08:00, outside [06:00, 07:45); M1 has no lifecycle
    # rows (presence unknown) and is traversed.
    _check(
        where_used(log, "M1", start=_h(0), end=_h(1.75)),
        [
            ("B1", "batch", "part_of", 1, "M1"),
            ("S1", "serial", "part_of", 2, "B1"),
            ("S2", "serial", "part_of", 2, "B1"),
            ("S4", "serial", "part_of", 2, "B1"),
        ],
    )
    # Touching the window at the release instant is not an overlap: B1 is
    # released at 08:00.
    _check(
        where_from(log, "S4", start=_h(2)),
        [
            ("B2", "batch", "part_of", 1, "S4"),
            ("HU2", "handling_unit", "part_of", 1, "S4"),
            ("M1", "material", "part_of", 2, "B2"),
        ],
    )
    # Naive bounds are UTC; a start object absent in the window reaches nothing.
    _check(where_used(log, "HU1", start="2025-01-01 09:00"), [])
    with pytest.raises(ValueError, match="empty time window"):
        where_used(log, "M1", start=_h(2), end=_h(1))


def test_lineage_window_excludes_absent_start(log):
    _check(lineage(log, "S1", start=_h(3)), [])


def test_qualifiers(log):
    _check(
        where_used(log, "B1", qualifiers="co_occurs"),
        [("A1", "asset", "co_occurs", 1, "B1")],
    )
    assert "A1" not in set(where_used(log, "B1")["oid"])
    _check(where_used(log, "M1", qualifiers=["contains"]), [])


def test_cycle_terminates():
    o2o = pd.DataFrame(
        {
            "ocel:oid": ["X", "Y", "Z", "Z"],
            "ocel:oid_2": ["Y", "Z", "X", "Z"],
            "ocel:qualifier": "part_of",
        }
    )
    cyc = EventLog(o2o=o2o)
    _check(
        where_from(cyc, "X"),
        [(n, None, "part_of", d, v) for n, d, v in [("Y", 1, "X"), ("Z", 2, "Y")]],
    )
    _check(
        lineage(cyc, "X"),
        [
            ("Z", None, "part_of", -2, "Y"),
            ("Y", None, "part_of", -1, "X"),
            ("X", None, None, 0, None),
            ("Z", None, "part_of", 1, "X"),
            ("Y", None, "part_of", 2, "Z"),
        ],
    )


def test_extra_o2o_columns_pass_through():
    o2o = pd.DataFrame(
        {
            "ocel:oid": ["S1", "B1"],
            "ocel:oid_2": ["B1", "M1"],
            "ocel:qualifier": "part_of",
            "since": [_h(1), _h(0)],
        }
    )
    got = where_from(EventLog(o2o=o2o), "S1")
    assert list(got.columns) == [
        "oid",
        "object_type",
        "qualifier",
        "depth",
        "via",
        "since",
    ]
    assert list(got["since"]) == [_h(1), _h(0)]


def test_unknown_oid_raises(log):
    with pytest.raises(ObjectNotFoundError, match="'nope' is not in the event log"):
        where_used(log, "nope")
    with pytest.raises(KeyError):
        log.lineage("nope")
    # An object without edges is known: empty result.
    _check(where_used(log, "A1"), [])


def test_empty_log():
    with pytest.raises(ObjectNotFoundError):
        where_from(EventLog(), "X")


# ---------------------------------------------------------------------------
# filter_by_object_type / filter_by_time
# ---------------------------------------------------------------------------


def _assert_consistent(sub: EventLog) -> None:
    schema.validate(sub)
    keys = set(zip(sub.objects["ocel:oid"], sub.objects["ocel:type"], strict=True))
    oids = set(sub.objects["ocel:oid"])
    assert set(sub.relations["ocel:eid"]) <= set(sub.events["ocel:eid"])
    assert (
        set(zip(sub.relations["ocel:oid"], sub.relations["ocel:type"], strict=True))
        <= keys
    )
    assert set(sub.o2o["ocel:oid"]) | set(sub.o2o["ocel:oid_2"]) <= oids
    ch = sub.object_changes
    assert set(zip(ch["ocel:oid"], ch["ocel:type"], strict=True)) <= keys


def test_filter_by_object_type_linked(log):
    sub = log.filter_by_object_type(["serial", "batch"])
    _assert_consistent(sub)
    assert sorted(sub.objects["ocel:oid"]) == ["B1", "B2", "S1", "S2", "S3", "S4"]
    assert list(sub.events["ocel:eid"]) == ["e1", "e2", "e3", "e4"]
    assert set(sub.relations["ocel:type"]) == {"serial", "batch"}
    assert len(sub.o2o) == 5  # serial -> batch edges only
    assert set(sub.object_changes["ocel:type"]) == {"serial", "batch"}


def test_filter_by_object_type_keep_events(log):
    sub = log.filter_by_object_type("handling_unit")
    _assert_consistent(sub)
    assert list(sub.events["ocel:eid"]) == ["e2", "e4"]
    assert sub.o2o.empty
    every = log.filter_by_object_type("handling_unit", keep_events="all")
    _assert_consistent(every)
    assert list(every.events["ocel:eid"]) == ["e1", "e2", "e3", "e4"]
    assert len(every.relations) == 2
    none = log.filter_by_object_type("unknown_type")
    _assert_consistent(none)
    assert none.objects.empty and none.events.empty
    with pytest.raises(ValueError, match="keep_events"):
        log.filter_by_object_type("serial", keep_events="some")  # type: ignore[arg-type]


def test_filter_by_time(log):
    sub = log.filter_by_time(_h(1.5), _h(3))
    _assert_consistent(sub)
    assert list(sub.events["ocel:eid"]) == ["e2", "e3"]
    # S1 left at 07:00 and HU2 arrives at 09:00 (= end, exclusive); M1 and A1
    # have no lifecycle and stay.
    assert sorted(sub.objects["ocel:oid"]) == sorted(
        ["M1", "B1", "B2", "S2", "S3", "S4", "HU1", "A1"]
    )
    assert sorted(map(tuple, sub.o2o[["ocel:oid", "ocel:oid_2"]].to_numpy())) == [
        ("A1", "B1"),  # the co_occurs edge: both ends kept
        ("B1", "M1"),
        ("B2", "M1"),
        ("S2", "B1"),
        ("S2", "HU1"),
        ("S3", "B2"),
        ("S4", "B1"),
        ("S4", "B2"),
    ]
    ch = sub.object_changes
    # The window's changes, plus each field's last value before 07:30.
    b1 = ch[ch["ocel:oid"] == "B1"]
    assert list(zip(b1["ocel:field"], b1["ocel:value"], strict=True)) == [
        ("lifecycle", "active"),
        ("lifecycle", "released"),
        ("recipe", "R2"),
    ]
    assert not (ch["ocel:timestamp"] >= _h(3)).any()
    assert len(ch[ch["ocel:oid"] == "HU1"]) == 2


def test_filter_by_time_keeps_linked_objects_and_open_bounds(log):
    whole = log.filter_by_time()
    _assert_consistent(whole)
    assert len(whole.events) == 4 and len(whole.objects) == len(log.objects)
    late = log.filter_by_time(start=_h(3))
    _assert_consistent(late)
    assert list(late.events["ocel:eid"]) == ["e4"]
    assert {"S4", "HU2", "A1"} <= set(late.objects["ocel:oid"])
    assert "S1" not in set(late.objects["ocel:oid"])
    with pytest.raises(ValueError, match="empty time window"):
        log.filter_by_time(_h(3), _h(3))


def test_filtered_log_supports_genealogy(log):
    sub = log.filter_by_time(_h(2), _h(4))
    _check(
        sub.where_from("S4"),
        [
            ("B2", "batch", "part_of", 1, "S4"),
            ("HU2", "handling_unit", "part_of", 1, "S4"),
            ("M1", "material", "part_of", 2, "B2"),
        ],
    )


# ---------------------------------------------------------------------------
# End to end: the object-signal benchmark with its labelled nesting
# ---------------------------------------------------------------------------


def test_benchmark_serials_trace_back_to_their_batch():
    frame, labels = make_object_signals(seed=1, days=1)
    ids = labels[labels["role"] == "identifier"]
    specs = [
        ObjectSpec(
            r.uuid,
            r.object_type,
            value_column=r.value_column,
            id_template="{type}:{value}",
        )
        for r in ids.itertuples()
    ]
    # Signal-level hierarchy from the generator's ground truth.
    hierarchy = {
        r.uuid: r.parent_uuid for r in ids.itertuples() if r.parent_uuid is not None
    }
    serial_signals = {
        u
        for u, p in hierarchy.items()
        if u in set(ids[ids["object_type"] == "serial"]["uuid"])
    }
    assert serial_signals
    log = detect_objects(frame, specs, hierarchy=hierarchy)
    signal_of = (
        object_intervals(frame, specs).drop_duplicates("oid").set_index("oid")["uuid"]
    )
    serials = log.objects[log.objects["ocel:type"] == "serial"]["ocel:oid"]
    sample = list(serials.iloc[:: max(len(serials) // 40, 1)])
    reached = 0
    for serial in sample:
        got = log.where_from(serial)
        batches = got[got["object_type"] == "batch"]
        if batches.empty:
            continue
        reached += 1
        assert (batches["depth"] == 1).all()
        # The batch comes from the serial's labelled parent signal ...
        assert {signal_of[b] for b in batches["oid"]} == {hierarchy[signal_of[serial]]}
        # ... and the batch's material is one step further.
        material = got[got["object_type"] == "material"]
        assert not material.empty and (material["depth"] == 2).all()
        down = log.where_used(batches["oid"].iloc[0])
        assert serial in set(down["oid"])
    assert reached >= 0.9 * len(sample)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


@pytest.mark.scale
def test_scale_100k_edges():
    n_batches, n_serials = 100, 100_000
    batches = [f"B{i}" for i in range(n_batches)]
    serials = [f"S{i}" for i in range(n_serials)]
    hus = [f"H{i}" for i in range(n_serials // 10)]
    child = batches + serials + serials
    parent = (
        ["M"] * n_batches
        + [batches[i % n_batches] for i in range(n_serials)]
        + [hus[i // 10] for i in range(n_serials)]
    )
    o2o = pd.DataFrame(
        {"ocel:oid": child, "ocel:oid_2": parent, "ocel:qualifier": "part_of"}
    ).astype("string")
    oids = ["M", *batches, *serials, *hus]
    types = (
        ["material"]
        + ["batch"] * n_batches
        + ["serial"] * n_serials
        + ["hu"] * len(hus)
    )
    objects = pd.DataFrame({"ocel:oid": oids, "ocel:type": types}).astype("string")
    start = T0 + pd.to_timedelta(np.arange(len(oids)), unit="s")
    changes = pd.concat(
        [
            pd.DataFrame(
                {
                    "ocel:oid": oids,
                    "ocel:type": types,
                    "ocel:field": "lifecycle",
                    "ocel:value": value,
                    "ocel:timestamp": start + pd.Timedelta(hours=offset),
                }
            )
            for value, offset in (("active", 0), ("released", 1))
        ],
        ignore_index=True,
    )
    log = EventLog(objects=objects, o2o=o2o, object_changes=changes)
    assert len(o2o) > 100_000

    t = time.perf_counter()
    used = log.where_used("M")
    came = log.where_from("S5")
    tree = log.lineage("B3")
    window = log.where_used("M", start=T0, end=T0 + pd.Timedelta(hours=2))
    elapsed = time.perf_counter() - t

    assert len(used) == n_batches + n_serials
    assert set(came["oid"]) == {"B5", "H0", "M"}
    assert (tree["depth"] > 0).sum() == n_serials // n_batches
    assert len(window) == 2 * 3600 - 1  # objects active in the first two hours
    assert elapsed < 4.0, f"genealogy queries took {elapsed:.2f}s"
