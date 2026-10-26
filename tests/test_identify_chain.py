"""identify_objects into the event log: namespaced specs, attributes,
``hierarchy()`` and ``to_event_log``.

Hand-built frames use single-letter uuids, so nothing can pass by reading a
name; the end-to-end cases use the labelled benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.eventlog import (
    EventLog,
    detect_objects,
    object_intervals,
    to_event_log,
)
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.signals import identify_objects
from ts_shape.signals.identify import REPORT_COLUMNS

START = pd.Timestamp("2025-03-03 06:00")


def _frame(uuid, times, values, column="value_string"):
    return pd.DataFrame({"systime": times, "uuid": uuid, column: values})


def _orders(n=40, seed=3):
    """Sales orders held 30-50 min, the customer of each from a pool of four,
    a description text per order, and two signals on clocks of their own."""
    rng = np.random.default_rng(seed)
    holds = rng.integers(30, 50, n)
    customers = rng.choice([f"CUST-{101 + 13 * k:04d}" for k in range(9)], n)
    order = [f"SO-{2500100 + 7 * i}" for i in range(n)]
    seq = np.repeat(np.arange(n), holds)
    t = pd.date_range(START, periods=len(seq), freq="1min")
    words = ["bracket", "housing", "cover", "flange", "shaft"]
    text = [f"Order {order[i]} for {words[i % 5]} assembly, rev {i % 3}" for i in seq]
    tools = [f"T{1 + (k // 97) % 4:02d}" for k in range(len(seq))]
    return pd.concat(
        [
            _frame("o", t, [order[i] for i in seq]),
            _frame("c", t, [customers[i] for i in seq]),
            _frame("d", t, text),
            _frame("x", t, tools),
            _frame("m", t, 60 + rng.normal(0, 0.5, len(seq)), "value_double"),
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def orders():
    df = _orders()
    return df, identify_objects(df)


@pytest.fixture(scope="module")
def bench():
    df, labels = ts_shape.make_object_signals(seed=1)
    return df, labels, identify_objects(df)


def _parts():
    """Serials in batches of varying size, packed 24 to a box."""
    t = pd.date_range(START, periods=1170, freq="1min")
    lengths = [20, 90, 35, 120, 50, 25, 100, 40, 75, 30] * 2
    batch = [i for i, n in enumerate(lengths) for _ in range(n)]
    return pd.concat(
        [
            _frame("s", t, [f"SN{100000 + i}" for i in range(1170)]),
            _frame("b", t, [f"B250303-{b + 1:03d}" for b in batch]),
            _frame("k", t, [f"PK{7000 + i // 24}" for i in range(1170)]),
        ],
        ignore_index=True,
    )


# -- the table ---------------------------------------------------------------


def test_new_columns_are_appended_at_the_end(orders):
    _, report = orders
    assert REPORT_COLUMNS[-4:] == ["evidence", "attribute_of", "cycle_uuid", "uses"]
    assert list(report.table.columns) == REPORT_COLUMNS


def test_customer_is_an_attribute_of_its_order(orders):
    _, report = orders
    host = report.table.set_index("uuid")["attribute_of"]
    assert report["o"] is not None
    assert host["c"] == "o"
    assert host["d"] == "o"  # a description text, one per order
    # the pool of an order held for 30-50 min is its partner, not a recipe
    # or material (the CUST- prefix and the sales order name the customer)
    assert report["o"] == "sales_order"
    assert report["c"] == "customer", report.explain("c")


def test_independent_signals_are_no_attributes(orders):
    _, report = orders
    host = report.table.set_index("uuid")["attribute_of"]
    assert host["x"] is None  # a tool on its own clock
    assert host["m"] is None  # a measurement
    assert host["o"] is None  # the order is no attribute of its customer


def test_independent_ids_do_not_pair():
    rng = np.random.default_rng(11)
    frames = []
    for uuid, pool, mean in (("p", "R", 70), ("q", "C", 53)):
        holds = rng.integers(mean // 2, mean * 2, 40)
        values = [f"{pool}-{rng.integers(100, 105)}" for _ in holds]
        seq = [v for v, h in zip(values, holds, strict=True) for _ in range(h)]
        t = pd.date_range(START, periods=len(seq), freq="1min")
        frames.append(_frame(uuid, t, seq))
    sn = _frame(
        "s",
        pd.date_range(START, periods=3000, freq="1min"),
        [f"SN{500000 + i // 2}" for i in range(3000)],
    )
    report = identify_objects(pd.concat([*frames, sn], ignore_index=True))
    assert report.table["attribute_of"].isna().all()


def test_nesting_parent_is_no_attribute():
    report = identify_objects(_parts())
    table = report.table.set_index("uuid")
    assert table.loc["s", "parent_uuid"] == "b"
    assert table.loc["s", "packing_parent_uuid"] == "k"
    # a batch and a box are constant within each serial and change at serial
    # boundaries, but they are the serial's parents, not its attributes
    assert table["attribute_of"].isna().all()


def test_empty_frame_has_the_column():
    df = pd.DataFrame({"systime": pd.DatetimeIndex([]), "uuid": [], "value_string": []})
    report = identify_objects(df)
    assert list(report.table.columns) == REPORT_COLUMNS
    assert report.hierarchy() == {}


# -- to_object_specs ---------------------------------------------------------


def test_default_specs_unchanged(orders):
    _, report = orders
    for spec in report.to_object_specs(0.0):
        assert spec.id_template == "{value}"
        assert dict(spec.attributes) == {}


def test_namespaced_ids(orders):
    df, report = orders
    specs = report.to_object_specs(0.0, namespace=True)
    assert specs and all(s.id_template == "{type}:{value}" for s in specs)
    intervals = object_intervals(df, specs)
    for spec in specs:
        oids = intervals.loc[intervals["uuid"] == spec.uuid, "oid"]
        assert len(oids) and oids.str.startswith(f"{spec.object_type}:").all()


def test_attributes_on_the_host_spec(orders):
    df, report = orders
    specs = {s.uuid: s for s in report.to_object_specs(0.0, attributes=True)}
    attrs = dict(specs["o"].attributes)
    assert sorted(attrs.values()) == ["c", "d"]
    assert attrs["d"] == "d"  # not an identifier: named by its uuid
    name_c = next(k for k, v in attrs.items() if v == "c")
    assert name_c in ("c", str(report["c"]))
    log = detect_objects(df, list(specs.values()))
    orders_tab = log.objects[log.objects["ocel:type"] == specs["o"].object_type]
    assert len(orders_tab) == 40
    fields = set(log.object_changes["ocel:field"])
    assert name_c in fields and "d" in fields


# -- hierarchy ---------------------------------------------------------------


def test_hierarchy_from_nesting():
    report = identify_objects(_parts())
    assert report.hierarchy() == {"s": ["b", "k"]}
    assert report.hierarchy(min_confidence=1.01) == {}


def test_hierarchy_keeps_only_emitted_signals(bench):
    _, _, report = bench
    for threshold in (0.0, 0.6, 0.7):
        kept = {s.uuid for s in report.to_object_specs(threshold)}
        edges = report.hierarchy(threshold)
        assert set(edges) <= kept
        assert all(p in kept for parents in edges.values() for p in parents)
    table = report.table.set_index("uuid")
    for child, parents in report.hierarchy().items():
        expected = [
            p
            for p in table.loc[child, ["parent_uuid", "packing_parent_uuid"]]
            if p is not None
        ]
        assert parents == [p for p in expected if p in parents]
        assert parents[0] in expected


def test_hierarchy_gives_labelled_part_of(bench):
    df, labels, report = bench
    truth = labels.set_index("uuid")["parent_uuid"]
    edges = report.hierarchy()
    right = {c: ps[0] for c, ps in edges.items() if truth.get(c) == ps[0]}
    assert len(right) >= 2
    specs = report.to_object_specs(namespace=True)
    log = detect_objects(df, specs, hierarchy=edges)
    intervals = object_intervals(df, specs)
    signal = intervals.groupby("oid")["uuid"].agg(lambda s: set(s)).to_dict()
    part_of = log.o2o[log.o2o["ocel:qualifier"] == "part_of"]
    pairs = {
        (c, p)
        for a, b in zip(part_of["ocel:oid"], part_of["ocel:oid_2"], strict=True)
        for c in signal[a]
        for p in signal[b]
    }
    declared = {(c, p) for c, ps in edges.items() for p in ps}
    assert pairs <= declared
    for child, parent in right.items():
        assert (child, parent) in pairs
        # every child presence that overlaps a parent presence is part of
        # an object of the parent signal (the parent may have gaps)
        kids = intervals[intervals["uuid"] == child]
        held = intervals[intervals["uuid"] == parent]
        overlaps = [
            ((held["start"] < end) & (held["end"] > start)).any()
            for start, end in zip(kids["start"], kids["end"], strict=True)
        ]
        expected = set(kids["oid"][overlaps])
        parent_oids = set(held["oid"])
        linked = set(part_of["ocel:oid"][part_of["ocel:oid_2"].isin(parent_oids)])
        assert expected and expected <= linked


# -- to_event_log ------------------------------------------------------------


def test_to_event_log_objects_only(bench):
    df, _, report = bench
    log = report.to_event_log(df)
    assert isinstance(log, EventLog)
    assert log.has_objects and log.events.empty and log.relations.empty
    assert (log.o2o["ocel:qualifier"] == "part_of").any()
    assert log.objects["ocel:oid"].str.contains(":").all()
    fields = set(log.object_changes["ocel:field"]) - {"lifecycle"}
    assert fields  # the attributes are captured


def test_to_event_log_with_events():
    df = _parts()
    report = identify_objects(df)
    t = pd.date_range(START, periods=1170, freq="1min")
    running = (np.arange(1170) // 45) % 4 != 3
    state = pd.DataFrame({"systime": t, "uuid": "e", "value_bool": running})
    events = MachineStateEvents(state, "e").detect_run_idle()
    detector = "MachineStateEvents.detect_run_idle"
    log = report.to_event_log(df, events, detector=detector)
    assert isinstance(log, EventLog)
    assert len(log.events) == len(events)
    assert set(log.relations["ocel:type"]) >= {"serial", "batch", "handling_unit"}
    assert (log.o2o["ocel:qualifier"] == "part_of").sum() == 2 * 1170
    # the detector's own asset, plus the identified objects
    assert set(log.objects["ocel:type"]) == {
        "asset",
        "serial",
        "batch",
        "handling_unit",
    }
    # the same events as an EventLog, a mapping or a list give the same log
    base = to_event_log(events, detector=detector)
    for given in (base, {detector: events}, [base]):
        other = report.to_event_log(df, given)
        assert len(other.relations) == len(log.relations)
    scoped = report.to_event_log(
        df, base, scope={"e": "L1", "s": "L1", "b": "L1", "k": "L1"}
    )
    assert len(scoped.relations) == len(log.relations)
    plain = report.to_event_log(df, base, namespace=False)
    assert not plain.objects["ocel:oid"].str.startswith("serial:").any()


def test_to_event_log_errors():
    df = _parts()
    report = identify_objects(df)
    with pytest.raises(ValueError, match="scope"):
        report.to_event_log(df, scope={"s": "L1"})
    with pytest.raises(ValueError, match="detector"):
        report.to_event_log(df, pd.DataFrame())
    with pytest.raises(TypeError, match="EventLog"):
        report.to_event_log(df, 42)


def test_to_event_log_links_usage_and_bounds_serials_by_cycle():
    """Discrete line: report.to_event_log() carries cycle_uuid into the specs
    and uses() into the o2o (made_from to the held inputs)."""
    from ts_shape._object_signals import generate

    frame, labels = generate(
        seed=7,
        days=1,
        include=None,
        noise=False,
        discrete_cycles=True,
        constant_materials=2,
        lines=1,
    )
    report = identify_objects(frame)
    uses = report.uses()
    assert uses, "the discrete line has usage links"
    log = report.to_event_log(frame)
    o2o = log.o2o
    assert "made_from" in set(o2o["ocel:qualifier"])
    specs = {s.uuid: s for s in report.to_object_specs()}
    cycled = [u for u in uses if specs.get(u) is not None and specs[u].cycle_uuid]
    assert cycled, "serials get their cycle signal"
    # Every child with usage links has o2o rows of those qualifiers.
    for child, parents in uses.items():
        for qualifier in set(parents.values()):
            assert (o2o["ocel:qualifier"] == qualifier).any()
