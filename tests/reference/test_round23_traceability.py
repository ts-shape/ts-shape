"""Round 23: traceability leftovers, hand-computed references.

``RoutingTraceabilityEvents`` and ``MultiProcessTraceabilityEvents`` (ids,
handovers, ``source_uuid``). Times are minutes after 08:00 (naive). Each test
names the former result it replaces.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.eventlog import to_event_log
from ts_shape.events.production import (
    MultiProcessTraceabilityEvents,
    RoutingTraceabilityEvents,
    ValueTraceabilityEvents,
)

T0 = pd.Timestamp("2026-01-01 08:00")
PROCESSES = [
    {"id_uuid": "a", "station": "A"},
    {"id_uuid": "b", "station": "B"},
]


def _t(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def _numeric(rows) -> pd.DataFrame:
    """``rows``: ``(minutes, uuid, value_double, value_integer)``."""
    return pd.DataFrame(
        [
            {"systime": _t(m), "uuid": u, "value_double": v, "value_integer": i}
            for m, u, v, i in rows
        ]
    )


def _text(rows) -> pd.DataFrame:
    """``rows``: ``(minutes, uuid, value_string, value_integer)``."""
    return pd.DataFrame(
        {
            "systime": [_t(m) for m, *_ in rows],
            "uuid": [u for _, u, _, _ in rows],
            "value_string": pd.array([s for _, _, s, _ in rows], dtype="object"),
            "value_integer": pd.array([i for *_, i in rows], dtype="Int64"),
        }
    )


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        return fn(*args, **kwargs)


def _serials(log) -> list[str]:
    rel = log.relations
    return rel.loc[rel["ocel:type"] == "serial", "ocel:oid"].tolist()


# ---------------------------------------------------------------------------
# 1. Integral numeric ids render without ".0"
# ---------------------------------------------------------------------------

#: Item 1001 (a float id signal) at A [0, 5] and B [10, 15]; the handover
#: B -> C rises at 13.
NUMERIC = [
    (0, "a", 1001.0, None),
    (5, "a", 1001.0, None),
    (10, "b", 1001.0, None),
    (15, "b", 1001.0, None),
    (12, "ho", None, 0),
    (13, "ho", None, 1),
]
NUMERIC_HANDOVERS = [{"uuid": "ho", "from_station": "B", "to_station": "C"}]


def _routing_numeric() -> RoutingTraceabilityEvents:
    # Item 1001 at step 1 (A) at 0, then step 2 (B) at 5; one non-integral
    # id 7.5 at step 1 at 10.
    df = _numeric(
        [
            (0, "id", 1001.0, None),
            (0, "st", None, 1),
            (5, "st", None, 2),
            (10, "id", 7.5, None),
            (10, "st", None, 1),
        ]
    )
    return RoutingTraceabilityEvents(
        df, "id", "st", state_map={1: "A", 2: "B"}, value_column_id="value_double"
    )


def test_routing_numeric_ids_render_as_integers():
    trace = _routing_numeric()
    tl = trace.build_routing_timeline()
    # Former: ["1001.0", "1001.0", "7.5"].
    assert tl["item_id"].tolist() == ["1001", "1001", "7.5"]
    assert tl["station_name"].tolist() == ["A", "B", "A"]
    # The column stays text (the dtype pandas gives a column of str).
    assert tl["item_id"].dtype == pd.DataFrame({"x": ["s"]})["x"].dtype
    lead = trace.lead_time()
    assert lead["item_id"].tolist() == ["1001", "7.5"]  # former "1001.0"
    assert lead["routing_path"].tolist() == ["A -> B", "A"]
    iv = trace.to_intervals()
    assert iv["oid"].tolist() == ["1001", "1001", "7.5"]  # former "1001.0"
    assert str(iv["oid"].dtype) == "string"
    paths = trace.routing_paths()
    assert sorted(paths["routing_path"]) == ["A", "A -> B"]
    assert paths["item_count"].tolist() == [1, 1]


def test_routing_numeric_and_text_ids_merge_into_one_visit():
    # 1001 read as a float, then as the text "1001": one item, one visit
    # (former: "1001.0" and "1001", two visits).
    df = pd.DataFrame(
        {
            "systime": [_t(0), _t(0), _t(5), _t(5)],
            "uuid": ["id", "st", "id", "st"],
            "value_string": pd.array([1001.0, None, "1001", None], dtype="object"),
            "value_integer": pd.array([None, 1, None, 1], dtype="Int64"),
        }
    )
    tl = RoutingTraceabilityEvents(
        df, "id", "st", state_map={1: "A"}
    ).build_routing_timeline()
    assert tl["item_id"].tolist() == ["1001"]
    assert tl["sample_count"].tolist() == [2]


def test_multi_handover_numeric_id_renders_as_integer():
    trace = MultiProcessTraceabilityEvents(
        _numeric(NUMERIC),
        PROCESSES,
        handovers=NUMERIC_HANDOVERS,
        value_column_id="value_double",
    )
    log = trace.handover_log()
    # The item read at B before 13 is 1001. Former: "1001.0".
    assert log["item_id"].tolist() == ["1001"]
    # The timeline keeps the float id (its dtype is unchanged).
    tl = trace.build_timeline()
    assert tl["item_id"].tolist() == [1001.0, 1001.0]
    assert tl["item_id"].dtype == "float64"
    assert trace.to_intervals()["oid"].tolist() == ["1001", "1001"]


@pytest.mark.parametrize(
    ("kind", "method"),
    [
        ("multi", "build_timeline"),
        ("multi", "lead_time"),
        ("multi", "handover_log"),
        ("multi", "to_intervals"),
        ("value", "build_timeline"),
        ("value", "lead_time"),
        ("value", "to_intervals"),
    ],
)
def test_event_log_serial_oids_have_no_trailing_zero(kind, method):
    df = _numeric(NUMERIC)
    if kind == "multi":
        trace = MultiProcessTraceabilityEvents(
            df, PROCESSES, handovers=NUMERIC_HANDOVERS, value_column_id="value_double"
        )
        det = f"MultiProcessTraceabilityEvents.{method}"
    else:
        trace = ValueTraceabilityEvents(
            df[df["uuid"] != "ho"], {"a": "A", "b": "B"}, value_column="value_double"
        )
        det = f"ValueTraceabilityEvents.{method}"
    log = to_event_log(getattr(trace, method)(), detector=det)
    # Former: "1001.0" for the timeline, lead time and handover logs (the
    # to_intervals logs already had "1001").
    assert set(_serials(log)) == {"1001"}


#: Event ids of the NUMERIC logs before round 23 (frozen): the float id
#: columns render as before in the id, and ``source_uuid`` is not in it.
FROZEN_NUMERIC_IDS = {
    "MultiProcessTraceabilityEvents.build_timeline": [
        "e-303e7f35-82ea-5938-a1d6-0d991982fd9b",
        "e-14864290-8288-529c-95d6-65df55691ba0",
    ],
    "MultiProcessTraceabilityEvents.lead_time": [
        "e-acd08bb3-e52b-51e8-be89-cb072a31dd4e"
    ],
    "ValueTraceabilityEvents.build_timeline": [
        "e-328db6b2-14af-5889-a1e0-9fed382904cb",
        "e-24a734cb-41dc-54b8-8367-3de53ac591ea",
    ],
}


@pytest.mark.parametrize("detector", sorted(FROZEN_NUMERIC_IDS))
def test_numeric_id_event_ids_unchanged(detector):
    df = _numeric(NUMERIC)
    cls, method = detector.split(".")
    if cls == "MultiProcessTraceabilityEvents":
        trace = MultiProcessTraceabilityEvents(
            df, PROCESSES, handovers=NUMERIC_HANDOVERS, value_column_id="value_double"
        )
    else:
        trace = ValueTraceabilityEvents(
            df[df["uuid"] != "ho"], {"a": "A", "b": "B"}, value_column="value_double"
        )
    log = to_event_log(getattr(trace, method)(), detector=detector)
    assert log.events["ocel:eid"].tolist() == FROZEN_NUMERIC_IDS[detector]


# ---------------------------------------------------------------------------
# 2. A station name shared by parallel cells
# ---------------------------------------------------------------------------

#: Cell w1 of "Weld" holds X (0, 3), cell w2 holds Y (1, 4); the handover
#: ho1 Weld -> Paint rises at 5.
SHARED = [
    (0, "w1", "X", None),
    (1, "w2", "Y", None),
    (3, "w1", "X", None),
    (4, "w2", "Y", None),
    (0, "ho1", None, 0),
    (5, "ho1", None, 1),
]
WELD = [
    {"id_uuid": "w1", "station": "Weld"},
    {"id_uuid": "w2", "station": "Weld"},
]


def _shared_log(**handover):
    ho = {"uuid": "ho1", "from_station": "Weld", "to_station": "Paint", **handover}
    trace = MultiProcessTraceabilityEvents(_text(SHARED), WELD, handovers=[ho])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = trace.handover_log()
    return log, [w for w in caught if issubclass(w.category, DataQualityWarning)]


@pytest.mark.parametrize(("cell", "item"), [("w1", "X"), ("w2", "Y")])
def test_handover_reads_the_named_cell(cell, item):
    log, caught = _shared_log(from_id_uuid=cell)
    # Former (no way to name the cell): Y for both, w2's reading at 4 being
    # the latest of any Weld cell.
    assert log["item_id"].tolist() == [item]
    assert caught == []


def test_handover_names_several_cells():
    log, caught = _shared_log(from_id_uuid=["w1", "w2"])
    # The latest reading of the two named cells: w2's Y at 4.
    assert log["item_id"].tolist() == ["Y"]
    assert caught == []


def test_shared_station_without_a_cell_warns():
    log, caught = _shared_log()
    assert log["item_id"].tolist() == ["Y"]  # as before
    assert len(caught) == 1
    assert "'Weld'" in str(caught[0].message)
    assert "from_id_uuid" in str(caught[0].message)


def test_single_cell_station_does_not_warn():
    ho = {"uuid": "ho1", "from_station": "Weld", "to_station": "Paint"}
    trace = MultiProcessTraceabilityEvents(_text(SHARED), WELD[:1], handovers=[ho])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        log = trace.handover_log()
    assert log["item_id"].tolist() == ["X"]


@pytest.mark.parametrize("cells", ["b", ["w1", "b"], []])
def test_from_id_uuid_must_name_cells_of_the_station(cells):
    ho = {
        "uuid": "ho1",
        "from_station": "Weld",
        "to_station": "Paint",
        "from_id_uuid": cells,
    }
    with pytest.raises(ValueError, match="from_id_uuid"):
        _quiet(
            MultiProcessTraceabilityEvents,
            _text(SHARED),
            [*WELD, {"id_uuid": "b", "station": "B"}],
            handovers=[ho],
        )


# ---------------------------------------------------------------------------
# 3. A handover from a station not in ``processes``
# ---------------------------------------------------------------------------


def test_handover_from_unknown_station_warns_once():
    handovers = [
        {"uuid": "ho1", "from_station": "Nowhere", "to_station": "Paint"},
        {"uuid": "ho1", "from_station": "Nowhere", "to_station": "Weld"},
        {"uuid": "ho1", "from_station": "Weld", "to_station": "Paint"},
    ]
    trace = MultiProcessTraceabilityEvents(_text(SHARED), WELD[:1], handovers=handovers)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = trace.handover_log()
    dq = [w for w in caught if issubclass(w.category, DataQualityWarning)]
    # Former: no warning. One warning per call, naming the station once.
    assert len(dq) == 1
    message = str(dq[0].message)
    assert message.count("'Nowhere'") == 1
    assert "not in processes" in message
    # The other handover is unchanged: X from w1 before 5.
    assert log["item_id"].tolist() == ["X"]
    assert log["from_station"].tolist() == ["Weld"]


# ---------------------------------------------------------------------------
# 4. source_uuid on the MultiProcess interval and point outputs
# ---------------------------------------------------------------------------

#: X at a [0, 20] and, in parallel, at b [5, 10]; Y at b [30, 40]. The
#: handover h b -> C rises at 35 (Y).
PARALLEL = [
    (0, "a", "X", None),
    (20, "a", "X", None),
    (5, "b", "X", None),
    (10, "b", "X", None),
    (30, "b", "Y", None),
    (40, "b", "Y", None),
    (34, "h", None, 0),
    (35, "h", None, 1),
]
PARALLEL_HANDOVERS = [{"uuid": "h", "from_station": "B", "to_station": "C"}]


def _parallel() -> MultiProcessTraceabilityEvents:
    return MultiProcessTraceabilityEvents(
        _text(PARALLEL), PROCESSES, handovers=PARALLEL_HANDOVERS
    )


#: The columns before round 23, in order; ``source_uuid`` is appended.
FORMER_COLUMNS = {
    "build_timeline": [
        "item_id",
        "station",
        "id_uuid",
        "start",
        "end",
        "duration_seconds",
        "sample_count",
        "step_sequence",
        "is_parallel",
        "uuid",
    ],
    "lead_time": [
        "item_id",
        "first_station",
        "last_station",
        "first_seen",
        "last_seen",
        "lead_time_seconds",
        "stations_visited",
        "station_path",
        "has_parallel",
        "uuid",
        "start",
        "end",
        "duration_seconds",
    ],
    "parallel_activity": [
        "item_id",
        "station_a",
        "station_b",
        "overlap_start",
        "overlap_end",
        "overlap_seconds",
        "uuid",
        "start",
        "end",
        "duration_seconds",
    ],
    "handover_log": [
        "timestamp",
        "item_id",
        "from_station",
        "to_station",
        "handover_uuid",
        "handover_value",
        "uuid",
        "systime",
    ],
}
SOURCE_UUIDS = {
    # Rows by item, then start: X@a, X@b, Y@b.
    "build_timeline": ["a", "b", "b"],
    # X visits a then b; Y only b.
    "lead_time": ["a,b", "b"],
    # X at a (station_a) overlaps X at b (station_b).
    "parallel_activity": ["a,b"],
    "handover_log": ["h"],
}


@pytest.mark.parametrize("method", sorted(FORMER_COLUMNS))
def test_source_uuid_is_appended(method):
    out = getattr(_parallel(), method)()
    # Former: no source_uuid column.
    assert list(out.columns) == [*FORMER_COLUMNS[method], "source_uuid"]
    assert out["source_uuid"].tolist() == SOURCE_UUIDS[method]


def test_source_uuid_values_of_the_other_columns():
    trace = _parallel()
    tl = trace.build_timeline()
    assert tl["source_uuid"].tolist() == tl["id_uuid"].tolist()
    pa = trace.parallel_activity()
    assert pa[["station_a", "station_b"]].values.tolist() == [["A", "B"]]
    assert pa["overlap_seconds"].tolist() == [300.0]
    ho = trace.handover_log()
    assert ho["item_id"].tolist() == ["Y"]
    assert ho["source_uuid"].tolist() == ho["handover_uuid"].tolist()


@pytest.mark.parametrize("method", sorted(FORMER_COLUMNS))
def test_empty_results_have_source_uuid(method):
    df = _text([(0, "zzz", "X", None)])
    trace = _quiet(MultiProcessTraceabilityEvents, df, PROCESSES)
    out = getattr(trace, method)()
    assert out.empty
    assert list(out.columns) == [*FORMER_COLUMNS[method], "source_uuid"]


#: Event ids of the PARALLEL logs before round 23 (frozen).
FROZEN_PARALLEL_IDS = {
    "build_timeline": [
        "e-b7052059-b053-500f-98f3-202af63360e6",
        "e-2193d904-92c7-5424-ac9d-ecf18c030c1a",
        "e-235dd769-f4ee-5d20-a240-296d67cd0d83",
    ],
    "lead_time": [
        "e-25575ecf-1bc7-5003-a48b-db9f817c73ad",
        "e-70ae5a16-6caa-5d34-a314-a70c969d5a2a",
    ],
    "parallel_activity": ["e-c699536a-bcb0-5239-9b9a-0ce541132495"],
    "handover_log": ["e-734124f3-1876-563c-ba3f-0445fadf79ae"],
}


@pytest.mark.parametrize("method", sorted(FROZEN_PARALLEL_IDS))
def test_event_ids_unchanged_by_source_uuid(method):
    frame = getattr(_parallel(), method)()
    log = to_event_log(frame, detector=f"MultiProcessTraceabilityEvents.{method}")
    # The ids of the former frame (without source_uuid).
    former = to_event_log(
        frame.drop(columns="source_uuid"),
        detector=f"MultiProcessTraceabilityEvents.{method}",
    )
    assert log.events["ocel:eid"].tolist() == former.events["ocel:eid"].tolist()
    assert log.events["ocel:eid"].tolist() == FROZEN_PARALLEL_IDS[method]


def test_handover_log_binds_the_handover_signal_as_asset():
    log = to_event_log(
        _parallel().handover_log(),
        detector="MultiProcessTraceabilityEvents.handover_log",
    )
    rel = log.relations
    # Former: no asset (the frame had no source_uuid).
    assert rel.loc[rel["ocel:type"] == "asset", "ocel:oid"].tolist() == ["h"]
    assert sorted(rel.loc[rel["ocel:type"] == "station", "ocel:oid"]) == ["B", "C"]
    assert _serials(log) == ["Y"]


def test_timeline_asset_binding_unchanged():
    log = to_event_log(
        _parallel().build_timeline(),
        detector="MultiProcessTraceabilityEvents.build_timeline",
    )
    rel = log.relations
    assert sorted(rel.loc[rel["ocel:type"] == "asset", "ocel:oid"]) == ["a", "b", "b"]
