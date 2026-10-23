"""Traceability classes in the object model.

``ValueTraceabilityEvents``, ``RoutingTraceabilityEvents`` and
``MultiProcessTraceabilityEvents``: auto-bound ``serial`` / ``station``
objects in their event logs, ``to_intervals()`` for the object analytics and
``interval_end="last_sample" | "next_change"``.

The station frame (minutes after 08:00, two id signals):

=========  ========================================  ==========
signal     readings                                  station
=========  ========================================  ==========
st1:id     0 SN-0001, 5 SN-0001, 10 SN-0002,         Station 1
           15 SN-0002, 20 "" (empty)
st2:id     12 SN-0001, 18 SN-0001, 25 SN-0002,       Station 2
           30 SN-0002
=========  ========================================  ==========

Visits by hand (minutes):

========  =========  ===============  ===============
serial    station    last_sample      next_change
========  =========  ===============  ===============
SN-0001   Station 1  [0, 5]           [0, 10)
SN-0001   Station 2  [12, 18]         [12, 25)
SN-0002   Station 1  [10, 15]         [10, 20)   (the empty reading)
SN-0002   Station 2  [25, 30]         [25, 30]   (end of the data)
========  =========  ===============  ===============
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ts_shape.eventlog import (
    EventLog,
    ObjectSpec,
    object_intervals,
    object_kpis,
    presence_events,
    relate,
    to_event_log,
    wip_over_time,
)
from ts_shape.eventlog import taxonomy as tx
from ts_shape.events.production import (
    MultiProcessTraceabilityEvents,
    RoutingTraceabilityEvents,
    ValueTraceabilityEvents,
)
from ts_shape.events.production._trace_objects import INTERVAL_COLUMNS

T0 = pd.Timestamp("2024-05-06 08:00")
STATIONS = {"st1:id": "Station 1", "st2:id": "Station 2"}
PROCESSES = [
    {"id_uuid": "st1:id", "station": "Station 1"},
    {"id_uuid": "st2:id", "station": "Station 2"},
]
INTERVAL_DTYPES = {
    "oid": "string",
    "type": "string",
    "start": "datetime64[ns, UTC]",
    "end": "datetime64[ns, UTC]",
    "duration_seconds": "float64",
    "uuid": "string",
    "source_uuid": "string",
    "station": "string",
}


def _m(minutes: float, tz: str | None = None) -> pd.Timestamp:
    t = T0 + pd.Timedelta(minutes=minutes)
    return t.tz_localize(tz) if tz else t


def _utc(minutes: float, tz: str | None = None) -> pd.Timestamp:
    t = _m(minutes, tz)
    return t.tz_convert("UTC") if tz else t.tz_localize("UTC")


def _frame(rows: list[tuple[float, str, object]], tz: str | None = None):
    return pd.DataFrame(
        {
            "systime": [_m(m, tz) for m, _, _ in rows],
            "uuid": [u for _, u, _ in rows],
            "value_string": [v for _, _, v in rows],
            "value_integer": pd.array([pd.NA] * len(rows), dtype="Int64"),
        }
    )


def _stations(tz: str | None = None) -> pd.DataFrame:
    return _frame(
        [
            (0, "st1:id", "SN-0001"),
            (5, "st1:id", "SN-0001"),
            (10, "st1:id", "SN-0002"),
            (12, "st2:id", "SN-0001"),
            (15, "st1:id", "SN-0002"),
            (18, "st2:id", "SN-0001"),
            (20, "st1:id", ""),
            (25, "st2:id", "SN-0002"),
            (30, "st2:id", "SN-0002"),
        ],
        tz,
    )


def _routing(tz: str | None = None) -> pd.DataFrame:
    """ID signal + step signal: SN-0001 at A [0, 5], B [10, 15]; SN-0002 at
    A [20, 20], B [25, 25] (next_change: [0, 10), [10, 20), [20, 25),
    [25, 25])."""
    rows = [
        (0, "line:serial", "SN-0001", None),
        (20, "line:serial", "SN-0002", None),
        (0, "line:step", None, 10),
        (5, "line:step", None, 10),
        (10, "line:step", None, 20),
        (15, "line:step", None, 20),
        (20, "line:step", None, 10),
        (25, "line:step", None, 20),
    ]
    return pd.DataFrame(
        {
            "systime": [_m(m, tz) for m, *_ in rows],
            "uuid": [u for _, u, _, _ in rows],
            "value_string": [s for _, _, s, _ in rows],
            "value_integer": pd.array([i for *_, i in rows], dtype="Int64"),
        }
    )


def _route(df: pd.DataFrame) -> RoutingTraceabilityEvents:
    return RoutingTraceabilityEvents(
        df, "line:serial", "line:step", state_map={10: "A", 20: "B"}
    )


def _visits(iv: pd.DataFrame) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    out = iv.sort_values(["oid", "start"])
    return list(zip(out["oid"], out["station"], out["start"], out["end"], strict=True))


LAST = [
    ("SN-0001", "Station 1", 0, 5),
    ("SN-0001", "Station 2", 12, 18),
    ("SN-0002", "Station 1", 10, 15),
    ("SN-0002", "Station 2", 25, 30),
]
NEXT = [
    ("SN-0001", "Station 1", 0, 10),
    ("SN-0001", "Station 2", 12, 25),
    ("SN-0002", "Station 1", 10, 20),
    ("SN-0002", "Station 2", 25, 30),
]


def _expected(rows, tz=None):
    return [(o, s, _utc(a, tz), _utc(b, tz)) for o, s, a, b in rows]


STATION_TRACES = {
    "value": lambda df: ValueTraceabilityEvents(df, STATIONS),
    "multi": lambda df: MultiProcessTraceabilityEvents(df, PROCESSES),
}


# ---------------------------------------------------------------------------
# to_intervals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(STATION_TRACES))
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize(
    ("interval_end", "rows"), [("last_sample", LAST), ("next_change", NEXT)]
)
def test_to_intervals_by_hand(kind, tz, interval_end, rows):
    trace = STATION_TRACES[kind](_stations(tz))
    iv = trace.to_intervals(interval_end=interval_end)
    assert tuple(iv.columns) == INTERVAL_COLUMNS
    assert iv.dtypes.astype(str).to_dict() == INTERVAL_DTYPES
    assert _visits(iv) == _expected(rows, tz)
    assert set(iv["type"]) == {"serial"}
    assert set(iv["uuid"]) == {"st1:id", "st2:id"}
    assert (iv["uuid"] == iv["source_uuid"]).all()
    span = (iv["end"] - iv["start"]).dt.total_seconds()
    np.testing.assert_allclose(iv["duration_seconds"], span)


def test_next_change_matches_object_intervals():
    df = _stations()
    iv = ValueTraceabilityEvents(df, STATIONS).to_intervals(interval_end="next_change")
    ref = object_intervals(
        df, [ObjectSpec("st1:id", "serial"), ObjectSpec("st2:id", "serial")]
    )
    cols = ["oid", "type", "start", "end", "duration_seconds", "uuid"]
    got = iv[cols].sort_values(["oid", "start"]).reset_index(drop=True)
    want = ref[cols].sort_values(["oid", "start"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want)


@pytest.mark.parametrize("kind", sorted(STATION_TRACES))
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_object_kpis_two_stations(kind, tz):
    trace = STATION_TRACES[kind](_stations(tz))

    # last_sample: SN-0001 0 -> 18 min, dwell 5 + 6; SN-0002 10 -> 30, 5 + 5.
    kpis = object_kpis(trace.to_intervals(), object_type="serial")
    obj = kpis.objects.set_index("oid")
    assert obj.loc["SN-0001", "lead_time_seconds"] == 18 * 60
    assert obj.loc["SN-0001", "dwell_seconds"] == 11 * 60
    assert obj.loc["SN-0002", "lead_time_seconds"] == 20 * 60
    assert obj.loc["SN-0002", "dwell_seconds"] == 10 * 60
    assert obj["presence_count"].tolist() == [2, 2]
    assert obj["station_count"].tolist() == [2, 2]
    assert obj.loc["SN-0001", "start"] == _utc(0, tz)
    st = kpis.stations.set_index(["oid", "station"])["dwell_seconds"].to_dict()
    assert st == {
        ("SN-0001", "Station 1"): 300.0,
        ("SN-0001", "Station 2"): 360.0,
        ("SN-0002", "Station 1"): 300.0,
        ("SN-0002", "Station 2"): 300.0,
    }

    # next_change: SN-0001 0 -> 25, dwell 10 + 13; SN-0002 10 -> 30, 10 + 5.
    kpis = object_kpis(
        trace.to_intervals(interval_end="next_change"), object_type="serial"
    )
    obj = kpis.objects.set_index("oid")
    assert obj.loc["SN-0001", "lead_time_seconds"] == 25 * 60
    assert obj.loc["SN-0001", "dwell_seconds"] == 23 * 60
    assert obj.loc["SN-0002", "lead_time_seconds"] == 20 * 60
    assert obj.loc["SN-0002", "dwell_seconds"] == 15 * 60


def test_lead_time_agrees_with_class():
    trace = ValueTraceabilityEvents(_stations(), STATIONS)
    kpis = object_kpis(trace.to_intervals(), object_type="serial").objects
    lead = trace.lead_time().set_index("identifier")["lead_time_seconds"]
    assert kpis.set_index("oid")["lead_time_seconds"].to_dict() == lead.to_dict()


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_routing_to_intervals(tz):
    trace = _route(_routing(tz))
    iv = trace.to_intervals()
    assert tuple(iv.columns) == INTERVAL_COLUMNS
    assert _visits(iv) == _expected(
        [
            ("SN-0001", "A", 0, 5),
            ("SN-0001", "B", 10, 15),
            ("SN-0002", "A", 20, 20),
            ("SN-0002", "B", 25, 25),
        ],
        tz,
    )
    assert set(iv["uuid"]) == {"line:serial"}
    assert set(iv["source_uuid"]) == {"line:serial,line:step"}
    nxt = trace.to_intervals(interval_end="next_change")
    assert _visits(nxt) == _expected(
        [
            ("SN-0001", "A", 0, 10),
            ("SN-0001", "B", 10, 20),
            ("SN-0002", "A", 20, 25),
            ("SN-0002", "B", 25, 25),
        ],
        tz,
    )
    obj = object_kpis(nxt, object_type="serial").objects.set_index("oid")
    assert obj.loc["SN-0001", "lead_time_seconds"] == 20 * 60
    assert obj.loc["SN-0001", "dwell_seconds"] == 20 * 60
    assert obj.loc["SN-0002", "lead_time_seconds"] == 5 * 60
    assert obj.loc["SN-0002", "dwell_seconds"] == 5 * 60
    st = object_kpis(nxt, object_type="serial").stations
    assert set(st["station"]) == {"A", "B"}


def test_routing_next_change_after_missing_item():
    """A routing sample whose current ID reading is missing ends the segment."""
    df = _frame(
        [
            (0, "line:serial", "SN-0001"),
            (8, "line:serial", None),
            (12, "line:serial", "SN-0001"),
        ]
    )
    steps = pd.DataFrame(
        {
            "systime": [_m(0), _m(5), _m(10), _m(15)],
            "uuid": "line:step",
            "value_string": None,
            "value_integer": pd.array([10, 10, 10, 20], dtype="Int64"),
        }
    )
    trace = _route(pd.concat([df, steps], ignore_index=True))
    tl = trace.build_routing_timeline(interval_end="next_change")
    # The 10-min step sample has no item: SN-0001 at A [0, 10), B [15, 15].
    got = list(zip(tl["station_name"], tl["start"], tl["end"], strict=True))
    assert got == [("A", _m(0), _m(10)), ("B", _m(15), _m(15))]


def test_multi_gap_split_keeps_last_sample():
    """The same item after a time gap is a new visit: no hold across it."""
    df = _frame(
        [
            (0, "st1:id", "SN-0001"),
            (1, "st1:id", "SN-0001"),
            (2, "st1:id", "SN-0001"),
            (30, "st1:id", "SN-0001"),
            (31, "st1:id", "SN-0001"),
            (40, "st2:id", "SN-0001"),
        ]
    )
    trace = MultiProcessTraceabilityEvents(df, PROCESSES)
    iv = trace.to_intervals(interval_end="next_change")
    assert _visits(iv) == _expected(
        [
            ("SN-0001", "Station 1", 0, 2),
            ("SN-0001", "Station 1", 30, 40),
            ("SN-0001", "Station 2", 40, 40),
        ]
    )


def test_custom_object_type_and_numeric_ids():
    df = _stations()
    df["value_double"] = df["value_string"].map({"SN-0001": 1001.0, "SN-0002": 1002.0})
    trace = ValueTraceabilityEvents(df, STATIONS, value_column="value_double")
    iv = trace.to_intervals(object_type="work_order")
    assert set(iv["type"]) == {"work_order"}
    assert sorted(set(iv["oid"])) == ["1001", "1002"]


@pytest.mark.parametrize("kind", ["value", "multi", "routing"])
def test_empty_inputs(kind):
    empty = _stations().iloc[0:0]
    if kind == "routing":
        trace = _route(empty)
    else:
        trace = STATION_TRACES[kind](empty)
    for interval_end in ("last_sample", "next_change"):
        iv = trace.to_intervals(interval_end=interval_end)
        assert iv.empty
        assert iv.dtypes.astype(str).to_dict() == INTERVAL_DTYPES
    kpis = object_kpis(iv, object_type="serial")
    assert kpis.objects.empty and kpis.stations.empty
    assert presence_events(iv).empty
    assert len(to_event_log(iv, detector=_detector(kind, "to_intervals"))) == 0


def test_interval_end_validated():
    trace = ValueTraceabilityEvents(_stations(), STATIONS)
    with pytest.raises(ValueError, match="interval_end"):
        trace.to_intervals(interval_end="next")
    with pytest.raises(ValueError, match="interval_end"):
        _route(_routing()).build_routing_timeline(interval_end="end")
    with pytest.raises(ValueError, match="interval_end"):
        MultiProcessTraceabilityEvents(_stations(), PROCESSES).build_timeline(
            interval_end="x"
        )


def test_default_timelines_unchanged():
    trace = ValueTraceabilityEvents(_stations(), STATIONS)
    pd.testing.assert_frame_equal(
        trace.build_timeline(), trace.build_timeline(interval_end="last_sample")
    )
    stats = trace.station_dwell_statistics(interval_end="next_change")
    dwell = stats.set_index("station_name")["total_dwell_seconds"].to_dict()
    assert dwell == {"Station 1": 20 * 60.0, "Station 2": 18 * 60.0}


def test_analysis_consumers():
    iv = ValueTraceabilityEvents(_stations(), STATIONS).to_intervals(
        interval_end="next_change"
    )
    pe = presence_events(iv, object_type="serial")
    assert pe["station"].tolist()[:2] == ["Station 1", "Station 1"]
    log = to_event_log(pe, detector="eventlog.presence_events")
    assert set(
        log.relations.loc[log.relations["ocel:type"] == "serial", "ocel:oid"]
    ) == {
        "SN-0001",
        "SN-0002",
    }
    via_log = object_kpis(log, object_type="serial").objects
    direct = object_kpis(iv, object_type="serial").objects
    pd.testing.assert_series_equal(
        via_log["lead_time_seconds"], direct["lead_time_seconds"]
    )
    # SN-0001 [0, 10) + [12, 25), SN-0002 [10, 20) + [25, 30]: both only in
    # [12, 20); in [20, 30) one hands over to the other at 25.
    wip = wip_over_time(iv, object_type="serial", window="10min")
    assert wip["wip_max"].tolist() == [1, 2, 1, 0]
    # relate(): the station is an object attribute at each visit.
    rel = relate(EventLog(), iv)
    changes = rel.object_changes
    stations = changes[changes["ocel:field"] == "station"]
    assert sorted(zip(stations["ocel:oid"], stations["ocel:value"], strict=True)) == [
        ("SN-0001", "Station 1"),
        ("SN-0001", "Station 2"),
        ("SN-0002", "Station 1"),
        ("SN-0002", "Station 2"),
    ]


# ---------------------------------------------------------------------------
# Event-log binding
# ---------------------------------------------------------------------------


def _detector(kind: str, method: str) -> str:
    cls = {
        "value": "ValueTraceabilityEvents",
        "multi": "MultiProcessTraceabilityEvents",
        "routing": "RoutingTraceabilityEvents",
    }[kind]
    return f"{cls}.{method}"


def _objects(log, otype):
    rel = log.relations
    return sorted(set(rel.loc[rel["ocel:type"] == otype, "ocel:oid"]))


@pytest.mark.parametrize(
    ("kind", "method", "serial", "station"),
    [
        ("value", "build_timeline", True, True),
        ("value", "lead_time", True, False),
        ("value", "current_status", True, False),
        ("value", "station_dwell_statistics", False, True),
        ("multi", "build_timeline", True, True),
        ("routing", "build_routing_timeline", True, True),
        ("routing", "lead_time", True, False),
        ("routing", "station_statistics", False, True),
    ],
)
def test_detector_frames_bind_serial_and_station(kind, method, serial, station):
    if kind == "routing":
        trace = _route(_routing())
        stations = ["A", "B"]
    else:
        trace = STATION_TRACES[kind](_stations())
        stations = ["Station 1", "Station 2"]
    log = to_event_log(getattr(trace, method)(), detector=_detector(kind, method))
    assert _objects(log, "serial") == (["SN-0001", "SN-0002"] if serial else [])
    assert _objects(log, "station") == (stations if station else [])
    objects = log.objects
    if serial:
        assert {"SN-0001", "SN-0002"} <= set(
            objects.loc[objects["ocel:type"] == "serial", "ocel:oid"]
        )


@pytest.mark.parametrize(
    ("kind", "method"),
    [
        ("value", "build_timeline"),
        ("value", "lead_time"),
        ("value", "station_dwell_statistics"),
        ("routing", "build_routing_timeline"),
        ("routing", "station_statistics"),
        ("value", "to_intervals"),
        ("multi", "build_timeline"),
    ],
)
def test_bound_objects_keep_event_ids(kind, method, monkeypatch):
    trace = (
        _route(_routing()) if kind == "routing" else STATION_TRACES[kind](_stations())
    )
    frame = getattr(trace, method)()
    detector = _detector(kind, method)
    with_objects = to_event_log(frame, detector=detector)
    key = tuple(detector.split("."))
    rule = tx.REGISTRY[key]
    monkeypatch.setitem(tx.REGISTRY, key, dataclasses.replace(rule, object_columns={}))
    without = to_event_log(frame, detector=detector)
    assert with_objects.events["ocel:eid"].tolist() == (
        without.events["ocel:eid"].tolist()
    )
    assert len(with_objects.relations) > len(without.relations) or (
        method == "to_intervals"
    )


def test_caller_binding_wins():
    trace = ValueTraceabilityEvents(_stations(), STATIONS)
    tl = trace.build_timeline()
    log = to_event_log(
        tl, detector="ValueTraceabilityEvents.build_timeline", objects={"serial": None}
    )
    assert _objects(log, "serial") == []
    assert _objects(log, "station") == ["Station 1", "Station 2"]


@pytest.mark.parametrize("kind", ["value", "multi", "routing"])
def test_to_intervals_event_log(kind):
    trace = (
        _route(_routing()) if kind == "routing" else STATION_TRACES[kind](_stations())
    )
    iv = trace.to_intervals()
    log = to_event_log(iv, detector=_detector(kind, "to_intervals"))
    assert len(log) == 4
    assert _objects(log, "serial") == ["SN-0001", "SN-0002"]
    assert len(_objects(log, "station")) == 2
    assert set(log.events["ts_shape:lifecycle_state"]) == set(iv["station"])


# ---------------------------------------------------------------------------
# MultiProcessTraceabilityEvents: lead_time, parallel_activity, handover_log
# ---------------------------------------------------------------------------


def _multi_parallel(tz: str | None = None) -> MultiProcessTraceabilityEvents:
    """SN-0001 at Station 1 [0, 20] and Station 2 [5, 10] (parallel); a
    handover Station 1 -> Station 2 fires at 15 (rising edge 0 -> 1)."""
    df = _frame(
        [
            (0, "st1:id", "SN-0001"),
            (20, "st1:id", "SN-0001"),
            (5, "st2:id", "SN-0001"),
            (10, "st2:id", "SN-0001"),
        ],
        tz,
    )
    ho = pd.DataFrame(
        {
            "systime": [_m(0, tz), _m(15, tz)],
            "uuid": "ho",
            "value_string": None,
            "value_integer": pd.array([0, 1], dtype="Int64"),
        }
    )
    return MultiProcessTraceabilityEvents(
        pd.concat([df, ho], ignore_index=True),
        PROCESSES,
        handovers=[
            {"uuid": "ho", "from_station": "Station 1", "to_station": "Station 2"}
        ],
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_multi_outputs_convert(tz):
    trace = _multi_parallel(tz)
    det = "MultiProcessTraceabilityEvents."

    pa = trace.parallel_activity()
    assert list(pa.columns)[-3:] == ["start", "end", "duration_seconds"]
    log = to_event_log(pa, detector=det + "parallel_activity")
    assert len(log) == 1
    ev = log.events.iloc[0]
    assert ev["ocel:timestamp"] == _utc(10, tz)
    assert ev["ts_shape:start_timestamp"] == _utc(5, tz)
    assert ev["ts_shape:duration_s"] == 300.0
    assert _objects(log, "serial") == ["SN-0001"]
    assert _objects(log, "station") == ["Station 1", "Station 2"]

    ho = trace.handover_log()
    assert list(ho.columns)[-1] == "systime"
    log = to_event_log(ho, detector=det + "handover_log")
    assert len(log) == 1
    assert log.events.iloc[0]["ocel:timestamp"] == _utc(15, tz)
    assert _objects(log, "serial") == ["SN-0001"]
    assert _objects(log, "station") == ["Station 1", "Station 2"]

    lead = trace.lead_time()
    assert list(lead.columns)[-3:] == ["start", "end", "duration_seconds"]
    log = to_event_log(lead, detector=det + "lead_time")
    assert len(log) == 1
    assert log.events.iloc[0]["ts_shape:start_timestamp"] == _utc(0, tz)
    assert _objects(log, "serial") == ["SN-0001"]
    assert _objects(log, "station") == []


def test_tuple_object_columns_link_once_and_caller_wins():
    trace = _multi_parallel()
    pa = trace.parallel_activity()
    same = pa.assign(station_b=pa["station_a"])
    log = to_event_log(
        same, detector="MultiProcessTraceabilityEvents.parallel_activity"
    )
    rel = log.relations
    assert rel.loc[rel["ocel:type"] == "station", "ocel:oid"].tolist() == ["Station 1"]
    log = to_event_log(
        pa,
        detector="MultiProcessTraceabilityEvents.parallel_activity",
        objects={"station": None},
    )
    assert _objects(log, "station") == []
    log = to_event_log(
        pa,
        detector="MultiProcessTraceabilityEvents.parallel_activity",
        objects={"station": "station_b"},
    )
    assert _objects(log, "station") == ["Station 2"]


def test_multi_outputs_empty_keep_columns():
    trace = MultiProcessTraceabilityEvents(_stations().iloc[0:0], PROCESSES)
    assert list(trace.lead_time().columns)[-3:] == ["start", "end", "duration_seconds"]
    assert list(trace.parallel_activity().columns)[-3:] == [
        "start",
        "end",
        "duration_seconds",
    ]
    assert list(trace.handover_log().columns)[-1] == "systime"
