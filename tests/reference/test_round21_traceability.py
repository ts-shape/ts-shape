"""Round 21: traceability audit, hand-computed references.

``ValueTraceabilityEvents``, ``RoutingTraceabilityEvents`` and
``MultiProcessTraceabilityEvents``. Times are minutes after 08:00 (naive) or
after 2024-03-31 00:50 UTC on the Berlin DST day (02:00 CET -> 03:00 CEST).
Each test names the former result it replaces.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.eventlog import ObjectSpec, object_intervals, object_kpis, to_event_log
from ts_shape.eventlog import taxonomy as tx
from ts_shape.events.production import (
    MultiProcessTraceabilityEvents,
    RoutingTraceabilityEvents,
    ValueTraceabilityEvents,
)

T0 = pd.Timestamp("2024-05-06 08:00")
#: 10 minutes before the Berlin clock jumps from 02:00 to 03:00.
DST0 = pd.Timestamp("2024-03-31 00:50", tz="UTC")
PROCESSES = [
    {"id_uuid": "a", "station": "A"},
    {"id_uuid": "b", "station": "B"},
]
STATIONS = {"a": "A", "b": "B"}


def _t(minutes: float, tz: str | None = None) -> pd.Timestamp:
    if tz is None:
        return T0 + pd.Timedelta(minutes=minutes)
    return (DST0 + pd.Timedelta(minutes=minutes)).tz_convert(tz)


def _frame(rows, tz: str | None = None) -> pd.DataFrame:
    """``rows``: ``(minutes, uuid, value_string, value_integer)``."""
    return pd.DataFrame(
        {
            "systime": [pd.NaT if m is None else _t(m, tz) for m, *_ in rows],
            "uuid": [u for _, u, _, _ in rows],
            "value_string": pd.array([s for _, _, s, _ in rows], dtype="object"),
            "value_integer": pd.array([i for *_, i in rows], dtype="Int64"),
        }
    )


def _visits(tl: pd.DataFrame, item: str, station: str):
    return list(
        zip(tl[item], tl[station], tl["start"], tl["end"], strict=True),
    )


# ---------------------------------------------------------------------------
# 1. lead_time ends at the latest visit end
# ---------------------------------------------------------------------------

#: X at A [0, 20] and, in parallel, at B [5, 10].
OVERLAP = [
    (0, "a", "X", None),
    (20, "a", "X", None),
    (5, "b", "X", None),
    (10, "b", "X", None),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("kind", ["multi", "value"])
def test_lead_time_takes_the_latest_visit_end(kind, tz):
    df = _frame(OVERLAP, tz)
    if kind == "multi":
        trace = MultiProcessTraceabilityEvents(df, PROCESSES)
    else:
        trace = ValueTraceabilityEvents(df, STATIONS)
    lead = trace.lead_time().iloc[0]
    # Former: last_seen 10 (end of the last-started visit), lead time 600 s.
    assert lead["first_seen"] == _t(0, tz)
    assert lead["last_seen"] == _t(20, tz)
    assert lead["lead_time_seconds"] == 1200.0
    assert lead["end"] == _t(20, tz)
    assert lead["duration_seconds"] == 1200.0
    assert lead["last_station"] == "B"  # the last visit by start, unchanged
    kpis = object_kpis(trace.to_intervals(), object_type="serial").objects
    assert kpis["lead_time_seconds"].tolist() == [1200.0]


def test_multi_routing_paths_use_the_lead_time():
    trace = MultiProcessTraceabilityEvents(_frame(OVERLAP), PROCESSES)
    paths = trace.routing_paths()
    assert paths["avg_lead_time_seconds"].tolist() == [1200.0]  # former 600.0


# ---------------------------------------------------------------------------
# 2. Routing: a sample without an item or routing value is no visit
# ---------------------------------------------------------------------------


def _route(rows, tz=None) -> RoutingTraceabilityEvents:
    return RoutingTraceabilityEvents(
        _frame(rows, tz), "id", "st", state_map={1: "A", 2: "B"}
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_routing_missing_item_ends_the_visit(tz):
    # SN1 read at 0, a missing reading at 8, SN1 again at 12; step A at
    # 0, 5, 10, 15: the step sample at 10 has no item.
    trace = _route(
        [
            (0, "id", "SN1", None),
            (8, "id", None, None),
            (12, "id", "SN1", None),
            (0, "st", None, 1),
            (5, "st", None, 1),
            (10, "st", None, 1),
            (15, "st", None, 1),
        ],
        tz,
    )
    # Former (both modes): one visit SN1 at A [0, 15] with 3 samples.
    tl = trace.build_routing_timeline()
    assert _visits(tl, "item_id", "station_name") == [
        ("SN1", "A", _t(0, tz), _t(5, tz)),
        ("SN1", "A", _t(15, tz), _t(15, tz)),
    ]
    assert tl["sample_count"].tolist() == [2, 1]
    assert tl["station_sequence"].tolist() == [1, 2]
    tl = trace.build_routing_timeline(interval_end="next_change")
    assert _visits(tl, "item_id", "station_name") == [
        ("SN1", "A", _t(0, tz), _t(10, tz)),
        ("SN1", "A", _t(15, tz), _t(15, tz)),
    ]


def test_routing_empty_item_and_missing_step_are_no_visits():
    trace = _route(
        [
            (0, "id", "SN1", None),
            (10, "id", "", None),
            (20, "id", "SN2", None),
            (0, "st", None, 1),
            (5, "st", None, 1),
            (10, "st", None, 2),
            (15, "st", None, None),
            (20, "st", None, 1),
            (25, "st", None, 2),
        ]
    )
    # Former: also item "" at B [10, 10] and item "" at "State nan" [15, 15]
    # (next_change [10, 15) and [15, 20)); lead_time had an item "".
    tl = trace.build_routing_timeline(interval_end="next_change")
    assert _visits(tl, "item_id", "station_name") == [
        ("SN1", "A", _t(0), _t(10)),
        ("SN2", "A", _t(20), _t(25)),
        ("SN2", "B", _t(25), _t(25)),
    ]
    assert trace.lead_time()["item_id"].tolist() == ["SN1", "SN2"]


def test_routing_missing_step_between_two_samples_of_one_step():
    # A, missing, A for SN1: two visits. Former: a third visit at
    # "State nan" [5, 5] between them.
    trace = _route(
        [
            (0, "id", "SN1", None),
            (0, "st", None, 1),
            (5, "st", None, None),
            (10, "st", None, 1),
        ]
    )
    tl = trace.build_routing_timeline()
    assert _visits(tl, "item_id", "station_name") == [
        ("SN1", "A", _t(0), _t(0)),
        ("SN1", "A", _t(10), _t(10)),
    ]


# ---------------------------------------------------------------------------
# 3. Rows without a timestamp no longer crash the asof joins
# ---------------------------------------------------------------------------


def test_nat_rows_do_not_crash_routing():
    trace = _route(
        [
            (0, "id", "SN1", None),
            (None, "id", "SN2", None),
            (0, "st", None, 1),
            (5, "st", None, 1),
            (None, "st", None, 2),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        # Former: ValueError "Merge keys contain null values".
        tl = trace.build_routing_timeline()
    assert _visits(tl, "item_id", "station_name") == [("SN1", "A", _t(0), _t(5))]


def test_nat_rows_do_not_crash_handover_log():
    df = _frame(
        [
            (0, "a", "X", None),
            (None, "a", "X", None),
            (10, "a", "Y", None),
            (0, "h", None, 0),
            (5, "h", None, 1),
            (None, "h", None, 0),
        ]
    )
    trace = MultiProcessTraceabilityEvents(
        df,
        PROCESSES,
        handovers=[{"uuid": "h", "from_station": "A", "to_station": "B"}],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        log = trace.handover_log()  # former: ValueError
    assert list(zip(log["systime"], log["item_id"], strict=True)) == [(_t(5), "X")]


# ---------------------------------------------------------------------------
# 4. Readings at the same instant keep their input order
# ---------------------------------------------------------------------------


def _ties() -> pd.DataFrame:
    """18 readings, two per minute 0..8, all X except the second at minute
    4, which is Y. A default (quicksort) sort swaps the two readings at
    minute 4 (rows 8 and 9)."""
    rows = []
    for k in range(18):
        rows.append((k // 2, "a", "Y" if k == 9 else "X", None))
    return _frame(rows)


@pytest.mark.parametrize("kind", ["multi", "value"])
def test_tied_readings_keep_input_order(kind):
    df = _ties()
    if kind == "multi":
        tl = MultiProcessTraceabilityEvents(df, PROCESSES).build_timeline()
        item, station = "item_id", "station"
    else:
        tl = ValueTraceabilityEvents(df, STATIONS).build_timeline()
        item, station = "identifier", "station_name"
    got = sorted(_visits(tl, item, station), key=lambda v: (v[2], v[0] != "X"))
    # Former: X [0, 3], Y [4, 4], X [4, 8] (Y read before X at minute 4).
    assert got == [
        ("X", "A", _t(0), _t(4)),
        ("Y", "A", _t(4), _t(4)),
        ("X", "A", _t(5), _t(8)),
    ]


def test_tied_readings_keep_input_order_routing():
    df = pd.concat(
        [
            _ties().assign(uuid="id"),
            _frame([(4.5, "st", None, 1)]),
        ],
        ignore_index=True,
    )
    trace = RoutingTraceabilityEvents(df, "id", "st", state_map={1: "A"})
    # The item at 4.5 is the last reading at minute 4 in input order: Y.
    # Former: X.
    assert trace.build_routing_timeline()["item_id"].tolist() == ["Y"]


# ---------------------------------------------------------------------------
# 5. MultiProcess build_timeline event log: station and step attributes
# ---------------------------------------------------------------------------


def test_multi_timeline_log_attributes(monkeypatch):
    df = _frame(
        [(0, "a", "X", None), (5, "a", "X", None), (10, "b", "X", None)],
    )
    tl = MultiProcessTraceabilityEvents(df, PROCESSES).build_timeline()
    det = "MultiProcessTraceabilityEvents.build_timeline"
    log = to_event_log(tl, detector=det)
    ev = log.events.sort_values("ocel:timestamp")
    # Former: the literals "station_name" / "station_sequence" on every row.
    assert ev["ts_shape:lifecycle_state"].tolist() == ["A", "B"]
    assert ev["ts_shape:direction"].tolist() == ["1", "2"]
    key = ("MultiProcessTraceabilityEvents", "build_timeline")
    old = dataclasses.replace(
        tx.REGISTRY[key],
        standard_attrs={
            "ts_shape:lifecycle_state": "station_name",
            "ts_shape:direction": "station_sequence",
        },
    )
    monkeypatch.setitem(tx.REGISTRY, key, old)
    before = to_event_log(tl, detector=det)
    assert before.events["ts_shape:lifecycle_state"].tolist() == ["station_name"] * 2
    # The event ids are unchanged.
    assert before.events["ocel:eid"].tolist() == log.events["ocel:eid"].tolist()


# ---------------------------------------------------------------------------
# 6. MultiProcess warns about process / handover signals not in the data
# ---------------------------------------------------------------------------


def test_multi_warns_about_missing_signals():
    df = _frame([(0, "a", "X", None), (1, "a", "X", None)])
    with pytest.warns(DataQualityWarning, match=r"processes='zz', handovers='hh'"):
        MultiProcessTraceabilityEvents(
            df,
            [*PROCESSES[:1], {"id_uuid": "zz", "station": "Z"}],
            handovers=[{"uuid": "hh", "from_station": "A", "to_station": "B"}],
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        MultiProcessTraceabilityEvents(df, PROCESSES[:1])


# ---------------------------------------------------------------------------
# Handover: the item before the trigger; a missing reading holds
# ---------------------------------------------------------------------------


def test_handover_item_before_trigger_and_missing_holds():
    # A: X at 0, 5; Y at 10, 15; empty at 20. B reads X at 10.
    # Handover A -> B: 0@0, 1@10, 0@12, 1@15, missing@16, 1@17, 0@21, 1@22.
    df = _frame(
        [
            (0, "a", "X", None),
            (5, "a", "X", None),
            (10, "a", "Y", None),
            (15, "a", "Y", None),
            (20, "a", "", None),
            (10, "b", "X", None),
            (0, "h", None, 0),
            (10, "h", None, 1),
            (12, "h", None, 0),
            (15, "h", None, 1),
            (16, "h", None, None),
            (17, "h", None, 1),
            (21, "h", None, 0),
            (22, "h", None, 1),
        ]
    )
    trace = MultiProcessTraceabilityEvents(
        df,
        PROCESSES,
        handovers=[{"uuid": "h", "from_station": "A", "to_station": "B"}],
    )
    log = trace.handover_log()
    # At 10 Y is read at the trigger instant: the transferred item is X (read
    # before). At 15: Y. 16 is missing (holds 1): 17 is no new transfer. At 22
    # the reading before is empty: skipped.
    # Former: (10, Y), (15, Y), (17, Y).
    assert list(zip(log["timestamp"], log["item_id"], strict=True)) == [
        (_t(10), "X"),
        (_t(15), "Y"),
    ]
    assert log["handover_value"].tolist() == [1.0, 1.0]
    assert (log["systime"] == log["timestamp"]).all()


def test_handover_leading_missing_reading_is_zero():
    df = _frame(
        [
            (0, "a", "X", None),
            (0, "h", None, None),
            (5, "h", None, 1),
        ]
    )
    trace = MultiProcessTraceabilityEvents(
        df,
        PROCESSES,
        handovers=[{"uuid": "h", "from_station": "A", "to_station": "B"}],
    )
    assert trace.handover_log()["item_id"].tolist() == ["X"]


# ---------------------------------------------------------------------------
# exclude= (opt-in)
# ---------------------------------------------------------------------------

MISREAD_ROWS = [
    (0, "a", "SN-1", None),
    (1, "a", "NOREAD", None),
    (2, "a", "SN-1", None),
    (3, "a", "#N/A", None),
    (4, "a", "SN-2", None),
    (6, "a", "SN-2", None),
]


@pytest.mark.parametrize("kind", ["multi", "value"])
def test_exclude_misreads(kind):
    df = _frame(MISREAD_ROWS)

    def make(**kw):
        if kind == "multi":
            return MultiProcessTraceabilityEvents(df, PROCESSES, **kw)
        return ValueTraceabilityEvents(df, STATIONS, **kw)

    def visits(trace, **kw):
        iv = trace.to_intervals(**kw)
        return list(zip(iv["oid"], iv["start"], iv["end"], strict=True))

    utc = lambda m: _t(m).tz_localize("UTC")  # noqa: E731
    # Default: NOREAD and #N/A are items (unchanged).
    assert sorted({o for o, *_ in visits(make())}) == ["#N/A", "NOREAD", "SN-1", "SN-2"]
    # NOREAD between two SN-1 is a gap; #N/A between SN-1 and SN-2 is empty.
    for exclude in (
        "misreads",
        ["NOREAD", "#N/A"],
        lambda v: v.isin(["NOREAD", "#N/A"]),
    ):
        trace = make(exclude=exclude)
        assert visits(trace) == [
            ("SN-1", utc(0), utc(2)),
            ("SN-2", utc(4), utc(6)),
        ]
        assert visits(trace, interval_end="next_change") == [
            ("SN-1", utc(0), utc(3)),
            ("SN-2", utc(4), utc(6)),
        ]


def test_exclude_next_change_matches_object_intervals():
    df = _frame(MISREAD_ROWS)
    iv = ValueTraceabilityEvents(df, STATIONS, exclude="misreads").to_intervals(
        interval_end="next_change"
    )
    ref = object_intervals(df, [ObjectSpec("a", "serial")])
    cols = ["oid", "start", "end"]
    pd.testing.assert_frame_equal(
        iv[cols].reset_index(drop=True), ref[cols].reset_index(drop=True)
    )


def test_exclude_routing_id_signal():
    rows = [
        (0, "id", "SN-1", None),
        (4, "id", "NOREAD", None),
        (8, "id", "SN-1", None),
        (0, "st", None, 1),
        (5, "st", None, 1),
        (10, "st", None, 1),
    ]
    default = _route(rows).build_routing_timeline()
    assert default["item_id"].tolist() == ["NOREAD", "SN-1", "SN-1"]
    trace = RoutingTraceabilityEvents(
        _frame(rows), "id", "st", state_map={1: "A"}, exclude="misreads"
    )
    tl = trace.build_routing_timeline()
    assert _visits(tl, "item_id", "station_name") == [("SN-1", "A", _t(0), _t(10))]


def test_exclude_invalid_text_raises():
    with pytest.raises(ValueError, match="misreads"):
        ValueTraceabilityEvents(_frame(MISREAD_ROWS), STATIONS, exclude="noread")


def test_exclude_keeps_numeric_ids():
    df = pd.DataFrame(
        {
            "systime": [_t(m) for m in range(4)],
            "uuid": "a",
            "value_double": [1001.0, 0.0, 1001.0, 65535.0],
        }
    )
    tl = ValueTraceabilityEvents(
        df, STATIONS, value_column="value_double", exclude="misreads"
    ).build_timeline()
    assert tl["identifier"].tolist() == [1001.0]
    assert tl["end"].tolist() == [_t(2)]
    assert np.isfinite(tl["duration_seconds"]).all()
