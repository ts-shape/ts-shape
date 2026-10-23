"""Object handling: overlap linking, scope, misread ids, signal hierarchy.

Hand-computed references for the event-to-object linking by interval
overlap (with ``scope`` and the fan-out / tz checks), the misread filter of
:class:`ObjectSpec` (``exclude``), signal-level ``hierarchy`` edges, opt-in
``co_occurs`` and the grouped ``_detect`` path (equivalence with the former
per-spec filtering).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape._object_signals as gen
from ts_shape.datasets import make_object_signals
from ts_shape.errors import DataQualityWarning
from ts_shape.eventlog import (
    EventLog,
    ObjectSpec,
    attach_objects,
    concat,
    derive_o2o,
    detect_objects,
    link_events_to_objects,
    object_intervals,
    relate,
    schema,
    to_event_log,
)
from ts_shape.eventlog import objects as om
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.utils.id_codes import is_misread, is_misread_value

T0 = pd.Timestamp("2025-01-01 06:00", tz="UTC")


def _m(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def _signal(uuid, minutes, values, column="value_string"):
    n = len(minutes)
    vals = (
        pd.array(values, dtype="Int64")
        if column == "value_integer"
        else pd.Series(values, dtype=object)
    )
    return pd.DataFrame(
        {"systime": [_m(x) for x in minutes], "uuid": [uuid] * n, column: vals}
    )


def _events(spans, prefix="e", source=None):
    """Events from ``[(start_min | None, end_min), ...]``."""
    n = len(spans)
    out = pd.DataFrame(
        {
            schema.OCEL_EID: [f"{prefix}{i}" for i in range(n)],
            schema.OCEL_ACTIVITY: ["run"] * n,
            schema.OCEL_TIMESTAMP: [_m(e) for _, e in spans],
            schema.TS_START_TIMESTAMP: pd.Series(
                [pd.NaT if s is None else _m(s) for s, _ in spans],
                dtype="datetime64[ns, UTC]",
            ),
            schema.TS_DETECTOR: ["d"] * n,
            schema.TS_PACK: ["production"] * n,
        }
    )
    if source is not None:
        out["production:source_uuid"] = source
    return out


def _links(relations):
    rel = relations[relations[schema.OCEL_TYPE] != "asset"]
    out: dict[str, list[str]] = {}
    for eid, oid in zip(rel[schema.OCEL_EID], rel[schema.OCEL_OID], strict=True):
        out.setdefault(eid, []).append(oid)
    return {k: sorted(v) for k, v in out.items()}


def _spans(iv):
    return [
        (o, (s - T0) / pd.Timedelta(minutes=1), (e - T0) / pd.Timedelta(minutes=1))
        for o, s, e in zip(iv["oid"], iv["start"], iv["end"], strict=True)
    ]


# ---------------------------------------------------------------------------
# Overlap linking
# ---------------------------------------------------------------------------


@pytest.fixture()
def batches():
    # B1 [0, 10), B2 [10, 20], one signal
    return _signal("batch", [0, 10, 20], ["B1", "B2", "B2"])


def test_interval_event_spanning_two_presences_links_both(batches):
    ev = EventLog(events=_events([(5, 15)]))
    log = attach_objects(ev, batches, [ObjectSpec("batch", "batch")])
    # end-timestamp linking (15) gave only B2
    assert _links(log.relations) == {"e0": ["B1", "B2"]}


def test_event_ending_after_object_left_still_links():
    # S1 [0, 10), then empty: nothing present at the event end (20)
    df = _signal("serial", [0, 10, 20], ["S1", "", ""])
    ev = EventLog(events=_events([(5, 20)]))
    log = attach_objects(ev, df, [ObjectSpec("serial", "serial")])
    assert _links(log.relations) == {"e0": ["S1"]}


def test_touching_is_not_overlap_and_points_keep_hand_over(batches):
    ev = EventLog(events=_events([(0, 10), (10, 20), (None, 10), (None, 20), (10, 10)]))
    log = attach_objects(ev, batches, [ObjectSpec("batch", "batch")])
    assert _links(log.relations) == {
        "e0": ["B1"],  # [0, 10) ends where B2 starts
        "e1": ["B2"],  # starts where B1 hands over
        "e2": ["B2"],  # point at the hand-over: the incoming object
        "e3": ["B2"],  # point at the last observation: closed end
        "e4": ["B2"],  # zero-length interval == point
    }


def test_point_event_before_and_after_presence_links_nothing(batches):
    ev = EventLog(events=_events([(None, -1), (None, 21), (-5, -1), (21, 30)]))
    log = attach_objects(ev, batches, [ObjectSpec("batch", "batch")])
    assert _links(log.relations) == {}


def test_link_events_to_objects_overlap_with_open_bounds():
    iv = pd.DataFrame(
        {
            "oid": ["A", "B"],
            "type": ["tool", "tool"],
            "start": [pd.NaT, _m(30)],
            "end": [_m(10), pd.NaT],
        }
    )
    rel = link_events_to_objects(_events([(5, 15), (20, 25), (28, 40)]), iv)
    assert _links(rel) == {"e0": ["A"], "e2": ["B"]}


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_lines():
    # the same timing on two lines
    df = pd.concat(
        [
            _signal("L1:batch", [0, 10, 20], ["B1", "B2", "B2"]),
            _signal("L2:batch", [0, 10, 20], ["C1", "C2", "C2"]),
        ],
        ignore_index=True,
    )
    scope = {
        "L1:batch": "L1",
        "L1:state": "L1",
        "L2:batch": "L2",
        "L2:state": "L2",
    }
    return df, [ObjectSpec("L1:batch", "batch"), ObjectSpec("L2:batch", "batch")], scope


def test_scope_restricts_links_to_the_same_line(two_lines):
    df, specs, scope = two_lines
    ev = EventLog(events=_events([(5, 15), (2, 4)], source=["L1:state", "L2:state"]))
    log = attach_objects(ev, df, specs, scope=scope)
    assert _links(log.relations) == {"e0": ["B1", "B2"], "e1": ["C1"]}


def test_scope_reads_plain_source_uuid_and_asset_relations(two_lines):
    df, specs, scope = two_lines
    events = _events([(5, 15)])
    events["source_uuid"] = ["L2:state"]
    rel = link_events_to_objects(events, object_intervals(df, specs), scope=scope)
    assert _links(rel) == {"e0": ["C1", "C2"]}
    # the asset relation of to_event_log names the source signal too
    state = pd.DataFrame(
        {
            "systime": [_m(0), _m(5), _m(15)],
            "uuid": "L1:state",
            "value_bool": [True, True, False],
        }
    )
    run = MachineStateEvents(state, "L1:state").detect_run_idle()
    log = to_event_log(run, detector="MachineStateEvents.detect_run_idle")
    log.events = log.events.drop(columns=["production:source_uuid"])
    out = attach_objects(log, df, specs, scope=scope)
    # run [0, 15] spans B1 and B2, idle is a point at 15; nothing of line 2
    assert sorted(_links(out.relations).values()) == [["B1", "B2"], ["B2"]]


def test_scope_from_to_event_log_source_column(two_lines):
    df, specs, scope = two_lines
    state = pd.DataFrame(
        {
            "systime": [_m(0), _m(15), _m(20)],
            "uuid": "L2:state",
            "value_bool": [True, False, False],
        }
    )
    run = MachineStateEvents(state, "L2:state").detect_run_idle()
    log = to_event_log(run, detector="MachineStateEvents.detect_run_idle")
    out = attach_objects(log, df, specs, scope=scope)
    links = _links(out.relations)
    # run [0, 15) spans C1 and C2, idle [15, 20] only C2; nothing of line 1
    assert sorted(links.values()) == [["C1", "C2"], ["C2"]]


def test_unscoped_event_and_object_warn(two_lines):
    df, specs, _ = two_lines
    ev = EventLog(events=_events([(5, 15)], source=["L9:state"]))
    with pytest.warns(DataQualityWarning, match="no scope") as rec:
        log = attach_objects(ev, df, specs, scope={"L1:batch": "L1"})
    messages = " ".join(str(w.message) for w in rec)
    assert "L9:state" in messages and "L2:batch" in messages
    assert _links(log.relations) == {}


def test_fanout_warning_without_scope(two_lines):
    df, specs, _ = two_lines
    ev = EventLog(events=_events([(None, 5)]))
    with pytest.warns(DataQualityWarning, match="pass scope="):
        log = attach_objects(ev, df, specs)
    assert _links(log.relations) == {"e0": ["B1", "C1"]}  # global linking kept
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        attach_objects(ev, df, specs, fanout_threshold=None)
        attach_objects(ev, df, specs, fanout_threshold=2)


def test_sequential_objects_of_one_signal_do_not_warn(batches):
    ev = EventLog(events=_events([(0, 20)]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        log = attach_objects(ev, batches, [ObjectSpec("batch", "batch")])
    assert _links(log.relations) == {"e0": ["B1", "B2"]}


def test_tz_mismatch_warns_and_reads_naive_as_utc(batches):
    iv = object_intervals(batches, [ObjectSpec("batch", "batch")])
    events = _events([(None, 5)])
    events[schema.OCEL_TIMESTAMP] = events[schema.OCEL_TIMESTAMP].dt.tz_localize(None)
    events = events.drop(columns=[schema.TS_START_TIMESTAMP])
    with pytest.warns(DataQualityWarning, match="tz-naive"):
        rel = link_events_to_objects(events, iv)
    assert _links(rel) == {"e0": ["B1"]}
    naive_iv = iv.assign(
        start=iv["start"].dt.tz_localize(None), end=iv["end"].dt.tz_localize(None)
    )
    with pytest.warns(DataQualityWarning, match="object timestamps are tz-naive"):
        log = relate(EventLog(events=_events([(None, 5)])), naive_iv)
    assert _links(log.relations) == {"e0": ["B1"]}


# ---------------------------------------------------------------------------
# Misread ids
# ---------------------------------------------------------------------------

TOKENS = [
    "NOREAD",
    "NO READ",
    "no_read",
    "ERR",
    "ERROR",
    "#N/A",
    "N/A",
    "NA",
    "NULL",
    "None",
    "nan",
    "????????",
    "?",
    "--------",
    "XXXXXXXX",
    "00000000",
    "0",
    "-1",
    "32767",
    "65535",
]


@pytest.mark.parametrize("token", TOKENS)
def test_each_misread_token_is_excluded_and_does_not_split(token):
    df = _signal("serial", [0, 5, 10, 20], ["A-100", token, "A-100", "A-100"])
    iv = object_intervals(df, [ObjectSpec("serial", "serial")])
    assert _spans(iv) == [("A-100", 0.0, 20.0)]


@pytest.mark.parametrize("token", [0, -1, 32767, 65535])
def test_integer_plc_defaults_are_excluded(token):
    df = _signal("serial", [0, 5, 10], [1001, token, 1001], column="value_integer")
    spec = ObjectSpec("serial", "serial", value_column="value_integer")
    assert _spans(object_intervals(df, [spec])) == [("1001", 0.0, 10.0)]


def test_misread_between_different_ids_ends_the_presence():
    df = _signal("serial", [0, 5, 10, 20], ["A", "NOREAD", "B", "B"])
    iv = object_intervals(df, [ObjectSpec("serial", "serial")])
    assert _spans(iv) == [("A", 0.0, 5.0), ("B", 10.0, 20.0)]


def test_misread_as_last_reading_ends_like_an_empty_one():
    df = pd.concat(
        [
            _signal("serial", [0, 10, 20], ["A", "A", "ERR"]),
            _signal("other", [30], ["X1"]),
        ],
        ignore_index=True,
    )
    iv = object_intervals(df, [ObjectSpec("serial", "serial")])
    assert _spans(iv) == [("A", 0.0, 20.0)]  # not held to the data end (30)


def test_blank_reading_keeps_ending_a_presence():
    df = _signal("serial", [0, 5, 10], ["A", "  ", "A"])
    iv = object_intervals(df, [ObjectSpec("serial", "serial")])
    assert _spans(iv) == [("A", 0.0, 5.0), ("A", 10.0, 10.0)]


def test_truncated_read_inside_a_presence_keeps_one_presence():
    df = _signal("serial", [0, 5, 10, 20], ["SN-0042", "SN-0", "SN-0042", "SN-0042"])
    iv = object_intervals(df, [ObjectSpec("serial", "serial")])
    assert _spans(iv) == [("SN-0042", 0.0, 20.0)]


def test_exclude_none_opts_out():
    df = _signal("serial", [0, 5, 10, 20], ["A", "NOREAD", "A", "A"])
    iv = object_intervals(df, [ObjectSpec("serial", "serial", exclude=None)])
    assert _spans(iv) == [("A", 0.0, 5.0), ("NOREAD", 5.0, 10.0), ("A", 10.0, 20.0)]


def test_exclude_collection_and_callable():
    df = _signal("serial", [0, 5, 10, 15, 20], ["A", "BAD", "A", "NOREAD", "A"])
    iv = object_intervals(df, [ObjectSpec("serial", "serial", exclude=["BAD"])])
    assert [o for o, _, _ in _spans(iv)] == ["A", "NOREAD", "A"]
    iv = object_intervals(
        df,
        [ObjectSpec("serial", "serial", exclude=lambda s: s.isin(["BAD", "NOREAD"]))],
    )
    assert _spans(iv) == [("A", 0.0, 20.0)]
    ints = _signal("n", [0, 5, 10], [7, 9, 7], column="value_integer")
    spec = ObjectSpec("n", "serial", value_column="value_integer", exclude={"9"})
    assert _spans(object_intervals(ints, [spec])) == [("7", 0.0, 10.0)]


def test_exclude_rejects_bad_rules():
    df = _signal("serial", [0, 5], ["A", "B"])
    with pytest.raises(ValueError, match="only text value"):
        object_intervals(df, [ObjectSpec("serial", "serial", exclude="NOREAD")])
    with pytest.raises(ValueError, match="mask of shape"):
        object_intervals(df, [ObjectSpec("serial", "serial", exclude=lambda s: [True])])


def test_is_misread_rules():
    values = ["B1", "NOREAD", "", "????", 0, "B2", None, np.nan, 1111, "1111"]
    assert is_misread(values).tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,  # an integer id is no filler
        True,  # a text of one repeated character is
    ]
    assert is_misread_value(65535.0) and is_misread_value("-32768")
    assert not is_misread_value("A") and not is_misread_value(True)
    assert not is_misread_value(pd.NA)
    # truncation: a prefix of the neighbour, shorter than the fixed length
    seq = ["SN-0001", "SN-00", "SN-0002", "SN-0003", "SN-0004"]
    assert is_misread(seq).tolist() == [False, True, False, False, False]
    assert not is_misread(seq, truncations=False).any()
    # variable-length ids: a shorter id is not a truncation of a longer one
    molds = ["M-1", "M-12", "M-2", "M-3", "M-4", "M-5"]
    assert not is_misread(molds).any()


# ---------------------------------------------------------------------------
# Hierarchy and o2o
# ---------------------------------------------------------------------------


@pytest.fixture()
def nested_lines():
    df = pd.concat(
        [
            _signal("L1:serial", [0, 10, 20], ["S1", "S2", "S2"]),
            _signal("L1:batch", [0, 20], ["B1", "B1"]),
            _signal("L1:hu", [0, 10, 20], ["H1", "H2", "H2"]),
            _signal("L2:serial", [0, 20], ["T1", "T1"]),
            # line 2's batch starts first, so on a tie a type-level pick
            # chooses it for line 1's serials too
            _signal("L2:batch", [-5, 20], ["C1", "C1"]),
        ],
        ignore_index=True,
    )
    specs = [
        ObjectSpec("L1:serial", "serial"),
        ObjectSpec("L1:batch", "batch"),
        ObjectSpec("L1:hu", "handling_unit"),
        ObjectSpec("L2:serial", "serial"),
        ObjectSpec("L2:batch", "batch"),
    ]
    return df, specs


def _pairs(o2o, qualifier="part_of"):
    sel = o2o[o2o[schema.OCEL_QUALIFIER] == qualifier]
    return set(zip(sel[schema.OCEL_OID], sel[schema.OCEL_OID2], strict=True))


def test_signal_level_hierarchy_two_parents_two_lines(nested_lines):
    df, specs = nested_lines
    log = detect_objects(
        df,
        specs,
        hierarchy={"L1:serial": ["L1:batch", "L1:hu"], "L2:serial": "L2:batch"},
    )
    assert _pairs(log.o2o) == {
        ("S1", "B1"),
        ("S1", "H1"),
        ("S2", "B1"),
        ("S2", "H2"),
        ("T1", "C1"),
    }
    assert set(log.o2o[schema.OCEL_QUALIFIER]) == {"part_of"}


def test_type_level_hierarchy_mixes_lines(nested_lines):
    df, specs = nested_lines
    log = detect_objects(df, specs, hierarchy={"serial": "batch"})
    # every serial overlaps C1 as long as B1 and C1 starts first
    assert _pairs(log.o2o) == {("S1", "C1"), ("S2", "C1"), ("T1", "C1")}
    # a list of parent types
    log = detect_objects(df, specs, hierarchy={"serial": ["handling_unit"]})
    assert _pairs(log.o2o) == {("S1", "H1"), ("S2", "H2"), ("T1", "H1")}


def test_default_o2o_has_no_co_occurs_and_opt_in_restores_it(nested_lines):
    df, specs = nested_lines
    assert detect_objects(df, specs).o2o.empty
    log = detect_objects(df, specs, infer_o2o="co_occurs")
    co = _pairs(log.o2o, "co_occurs")
    # batch < handling_unit < serial; touching end-to-start does not count
    assert co == {
        ("B1", "H1"),
        ("B1", "H2"),
        ("C1", "H1"),
        ("C1", "H2"),
        ("B1", "S1"),
        ("B1", "S2"),
        ("B1", "T1"),
        ("C1", "S1"),
        ("C1", "S2"),
        ("C1", "T1"),
        ("H1", "S1"),
        ("H1", "T1"),
        ("H2", "S2"),
        ("H2", "T1"),
    }
    assert (
        _pairs(derive_o2o(object_intervals(df, specs), co_occurs=True), "co_occurs")
        == co
    )
    # with a hierarchy, pairs related by part_of are not repeated as co_occurs
    both = detect_objects(
        df, specs, hierarchy={"L2:serial": "L2:batch"}, infer_o2o="co_occurs"
    )
    assert _pairs(both.o2o) == {("T1", "C1")}
    assert _pairs(both.o2o, "co_occurs") == co - {("C1", "T1")}
    assert detect_objects(df, specs, infer_o2o=False).o2o.empty
    with pytest.raises(ValueError, match="infer_o2o"):
        detect_objects(df, specs, infer_o2o="all")


# ---------------------------------------------------------------------------
# Grouped _detect == the former per-spec filtering
# ---------------------------------------------------------------------------


def _ref_segments(sig, value_column, time_column, min_duration):
    times = om._ns(sig[time_column])
    codes, uniques = om._id_codes(sig[value_column].reset_index(drop=True))
    rows = np.arange(len(codes), dtype=np.int64)
    c, s, e, first = om._rle(codes, times, times)
    first = rows[first]
    if min_duration is not None and len(c) > 1:
        threshold = pd.Timedelta(min_duration).value
        while len(c) > 1:
            dur = e - s
            short = (c >= 0) & (dur < threshold)
            short[-1] = False
            if not short.any():
                break
            big = np.iinfo(np.int64).max
            d = np.where(short, dur, big)
            left = np.append(big, d[:-1])
            right = np.append(d[1:], big)
            keep = ~(short & (d < left) & (d <= right))
            c, s, last, first = c[keep], s[keep], e[keep], first[keep]
            c, s, e, k = om._rle(c, s, last)
            first = first[k]
    ids = c >= 0
    return c[ids], s[ids], e[ids], first[ids], uniques


def _ref_attribute(df, attr_uuid, starts, ends, is_last, value_column):
    sig = df[df["uuid"] == attr_uuid]
    if sig.empty:
        return pd.Series([pd.NA] * len(starts), dtype="object"), None
    col = next(
        (
            c
            for c in (value_column, "value_string", "value_double", "value_integer")
            if c in sig.columns and sig[c].notna().any()
        ),
        value_column,
    )
    sig = sig[sig["systime"].notna() & sig[col].notna()]
    sig = sig.sort_values("systime", kind="stable")
    t = om._ns(sig["systime"])
    vals = sig[col].to_numpy(dtype=object)
    if len(t) == 0:
        return pd.Series([pd.NA] * len(starts), dtype="object"), None
    pos = np.searchsorted(t, starts, side="right") - 1
    at_start = np.where(pos >= 0, vals[pos.clip(min=0)], pd.NA)
    changed = np.ones(len(vals), dtype=bool)
    changed[1:] = vals[1:] != vals[:-1]
    ct, cv = t[changed], vals[changed]
    owner = np.searchsorted(starts, ct, side="right") - 1
    o = owner.clip(min=0)
    inside = (
        (owner >= 0)
        & (ct > starts[o])
        & ((ct < ends[o]) | (is_last[o] & (ct == ends[o])))
    )
    return pd.Series(at_start, dtype="object"), (o[inside], cv[inside], ct[inside])


def _reference_detect(df, specs):
    """The former per-spec path: filter the frame by uuid for every spec."""
    frames, changes = [], []
    available = set(df["uuid"].unique())
    data_end = int(np.max(om._ns(df["systime"].dropna())))
    for spec in specs:
        if spec.uuid not in available:
            continue
        sig = df[df["uuid"] == spec.uuid]
        sig = sig[sig["systime"].notna()].sort_values("systime", kind="stable")
        sig = sig.reset_index(drop=True)
        codes, start, end, first, uniques = _ref_segments(
            sig, spec.value_column, "systime", spec.min_duration
        )
        if len(codes) == 0:
            continue
        last = sig[spec.value_column].iloc[-1]
        still = not (
            pd.isna(last) or (isinstance(last, str) and last.strip(" \t\r\n\x00") == "")
        )
        if still and end[-1] < data_end:
            end = end.copy()
            end[-1] = data_end
        oid = om._render_oids(uniques, spec.id_template, spec.object_type).to_numpy()[
            codes
        ]
        out = pd.DataFrame(
            {
                "oid": pd.array(oid, dtype="string"),
                "type": pd.array([spec.object_type] * len(codes), dtype="string"),
                "start": om._from_ns(start),
                "end": om._from_ns(end),
                "duration_seconds": (end - start) / 1e9,
                "uuid": pd.array([spec.uuid] * len(codes), dtype="string"),
                "source_uuid": pd.array([pd.NA] * len(codes), dtype="string"),
            }
        )
        is_last = np.zeros(len(codes), dtype=bool)
        is_last[-1] = True
        for name, attr in spec.attributes.items():
            at_start, during = _ref_attribute(
                df, attr, start, end, is_last, spec.value_column
            )
            out[name] = at_start
            if during is not None and len(during[0]):
                changes.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_OID: out["oid"].to_numpy()[during[0]],
                            schema.OCEL_TYPE: spec.object_type,
                            schema.OCEL_FIELD: name,
                            schema.OCEL_VALUE: during[1],
                            schema.OCEL_TIMESTAMP: om._from_ns(during[2]).to_numpy(),
                        }
                    )
                )
        frames.append(out)
    return pd.concat(frames, ignore_index=True), pd.concat(changes, ignore_index=True)


def test_grouped_detect_matches_per_spec_filtering():
    frame, labels = make_object_signals(seed=1, days=1)
    ids = labels[labels["role"] == "identifier"].reset_index(drop=True)
    uuids = list(labels["uuid"])
    specs = []
    for i, row in enumerate(ids.itertuples()):
        attributes = (
            {"a1": uuids[(7 * i) % len(uuids)], "a2": uuids[(11 * i) % len(uuids)]}
            if i % 4 == 0
            else {}
        )
        specs.append(
            ObjectSpec(
                row.uuid,
                row.object_type,
                value_column=row.value_column,
                min_duration="30s" if i % 3 == 0 else None,
                attributes=attributes,
                exclude=None,
            )
        )
    got, got_changes = om._detect(
        frame, specs, uuid_column="uuid", time_column="systime"
    )
    want, want_changes = _reference_detect(frame, specs)
    pd.testing.assert_frame_equal(got, want)
    pd.testing.assert_frame_equal(got_changes, want_changes)


# ---------------------------------------------------------------------------
# The benchmark chain: scoped linking keeps the lines apart
# ---------------------------------------------------------------------------


@pytest.fixture()
def benchmark(monkeypatch):
    """make_object_signals(seed=1) plus each uuid's line (ground truth)."""
    key_of: dict[str, str] = {}
    current: dict[str, str] = {}
    sample, frame_fn = gen._sample, gen._frame

    def _sample(rng, truth, *args, **kwargs):
        current["key"] = truth.key
        return sample(rng, truth, *args, **kwargs)

    def _frame(t0, uid, *args, **kwargs):
        key_of[uid] = current["key"]
        return frame_fn(t0, uid, *args, **kwargs)

    monkeypatch.setattr(gen, "_sample", _sample)
    monkeypatch.setattr(gen, "_frame", _frame)
    frame, labels = make_object_signals(seed=1, days=1)
    return frame, labels, {u: k.split(".")[0] for u, k in key_of.items()}


def test_benchmark_chain_scoped_links_stay_on_their_line(benchmark):
    frame, labels, line_of = benchmark
    assert {"L0", "L1"} <= set(line_of.values())
    ids = labels[labels["role"] == "identifier"]
    specs = [
        ObjectSpec(r.uuid, r.object_type, value_column=r.value_column)
        for r in ids.itertuples()
    ]
    logs = []
    for uid, line in line_of.items():
        if labels.set_index("uuid").loc[uid, "variant"].startswith("machine_state"):
            sig = frame[frame["uuid"] == uid]
            col = labels.set_index("uuid").loc[uid, "value_column"]
            run = sig[col].astype("string").isin(["RUN", "Running", "PRODUKTION"])
            st = pd.DataFrame(
                {
                    "systime": sig["systime"],
                    "uuid": uid,
                    "value_bool": run.fillna(False).astype(bool),
                }
            )
            ev = MachineStateEvents(st, uid).detect_run_idle()
            logs.append(to_event_log(ev, detector="MachineStateEvents.detect_run_idle"))
    assert len(logs) == 2
    events = concat(logs[0], logs[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        unscoped = attach_objects(events, frame, specs)
        scoped = attach_objects(events, frame, specs, scope=line_of)
    oid_line = {}
    iv = object_intervals(frame, specs)
    for oid, uuid in zip(iv["oid"], iv["uuid"], strict=True):
        oid_line.setdefault(oid, set()).add(line_of[uuid])
    ev_line = {}
    rel = events.relations
    for eid, oid in zip(rel[schema.OCEL_EID], rel[schema.OCEL_OID], strict=True):
        ev_line[eid] = line_of[oid]

    def cross(log):
        r = log.relations[log.relations[schema.OCEL_TYPE] != "asset"]
        return sum(
            ev_line[e] not in oid_line[o]
            for e, o in zip(r[schema.OCEL_EID], r[schema.OCEL_OID], strict=True)
        )

    assert cross(unscoped) > 0
    assert cross(scoped) == 0
    # no misread became an object
    assert not is_misread(iv["oid"].astype(object), truncations=False).any()
