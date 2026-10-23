"""Verification of the object-detection layer (ts_shape.eventlog.objects).

Realistic id-signal scenarios with hand-derived expected outputs: exact
presence boundaries, object identity, event links and relations.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.eventlog import (
    EventLog,
    ObjectSpec,
    attach_objects,
    detect_objects,
    link_events_to_objects,
    object_intervals,
    relate,
    schema,
    to_event_log_ocel,
    validate,
)

T0 = pd.Timestamp("2025-01-01 06:00", tz="UTC")
INTERVAL_COLUMNS = [
    "oid",
    "type",
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
]


def _m(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def _signal(uuid, minutes, values, column="value_string", dtype=None, source=None):
    vals = pd.array(values, dtype=dtype) if dtype else list(values)
    frame = pd.DataFrame(
        {"systime": [_m(x) for x in minutes], "uuid": uuid, column: vals}
    )
    if source is not None:
        frame["source_uuid"] = source
    return frame


def _events(minutes, prefix="e"):
    n = len(minutes)
    return pd.DataFrame(
        {
            schema.OCEL_EID: [f"{prefix}{i}" for i in range(n)],
            schema.OCEL_ACTIVITY: ["inspect"] * n,
            schema.OCEL_TIMESTAMP: [_m(x) for x in minutes],
            schema.TS_DETECTOR: ["d"] * n,
            schema.TS_PACK: ["quality"] * n,
        }
    )


def _spans(iv):
    return [(o, s, e) for o, s, e in zip(iv["oid"], iv["start"], iv["end"])]


def _links(rel):
    return {eid: sorted(g[schema.OCEL_OID]) for eid, g in rel.groupby(schema.OCEL_EID)}


def _lifecycle(log, oid):
    oc = log.object_changes
    sel = oc[(oc[schema.OCEL_OID] == oid) & (oc[schema.OCEL_FIELD] == "lifecycle")]
    return list(zip(sel[schema.OCEL_VALUE], sel[schema.OCEL_TIMESTAMP]))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [None, "object", "str", "string", "string[pyarrow]"])
def test_every_string_dtype_is_segmented(dtype):
    # Regression: a plain object / str column crashed (read-only numpy view).
    df = _signal("s", [0, 5, 10, 15], ["A", "A", "B", "B"], dtype=dtype)
    iv = object_intervals(df, [ObjectSpec("s", "part")])
    assert _spans(iv) == [("A", _m(0), _m(10)), ("B", _m(10), _m(15))]
    assert list(iv.columns) == INTERVAL_COLUMNS
    assert iv["duration_seconds"].tolist() == [600.0, 300.0]


@pytest.mark.parametrize("empty", ["", "   ", None, np.nan, "\x00\x00"])
def test_empty_or_missing_id_ends_the_object(empty):
    df = _signal("s", [0, 10, 20, 30], ["A", empty, "A", "B"], dtype="object")
    iv = object_intervals(df, [ObjectSpec("s", "part")])
    # A is gone while the reading is empty; it re-appears as the SAME object.
    assert _spans(iv) == [
        ("A", _m(0), _m(10)),
        ("A", _m(20), _m(30)),
        ("B", _m(30), _m(30)),
    ]
    log = detect_objects(df, [ObjectSpec("s", "part")])
    assert sorted(log.objects[schema.OCEL_OID]) == ["A", "B"]
    assert _lifecycle(log, "A") == [
        ("active", _m(0)),
        ("released", _m(10)),
        ("active", _m(20)),
        ("released", _m(30)),
    ]


def test_id_held_across_a_data_gap():
    df = _signal("s", [0, 1, 180, 181], ["A", "A", "A", "B"])
    iv = object_intervals(df, [ObjectSpec("s", "carrier")])
    assert _spans(iv) == [("A", _m(0), _m(181)), ("B", _m(181), _m(181))]
    rel = link_events_to_objects(_events([90]), iv)
    assert _links(rel) == {"e0": ["A"]}


def test_integer_ids_with_missing_values_render_as_integers():
    df = _signal("s", [0, 10, 20, 30], [1001, np.nan, 1002, 1003], "value_integer")
    assert df["value_integer"].dtype == "float64"
    iv = object_intervals(df, [ObjectSpec("s", "order", value_column="value_integer")])
    assert iv["oid"].tolist() == ["1001", "1002", "1003"]
    assert iv["end"].tolist() == [_m(10), _m(30), _m(30)]


def test_nullable_integer_ids():
    df = _signal("s", [0, 10, 20], [7, pd.NA, 8], "value_integer", dtype="Int64")
    iv = object_intervals(
        df,
        [
            ObjectSpec(
                "s", "order", value_column="value_integer", id_template="{type}:{value}"
            )
        ],
    )
    assert _spans(iv) == [("order:7", _m(0), _m(10)), ("order:8", _m(20), _m(20))]


def test_unsorted_input_and_duplicate_rows():
    df = _signal("s", [10, 0, 5, 5, 20], ["B", "A", "A", "A", "C"])
    iv = object_intervals(df, [ObjectSpec("s", "part")])
    assert _spans(iv) == [
        ("A", _m(0), _m(10)),
        ("B", _m(10), _m(20)),
        ("C", _m(20), _m(20)),
    ]


def test_min_duration_on_change_encoded_signal_keeps_every_part():
    # One sample per change: each id is present until the next change.
    df = _signal("s", [0, 10, 20, 30], ["A", "B", "C", "D"])
    iv = object_intervals(df, [ObjectSpec("s", "part", min_duration="1min")])
    assert iv["oid"].tolist() == ["A", "B", "C", "D"]  # D: open-ended, kept


def test_min_duration_debounces_a_blip_and_merges_around_it():
    df = _signal("s", [0, 10, 10 + 5 / 60, 20], ["A", "X", "A", "B"])
    iv = object_intervals(df, [ObjectSpec("s", "part", min_duration="1min")])
    assert _spans(iv) == [("A", _m(0), _m(20)), ("B", _m(20), _m(20))]


def test_tz_aware_berlin_across_dst_and_naive_input():
    berlin = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2025-03-30 01:30", "2025-03-30 03:30", "2025-03-30 04:00"]
            ).tz_localize("Europe/Berlin"),
            "uuid": "s",
            "value_string": ["X", "Y", "Y"],
        }
    )
    iv = object_intervals(berlin, [ObjectSpec("s", "batch")])
    assert str(iv["start"].dt.tz) == "UTC"
    assert iv["start"].tolist() == [
        pd.Timestamp("2025-03-30 00:30", tz="UTC"),
        pd.Timestamp("2025-03-30 01:30", tz="UTC"),
    ]
    assert iv["duration_seconds"].tolist() == [3600.0, 1800.0]  # 1 h across DST

    naive = berlin.assign(systime=berlin["systime"].dt.tz_localize(None))
    iv2 = object_intervals(naive, [ObjectSpec("s", "batch")])
    assert iv2["start"].iloc[0] == pd.Timestamp("2025-03-30 01:30", tz="UTC")


def test_source_uuid_and_signal_are_carried():
    df = _signal("st1:serial", [0, 5], ["P1", "P2"], source="station-1")
    iv = object_intervals(df, [ObjectSpec("st1:serial", "part")])
    assert iv["uuid"].tolist() == ["st1:serial"] * 2
    assert iv["source_uuid"].tolist() == ["station-1"] * 2


def test_empty_frame_and_unknown_signal_give_full_schema():
    empty = pd.DataFrame(columns=["systime", "uuid", "value_string"])
    iv = object_intervals(empty, [ObjectSpec("s", "part")])
    assert list(iv.columns) == INTERVAL_COLUMNS and iv.empty
    assert str(iv["start"].dtype) == "datetime64[ns, UTC]"

    df = _signal("other", [0], ["A"])
    with pytest.warns(DataQualityWarning, match="'s' is not in the frame"):
        iv = object_intervals(df, [ObjectSpec("s", "part")])
    assert iv.empty and list(iv.columns) == INTERVAL_COLUMNS


def test_missing_columns_raise():
    df = _signal("s", [0], ["A"])
    with pytest.raises(ColumnNotFoundError, match="value_integer"):
        object_intervals(df, [ObjectSpec("s", "part", value_column="value_integer")])
    with pytest.raises(ColumnNotFoundError, match="ts"):
        object_intervals(df, [ObjectSpec("s", "part")], time_column="ts")
    with pytest.raises(ValueError, match="reserved"):
        object_intervals(df, [ObjectSpec("s", "part", attributes={"start": "x"})])


# ---------------------------------------------------------------------------
# Event linking at boundaries
# ---------------------------------------------------------------------------


def test_event_boundaries_hand_over_clear_and_last_sample():
    df = _signal("s", [0, 10, 20, 30, 40], ["A", "B", "", "C", "C"])
    iv = object_intervals(df, [ObjectSpec("s", "part")])
    assert _spans(iv) == [
        ("A", _m(0), _m(10)),
        ("B", _m(10), _m(20)),
        ("C", _m(30), _m(40)),
    ]
    ev = _events([-1, 0, 10, 20, 25, 30, 40, 41])
    rel = link_events_to_objects(ev, iv)
    assert _links(rel) == {
        "e1": ["A"],  # first sample
        "e2": ["B"],  # hand-over instant belongs to the incoming id only
        "e3": ["B"],  # id cleared at 20: the outgoing part owns that instant
        "e5": ["C"],
        "e6": ["C"],  # last observation is inside the last presence
    }  # e0 before data, e4 while empty, e7 after data: no object


def test_simultaneous_hand_over_on_two_stations():
    st1 = _signal("st1", [0, 10, 20], ["B1", "B2", "B2"])
    st2 = _signal("st2", [0, 10, 20], ["C1", "C2", "C2"])
    df = pd.concat([st1, st2], ignore_index=True)
    specs = [ObjectSpec("st1", "batch"), ObjectSpec("st2", "carrier")]
    ev = EventLog(events=_events([10, 5]))
    log = attach_objects(ev, df, specs, infer_o2o="co_occurs")
    assert _links(log.relations) == {"e0": ["B2", "C2"], "e1": ["B1", "C1"]}
    pairs = set(zip(log.o2o[schema.OCEL_OID], log.o2o[schema.OCEL_OID2]))
    assert pairs == {("B1", "C1"), ("B2", "C2")}  # touching is not overlapping
    assert set(log.o2o[schema.OCEL_QUALIFIER]) == {"co_occurs"}


def test_same_serial_handed_from_station_to_station():
    st1 = _signal("st1", [0, 10, 20], ["P1", "P2", "P3"])
    st2 = _signal("st2", [10, 20, 30], ["P1", "P2", "P3"])
    df = pd.concat([st1, st2], ignore_index=True)
    specs = [ObjectSpec("st1", "part"), ObjectSpec("st2", "part")]
    ev = EventLog(events=_events([10, 15]))
    # Two parts present at once from two signals: the fan-out check speaks up.
    with pytest.warns(DataQualityWarning, match="present at the same time"):
        log = attach_objects(ev, df, specs)
    # At 10 P1 arrives at st2 while P2 enters st1: both present.
    assert _links(log.relations) == {"e0": ["P1", "P2"], "e1": ["P1", "P2"]}
    assert sorted(log.objects[schema.OCEL_OID]) == ["P1", "P2", "P3"]
    # Continuous presence across the two stations: one active, one released.
    assert _lifecycle(log, "P1") == [("active", _m(0)), ("released", _m(20))]
    assert _lifecycle(log, "P2") == [("active", _m(10)), ("released", _m(30))]


# ---------------------------------------------------------------------------
# Object-to-object relations
# ---------------------------------------------------------------------------


@pytest.fixture()
def nested():
    """Parts on carriers within batches; part P3 starts 2 s before its carrier."""
    batch = _signal("batch", [0, 60, 120], ["B1", "B2", "B2"])
    carrier = _signal("carrier", [0, 30, 60, 90, 120], ["K1", "K2", "K1", "K3", "K3"])
    jitter = 30 - 2 / 60
    part = _signal(
        "part",
        [0, 10, jitter, 40, 60, 75, 90, 105, 120],
        ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P8"],
    )
    df = pd.concat([batch, carrier, part], ignore_index=True)
    specs = [
        ObjectSpec("batch", "batch"),
        ObjectSpec("carrier", "carrier"),
        ObjectSpec("part", "part"),
    ]
    return df, specs


def test_nested_hierarchy_part_of_picks_the_owning_parent(nested):
    df, specs = nested
    log = detect_objects(df, specs, hierarchy={"part": "carrier", "carrier": "batch"})
    got = set(zip(log.o2o[schema.OCEL_OID], log.o2o[schema.OCEL_OID2]))
    assert got == {
        ("P1", "K1"),
        ("P2", "K1"),
        ("P3", "K2"),  # overlaps K1 for 2 s, K2 for 10 min
        ("P4", "K2"),
        ("P5", "K1"),  # K1 re-used in batch B2
        ("P6", "K1"),
        ("P7", "K3"),
        ("P8", "K3"),
        ("K1", "B1"),
        ("K2", "B1"),
        ("K1", "B2"),
        ("K3", "B2"),
    }
    assert set(log.o2o[schema.OCEL_QUALIFIER]) == {"part_of"}
    assert not log.o2o.duplicated().any()


def test_nested_without_hierarchy_is_co_occurs(nested):
    df, specs = nested
    assert detect_objects(df, specs).o2o.empty  # co_occurs is opt-in
    log = detect_objects(df, specs, infer_o2o="co_occurs")
    got = set(zip(log.o2o[schema.OCEL_OID], log.o2o[schema.OCEL_OID2]))
    # Types sort batch < carrier < part; pairs share time (touching excluded).
    expected_bk = {("B1", "K1"), ("B1", "K2"), ("B2", "K1"), ("B2", "K3")}
    expected_bp = {("B1", f"P{i}") for i in (1, 2, 3, 4)} | {
        ("B2", f"P{i}") for i in (5, 6, 7, 8)
    }
    expected_kp = {
        ("K1", "P1"),
        ("K1", "P2"),
        ("K1", "P3"),
        ("K2", "P3"),
        ("K2", "P4"),
        ("K1", "P5"),
        ("K1", "P6"),
        ("K3", "P7"),
        ("K3", "P8"),
    }
    assert got == expected_bk | expected_bp | expected_kp
    assert set(log.o2o[schema.OCEL_QUALIFIER]) == {"co_occurs"}


def test_attribute_value_at_start_and_changes_during_presence():
    batch = _signal("batch", [0, 60, 90], ["B1", "B2", "B2"])
    recipe = _signal("recipe", [-5, 30, 45, 70], ["R1", "R2", None, "R3"])
    df = pd.concat([batch, recipe], ignore_index=True)
    spec = ObjectSpec("batch", "batch", attributes={"recipe": "recipe"})
    iv = object_intervals(df, [spec])
    assert iv["recipe"].tolist() == ["R1", "R2"]
    log = detect_objects(df, [spec])
    oc = log.object_changes
    rc = oc[oc[schema.OCEL_FIELD] == "recipe"]
    assert list(
        zip(rc[schema.OCEL_OID], rc[schema.OCEL_VALUE], rc[schema.OCEL_TIMESTAMP])
    ) == [
        ("B1", "R1", _m(0)),
        ("B1", "R2", _m(30)),
        ("B2", "R2", _m(60)),
        ("B2", "R3", _m(70)),
    ]


# ---------------------------------------------------------------------------
# relate() with a user object list
# ---------------------------------------------------------------------------


def test_relate_open_ended_object_and_invalid_interval():
    iv = pd.DataFrame(
        {
            "oid": ["WO1", "WO2"],
            "type": ["work_order", "work_order"],
            "start": [_m(0), _m(60)],
            "end": [_m(60), pd.NaT],
        }
    )
    log = relate(EventLog(events=_events([30, 60, 600])), iv)
    assert _links(log.relations) == {"e0": ["WO1"], "e1": ["WO2"], "e2": ["WO2"]}
    assert _lifecycle(log, "WO2") == [("active", _m(60))]  # still active

    bad = iv.assign(end=[_m(-1), pd.NaT])
    with pytest.raises(ValueError, match="ends before it starts"):
        relate(EventLog(events=_events([30])), bad)


def test_oid_shared_by_two_types_warns():
    df = pd.concat(
        [_signal("a", [0, 5], ["100", "100"]), _signal("b", [0, 5], ["100", "100"])]
    )
    with pytest.warns(DataQualityWarning, match="more than one object type"):
        detect_objects(df, [ObjectSpec("a", "batch"), ObjectSpec("b", "order")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect_objects(
            df,
            [
                ObjectSpec("a", "batch", id_template="{type}:{value}"),
                ObjectSpec("b", "order", id_template="{type}:{value}"),
            ],
        )


# ---------------------------------------------------------------------------
# Scale and round trip
# ---------------------------------------------------------------------------


def test_round_trip_10k_short_lived_objects():
    rng = np.random.default_rng(7)
    n_parts = 12_000
    # Irregular part durations (2..40 s), 4 samples per part, ~5% empty gaps.
    durations = rng.integers(2, 41, n_parts)
    starts = np.concatenate([[0], np.cumsum(durations)[:-1]]).astype(float)
    gap = rng.random(n_parts) < 0.05
    rows_t, rows_v = [], []
    for i in range(n_parts):
        for k in range(4):
            rows_t.append(starts[i] + durations[i] * k / 4)
            rows_v.append(f"S{i % 9000}")  # ids repeat later: same object
        if gap[i]:
            rows_t[-1] = starts[i] + durations[i] * 0.9
            rows_v[-1] = ""
    t = T0 + pd.to_timedelta(np.asarray(rows_t), unit="s")
    part = pd.DataFrame({"systime": t, "uuid": "serial", "value_string": rows_v})
    # Batches of 100 parts on a second signal.
    b_idx = np.arange(0, n_parts, 100)
    batch = pd.DataFrame(
        {
            "systime": T0 + pd.to_timedelta(starts[b_idx], unit="s"),
            "uuid": "batch",
            "value_string": [f"B{i}" for i in range(len(b_idx))],
        }
    )
    df = pd.concat([part, batch], ignore_index=True).sample(frac=1, random_state=1)

    # Events: 20k random instants over the whole run.
    total = float(starts[-1] + durations[-1])
    ev_s = np.sort(rng.uniform(-10, total + 10, 20_000))
    events = pd.DataFrame(
        {
            schema.OCEL_EID: [f"e{i}" for i in range(len(ev_s))],
            schema.OCEL_ACTIVITY: "inspect",
            schema.OCEL_TIMESTAMP: T0 + pd.to_timedelta(ev_s, unit="s"),
            schema.TS_DETECTOR: "d",
            schema.TS_PACK: "quality",
        }
    )
    specs = [ObjectSpec("serial", "serial"), ObjectSpec("batch", "batch")]
    began = time.perf_counter()
    log = attach_objects(
        EventLog(events=events), df, specs, hierarchy={"serial": "batch"}
    )
    elapsed = time.perf_counter() - began
    assert elapsed < 10, elapsed

    iv = object_intervals(df, specs)
    serial = iv[iv["type"] == "serial"].reset_index(drop=True)
    # Every part occurrence is exactly one presence.
    assert len(serial) == n_parts
    assert serial["oid"].tolist() == [f"S{i % 9000}" for i in range(n_parts)]
    assert set(log.objects[schema.OCEL_OID]) == {f"S{i}" for i in range(9000)} | {
        f"B{i}" for i in range(len(b_idx))
    }

    # Expected serial per event, computed independently.
    p_start = starts
    p_end = np.where(gap, starts + durations * 0.9, np.append(starts[1:], np.nan))
    p_end[-1] = starts[-1] + durations[-1] * (0.9 if gap[-1] else 0.75)
    i = np.searchsorted(p_start, ev_s, side="right") - 1
    ok = i >= 0
    ic = i.clip(min=0)
    last = ic == n_parts - 1
    inside = ok & ((ev_s < p_end[ic]) | ((ev_s <= p_end[ic]) & (last | gap[ic])))
    expected = {f"e{k}": f"S{ic[k] % 9000}" for k in range(len(ev_s)) if inside[k]}
    rel = log.relations[log.relations[schema.OCEL_TYPE] == "serial"]
    assert dict(zip(rel[schema.OCEL_EID], rel[schema.OCEL_OID])) == expected
    assert not rel[schema.OCEL_EID].duplicated().any()
    assert not log.relations.duplicated().any()
    # Each serial presence is part_of exactly the batch that owns its start.
    part_of = log.o2o
    assert set(part_of[schema.OCEL_QUALIFIER]) == {"part_of"}
    assert not part_of.duplicated().any()
    validate(log)


def test_100k_rows_is_fast():
    n = 25_000
    t = T0 + pd.to_timedelta(np.arange(n * 4) * 2, unit="s")
    vals = np.repeat([f"P{i}" for i in range(n)], 4)
    df = pd.DataFrame({"systime": t, "uuid": "serial", "value_string": vals})
    began = time.perf_counter()
    log = detect_objects(df, [ObjectSpec("serial", "serial")])
    assert time.perf_counter() - began < 5
    assert len(log.objects) == n
    assert (log.object_changes[schema.OCEL_FIELD] == "lifecycle").sum() == 2 * n


# ---------------------------------------------------------------------------
# OCEL export
# ---------------------------------------------------------------------------


def test_pm4py_layout_of_ocel_tables(nested):
    df, specs = nested
    spec = ObjectSpec("batch", "batch", attributes={"first_part": "part"})
    log = attach_objects(EventLog(events=_events([5, 65])), df, [spec, *specs[1:]])
    plain = to_event_log_ocel(log)
    assert schema.OCEL_VALUE in plain.object_changes.columns
    t = to_event_log_ocel(log, pm4py=True)
    assert list(t.relations.columns[:6]) == [
        schema.OCEL_EID,
        schema.OCEL_ACTIVITY,
        schema.OCEL_TIMESTAMP,
        schema.OCEL_OID,
        schema.OCEL_TYPE,
        schema.OCEL_QUALIFIER,
    ]
    assert t.relations[schema.OCEL_ACTIVITY].eq("inspect").all()
    oc = t.object_changes
    assert schema.OCEL_VALUE not in oc.columns
    assert {"lifecycle", "first_part"} <= set(oc.columns)
    first = oc[oc[schema.OCEL_FIELD] == "first_part"]
    at_start = first[first[schema.OCEL_TIMESTAMP].isin([_m(0), _m(60)])]
    assert at_start["first_part"].tolist() == ["P1", "P5"]
    assert first["lifecycle"].isna().all()


def test_pyarrow_timestamps_and_ids():
    pa = pytest.importorskip("pyarrow")
    times = pd.Series(
        pd.date_range("2025-01-01 07:00", periods=4, freq="5min", tz="Europe/Berlin")
    ).astype(pd.ArrowDtype(pa.timestamp("us", tz="Europe/Berlin")))
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": "s",
            "value_integer": pd.Series(
                [1, 1, None, 2], dtype=pd.ArrowDtype(pa.int64())
            ),
        }
    )
    iv = object_intervals(df, [ObjectSpec("s", "order", value_column="value_integer")])
    assert _spans(iv) == [("1", _m(0), _m(10)), ("2", _m(15), _m(15))]


def test_last_id_on_a_change_encoded_signal_stays_present_to_the_data_end():
    import pandas as pd

    from ts_shape.eventlog import ObjectSpec, object_intervals

    t = pd.date_range("2024-01-01 06:00", periods=6, freq="10min", tz="UTC")
    df = pd.concat(
        [
            pd.DataFrame(
                {"systime": t[[0, 3]], "uuid": "batch", "value_string": ["B1", "B2"]}
            ),
            pd.DataFrame({"systime": t, "uuid": "temp", "value_string": "x"}),
        ],
        ignore_index=True,
    )
    out = object_intervals(df, [ObjectSpec("batch", "batch")])
    assert out["end"].tolist() == [t[3], t[5]]  # B2 is still running
    # A cleared id is not extended.
    cleared = pd.concat(
        [df, pd.DataFrame({"systime": [t[4]], "uuid": "batch", "value_string": [""]})]
    )
    out = object_intervals(cleared, [ObjectSpec("batch", "batch")])
    assert out["end"].tolist() == [t[3], t[4]]
