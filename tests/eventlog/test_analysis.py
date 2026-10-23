"""Object analytics (ts_shape.eventlog.analysis) on a hand-built line.

Two stations, three serials. Every KPI below is computed by hand from the
presences, which are (minutes after 06:00 UTC):

=====  ===============  ===============
serial station st1        station st2
=====  ===============  ===============
S1     [0, 10)          [12, 25)
S2     [10, 20)         [25, 40)
S3     [20, 40)         [40, 45)
=====  ===============  ===============
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import DataQualityWarning
from ts_shape.eventlog import (
    CarrierLoops,
    EventLog,
    ObjectKPIs,
    ObjectSpec,
    attach_counts,
    carrier_loops,
    concat,
    detect_objects,
    object_intervals,
    object_kpis,
    presence_events,
    schema,
    to_event_log,
    to_event_log_ocel,
    to_event_log_xes,
    wip_over_time,
)
from ts_shape.eventlog.analysis import (
    CARRIER_COLUMNS,
    COUNT_COLUMNS,
    LOOP_COLUMNS,
    OBJECT_COLUMNS,
    PRESENCE_EVENT_COLUMNS,
    STATION_COLUMNS,
    THROUGHPUT_COLUMNS,
    WIP_COLUMNS,
)

T0 = pd.Timestamp("2025-03-10 06:00", tz="UTC")
SPECS = [ObjectSpec("st1:serial", "serial"), ObjectSpec("st2:serial", "serial")]


def _m(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def _signal(uuid, source, minutes, values):
    return pd.DataFrame(
        {
            "systime": [_m(x) for x in minutes],
            "uuid": uuid,
            "source_uuid": source,
            "value_string": values,
        }
    )


@pytest.fixture
def line() -> pd.DataFrame:
    return pd.concat(
        [
            _signal("st1:serial", "st1", [0, 10, 20, 40], ["S1", "S2", "S3", ""]),
            _signal("st2:serial", "st2", [12, 25, 40, 45], ["S1", "S2", "S3", ""]),
        ],
        ignore_index=True,
    )


@pytest.fixture
def intervals(line) -> pd.DataFrame:
    return object_intervals(line, SPECS)


def _naive(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in ("start", "end"):
        out[c] = out[c].dt.tz_localize(None)
    return out


# ---------------------------------------------------------------------------
# Presence events and the event log
# ---------------------------------------------------------------------------


def test_presence_events_one_enter_and_leave_per_presence(intervals):
    pe = presence_events(intervals)
    assert list(pe.columns) == PRESENCE_EVENT_COLUMNS
    assert len(pe) == 12
    s3 = pe[pe["oid"] == "S3"]
    assert list(zip(s3["event"], s3["station"], s3["presence_index"])) == [
        ("enter", "st1", 0),
        ("leave", "st1", 0),
        ("enter", "st2", 1),
        ("leave", "st2", 1),
    ]
    assert s3["systime"].tolist() == [_m(20), _m(40), _m(40), _m(45)]
    assert pe["systime"].is_monotonic_increasing
    assert str(pe["systime"].dtype) == "datetime64[ns, UTC]"


def test_presence_events_object_type_filter_and_naive_times(intervals):
    assert presence_events(intervals, object_type="carrier").empty
    pe = presence_events(_naive(intervals), object_type="serial")
    assert pe["systime"].dt.tz is None
    assert pe["systime"].iloc[0] == pd.Timestamp("2025-03-10 06:00")


def test_presence_events_to_xes_gives_one_trace_per_serial(intervals):
    log = to_event_log(presence_events(intervals), detector="eventlog.presence_events")
    xes = to_event_log_xes(log, case_object_type="serial")
    traces = xes.groupby("case:concept:name", sort=True)["concept:name"].agg(list)
    expected = [
        "production.presence.enter.st1",
        "production.presence.leave.st1",
        "production.presence.enter.st2",
        "production.presence.leave.st2",
    ]
    assert traces.to_dict() == {"S1": expected, "S2": expected, "S3": expected}
    # Linked to the serial and to the station asset.
    objs = set(zip(log.objects[schema.OCEL_OID], log.objects[schema.OCEL_TYPE]))
    assert objs == {
        ("S1", "serial"),
        ("S2", "serial"),
        ("S3", "serial"),
        ("st1", "asset"),
        ("st2", "asset"),
    }
    assert set(log.events["ts_shape:lifecycle_state"]) == {"enter", "leave"}
    tables = to_event_log_ocel(log)
    assert len(tables.relations) == 24  # 12 events x (serial + asset)


def test_object_kpis_from_presence_event_log(line, intervals):
    # The presence events win over the lifecycle of the objects log.
    log = concat(
        to_event_log(presence_events(intervals), detector="eventlog.presence_events"),
        detect_objects(line, SPECS),
    )
    direct = object_kpis(intervals, object_type="serial")
    via_log = object_kpis(log, object_type="serial")
    pd.testing.assert_frame_equal(via_log.objects, direct.objects)
    pd.testing.assert_frame_equal(via_log.stations, direct.stations)


def test_object_kpis_from_detect_objects_lifecycle(line):
    log = detect_objects(line, SPECS)
    kpis = object_kpis(log, object_type="serial")
    assert kpis.objects["lead_time_seconds"].tolist() == [1500.0, 1800.0, 1500.0]
    assert kpis.objects["dwell_seconds"].tolist() == [1380.0, 1500.0, 1500.0]
    # The lifecycle has no station.
    assert kpis.objects["station_count"].tolist() == [0, 0, 0]
    assert kpis.stations["station"].isna().all()


# ---------------------------------------------------------------------------
# Lifecycle: a hand-over to another station is not merged
# ---------------------------------------------------------------------------


def _lifecycle(log, oid):
    ch = log.object_changes
    ch = ch[(ch[schema.OCEL_OID] == oid) & (ch[schema.OCEL_FIELD] == "lifecycle")]
    return list(zip(ch[schema.OCEL_VALUE], ch[schema.OCEL_TIMESTAMP]))


def test_lifecycle_starts_a_new_presence_at_a_station_hand_over(line):
    log = detect_objects(line, SPECS)
    assert _lifecycle(log, "S3") == [
        ("active", _m(20)),
        ("active", _m(40)),
        ("released", _m(45)),
    ]
    # Not touching: released at st1, active again at st2.
    assert _lifecycle(log, "S1") == [
        ("active", _m(0)),
        ("released", _m(10)),
        ("active", _m(12)),
        ("released", _m(25)),
    ]


def test_lifecycle_merges_overlapping_presences_at_two_stations():
    df = pd.concat(
        [
            _signal("a", "st1", [0, 15], ["P", ""]),
            _signal("b", "st2", [10, 20], ["P", ""]),
        ],
        ignore_index=True,
    )
    log = detect_objects(df, [ObjectSpec("a", "part"), ObjectSpec("b", "part")])
    assert _lifecycle(log, "P") == [("active", _m(0)), ("released", _m(20))]


# ---------------------------------------------------------------------------
# Lead time, dwell, throughput
# ---------------------------------------------------------------------------


def test_object_kpis_hand_computed(intervals):
    kpis = object_kpis(intervals, object_type="serial", window="15min")
    assert isinstance(kpis, ObjectKPIs)
    objects = kpis.objects
    assert list(objects.columns) == OBJECT_COLUMNS
    assert objects["oid"].tolist() == ["S1", "S2", "S3"]
    assert objects["start"].tolist() == [_m(0), _m(10), _m(20)]
    assert objects["end"].tolist() == [_m(25), _m(40), _m(45)]
    assert objects["lead_time_seconds"].tolist() == [1500.0, 1800.0, 1500.0]
    assert objects["presence_count"].tolist() == [2, 2, 2]
    assert objects["station_count"].tolist() == [2, 2, 2]
    assert objects["dwell_seconds"].tolist() == [1380.0, 1500.0, 1500.0]

    stations = kpis.stations
    assert list(stations.columns) == STATION_COLUMNS
    assert list(
        zip(stations["oid"], stations["station"], stations["dwell_seconds"])
    ) == [
        ("S1", "st1", 600.0),
        ("S1", "st2", 780.0),
        ("S2", "st1", 600.0),
        ("S2", "st2", 900.0),
        ("S3", "st1", 1200.0),
        ("S3", "st2", 300.0),
    ]

    tp = kpis.throughput
    assert list(tp.columns) == THROUGHPUT_COLUMNS
    assert tp["start"].tolist() == [_m(0), _m(15), _m(30), _m(45)]
    assert tp["completed_count"].tolist() == [0, 1, 1, 1]
    np.testing.assert_array_equal(
        tp["mean_lead_time_seconds"], [np.nan, 1500.0, 1800.0, 1500.0]
    )
    assert object_kpis(intervals, object_type="serial").throughput is None


def test_object_kpis_naive_and_repeated_presences():
    iv = pd.DataFrame(
        {
            "oid": ["A", "A", "A"],
            "type": "part",
            "start": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 03:00"]
            ),
            "end": pd.to_datetime(
                ["2025-01-01 00:30", "2025-01-01 02:00", "2025-01-01 03:15"]
            ),
            "uuid": ["x", "y", "x"],
        }
    )
    kpis = object_kpis(iv, object_type="part")
    row = kpis.objects.iloc[0]
    assert row["lead_time_seconds"] == 3.25 * 3600
    assert row["presence_count"] == 3
    assert row["station_count"] == 2
    assert row["dwell_seconds"] == (30 + 60 + 15) * 60
    assert kpis.objects["start"].dt.tz is None
    by_station = kpis.stations.set_index("station")
    assert by_station.loc["x", "presence_count"] == 2
    assert by_station.loc["x", "dwell_seconds"] == 45 * 60


def test_backwards_presence_raises():
    iv = pd.DataFrame(
        {"oid": ["A"], "type": ["part"], "start": [_m(10)], "end": [_m(0)]}
    )
    with pytest.raises(ValueError, match="ends before it starts"):
        object_kpis(iv, object_type="part")


# ---------------------------------------------------------------------------
# WIP
# ---------------------------------------------------------------------------


def test_wip_over_time_hand_computed(intervals):
    wip = wip_over_time(intervals, object_type="serial", window="15min")
    assert list(wip.columns) == WIP_COLUMNS
    assert wip["start"].tolist() == [_m(0), _m(15), _m(30), _m(45)]
    # [0,15): S1 13 min + S2 5 min; [15,30): 10 + 10 + 10; [30,45): 10 + 15.
    assert wip["object_seconds"].tolist() == [1080.0, 1800.0, 1500.0, 0.0]
    np.testing.assert_allclose(wip["wip_mean"], [1.2, 2.0, 1500 / 900, 0.0])
    # S3 handed from st1 to st2 at 40 counts once; S2 leaves as S3 arrives.
    assert wip["wip_max"].tolist() == [2, 2, 2, 0]


def test_wip_counts_an_object_at_two_stations_once():
    iv = pd.DataFrame(
        {
            "oid": ["A", "A", "B"],
            "type": "part",
            "start": [_m(0), _m(5), _m(10)],
            "end": [_m(10), _m(15), _m(15)],
            "uuid": ["st1", "st2", "st1"],
        }
    )
    wip = wip_over_time(iv, object_type="part", window="15min")
    assert wip["object_seconds"].tolist() == [(15 + 5) * 60.0, 0.0]
    assert wip["wip_max"].tolist() == [2, 0]


def _dst_intervals(tz="Europe/Berlin"):
    local = pd.to_datetime(
        ["2025-03-29 12:00", "2025-03-31 12:00", "2025-03-30 00:00", "2025-03-30 12:00"]
    )
    local = local.tz_localize(tz) if tz else local
    return pd.DataFrame(
        {
            "oid": ["A", "B"],
            "type": "part",
            "start": [local[0], local[2]],
            "end": [local[1], local[3]],
        }
    )


def test_wip_daily_windows_on_a_dst_day():
    wip = wip_over_time(_dst_intervals(), object_type="part", window="1D")
    assert wip["duration_seconds"].tolist() == [24 * 3600.0, 23 * 3600.0, 24 * 3600.0]
    # 30 March (23 h): A all day, B 00:00-12:00 local = 11 h.
    assert wip["object_seconds"].tolist() == [12 * 3600.0, 34 * 3600.0, 12 * 3600.0]
    np.testing.assert_allclose(wip["wip_mean"], [0.5, 34 / 23, 0.5])
    assert wip["wip_max"].tolist() == [1, 2, 1]
    assert str(wip["start"].dt.tz) == "Europe/Berlin"


def test_wip_tz_argument_converts_utc_intervals():
    utc = _dst_intervals()
    for c in ("start", "end"):
        utc[c] = utc[c].dt.tz_convert("UTC")
    wip = wip_over_time(utc, object_type="part", window="1D", tz="Europe/Berlin")
    assert wip["duration_seconds"].tolist() == [24 * 3600.0, 23 * 3600.0, 24 * 3600.0]
    assert wip["object_seconds"].tolist() == [12 * 3600.0, 34 * 3600.0, 12 * 3600.0]


def test_wip_naive_days_are_24_hours():
    wip = wip_over_time(_dst_intervals(tz=None), object_type="part", window="1D")
    assert wip["duration_seconds"].tolist() == [24 * 3600.0] * 3
    assert wip["object_seconds"].tolist() == [12 * 3600.0, 36 * 3600.0, 12 * 3600.0]
    assert wip["start"].dt.tz is None


# ---------------------------------------------------------------------------
# Carrier loops
# ---------------------------------------------------------------------------


def _carriers() -> pd.DataFrame:
    rows = [
        ("C1", "st1", 0, 3),
        ("C1", "st2", 5, 8),
        ("C1", "st1", 20, 23),
        ("C1", "st2", 26, 29),
        ("C1", "st1", 43, 46),
        ("C2", "st2", 2, 4),
        ("C2", "st1", 10, 12),
        ("C2", "st2", 30, 33),
    ]
    return pd.DataFrame(
        {
            "oid": [r[0] for r in rows],
            "type": "carrier",
            "start": [_m(r[2]) for r in rows],
            "end": [_m(r[3]) for r in rows],
            "uuid": [f"{r[1]}:reader" for r in rows],
            "source_uuid": [r[1] for r in rows],
        }
    ).sample(frac=1.0, random_state=1)


def test_carrier_loops_start_where_first_seen():
    result = carrier_loops(_carriers())
    assert isinstance(result, CarrierLoops)
    loops = result.loops
    assert list(loops.columns) == LOOP_COLUMNS
    assert list(zip(loops["oid"], loops["station"], loops["loop_index"])) == [
        ("C1", "st1", 0),
        ("C1", "st1", 1),
        ("C2", "st2", 0),
    ]
    assert loops["duration_seconds"].tolist() == [1200.0, 1380.0, 1680.0]
    assert loops["start"].tolist() == [_m(0), _m(20), _m(2)]
    assert loops["end"].tolist() == [_m(20), _m(43), _m(30)]
    carriers = result.carriers
    assert list(carriers.columns) == CARRIER_COLUMNS
    assert carriers["loop_count"].tolist() == [2, 1]
    assert carriers["mean_loop_seconds"].tolist() == [1290.0, 1680.0]


def test_carrier_loops_with_a_given_start_station():
    result = carrier_loops(_carriers(), start_station="st1")
    assert result.loops["oid"].tolist() == ["C1", "C1"]
    carriers = result.carriers.set_index("oid")
    assert carriers.loc["C2", "loop_count"] == 0
    assert np.isnan(carriers.loc["C2", "mean_loop_seconds"])
    assert carriers["station"].tolist() == ["st1", "st1"]
    none = carrier_loops(_carriers(), start_station="nowhere")
    assert none.loops.empty
    assert none.carriers["loop_count"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Counts per object
# ---------------------------------------------------------------------------


def _counter(values=(100, 106, 112, 2, 6, 9), minutes=(12, 18, 30, 32, 40, 47)):
    return pd.DataFrame(
        {
            "systime": [_m(x) for x in minutes],
            "uuid": "st2:good",
            "source_uuid": "st2",
            "value_integer": list(values),
        }
    )


def test_attach_counts_pro_rata_with_a_reset(intervals):
    counts = attach_counts(intervals, _counter(), "st2:good", object_type="serial")
    assert list(counts.columns) == COUNT_COLUMNS
    # (12,18]: +6 -> S1. (18,30]: +6, S1 7 min / S2 5 min -> 3.5 / 2.5.
    # 32: reset to 2 -> +2 for S2. (32,40]: +4 -> S2. (40,47]: +3, S3 present
    # 5 of the 7 min -> 15/7, the other 6/7 unattributed.
    assert counts["oid"].iloc[:3].tolist() == ["S1", "S2", "S3"]
    assert pd.isna(counts["oid"].iloc[3])
    np.testing.assert_allclose(counts["quantity"], [9.5, 8.5, 15 / 7, 6 / 7])
    assert counts["quantity"].sum() == pytest.approx(21.0)


def test_attach_counts_station_and_naive_counter(intervals):
    counter = _counter().drop(columns="source_uuid")
    counter["systime"] = counter["systime"].dt.tz_localize(None)
    # No source_uuid: objects at any station; naive times read as UTC.
    with pytest.warns(DataQualityWarning, match="tz-naive"):
        counts = attach_counts(
            intervals, counter, "st2:good", object_type="serial", station="st2"
        )
    np.testing.assert_allclose(counts["quantity"], [9.5, 8.5, 15 / 7, 6 / 7])
    # At st1 nobody is present after 40: S3 st1 [20,40) gets its share.
    at_st1 = attach_counts(
        intervals, _counter(), "st2:good", object_type="serial", station="st1"
    )
    by_oid = dict(zip(at_st1["oid"].fillna("-"), at_st1["quantity"]))
    # (12,18]: S2 -> 6. (18,30]: S2 2 min, S3 10 min -> 1 / 5. 32, 40: S3.
    assert by_oid["S1"] == 0.0
    assert by_oid["S2"] == pytest.approx(7.0)
    assert by_oid["S3"] == pytest.approx(11.0)
    assert by_oid["-"] == pytest.approx(3.0)


def test_attach_counts_shares_simultaneous_objects():
    iv = pd.DataFrame(
        {
            "oid": ["A", "B"],
            "type": "part",
            "start": [_m(0), _m(0)],
            "end": [_m(10), _m(10)],
        }
    )
    counter = _counter(values=(0, 10), minutes=(0, 10)).drop(columns="source_uuid")
    counts = attach_counts(iv, counter, "st2:good", object_type="part")
    assert counts["quantity"].tolist() == [5.0, 5.0]


def test_attach_counts_rejects_an_ambiguous_counter_source(intervals):
    counter = _counter()
    counter.loc[0, "source_uuid"] = "st1"
    with pytest.raises(ValueError, match="several source_uuid"):
        attach_counts(intervals, counter, "st2:good", object_type="serial")


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_empty_inputs_give_the_documented_columns(intervals):
    empty = intervals.iloc[0:0]
    assert list(presence_events(empty).columns) == PRESENCE_EVENT_COLUMNS
    kpis = object_kpis(empty, object_type="serial", window="1h")
    assert list(kpis.objects.columns) == OBJECT_COLUMNS
    assert list(kpis.stations.columns) == STATION_COLUMNS
    assert list(kpis.throughput.columns) == THROUGHPUT_COLUMNS
    assert list(wip_over_time(empty, object_type="serial", window="1h").columns) == (
        WIP_COLUMNS
    )
    loops = carrier_loops(empty)
    assert list(loops.loops.columns) == LOOP_COLUMNS
    assert list(loops.carriers.columns) == CARRIER_COLUMNS
    counts = attach_counts(
        empty, _counter().iloc[0:0], "st2:good", object_type="serial"
    )
    assert list(counts.columns) == COUNT_COLUMNS and counts.empty
    for frame in (
        presence_events(EventLog()),
        object_kpis(EventLog(), object_type="x").objects,
    ):
        assert frame.empty
    assert str(presence_events(empty)["systime"].dtype) == "datetime64[ns, UTC]"


def test_top_level_exports():
    for name in (
        "presence_events",
        "object_kpis",
        "wip_over_time",
        "carrier_loops",
        "attach_counts",
        "ObjectKPIs",
        "CarrierLoops",
    ):
        assert getattr(ts_shape, name) is getattr(ts_shape.eventlog, name)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def _large_intervals(n_rows: int = 1_000_000) -> pd.DataFrame:
    """make_object_signals serial / carrier presences, tiled to ``n_rows``."""
    frame, labels = ts_shape.make_object_signals(
        seed=0, days=3, include=["serial", "carrier"]
    )
    ids = labels[labels["role"] == "identifier"]
    specs = [
        ObjectSpec(r.uuid, r.object_type, value_column=r.value_column)
        for r in ids.itertuples()
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = object_intervals(frame, specs)
    copies = -(-n_rows // len(base))
    span = base["end"].max() - base["start"].min()
    parts = []
    for i in range(copies):
        part = base.assign(
            oid=base["oid"] + f"#{i % 50}",
            start=base["start"] + span * (i // 50),
            end=base["end"] + span * (i // 50),
            uuid=base["uuid"] + f":st{i % 7}",
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).iloc[:n_rows]


@pytest.mark.scale
def test_object_kpis_and_wip_scale_to_a_million_presences():
    iv = _large_intervals()
    assert len(iv) == 1_000_000
    t = time.perf_counter()
    kpis = object_kpis(iv, object_type="serial", window="1D")
    wip = wip_over_time(iv, object_type="serial", window="1h")
    elapsed = time.perf_counter() - t
    assert len(kpis.objects) > 1000
    assert not wip.empty and wip["wip_max"].max() > 0
    assert elapsed < 10.0, f"object_kpis + wip_over_time took {elapsed:.1f} s"
