"""Change flags (``is_delta``), cycle-bounded presences and usage links.

Hand-computed references for :class:`ObjectSpec` ``delta_column`` (a
same-value re-setup starts a new presence), ``cycle_uuid`` (a serial scanned
mid-cycle is present for exactly that cycle) and the ``uses`` links of
:func:`detect_objects` (every serial made while a material is held is
``made_from`` it), plus the genealogy over them.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.datasets import make_object_signals
from ts_shape.errors import DataQualityWarning
from ts_shape.eventlog import (
    EventLog,
    ObjectSpec,
    attach_objects,
    derive_o2o,
    detect_objects,
    object_intervals,
    relate,
    schema,
    where_from,
    where_used,
)

T0 = pd.Timestamp("2025-03-01 08:00")


def _t(seconds: float, tz: str | None = None) -> pd.Timestamp:
    ts = T0 + pd.Timedelta(seconds=seconds)
    return ts.tz_localize(tz) if tz else ts


def _utc(seconds: float, tz: str | None = None) -> pd.Timestamp:
    return _t(seconds, tz).tz_localize("UTC") if tz is None else _t(seconds, tz)


def _rows(uuid, seconds, values, *, column="value_string", delta=None, tz=None):
    n = len(seconds)
    frame = pd.DataFrame(
        {
            "systime": pd.DatetimeIndex([_t(s, tz) for s in seconds]),
            "uuid": [uuid] * n,
            column: (
                pd.array(values, dtype="boolean")
                if column == "value_bool"
                else pd.Series(values, dtype=object)
            ),
        }
    )
    if delta is not None:
        frame["is_delta"] = delta
    return frame


def _spans(iv: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [(r.oid, r.start, r.end) for r in iv.itertuples()]


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# A. is_delta
# ---------------------------------------------------------------------------

MAT = "line:material"


def _resetup_frame(with_flag: bool = True) -> pd.DataFrame:
    # M1 set at 0, repeated, set up again at 6 with the same number, M2 at 10.
    return _rows(
        MAT,
        [0, 2, 4, 6, 8, 10, 12],
        ["M1", "M1", "M1", "M1", "M1", "M2", "M2"],
        delta=[True, False, False, True, False, True, False] if with_flag else None,
    )


def test_same_value_resetup_starts_a_new_presence():
    iv = object_intervals(_resetup_frame(), [ObjectSpec(MAT, "material")])
    assert _spans(iv) == [
        ("M1", _utc(0), _utc(6)),
        ("M1", _utc(6), _utc(10)),
        ("M2", _utc(10), _utc(12)),
    ]
    # one object, two presences; the lifecycle joins the touching presences
    log = detect_objects(_resetup_frame(), [ObjectSpec(MAT, "material")])
    assert sorted(log.objects["ocel:oid"]) == ["M1", "M2"]


def test_without_flag_column_or_with_delta_none_value_changes_only():
    want = [("M1", _utc(0), _utc(10)), ("M2", _utc(10), _utc(12))]
    spec = ObjectSpec(MAT, "material")
    assert _spans(object_intervals(_resetup_frame(False), [spec])) == want
    off = ObjectSpec(MAT, "material", delta_column=None)
    assert _spans(object_intervals(_resetup_frame(), [off])) == want
    # a column with no True row carries no information
    frame = _resetup_frame()
    frame["is_delta"] = False
    assert _spans(object_intervals(frame, [spec])) == want


def test_value_change_without_flag_still_splits():
    frame = _rows(
        MAT, [0, 2, 4, 6], ["M1", "M1", "M2", "M2"], delta=[True, False, False, False]
    )
    iv = object_intervals(frame, [ObjectSpec(MAT, "material")])
    assert _spans(iv) == [("M1", _utc(0), _utc(4)), ("M2", _utc(4), _utc(6))]


def test_flag_after_a_misread_does_not_split():
    # the PLC flags the change back from NOREAD; the misread is a gap
    frame = _rows(
        MAT,
        [0, 2, 4, 6, 8],
        ["M1", "M1", "NOREAD", "M1", "M1"],
        delta=[True, False, True, True, False],
    )
    iv = object_intervals(frame, [ObjectSpec(MAT, "material")])
    assert _spans(iv) == [("M1", _utc(0), _utc(8))]


def test_missing_flags_count_as_false_and_custom_column():
    frame = _resetup_frame()
    frame["is_delta"] = pd.array(
        [True, None, None, True, None, True, None], dtype="boolean"
    )
    iv = object_intervals(frame, [ObjectSpec(MAT, "material")])
    assert len(iv) == 3
    frame = frame.rename(columns={"is_delta": "set_flag"})
    assert len(object_intervals(frame, [ObjectSpec(MAT, "material")])) == 2
    spec = ObjectSpec(MAT, "material", delta_column="set_flag")
    assert len(object_intervals(frame, [spec])) == 3


def test_flag_with_min_duration_keeps_the_resetup():
    spec = ObjectSpec(MAT, "material", min_duration="1s")
    iv = object_intervals(_resetup_frame(), [spec])
    assert len(iv) == 3


def test_generator_frames_without_flags_are_unchanged():
    frame, labels = make_object_signals(seed=2, days=0.5)
    ids = labels[labels["role"] == "identifier"]
    on = [
        ObjectSpec(r.uuid, r.object_type, value_column=r.value_column)
        for r in ids.itertuples()
    ]
    off = [
        ObjectSpec(
            r.uuid, r.object_type, value_column=r.value_column, delta_column=None
        )
        for r in ids.itertuples()
    ]
    without = _quiet(object_intervals, frame.drop(columns="is_delta"), on)
    pd.testing.assert_frame_equal(without, _quiet(object_intervals, frame, off))
    # with the generator's flags (every on-change row, keep-alives included)
    # presences only split: the objects are the same
    flagged = _quiet(object_intervals, frame, on)
    assert len(flagged) >= len(without)
    assert set(flagged["oid"]) == set(without["oid"])


# ---------------------------------------------------------------------------
# B. cycles
# ---------------------------------------------------------------------------

DOOR = "cell:door_closed"
SER = "cell:serial"


def _cycle_frame(tz: str | None = None) -> pd.DataFrame:
    # cycles [0, 60) and [70, 120); S1 scanned at 20, S2 at 85, repeated
    # every 10 s (S1 is still repeated at the start of the second cycle)
    door = _rows(
        DOOR, [0, 60, 70, 120], [True, False, True, False], column="value_bool", tz=tz
    )
    ser_t = [20, 30, 40, 50, 60, 70, 80, 85, 95, 105, 115, 125]
    ser_v = ["S1"] * 7 + ["S2"] * 5
    ser_d = [True] + [False] * 6 + [True] + [False] * 4
    ser = _rows(SER, ser_t, ser_v, delta=ser_d, tz=tz)
    door["is_delta"] = True
    return pd.concat([door, ser], ignore_index=True)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_serial_presence_is_its_cycle(tz):
    spec = ObjectSpec(SER, "serial", cycle_uuid=DOOR)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        iv = object_intervals(_cycle_frame(tz), [spec])
    assert _spans(iv) == [
        ("S1", _utc(0, tz), _utc(60, tz)),
        ("S2", _utc(70, tz), _utc(120, tz)),
    ]
    assert str(iv["start"].dt.tz) == "UTC"


def test_rescans_in_one_cycle_are_one_presence():
    frame = _cycle_frame()
    # S1 scanned again (same value, flagged) at 40
    frame.loc[(frame["uuid"] == SER) & (frame["systime"] == _t(40)), "is_delta"] = True
    iv = object_intervals(frame, [ObjectSpec(SER, "serial", cycle_uuid=DOOR)])
    assert _spans(iv) == [("S1", _utc(0), _utc(60)), ("S2", _utc(70), _utc(120))]


def test_scan_outside_every_cycle_warns_and_keeps_its_span():
    frame = pd.concat(
        [_cycle_frame(), _rows(SER, [130, 140], ["S3", "S3"], delta=[True, False])],
        ignore_index=True,
    )
    spec = ObjectSpec(SER, "serial", cycle_uuid=DOOR)
    with pytest.warns(DataQualityWarning, match="1 presence"):
        iv = object_intervals(frame, [spec])
    assert _spans(iv)[-1] == ("S3", _utc(130), _utc(140))
    assert len(iv) == 3


def test_two_serials_in_one_cycle_each_get_the_cycle():
    frame = _cycle_frame()
    frame.loc[(frame["uuid"] == SER) & (frame["systime"] == _t(40)), "value_string"] = (
        "SX"
    )
    spec = ObjectSpec(SER, "serial", cycle_uuid=DOOR)
    with pytest.warns(DataQualityWarning, match="1 cycle"):
        iv = object_intervals(frame, [spec])
    # S1, SX, S1 in cycle 1: S1 once, SX once
    assert sorted(_spans(iv)) == [
        ("S1", _utc(0), _utc(60)),
        ("S2", _utc(70), _utc(120)),
        ("SX", _utc(0), _utc(60)),
    ]


def test_open_last_cycle_ends_at_the_data_end():
    frame = _cycle_frame()
    frame = frame[~((frame["uuid"] == DOOR) & (frame["systime"] == _t(120)))]
    iv = object_intervals(frame, [ObjectSpec(SER, "serial", cycle_uuid=DOOR)])
    assert _spans(iv)[-1] == ("S2", _utc(70), _utc(125))


def test_numeric_cycle_signal_and_missing_cycle_signal():
    frame = _cycle_frame()
    door = frame["uuid"] == DOOR
    frame["value_integer"] = pd.array([pd.NA] * len(frame), dtype="Int64")
    frame.loc[door, "value_integer"] = [1, 0, 1, 0]
    spec = ObjectSpec(
        SER, "serial", cycle_uuid=DOOR, cycle_value_column="value_integer"
    )
    assert len(object_intervals(frame, [spec])) == 2
    with pytest.raises(ValueError, match="cycle_uuid"):
        object_intervals(frame, [ObjectSpec(SER, "serial", cycle_uuid="nope")])


# ---------------------------------------------------------------------------
# C. usage links
# ---------------------------------------------------------------------------

MAT1, MAT2, TOOL = "line:mat1", "line:mat2", "line:tool"
N_CYCLES = 100


def _bom_frame(
    change_at: float | None = None, *, tz: str | None = None
) -> pd.DataFrame:
    """100 cycles of 60 s ([60k, 60k+40)); serial SN{k} scanned at 60k+10.

    mat1 holds MAT-A (repeated every 30 s; MAT-X from ``change_at``), mat2
    holds MAT-B read once at 0, the tool T-7 read every 30 s.
    """
    end = 60 * N_CYCLES
    starts = np.arange(N_CYCLES) * 60
    door_t = np.sort(np.r_[starts, starts + 40])
    door_v = [bool(i % 2 == 0) for i in range(len(door_t))]
    frames = [
        _rows(DOOR, door_t, door_v, column="value_bool", tz=tz),
        _rows(
            SER,
            starts + 10,
            [f"SN{k:03d}" for k in range(N_CYCLES)],
            delta=True,
            tz=tz,
        ),
    ]
    rep = np.arange(0, end, 30)
    mat = ["MAT-A" if change_at is None or t < change_at else "MAT-X" for t in rep]
    mat_t = list(rep)
    if change_at is not None and change_at not in rep:
        mat_t.append(change_at)
        mat.append("MAT-X")
        order = np.argsort(mat_t)
        mat_t = [mat_t[i] for i in order]
        mat = [mat[i] for i in order]
    flags = [i == 0 or mat[i] != mat[i - 1] for i in range(len(mat))]
    frames.append(_rows(MAT1, mat_t, mat, delta=flags, tz=tz))
    frames.append(_rows(MAT2, [0], ["MAT-B"], delta=[True], tz=tz))
    frames.append(_rows(TOOL, rep, ["T-7"] * len(rep), delta=False, tz=tz))
    return pd.concat(frames, ignore_index=True)


SPECS = [
    ObjectSpec(SER, "serial", cycle_uuid=DOOR),
    ObjectSpec(MAT1, "material"),
    ObjectSpec(MAT2, "material"),
    ObjectSpec(TOOL, "tool"),
]
USES = {SER: {MAT1: "made_from", MAT2: "made_from", TOOL: "made_with"}}


def _links(log, qualifier: str) -> dict[str, set[str]]:
    o2o = log.o2o[log.o2o["ocel:qualifier"] == qualifier]
    out: dict[str, set[str]] = {}
    for child, parent in zip(o2o["ocel:oid"], o2o["ocel:oid_2"], strict=True):
        out.setdefault(child, set()).add(parent)
    return out


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_constant_materials_link_every_serial(tz):
    log = detect_objects(_bom_frame(tz=tz), SPECS, uses=USES)
    made_from = _links(log, "made_from")
    assert len(made_from) == N_CYCLES
    assert all(parents == {"MAT-A", "MAT-B"} for parents in made_from.values())
    made_with = _links(log, "made_with")
    assert all(parents == {"T-7"} for parents in made_with.values())
    assert len(made_with) == N_CYCLES
    # one row per (child, parent, qualifier)
    assert len(log.o2o) == 3 * N_CYCLES
    assert not log.o2o.duplicated().any()


def test_constant_parent_read_once_is_one_presence_to_the_end():
    iv = object_intervals(_bom_frame(), [ObjectSpec(MAT2, "material")])
    assert _spans(iv) == [("MAT-B", _utc(0), _utc(60 * (N_CYCLES - 1) + 40))]


def test_material_change_between_cycles():
    # cycle 49 is [2940, 2980), cycle 50 [3000, 3040): change at 2990
    log = detect_objects(_bom_frame(change_at=2990), SPECS, uses=USES)
    made_from = _links(log, "made_from")
    for k in range(N_CYCLES):
        want = {"MAT-A" if k < 50 else "MAT-X", "MAT-B"}
        assert made_from[f"SN{k:03d}"] == want, k


def test_material_change_inside_a_cycle_links_both():
    # change at 3020, inside cycle 50 [3000, 3040): SN050 used both lots
    log = detect_objects(_bom_frame(change_at=3020), SPECS, uses=USES)
    made_from = _links(log, "made_from")
    assert made_from["SN049"] == {"MAT-A", "MAT-B"}
    assert made_from["SN050"] == {"MAT-A", "MAT-X", "MAT-B"}
    assert made_from["SN051"] == {"MAT-X", "MAT-B"}


def test_where_used_and_where_from_follow_usage_links():
    log = detect_objects(_bom_frame(change_at=2990), SPECS, uses=USES)
    used = where_used(log, "MAT-A")
    assert set(used["oid"]) == {f"SN{k:03d}" for k in range(50)}
    assert set(used["qualifier"]) == {"made_from"}
    assert len(where_used(log, "MAT-B")) == N_CYCLES
    origin = where_from(log, "SN060")
    assert set(origin["oid"]) == {"MAT-X", "MAT-B", "T-7"}


def test_uses_with_hierarchy_and_object_types():
    log = detect_objects(
        _bom_frame(),
        SPECS,
        hierarchy={"serial": "tool"},
        uses={"serial": {"material": "made_from"}},
    )
    assert len(_links(log, "made_from")) == N_CYCLES
    assert len(_links(log, "part_of")) == N_CYCLES
    # part_of to the tool (longest overlap), made_from to both materials
    assert set(log.o2o["ocel:qualifier"]) == {"made_from", "part_of"}


def test_uses_none_changes_nothing_and_co_occurs_skips_used_pairs():
    frame = _bom_frame()
    a = detect_objects(frame, SPECS)
    assert a.o2o.empty
    b = detect_objects(frame, SPECS, uses=USES, infer_o2o="co_occurs")
    pairs = set(zip(b.o2o["ocel:oid"], b.o2o["ocel:oid_2"], strict=True))
    co = b.o2o[b.o2o["ocel:qualifier"] == "co_occurs"]
    for x, y in zip(co["ocel:oid"], co["ocel:oid_2"], strict=True):
        assert not (x.startswith("SN") and y in {"MAT-A", "MAT-B", "T-7"})
        assert not (y.startswith("SN") and x in {"MAT-A", "MAT-B", "T-7"})
    assert len(pairs) == len(b.o2o)
    assert detect_objects(frame, SPECS, uses=USES, infer_o2o=False).o2o.empty


def test_uses_through_derive_o2o_relate_and_attach_objects():
    frame = _bom_frame()
    iv = object_intervals(frame, SPECS)
    o2o = derive_o2o(iv, uses=USES)
    assert len(o2o) == 3 * N_CYCLES
    events = EventLog(
        events=pd.DataFrame(
            {
                schema.OCEL_EID: ["e0"],
                schema.OCEL_ACTIVITY: ["check"],
                schema.OCEL_TIMESTAMP: [_utc(15)],
                schema.TS_DETECTOR: ["d"],
                schema.TS_PACK: ["production"],
            }
        )
    )
    rel = relate(events, iv, uses=USES, fanout_threshold=None)
    assert len(rel.o2o) == 3 * N_CYCLES
    att = attach_objects(events, frame, SPECS, uses=USES, fanout_threshold=None)
    assert len(att.o2o) == 3 * N_CYCLES
    linked = set(att.relations["ocel:oid"])
    assert {"SN000", "MAT-A", "MAT-B", "T-7"} <= linked


def test_bad_uses_shape():
    iv = object_intervals(_bom_frame(), SPECS)
    with pytest.raises(TypeError, match="uses"):
        derive_o2o(iv, uses={SER: MAT1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="qualifier"):
        derive_o2o(iv, uses={SER: {MAT1: ""}})


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


@pytest.mark.scale
def test_cycles_and_uses_at_one_million_rows():
    n_cycles = 20_000
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2025-01-01", tz="UTC").value
    starts = np.arange(n_cycles, dtype=np.int64) * 60_000_000_000 + base
    door_t = np.sort(np.r_[starts, starts + 40_000_000_000])
    door_v = np.tile([True, False], n_cycles)
    # serial: scan at +10 s, then repeated every 2 s for 50 s (25 rows)
    rep = np.arange(25, dtype=np.int64) * 2_000_000_000 + 10_000_000_000
    ser_t = (starts[:, None] + rep[None, :]).ravel()
    ser_v = np.repeat(np.array([f"SN{k:06d}" for k in range(n_cycles)]), 25)
    ser_d = np.tile(np.r_[True, np.zeros(24, dtype=bool)], n_cycles)
    # two materials repeated every 5 s, changing every ~500 cycles
    span = int(n_cycles * 60 / 5)
    mat_t = base + np.arange(span, dtype=np.int64) * 5_000_000_000
    lots = np.cumsum(rng.random(span) < 1 / 6000)
    frames = [
        pd.DataFrame({"systime": door_t, "uuid": DOOR, "value_bool": door_v}),
        pd.DataFrame(
            {"systime": ser_t, "uuid": SER, "value_string": ser_v, "is_delta": ser_d}
        ),
    ]
    for uuid, prefix in ((MAT1, "A"), (MAT2, "B")):
        vals = np.array([f"{prefix}{x:04d}" for x in lots])
        flag = np.r_[True, vals[1:] != vals[:-1]]
        frames.append(
            pd.DataFrame(
                {"systime": mat_t, "uuid": uuid, "value_string": vals, "is_delta": flag}
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["systime"] = pd.to_datetime(frame["systime"], utc=True)
    assert len(frame) >= 900_000
    specs = [
        ObjectSpec(SER, "serial", cycle_uuid=DOOR),
        ObjectSpec(MAT1, "material"),
        ObjectSpec(MAT2, "material"),
    ]
    uses = {SER: {MAT1: "made_from", MAT2: "made_from"}}
    t0 = time.perf_counter()
    log = detect_objects(frame, specs, uses=uses)
    elapsed = time.perf_counter() - t0
    made_from = log.o2o[log.o2o["ocel:qualifier"] == "made_from"]
    assert made_from["ocel:oid"].nunique() == n_cycles
    assert len(made_from) >= 2 * n_cycles
    assert elapsed < 10, elapsed
