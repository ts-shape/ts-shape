"""Round 17: event log and value mapping fixes, hand-computed.

Covers caller-bound objects in the event ids, the order of the two-row XES
lifecycle, ``concat`` / sub-log deduplication keys, per-rule numeric
severity thresholds, case-insensitive severity labels, rows without an
``ocel:timestamp``, the empty XES export, and text keys in ``ValueMapper``.

Each test states the definition it checks in its docstring; the expected
values follow from that definition, not from the implementation.
"""

from __future__ import annotations

import json
import uuid
import warnings

import pandas as pd
import pytest

from ts_shape.context.value_mapping import ValueMapper
from ts_shape.errors import DataQualityWarning
from ts_shape.eventlog import EventLog, concat, to_event_log, to_event_log_xes
from ts_shape.eventlog.adapters import _NAMESPACE, _classify_severity
from ts_shape.eventlog.taxonomy import REGISTRY
from ts_shape.events._output import add_severity_level

ZSCORE = "OutlierDetectionEvents.detect_outliers_zscore"
IQR = "OutlierDetectionEvents.detect_outliers_iqr"
IFOREST = "OutlierDetectionEvents.detect_outliers_isolation_forest"
T0 = pd.Timestamp("2025-01-01T00:00:00Z")


def _points(severity: list[float]) -> pd.DataFrame:
    """A point detector frame: one row per severity, one minute apart."""
    n = len(severity)
    return pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(minutes=i) for i in range(n)],
            "uuid": [f"u{i}" for i in range(n)],
            "source_uuid": ["A"] * n,
            "is_delta": [True] * n,
            "value_double": [10.0] * n,
            "severity": severity,
        }
    )


def _eid(key: str) -> str:
    return "e-" + str(uuid.uuid5(_NAMESPACE, key))


def _events(eids: list[str], ts: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    ts = ts or [T0] * len(eids)
    return pd.DataFrame(
        {
            "ocel:eid": pd.Series(eids, dtype="string"),
            "ocel:activity": pd.Series(["a"] * len(eids), dtype="string"),
            "ocel:timestamp": pd.Series(ts, dtype="datetime64[ns, UTC]"),
        }
    )


def _objects(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ocel:oid": pd.Series([p[0] for p in pairs], dtype="string"),
            "ocel:type": pd.Series([p[1] for p in pairs], dtype="string"),
        }
    )


def _changes(rows: list[tuple[str, str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ocel:oid": pd.Series([r[0] for r in rows], dtype="string"),
            "ocel:type": pd.Series([r[1] for r in rows], dtype="string"),
            "ocel:field": pd.Series(["f"] * len(rows), dtype="string"),
            "ocel:value": pd.Series([r[2] for r in rows], dtype="object"),
            "ocel:timestamp": pd.Series([T0] * len(rows), dtype="datetime64[ns, UTC]"),
        }
    )


def _relations(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ocel:eid": [r[0] for r in rows],
            "ocel:oid": [r[1] for r in rows],
            "ocel:type": [r[2] for r in rows],
            "ocel:qualifier": [r[3] for r in rows],
        }
    ).astype("string")


# --- 1. caller-bound objects are part of the event id ----------------------


def _zscore_key(row: int, severity: str) -> str:
    """The hash input of row ``row`` of ``_points``: detector, source, start
    (missing for a point), end, activity, then the payload columns sorted."""
    end = (T0 + pd.Timedelta(minutes=row)).isoformat()
    return "|".join(
        [
            ZSCORE,
            "A",
            "\x00",
            end,
            "quality.outlier.zscore",
            "is_delta=True",
            f"severity={severity}",
            "source_uuid=A",
            f"uuid=u{row}",
            "value_double=10",
        ]
    )


def test_eid_without_objects_is_the_former_key():
    """No ``objects=``: the id is uuid5 of the former key (nothing added)."""
    log = to_event_log(_points([3.5]), detector=ZSCORE)
    assert log.events["ocel:eid"].tolist() == [_eid(_zscore_key(0, "3.5"))]


def test_eid_hashes_the_caller_bound_object():
    """``objects={"asset": "M1"}`` appends ``|obj:asset=M1`` to the key."""
    log = to_event_log(_points([3.5]), detector=ZSCORE, objects={"asset": "M1"})
    assert log.events["ocel:eid"].tolist() == [
        _eid(_zscore_key(0, "3.5") + "|obj:asset=M1")
    ]


def test_eid_binding_types_are_sorted():
    """Several bound types are appended as ``type=oid`` in sorted type order,
    whatever order the caller passes them in."""
    log = to_event_log(
        _points([3.5]), detector=ZSCORE, objects={"line": "L1", "batch": "B7"}
    )
    assert log.events["ocel:eid"].tolist() == [
        _eid(_zscore_key(0, "3.5") + "|obj:batch=B7|obj:line=L1")
    ]


def test_concat_of_two_bindings_keeps_every_event():
    """2 rows bound to M1 + the same 2 rows bound to M2 = 4 events, each
    linked to exactly one asset (2 per asset)."""
    out = _points([3.5, 5.0])
    m1 = to_event_log(out, detector=ZSCORE, objects={"asset": "M1"})
    m2 = to_event_log(out, detector=ZSCORE, objects={"asset": "M2"})
    both = concat(m1, m2)
    assert len(both.events) == 4
    assert both.relations.groupby("ocel:eid").size().tolist() == [1, 1, 1, 1]
    assert both.relations["ocel:oid"].value_counts().to_dict() == {"M1": 2, "M2": 2}
    # The same binding twice still deduplicates.
    assert len(concat(m1, m1).events) == 2


# --- 2. two-row XES: start before complete at equal times ------------------


def test_two_row_zero_duration_start_first():
    """A zero-length interval gives ``start`` then ``complete`` at the same
    time (rank start=0, complete=1, not the alphabetical label order)."""
    frame = pd.DataFrame(
        {
            "start": [T0],
            "end": [T0],
            "uuid": ["u"],
            "source_uuid": ["A"],
            "is_delta": [True],
            "state": ["run"],
        }
    )
    log = to_event_log(frame, detector="MachineStateEvents.detect_run_idle")
    xes = to_event_log_xes(log, lifecycle="two_row")
    assert xes["lifecycle:transition"].tolist() == ["start", "complete"]
    assert xes["time:timestamp"].tolist() == [T0, T0]


# --- 3. deduplication keys -------------------------------------------------


def test_concat_keeps_changes_of_same_oid_other_type():
    """Changes are deduplicated on (oid, type, field, timestamp): ``X`` the
    asset and ``X`` the batch changing ``f`` at the same time are 2 rows."""
    objs = _objects([("X", "asset"), ("X", "batch")])
    a = EventLog(
        events=_events(["e1"]),
        objects=objs,
        object_changes=_changes([("X", "asset", 1)]),
    )
    b = EventLog(
        events=_events(["e1"]),
        objects=objs,
        object_changes=_changes([("X", "batch", 2)]),
    )
    out = concat(a, b).object_changes
    assert sorted(zip(out["ocel:type"], out["ocel:value"])) == [
        ("asset", 1),
        ("batch", 2),
    ]
    # The very same change twice is still one row.
    assert len(concat(a, a).object_changes) == 1


def test_concat_keeps_relations_with_other_qualifier():
    """Relations are deduplicated on (eid, oid, type, qualifier): one event
    linked to ``X`` as input and as output keeps both roles."""
    objs = _objects([("X", "asset")])
    a = EventLog(
        events=_events(["e1"]),
        objects=objs,
        relations=_relations([("e1", "X", "asset", "input")]),
    )
    b = EventLog(
        events=_events(["e1"]),
        objects=objs,
        relations=_relations([("e1", "X", "asset", "output")]),
    )
    assert sorted(concat(a, b).relations["ocel:qualifier"]) == ["input", "output"]
    assert len(concat(a, a).relations) == 1


def test_sub_log_keeps_changes_of_kept_type_only():
    """``filter_by_object("X")`` keeps object (X, asset) only (the event is
    related to it alone), so of the changes of X only the asset one stays."""
    log = EventLog(
        events=_events(["e1"]),
        objects=_objects([("X", "asset"), ("X", "batch")]),
        relations=_relations([("e1", "X", "asset", "input")]),
        object_changes=_changes([("X", "asset", 1), ("X", "batch", 2)]),
    )
    sub = log.filter_by_object("X")
    assert sub.objects["ocel:type"].tolist() == ["asset"]
    assert sub.object_changes["ocel:type"].tolist() == ["asset"]
    assert sub.object_changes["ocel:value"].tolist() == [1]


# --- 4. per-rule severity thresholds ---------------------------------------


def test_registry_thresholds():
    """IQR: (1.5, 3.0) IQR beyond the fence; isolation forest: (0.6, 0.7) of
    ``-score_samples``; z-score and MAD keep the default (3.0, 4.5)."""
    rules = {
        name: REGISTRY[("OutlierDetectionEvents", name)].severity_thresholds
        for name in (
            "detect_outliers_iqr",
            "detect_outliers_isolation_forest",
            "detect_outliers_zscore",
            "detect_outliers_mad",
        )
    }
    assert rules == {
        "detect_outliers_iqr": (1.5, 3.0),
        "detect_outliers_isolation_forest": (0.6, 0.7),
        "detect_outliers_zscore": None,
        "detect_outliers_mad": None,
    }


@pytest.mark.parametrize(
    ("detector", "scores", "levels"),
    [
        # Isolation forest: < 0.6 info, [0.6, 0.7) warn, >= 0.7 critical.
        (
            IFOREST,
            [0.55, 0.6, 0.65, 0.7, 0.85],
            ["info", "warn", "warn", "critical", "critical"],
        ),
        # IQR units beyond the fence: < 1.5 info, [1.5, 3) warn, >= 3 critical.
        (
            IQR,
            [0.2, 1.5, 2.9, 3.0, 6.6],
            ["info", "warn", "warn", "critical", "critical"],
        ),
        # z-score: the default < 3 info, [3, 4.5) warn, >= 4.5 critical.
        (
            ZSCORE,
            [0.85, 2.9, 3.0, 4.4, 4.5],
            ["info", "info", "warn", "warn", "critical"],
        ),
    ],
)
def test_numeric_severity_buckets(detector, scores, levels):
    """Each detector's scores are bucketed with its own ``(warn, critical)``."""
    log = to_event_log(_points(scores), detector=detector)
    by_time = log.events.sort_values("ocel:timestamp")
    assert by_time["ts_shape:severity"].tolist() == levels


def test_classify_severity_default_thresholds():
    """Without thresholds the z-score bounds apply: 2.99 info, 3 warn, 4.5
    critical."""
    assert [_classify_severity(v) for v in (2.99, 3.0, 4.5)] == [
        "info",
        "warn",
        "critical",
    ]


# --- 5. severity labels without case / blanks ------------------------------


def test_severity_level_ignores_case_and_blanks():
    """``"High"``, ``" high "`` and ``"HIGH"`` map like ``"high"``
    (critical); an unknown label stays missing."""
    out = add_severity_level(
        pd.DataFrame({"severity": ["high", "High", " high ", "HIGH", "nope"]})
    )
    assert out["severity_level"].tolist()[:4] == ["critical"] * 4
    assert pd.isna(out["severity_level"].iloc[4])


def test_severity_level_overrides_ignore_case():
    """An override key ``"Minor"`` applies to the label ``"minor"``."""
    out = add_severity_level(
        pd.DataFrame({"severity": ["minor"]}), overrides={"Minor": "warn"}
    )
    assert out["severity_level"].tolist() == ["warn"]


# --- 6. rows without an ocel:timestamp -------------------------------------


def test_point_row_without_systime_is_dropped():
    """3 point rows, 1 with a missing ``systime``: 2 events and a warning."""
    frame = _points([3.5, 3.5, 3.5])
    frame["systime"] = pd.Series(
        [T0, pd.NaT, T0 + pd.Timedelta(minutes=2)], dtype="datetime64[ns, UTC]"
    )
    with pytest.warns(DataQualityWarning, match="1 row\\(s\\) have no 'systime'"):
        log = to_event_log(frame, detector=ZSCORE)
    assert len(log.events) == 2
    assert log.events["ocel:timestamp"].notna().all()
    assert len(log.relations) == 2


def test_interval_row_without_end_is_dropped():
    """2 interval rows, 1 with a missing ``end``: 1 event and a warning; an
    all-missing ``end`` gives an empty log."""
    frame = pd.DataFrame(
        {
            "start": [T0, T0],
            "end": pd.Series(
                [T0 + pd.Timedelta(minutes=1), pd.NaT], dtype="datetime64[ns, UTC]"
            ),
            "uuid": ["u0", "u1"],
            "source_uuid": ["A", "A"],
            "is_delta": [True, True],
            "state": ["run", "idle"],
        }
    )
    with pytest.warns(DataQualityWarning, match="1 row\\(s\\) have no 'end'"):
        log = to_event_log(frame, detector="MachineStateEvents.detect_run_idle")
    assert log.events["ocel:activity"].tolist() == ["production.machine_state.run"]
    assert log.events["ts_shape:duration_s"].tolist() == [60.0]
    with pytest.warns(DataQualityWarning, match="2 row\\(s\\) have no 'end'"):
        empty = to_event_log(
            frame.assign(end=pd.NaT), detector="MachineStateEvents.detect_run_idle"
        )
    assert len(empty.events) == 0


# --- 7. empty XES export ---------------------------------------------------


@pytest.mark.parametrize("lifecycle", ["single", "two_row"])
def test_empty_log_gives_empty_xes(lifecycle):
    """An event log without events exports as an empty XES frame."""
    xes = to_event_log_xes(EventLog(), lifecycle=lifecycle)
    assert len(xes) == 0
    expected = [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
        "lifecycle:transition",
        "start_timestamp",
    ]
    if lifecycle == "two_row":
        expected.append("concept:instance")
    assert xes.columns.tolist() == expected


# --- 9. ValueMapper text keys ----------------------------------------------


def _frame(codes: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range(
                "2025-01-01", periods=len(codes), freq="1min", tz="UTC"
            ),
            "code": codes,
        }
    )


def _map(tmp_path, codes, rows, file_type="csv"):
    if file_type == "csv":
        path = tmp_path / "map.csv"
        path.write_text("code,text\n" + "".join(f"{k},{v}\n" for k, v in rows))
    else:
        path = tmp_path / "map.json"
        path.write_text(json.dumps([{"code": k, "text": v} for k, v in rows]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        mapper = ValueMapper(
            _frame(codes), str(path), "code", "code", "text", file_type=file_type
        )
        return mapper.map_values()["code"].tolist()


@pytest.mark.parametrize("file_type", ["csv", "json"])
def test_leading_zero_keys_stay_distinct(tmp_path, file_type):
    """Keys ``007`` and ``7`` are two keys: ``"007"`` -> bond, ``"7"`` and
    ``7`` -> seven."""
    rows = [("007", "bond"), ("7", "seven")]
    assert _map(tmp_path, ["007", "7", 7], rows, file_type) == [
        "bond",
        "seven",
        "seven",
    ]


def test_seven_does_not_pick_up_007(tmp_path):
    """With only the key ``007``, the codes ``"7"`` and ``7`` stay unmapped."""
    assert _map(tmp_path, ["007", "7", 7], [("007", "bond")]) == ["bond", "7", 7]


def test_integral_codes_match_text_key(tmp_path):
    """Key ``1`` matches the codes ``1``, ``1.0`` and ``"1"``; the text code
    ``"1.0"`` is not the key ``1``."""
    assert _map(tmp_path, [1, 1.0, "1", "1.0"], [("1", "one")]) == [
        "one",
        "one",
        "one",
        "1.0",
    ]


def test_json_number_keys_become_text(tmp_path):
    """A JSON number key ``7`` is the text key ``"7"``."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps([{"code": 7, "text": "seven"}]))
    mapper = ValueMapper(
        _frame(["7", 7]), str(path), "code", "code", "text", file_type="json"
    )
    assert mapper.map_values()["code"].tolist() == ["seven", "seven"]
