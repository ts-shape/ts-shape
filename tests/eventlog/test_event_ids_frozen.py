"""Event ids, activities and timestamps == a frozen copy of the former code.

``adapt`` renders activities, UTC timestamps and the content hash of the
event ids column by column. The functions below are frozen copies of the
row-by-row implementation they replaced; the ids must stay byte-identical
(they are persisted and joined on), and so must activities and timestamps.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.eventlog import to_event_log
from ts_shape.eventlog.adapters import adapt
from ts_shape.eventlog.taxonomy import LabelRule, render_activity

_NAMESPACE = _uuid.UUID("8a4d3f1c-9b2e-4d3a-8f1e-c0ffee123456")


# --- frozen reference (the former implementation) -------------------------


def _ref_stable(value: object) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return "\x00"
    if isinstance(value, bool | np.bool_):
        return "True" if value else "False"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        f = float(value)
        if np.isnan(f):
            return "\x00"
        return str(int(f)) if f.is_integer() and abs(f) < 2**63 else repr(f)
    if isinstance(value, pd.Timestamp | np.datetime64 | _dt.datetime):
        ts = pd.Timestamp(value)
        if ts is pd.NaT:
            return "\x00"
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return str(ts.isoformat())
    if isinstance(value, pd.Timedelta | np.timedelta64 | _dt.timedelta):
        td = pd.Timedelta(value)
        return "\x00" if td is pd.NaT else str(td.value)
    return str(value)


def _ref_content_eids(detector, frame, ts_start, ts_end, activities, time_cols):
    n = len(frame)
    payload_cols = sorted(str(c) for c in frame.columns if c not in time_cols)
    rendered = [
        [f"{c}={_ref_stable(v)}" for v in frame[c].tolist()] for c in payload_cols
    ]
    source = (
        [_ref_stable(v) for v in frame["source_uuid"].tolist()]
        if "source_uuid" in frame.columns
        else ["\x00"] * n
    )
    starts = [_ref_stable(v) for v in ts_start.tolist()]
    ends = [_ref_stable(v) for v in ts_end.tolist()]
    acts = [str(a) for a in activities.tolist()]
    seen: dict[str, int] = {}
    out: list[str] = []
    for i in range(n):
        key = "|".join(
            [detector, source[i], starts[i], ends[i], acts[i]]
            + [col[i] for col in rendered]
        )
        k = seen.get(key, 0)
        seen[key] = k + 1
        if k:
            key = f"{key}|#{k}"
        out.append("e-" + str(_uuid.uuid5(_NAMESPACE, key)))
    return pd.Series(out, dtype="string")


def _ref_to_utc_ts(value):
    if value is None:
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        return pd.NaT
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _reference(frame: pd.DataFrame, rule: LabelRule, detector: str):
    """(eids, activities, timestamp, start) as the former ``adapt`` built them."""
    frame = frame.reset_index(drop=True)
    n = len(frame)
    if rule.shape in ("interval", "summary"):
        ts_end = frame["end"].apply(_ref_to_utc_ts)
        ts_start = frame["start"].apply(_ref_to_utc_ts)
        keep = ~(ts_end < ts_start).fillna(False).astype(bool).to_numpy()
        frame = frame.loc[keep].reset_index(drop=True)
        ts_end = ts_end.loc[keep].reset_index(drop=True)
        ts_start = ts_start.loc[keep].reset_index(drop=True)
        time_cols = {"start", "end"}
    else:
        ts_end = frame["systime"].apply(_ref_to_utc_ts)
        ts_start = pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
        time_cols = {"systime"}
    activities = frame.apply(lambda r: render_activity(rule, r), axis=1).astype(
        "string"
    )
    eids = _ref_content_eids(detector, frame, ts_start, ts_end, activities, time_cols)
    return (
        eids,
        activities.reset_index(drop=True),
        ts_end.astype("datetime64[ns, UTC]"),
        ts_start.astype("datetime64[ns, UTC]"),
    )


# --- inputs ----------------------------------------------------------------


def _mixed_frame(n: int, seed: int, unit: str, tz: str | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Series(
        pd.date_range("1969-12-31 23:59:58.5", periods=n, freq="1234567891ns")
    ).dt.as_unit(unit)
    if tz:
        start = start.dt.tz_localize(tz)
    floats = rng.normal(0, 1e3, n)
    floats[::7] = np.round(floats[::7])  # integral floats render as ints
    floats[::11] = np.nan
    floats[1::13] = [-0.0, np.inf, -np.inf, 1e20, 2.0**63, 1e-300][: len(floats[1::13])]
    obj = np.array([1, 1.0, True, None, "x", pd.Timestamp("2024", tz="UTC")] * n)[:n]
    frame = pd.DataFrame(
        {
            "start": start,
            "end": start + pd.to_timedelta(rng.integers(0, 10**9, n), unit="us"),
            "uuid": "ev",
            "source_uuid": rng.choice(["a", "b", None], n),
            "state": rng.choice(["run", "idle"], n),
            "level": rng.integers(-3, 3, n),
            "value": floats,
            "f32": rng.normal(0, 1, n).astype(np.float32),
            "flag": rng.random(n) < 0.5,
            "mixed": obj,
            "text": pd.array(rng.choice(["p", "q", None], n), dtype="string"),
            "count": pd.array(rng.integers(0, 5, n), dtype="Int64"),
            "cat": pd.Categorical(rng.choice(["u", "v"], n)),
            "lag": pd.to_timedelta(rng.integers(0, 100, n), unit="s"),
            "when": pd.Series(pd.date_range("2024", periods=n, freq="h")).where(
                rng.random(n) < 0.8
            ),
        }
    )
    frame.loc[3, "state"] = None  # missing template field
    # Exact duplicate rows get occurrence suffixes.
    return pd.concat([frame, frame.iloc[:5]], ignore_index=True)


RULES = {
    "fields": LabelRule(
        template="test.{state}.{level}.{missing}", pack="quality", shape="interval"
    ),
    "numeric_fields": LabelRule(
        template="{level}-{value}", pack="quality", shape="interval"
    ),
    "literal": LabelRule(template="test.constant", pack="quality", shape="interval"),
    "point": LabelRule(template="pt.{state}", pack="quality", shape="point"),
    "unclosed": LabelRule(template="a{state}b{level", pack="quality", shape="interval"),
}


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("rule_name", sorted(RULES))
def test_ids_activities_and_times_match_frozen_reference(unit, tz, rule_name):
    rule = RULES[rule_name]
    frame = _mixed_frame(40, seed=len(unit), unit=unit, tz=tz)
    if rule.shape == "point":
        frame = frame.rename(columns={"start": "systime"}).drop(columns="end")
    if rule_name == "numeric_fields":
        # Only numbers: rows are cast to float, an int renders as "1.0".
        frame = frame[["start", "end", "level", "value"]]
    detector = "Test.method"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eids, acts, ts, start = _reference(frame, rule, detector)
        log = adapt(frame, rule=rule, detector=detector)
    ev = log.events
    assert ev["ocel:eid"].tolist() == eids.tolist()
    assert ev["ocel:activity"].tolist() == acts.tolist()
    pd.testing.assert_series_equal(
        ev["ocel:timestamp"], ts, check_names=False, check_exact=True
    )
    pd.testing.assert_series_equal(
        ev["ts_shape:start_timestamp"], start, check_names=False, check_exact=True
    )


def test_detector_frames_match_frozen_reference():
    from ts_shape.eventlog.taxonomy import get
    from ts_shape.events.quality.change_point import ChangePointEvents

    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(m, 1, 200) for m in (0, 3, 1)])
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2024", periods=len(x), freq="1s", tz="UTC"),
            "uuid": "c",
            "value_double": x,
        }
    )
    out = ChangePointEvents(df, "c").detect_change_points()
    seg = ChangePointEvents(df, "c").segment_statistics()
    for frame, method in (
        (out, "detect_change_points"),
        (seg, "segment_statistics"),
    ):
        detector = f"ChangePointEvents.{method}"
        eids, acts, _, _ = _reference(frame, get("ChangePointEvents", method), detector)
        log = to_event_log(frame, detector=detector)
        assert log.events["ocel:eid"].tolist() == eids.tolist()
        assert log.events["ocel:activity"].tolist() == acts.tolist()
