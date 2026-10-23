"""Object analytics: lead time, dwell, WIP, carrier loops and counts per object.

The object layer (:mod:`ts_shape.eventlog.objects`) says *when* each serial,
carrier or batch was present *where*. This module turns those presences into
the numbers a line is run by:

* :func:`presence_events` -- one ``enter`` and one ``leave`` point event per
  presence. Through :func:`~ts_shape.eventlog.to_event_log` (detector
  ``"eventlog.presence_events"``) every event links to its object, so a
  serial-centric XES log has real traces: enter station A, leave A, enter B.
* :func:`object_kpis` -- per object lead time, presence count and dwell,
  dwell per station, and (with ``window=``) throughput per window.
* :func:`wip_over_time` -- time-weighted average and maximum work in process
  per window.
* :func:`carrier_loops` -- loop count and loop times of circulating carriers.
* :func:`attach_counts` -- parts (or scrap) per object from a counter signal.

Input
-----
Every function takes the interval table of
:func:`~ts_shape.eventlog.object_intervals` (columns ``oid``, ``type``,
``start``, ``end``; ``uuid`` / ``source_uuid`` when present) or an
:class:`~ts_shape.eventlog.EventLog`:

* a log built from :func:`presence_events` (``to_event_log(...,
  detector="eventlog.presence_events")``, alone or concatenated with other
  logs): the presences are rebuilt from its enter / leave events, stations
  included;
* else the ``lifecycle`` rows of ``object_changes`` (the
  :func:`~ts_shape.eventlog.detect_objects` log): one presence per
  ``active`` until the next lifecycle change. Those presences have no
  station (``<NA>``).

Rows without an ``oid``, ``type``, ``start`` or ``end`` are skipped; a
presence that ends before it starts raises ``ValueError``.

Stations
--------
The **station** of a presence is where the object is: the presence's
``source_uuid``, or its id signal ``uuid`` when it has none (one reader per
station). Presences are half-open ``[start, end)``, as everywhere in the
object layer.

Time zones
----------
Times keep the zone of the input (tz-aware stays in its zone, naive stays
naive). ``tz=`` converts them first (naive times are read as UTC, like the
rest of the event log); windows are laid out in that zone, so a daily window
on a DST day lasts 23 or 25 hours. The interval table of
:func:`~ts_shape.eventlog.object_intervals` is in UTC: pass ``tz=`` for
local days and shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.coerce import to_numeric_checked
from ..utils.counters import counter_increments
from ..utils.frames import require_columns, to_ns
from ..utils.time_accounting import grouper_periods, split_intervals
from . import schema
from .model import EventLog
from .objects import _presence_station, _to_utc, _warn_tz_mismatch

__all__ = [
    "PRESENCE_DETECTOR",
    "CarrierLoops",
    "ObjectKPIs",
    "attach_counts",
    "carrier_loops",
    "object_kpis",
    "presence_events",
    "wip_over_time",
]

_I64: TypeAlias = "npt.NDArray[np.int64]"
_F64: TypeAlias = "npt.NDArray[np.float64]"

#: Interval tables or an event log (see the module docstring).
ObjectData: TypeAlias = "pd.DataFrame | EventLog"

#: ``detector`` name of :func:`presence_events` frames in
#: :func:`~ts_shape.eventlog.to_event_log`.
PRESENCE_DETECTOR = "eventlog.presence_events"

_PACK = "production"
_LIFECYCLE_STATE = "ts_shape:lifecycle_state"

PRESENCE_EVENT_COLUMNS = [
    "systime",
    "oid",
    "type",
    "event",
    "station",
    "presence_index",
    "uuid",
    "source_uuid",
]
OBJECT_COLUMNS = [
    "oid",
    "type",
    "start",
    "end",
    "lead_time_seconds",
    "presence_count",
    "station_count",
    "dwell_seconds",
]
STATION_COLUMNS = [
    "oid",
    "type",
    "station",
    "start",
    "end",
    "presence_count",
    "dwell_seconds",
]
THROUGHPUT_COLUMNS = [
    "start",
    "end",
    "duration_seconds",
    "completed_count",
    "mean_lead_time_seconds",
]
WIP_COLUMNS = [
    "start",
    "end",
    "duration_seconds",
    "object_seconds",
    "wip_mean",
    "wip_max",
]
LOOP_COLUMNS = [
    "oid",
    "type",
    "station",
    "loop_index",
    "start",
    "end",
    "duration_seconds",
]
CARRIER_COLUMNS = ["oid", "type", "station", "loop_count", "mean_loop_seconds"]
COUNT_COLUMNS = ["oid", "type", "quantity"]

_STRING = ("oid", "type", "station", "uuid", "source_uuid", "event")


# ---------------------------------------------------------------------------
# Results with several tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectKPIs:
    """Result of :func:`object_kpis`.

    Attributes:
        objects: One row per object: ``oid``, ``type``, ``start`` (first
            enter), ``end`` (last leave), ``lead_time_seconds``,
            ``presence_count``, ``station_count`` (known stations),
            ``dwell_seconds`` (the sum of its presence durations; presences
            at two stations at once both count, so it can exceed the lead
            time).
        stations: One row per object and station, in order of arrival:
            ``oid``, ``type``, ``station``, ``start``, ``end``,
            ``presence_count``, ``dwell_seconds``.
        throughput: With ``window=``, one row per window: ``start``, ``end``,
            ``duration_seconds``, ``completed_count`` (objects whose last
            leave falls in the window) and ``mean_lead_time_seconds`` of
            those (NaN when none). ``None`` without ``window``.
    """

    objects: pd.DataFrame
    stations: pd.DataFrame
    throughput: pd.DataFrame | None = None


@dataclass(frozen=True)
class CarrierLoops:
    """Result of :func:`carrier_loops`.

    Attributes:
        loops: One row per loop: ``oid``, ``type``, ``station`` (the start
            station), ``loop_index`` (0-based per carrier), ``start`` (an
            arrival at the start station), ``end`` (the next arrival there)
            and ``duration_seconds`` (the loop time).
        carriers: One row per carrier: ``oid``, ``type``, ``station``,
            ``loop_count`` and ``mean_loop_seconds`` (NaN without a loop).
    """

    loops: pd.DataFrame
    carriers: pd.DataFrame


# ---------------------------------------------------------------------------
# Presences: the common input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Presences:
    """Presences sorted by object, then start: one group per object."""

    frame: pd.DataFrame  # oid, type, station, start, end, uuid, source_uuid
    start: _I64
    end: _I64
    key: _I64  # object group, 0..n_objects-1, contiguous
    first: _I64  # first row of each object group
    station: _I64  # station code (``<NA>`` is a code of its own)
    tz: Any

    @property
    def seconds(self) -> _F64:
        return (self.end - self.start).astype("float64") / 1e9


def _times(ns: npt.ArrayLike, tz: Any) -> pd.Series:
    """Datetimes from int64 ns (UTC instants when ``tz`` is set)."""
    values = pd.Series(np.asarray(ns, dtype=np.int64).view("datetime64[ns]"))
    if tz is None:
        return values
    return values.dt.tz_localize("UTC").dt.tz_convert(tz)


def _as_time(values: pd.Series, tz: Any) -> pd.Series:
    times = pd.to_datetime(values.reset_index(drop=True))
    if tz is not None:
        return _to_utc(times).dt.tz_convert(tz)
    return times.dt.as_unit("ns")


def _strings(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].reset_index(drop=True).astype("string")
    return pd.Series(pd.NA, index=pd.RangeIndex(len(frame)), dtype="string")


def _take(frame: pd.DataFrame, column: str, rows: npt.ArrayLike) -> pd.Series:
    """Rows of one column, dtype kept, on a fresh RangeIndex."""
    return frame[column].take(np.asarray(rows, dtype=np.intp)).reset_index(drop=True)


def _log_presences(log: EventLog) -> pd.DataFrame:
    """Presence intervals of an event log (see the module docstring)."""
    events = log.events
    col = {name: f"{_PACK}:{name}" for name in ("oid", "type", "presence_index")}
    if schema.TS_DETECTOR in events.columns and set(col.values()) <= set(
        events.columns
    ):
        sel = events[events[schema.TS_DETECTOR] == PRESENCE_DETECTOR]
        if not sel.empty and _LIFECYCLE_STATE in sel.columns:
            frame = pd.DataFrame(
                {
                    "oid": _strings(sel, col["oid"]),
                    "type": _strings(sel, col["type"]),
                    "station": _strings(sel, f"{_PACK}:station"),
                    "uuid": _strings(sel, f"{_PACK}:uuid"),
                    "source_uuid": _strings(sel, f"{_PACK}:source_uuid"),
                    "presence_index": pd.to_numeric(
                        sel[col["presence_index"]].reset_index(drop=True)
                    ),
                    "event": _strings(sel, _LIFECYCLE_STATE),
                    "time": sel[schema.OCEL_TIMESTAMP].reset_index(drop=True),
                }
            )
            keys = ["oid", "type", "presence_index"]
            is_enter = (frame["event"] == "enter").fillna(False).astype(bool)
            is_leave = (frame["event"] == "leave").fillna(False).astype(bool)
            enter = frame[is_enter].drop(columns="event")
            leave = frame.loc[is_leave, [*keys, "time"]]
            pairs = enter.merge(leave, on=keys, how="inner", suffixes=("", "_end"))
            return pairs.rename(columns={"time": "start", "time_end": "end"})
    changes = log.object_changes
    life = changes[changes[schema.OCEL_FIELD] == "lifecycle"]
    life = life.sort_values(
        [schema.OCEL_OID, schema.OCEL_TYPE, schema.OCEL_TIMESTAMP], kind="stable"
    ).reset_index(drop=True)
    oid = life[schema.OCEL_OID].astype("string")
    typ = life[schema.OCEL_TYPE].astype("string")
    nxt = life[schema.OCEL_TIMESTAMP].shift(-1)
    same = (oid == oid.shift(-1)) & (typ == typ.shift(-1))
    active = (life[schema.OCEL_VALUE].astype("string") == "active") & same
    active = active.fillna(False).astype(bool)
    return pd.DataFrame(
        {
            "oid": oid[active],
            "type": typ[active],
            "station": pd.Series(pd.NA, index=oid.index, dtype="string")[active],
            "start": life.loc[active, schema.OCEL_TIMESTAMP],
            "end": nxt[active],
        }
    ).reset_index(drop=True)


def _presences(
    data: ObjectData, object_type: str | None, *, tz: Any, where: str
) -> _Presences:
    if isinstance(data, EventLog):
        frame = _log_presences(data)
    elif isinstance(data, pd.DataFrame):
        require_columns(data, ("oid", "type", "start", "end"), where=where)
        frame = data.reset_index(drop=True)
    else:
        raise TypeError(
            f"{where}: expected an object interval table (DataFrame) or an "
            f"EventLog, got {type(data).__name__}"
        )
    if object_type is not None:
        # Select the type first: the string columns below are costly.
        is_type = (_strings(frame, "type") == object_type).fillna(False)
        frame = frame[is_type.to_numpy(dtype=bool)].reset_index(drop=True)
    oid, typ = _strings(frame, "oid"), _strings(frame, "type")
    start, end = _as_time(frame["start"], tz), _as_time(frame["end"], tz)
    keep = oid.notna() & typ.notna() & start.notna() & end.notna()
    keep = keep.fillna(False).to_numpy(dtype=bool)
    if not keep.all():
        frame = frame[keep].reset_index(drop=True)
        oid = oid[keep].reset_index(drop=True)
        typ = typ[keep].reset_index(drop=True)
        start = start[keep].reset_index(drop=True)
        end = end[keep].reset_index(drop=True)
    station = (
        _strings(frame, "station")
        if "station" in frame.columns
        else _presence_station(frame).reset_index(drop=True)
    )
    out = pd.DataFrame(
        {
            "oid": oid,
            "type": typ,
            "station": station,
            "start": start,
            "end": end,
            "uuid": _strings(frame, "uuid"),
            "source_uuid": _strings(frame, "source_uuid"),
        }
    )
    s, e = to_ns(out["start"]), to_ns(out["end"])
    if (e < s).any():
        row = out.iloc[int(np.flatnonzero(e < s)[0])]
        raise ValueError(
            f"{where}: object {row['oid']!r} ends before it starts "
            f"({row['start']} > {row['end']})"
        )
    type_code = pd.factorize(out["type"], sort=True)[0]
    oid_code = pd.factorize(out["oid"], sort=True)[0]
    station_code = pd.factorize(out["station"], sort=True, use_na_sentinel=False)[0]
    order = np.lexsort((station_code, e, s, oid_code, type_code))
    out = out.iloc[order].reset_index(drop=True)
    s, e, station_code = s[order], e[order], station_code[order]
    type_code, oid_code = type_code[order], oid_code[order]
    new = np.ones(len(out), dtype=bool)
    new[1:] = (type_code[1:] != type_code[:-1]) | (oid_code[1:] != oid_code[:-1])
    key = np.cumsum(new) - 1
    zone = out["start"].dt.tz
    return _Presences(
        frame=out,
        start=s.astype(np.int64),
        end=e.astype(np.int64),
        key=key.astype(np.int64),
        first=np.flatnonzero(new).astype(np.int64),
        station=station_code.astype(np.int64),
        tz=zone,
    )


def _empty(columns: list[str], tz: Any = "UTC") -> pd.DataFrame:
    dtype: Any = pd.DatetimeTZDtype("ns", tz) if tz is not None else "datetime64[ns]"
    cols: dict[str, pd.Series] = {}
    for c in columns:
        if c in ("start", "end", "systime"):
            cols[c] = pd.Series(dtype=dtype)
        elif c in _STRING:
            cols[c] = pd.Series(dtype="string")
        elif c.endswith("_count") or c in ("presence_index", "loop_index", "wip_max"):
            cols[c] = pd.Series(dtype="int64")
        else:
            cols[c] = pd.Series(dtype="float64")
    return pd.DataFrame(cols)


def _zone(data: ObjectData, tz: Any) -> Any:
    """Time zone of empty results: ``tz``, else the input's (UTC for a log)."""
    if tz is not None:
        return tz
    if isinstance(data, pd.DataFrame) and "start" in data.columns:
        dtype = data["start"].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            return dtype.tz
        if pd.api.types.is_datetime64_dtype(dtype):
            return None
    return "UTC"


def _windows(p: _Presences, window: Any) -> pd.DataFrame:
    """``start`` / ``end`` of the windows covering the presences."""
    lo, hi = int(p.start.min()), int(p.end.max())
    t0, t1 = _times([lo], p.tz).iloc[0], _times([hi], p.tz).iloc[0]
    periods = grouper_periods(t0, t1, window)[["start", "end"]]
    return periods.sort_values("start", kind="stable").reset_index(drop=True)


def _window_frame(periods: pd.DataFrame) -> pd.DataFrame:
    out = periods[["start", "end"]].copy()
    out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
    return out


# ---------------------------------------------------------------------------
# Presence events
# ---------------------------------------------------------------------------


def presence_events(
    intervals: ObjectData, *, object_type: str | None = None
) -> pd.DataFrame:
    """One ``enter`` and one ``leave`` point event per object presence.

    Turn them into an event log with ``to_event_log(frame,
    detector="eventlog.presence_events")``: the activity is
    ``production.presence.<event>.<station>`` and every event links to its
    object (``oid`` of type ``type``) and to the ``source_uuid`` asset, so
    ``to_event_log_xes(log, case_object_type="serial")`` gives one trace per
    serial: enter station A, leave A, enter B, leave B.

    Args:
        intervals: :func:`~ts_shape.eventlog.object_intervals` output or an
            event log (see the module docstring).
        object_type: Keep only this object type; ``None`` keeps every type.

    Returns:
        DataFrame with ``systime``, ``oid``, ``type``, ``event``
        (``"enter"`` at the presence start, ``"leave"`` at its end),
        ``station`` (``source_uuid``, else the id signal ``uuid``),
        ``presence_index`` (0-based per object, in order of entry), ``uuid``
        and ``source_uuid``. Sorted by time; at one instant the events of an
        object follow its presences (leave A before enter B at a hand-over).
        Times keep the input's zone. An empty input gives an empty frame
        with these columns.
    """
    p = _presences(intervals, object_type, tz=None, where="presence_events")
    n = len(p.frame)
    if n == 0:
        return _empty(PRESENCE_EVENT_COLUMNS, _zone(intervals, None))
    index = np.arange(n, dtype=np.int64) - p.first[p.key]
    rows = np.concatenate([np.arange(n), np.arange(n)])
    times = np.concatenate([p.start, p.end])
    rank = np.repeat(np.array([0, 1], dtype=np.int64), n)
    order = np.lexsort((rank, index[rows], p.key[rows], times))
    rows = rows[order]
    frame = p.frame
    out = pd.DataFrame(
        {
            "systime": _times(times[order], p.tz),
            "oid": _take(frame, "oid", rows),
            "type": _take(frame, "type", rows),
            "event": np.where(rank[order] == 0, "enter", "leave"),
            "station": _take(frame, "station", rows),
            "presence_index": index[rows],
            "uuid": _take(frame, "uuid", rows),
            "source_uuid": _take(frame, "source_uuid", rows),
        }
    )
    for c in ("oid", "type", "event", "station", "uuid", "source_uuid"):
        out[c] = out[c].astype("string")
    return out


# ---------------------------------------------------------------------------
# Lead time, dwell, throughput
# ---------------------------------------------------------------------------


def object_kpis(
    intervals: ObjectData,
    *,
    object_type: str,
    window: Any = None,
    tz: Any = None,
) -> ObjectKPIs:
    """Lead time, presence count and dwell per object, dwell per station.

    The lead time of an object runs from its first enter to its last leave.
    An object still present at the end of the data ends there (its last
    presence ends at the last observation).

    Args:
        intervals: :func:`~ts_shape.eventlog.object_intervals` output or an
            event log (see the module docstring).
        object_type: The object type to report (``"serial"``).
        window: Optional window (``"1h"``, ``"1D"``, ``"W"``, ...) for the
            throughput table: objects completing (last leave) per window.
            The windows cover the data from the first enter to the last
            leave, windows without completions included.
        tz: Time zone to convert the times to and lay the windows out in;
            ``None`` keeps the input's.

    Returns:
        :class:`ObjectKPIs` with ``objects``, ``stations`` and
        ``throughput`` (``None`` without ``window``). Empty input gives
        empty tables with the documented columns.
    """
    p = _presences(intervals, object_type, tz=tz, where="object_kpis")
    if len(p.frame) == 0:
        zone = _zone(intervals, tz)
        return ObjectKPIs(
            objects=_empty(OBJECT_COLUMNS, zone),
            stations=_empty(STATION_COLUMNS, zone),
            throughput=None if window is None else _empty(THROUGHPUT_COLUMNS, zone),
        )
    frame, first = p.frame, p.first
    seconds = p.seconds
    n_objects = len(first)
    first_start = p.start[first]
    last_end = np.maximum.reduceat(p.end, first)
    counts = np.diff(np.append(first, len(frame)))
    n_stations = int(p.station.max()) + 1
    known = frame["station"].notna().to_numpy(dtype=bool)
    pairs = np.unique(p.key[known] * n_stations + p.station[known])
    station_count = np.bincount(pairs // n_stations, minlength=n_objects)
    lead = (last_end - first_start).astype("float64") / 1e9
    objects = pd.DataFrame(
        {
            "oid": _take(frame, "oid", first),
            "type": _take(frame, "type", first),
            "start": _times(first_start, p.tz),
            "end": _times(last_end, p.tz),
            "lead_time_seconds": lead,
            "presence_count": counts.astype("int64"),
            "station_count": station_count.astype("int64"),
            "dwell_seconds": np.add.reduceat(seconds, first),
        }
    )
    objects = objects.sort_values(["start", "type", "oid"], kind="stable")
    objects = objects.reset_index(drop=True)
    for c in ("oid", "type"):
        objects[c] = objects[c].astype("string")

    # Per (object, station), in order of arrival (rows are object, start).
    pair = pd.factorize(p.key * n_stations + p.station)[0]
    grouped = pd.DataFrame(
        {"g": pair, "s": p.start, "e": p.end, "d": seconds, "row": np.arange(len(pair))}
    ).groupby("g", sort=True)
    agg = grouped.agg(
        s=("s", "min"),
        e=("e", "max"),
        d=("d", "sum"),
        n=("s", "size"),
        row=("row", "first"),
    )
    row = agg["row"].to_numpy()
    stations = pd.DataFrame(
        {
            "oid": _take(frame, "oid", row),
            "type": _take(frame, "type", row),
            "station": _take(frame, "station", row),
            "start": _times(agg["s"].to_numpy(), p.tz),
            "end": _times(agg["e"].to_numpy(), p.tz),
            "presence_count": agg["n"].to_numpy(dtype="int64"),
            "dwell_seconds": agg["d"].to_numpy(dtype="float64"),
        }
    )
    for c in ("oid", "type", "station"):
        stations[c] = stations[c].astype("string")

    throughput = None
    if window is not None:
        periods = _window_frame(_windows(p, window))
        ps, pe = to_ns(periods["start"]), to_ns(periods["end"])
        w = np.searchsorted(ps, last_end, side="right") - 1
        hit = (w >= 0) & (last_end < pe[np.clip(w, 0, None)])
        done = np.bincount(w[hit], minlength=len(periods))
        total = np.bincount(w[hit], weights=lead[hit], minlength=len(periods))
        mean = np.full(len(periods), np.nan)
        np.divide(total, done, out=mean, where=done > 0)
        throughput = periods.assign(
            completed_count=done.astype("int64"), mean_lead_time_seconds=mean
        )
    return ObjectKPIs(objects=objects, stations=stations, throughput=throughput)


# ---------------------------------------------------------------------------
# Work in process
# ---------------------------------------------------------------------------


def _union(p: _Presences) -> tuple[_I64, _I64]:
    """Each object's presences merged where they overlap or touch."""
    k, s, e = p.key, p.start, p.end
    runmax = pd.Series(e).groupby(k).cummax().to_numpy()
    new = np.ones(len(k), dtype=bool)
    new[1:] = (k[1:] != k[:-1]) | (s[1:] > runmax[:-1])
    first = np.flatnonzero(new)
    return s[first], np.maximum.reduceat(e, first)


def wip_over_time(
    intervals: ObjectData,
    *,
    object_type: str,
    window: Any,
    tz: Any = None,
) -> pd.DataFrame:
    """Time-weighted average and maximum work in process (WIP) per window.

    WIP at an instant is the number of objects of ``object_type`` present
    anywhere: an object counts once while at two stations at once, and not
    twice at a hand-over (presences are half-open). Presences are split at
    the window bounds (:func:`~ts_shape.utils.time_accounting.split_intervals`)
    and the windows come from
    :func:`~ts_shape.utils.time_accounting.grouper_periods`, so a daily
    window on a DST day lasts 23 or 25 hours.

    Args:
        intervals: :func:`~ts_shape.eventlog.object_intervals` output or an
            event log (see the module docstring).
        object_type: The object type to count (``"serial"``).
        window: Window (``"1h"``, ``"8h"``, ``"1D"``, ``"W"``, ...).
        tz: Time zone to lay the windows out in; ``None`` keeps the input's.

    Returns:
        One row per window from the first enter to the last leave, with
        ``start``, ``end``, ``duration_seconds``, ``object_seconds`` (object
        presence time in the window), ``wip_mean`` (``object_seconds /
        duration_seconds``, averaged over the whole window) and ``wip_max``
        (the most objects present at once in the window). Empty input gives
        an empty frame with these columns.
    """
    p = _presences(intervals, object_type, tz=tz, where="wip_over_time")
    if len(p.frame) == 0:
        return _empty(WIP_COLUMNS, _zone(intervals, tz))
    periods = _window_frame(_windows(p, window))
    u_start, u_end = _union(p)
    pieces = split_intervals(
        pd.DataFrame({"start": _times(u_start, p.tz), "end": _times(u_end, p.tz)}),
        periods.assign(_p=np.arange(len(periods))),
        keys=["_p"],
    )
    object_seconds = np.bincount(
        pieces["_p"].to_numpy(dtype=np.int64),
        weights=pieces["seconds"].to_numpy(dtype="float64"),
        minlength=len(periods),
    )
    # Step function of the WIP level: +1 at each start, -1 at each end.
    times, inverse = np.unique(np.concatenate([u_start, u_end]), return_inverse=True)
    delta = np.bincount(
        inverse,
        weights=np.repeat([1.0, -1.0], len(u_start)),
        minlength=len(times),
    )
    level = np.rint(np.cumsum(delta)).astype(np.int64)
    ps, pe = to_ns(periods["start"]), to_ns(periods["end"])
    lo = np.searchsorted(times, ps, side="right") - 1
    wip_max = np.where(lo >= 0, level[np.clip(lo, 0, None)], 0)
    w = np.searchsorted(ps, times, side="right") - 1
    inside = (w >= 0) & (times < pe[np.clip(w, 0, None)])
    peaks = pd.Series(level[inside]).groupby(w[inside]).max()
    idx = peaks.index.to_numpy(dtype=np.int64)
    wip_max[idx] = np.maximum(wip_max[idx], peaks.to_numpy(dtype=np.int64))
    duration = periods["duration_seconds"].to_numpy(dtype="float64")
    return periods.assign(
        object_seconds=object_seconds,
        wip_mean=object_seconds / duration,
        wip_max=wip_max.astype("int64"),
    )[WIP_COLUMNS]


# ---------------------------------------------------------------------------
# Carrier loops
# ---------------------------------------------------------------------------


def carrier_loops(
    intervals: ObjectData,
    *,
    object_type: str = "carrier",
    start_station: str | None = None,
) -> CarrierLoops:
    """Loop count and loop times of carriers circulating on a line.

    A loop is the time between two successive arrivals of a carrier at the
    loop's start station; every presence there is an arrival.

    Args:
        intervals: :func:`~ts_shape.eventlog.object_intervals` output or an
            event log (see the module docstring).
        object_type: The carrier object type.
        start_station: Station the loops start at (a ``source_uuid``, or the
            id signal ``uuid`` of a reader without one). ``None``: each
            carrier's first station, the one it is first seen at.

    Returns:
        :class:`CarrierLoops` with ``loops`` (one row per loop) and
        ``carriers`` (one row per carrier, including carriers without a full
        loop). Empty input gives empty tables with the documented columns.
    """
    p = _presences(intervals, object_type, tz=None, where="carrier_loops")
    zone = _zone(intervals, None)
    if len(p.frame) == 0:
        return CarrierLoops(_empty(LOOP_COLUMNS, zone), _empty(CARRIER_COLUMNS, zone))
    frame = p.frame
    station_text = frame["station"].to_numpy(dtype=object)
    if start_station is None:
        target = p.station[p.first]
        label = station_text[p.first]
    else:
        is_start = (frame["station"] == start_station).fillna(False)
        match = np.flatnonzero(is_start.to_numpy(dtype=bool))
        code = p.station[match[0]] if len(match) else -2
        target = np.full(len(p.first), code, dtype=np.int64)
        label = np.full(len(p.first), start_station, dtype=object)
    arrivals = np.flatnonzero(p.station == target[p.key])
    a_key = p.key[arrivals]
    same = a_key[1:] == a_key[:-1]
    cur, nxt = arrivals[:-1][same], arrivals[1:][same]
    loop_key = p.key[cur]
    new = np.ones(len(cur), dtype=bool)
    new[1:] = loop_key[1:] != loop_key[:-1]
    group_first = np.flatnonzero(new)
    loop_index = np.arange(len(cur)) - np.repeat(
        group_first, np.diff(np.append(group_first, len(cur)))
    )
    seconds = (p.start[nxt] - p.start[cur]).astype("float64") / 1e9
    loops = pd.DataFrame(
        {
            "oid": _take(frame, "oid", cur),
            "type": _take(frame, "type", cur),
            "station": label[loop_key],
            "loop_index": loop_index.astype("int64"),
            "start": _times(p.start[cur], p.tz),
            "end": _times(p.start[nxt], p.tz),
            "duration_seconds": seconds,
        }
    )
    n_objects = len(p.first)
    loop_count = np.bincount(loop_key, minlength=n_objects)
    total = np.bincount(loop_key, weights=seconds, minlength=n_objects)
    mean = np.full(n_objects, np.nan)
    np.divide(total, loop_count, out=mean, where=loop_count > 0)
    carriers = pd.DataFrame(
        {
            "oid": _take(frame, "oid", p.first),
            "type": _take(frame, "type", p.first),
            "station": label,
            "loop_count": loop_count.astype("int64"),
            "mean_loop_seconds": mean,
        }
    )
    for table in (loops, carriers):
        for c in ("oid", "type", "station"):
            table[c] = table[c].astype("string")
    return CarrierLoops(loops=loops, carriers=carriers)


# ---------------------------------------------------------------------------
# Counts per object
# ---------------------------------------------------------------------------


def _counter_station(
    readings: pd.DataFrame, station: str | None, counter_uuid: str
) -> str | None:
    if station is not None or "source_uuid" not in readings.columns:
        return station
    sources = readings["source_uuid"].dropna().astype(str).unique()
    if len(sources) > 1:
        raise ValueError(
            f"attach_counts: counter {counter_uuid!r} has several source_uuid "
            f"values {sorted(sources)[:5]}; pass station= to choose one"
        )
    return str(sources[0]) if len(sources) else None


def attach_counts(
    intervals: ObjectData,
    dataframe: pd.DataFrame,
    counter_uuid: str,
    *,
    object_type: str,
    value_column: str = "value_integer",
    station: str | None = None,
    uuid_column: str = "uuid",
    time_column: str = "systime",
) -> pd.DataFrame:
    """Quantity of a counter (parts, scrap) produced per object.

    The increment a counter reports at reading ``t_i`` was produced during
    ``(t_{i-1}, t_i]``. It is shared between the objects present at the
    counter's station in that time, in proportion to the time each was
    present (constant production rate between two readings). When several
    objects are present at once, the shares are scaled down so that they
    never add up to more than the increment. The part of an increment
    produced while no object was present is reported as unattributed.

    Counter resets follow :func:`~ts_shape.utils.counters.counter_increments`:
    a reading lower than the one before is a reset, and its value is the
    production since the reset. The first reading is the baseline (no
    production); missing or non-numeric readings are skipped. Readings at
    the same time are summed.

    Args:
        intervals: :func:`~ts_shape.eventlog.object_intervals` output or an
            event log (see the module docstring).
        dataframe: Long-format frame holding the counter signal.
        counter_uuid: uuid of the counter signal.
        object_type: Object type the counts are attributed to.
        value_column: Column of the counter readings.
        station: Station whose objects receive the counts. ``None``: the
            counter's ``source_uuid``; a counter without one counts for the
            objects present at any station.
        uuid_column: Signal identifier column of ``dataframe``.
        time_column: Timestamp column of ``dataframe``.

    Returns:
        One row per object present at the station, in order of first
        presence: ``oid``, ``type``, ``quantity`` (float; 0 for an object
        present while nothing was produced). A last row with ``oid`` <NA>
        holds the unattributed quantity, when there is any. The quantities
        add up to the counter's total. Empty inputs give an empty frame with
        these columns.

    Raises:
        ColumnNotFoundError: When ``dataframe`` lacks a named column.
        ValueError: When ``station`` is ``None`` and the counter has several
            ``source_uuid`` values.
    """
    require_columns(
        dataframe, (uuid_column, time_column, value_column), where="attach_counts"
    )
    readings = dataframe[dataframe[uuid_column] == counter_uuid]
    station = _counter_station(readings, station, counter_uuid)
    p = _presences(intervals, object_type, tz="UTC", where="attach_counts")
    if station is not None:
        keep = (p.frame["station"] == station).fillna(False).to_numpy(dtype=bool)
    else:
        keep = np.ones(len(p.frame), dtype=bool)
    if isinstance(intervals, pd.DataFrame) and "start" in intervals.columns:
        _warn_tz_mismatch(readings[time_column], intervals["start"])

    values = to_numeric_checked(
        readings[value_column], what=f"counter column {value_column!r}"
    ).astype("float64")
    times = _to_utc(readings[time_column])
    ok = values.notna().to_numpy() & times.notna().to_numpy()
    t = to_ns(times[ok])
    v = values.to_numpy()[ok]
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    inc = counter_increments(pd.Series(v)).to_numpy(dtype="float64")

    rows = np.flatnonzero(keep)
    key = p.key[rows]
    objects, slot = np.unique(key, return_inverse=True)  # in order of key
    first_row = rows[np.unique(key, return_index=True)[1]]
    quantity = np.zeros(len(objects), dtype="float64")
    unattributed = 0.0
    if len(t):
        tu, inverse = np.unique(t, return_inverse=True)
        q = np.bincount(inverse, weights=inc, minlength=len(tu))
        # The first instant has no interval before it: a point share.
        present = (p.start[rows] <= tu[0]) & (tu[0] < p.end[rows])
        if q[0] and present.any():
            np.add.at(quantity, slot[present], q[0] / int(present.sum()))
        elif q[0]:
            unattributed += q[0]
        if len(tu) > 1:
            readings_iv = pd.DataFrame(
                {
                    "start": _times(tu[:-1], "UTC"),
                    "end": _times(tu[1:], "UTC"),
                    "_r": np.arange(len(tu) - 1),
                }
            )
            pieces = split_intervals(
                pd.DataFrame(
                    {
                        "start": _times(p.start[rows], "UTC"),
                        "end": _times(p.end[rows], "UTC"),
                        "_slot": slot,
                    }
                ),
                readings_iv,
                keys=["_r"],
            )
            r = pieces["_r"].to_numpy(dtype=np.int64)
            sec = pieces["seconds"].to_numpy(dtype="float64")
            length = (tu[1:] - tu[:-1]).astype("float64") / 1e9
            covered = np.bincount(r, weights=sec, minlength=len(length))
            denom = np.maximum(length, covered)
            produced = q[1:]
            share = produced[r] * sec / denom[r]
            np.add.at(quantity, pieces["_slot"].to_numpy(dtype=np.int64), share)
            unattributed += float((produced * (1.0 - covered / denom)).sum())
    out = pd.DataFrame(
        {
            "oid": _take(p.frame, "oid", first_row),
            "type": _take(p.frame, "type", first_row),
            "quantity": quantity,
            "_start": p.start[first_row],
        }
    )
    out = out.sort_values("_start", kind="stable").drop(columns="_start")
    if unattributed > 1e-9 * max(1.0, float(np.abs(inc).sum())):
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {"oid": [pd.NA], "type": [object_type], "quantity": [unattributed]}
                ),
            ],
            ignore_index=True,
        )
    out = out.reset_index(drop=True)
    out["oid"] = out["oid"].astype("string")
    out["type"] = out["type"].astype("string")
    out["quantity"] = out["quantity"].astype("float64")
    return out[COUNT_COLUMNS]
