"""Object detection — turn identifier-bearing timeseries signals into OCEL 2.0
objects, relations, and time-varying attributes.

The event packs answer *"what happened?"*. This module answers *"to which
**things**?"* — the batches, serials, coils, recipes, tools, materials, drums,
bundles, families, customer part-numbers (and any other object out there) that
events refer to.

It is deliberately **one generic, declarative layer**, not another family of
detector classes: you describe which signals carry which object type with an
:class:`ObjectSpec` (or discover them from metadata), and a run-length kernel
(the same value-change segmentation that powers batch/traceability detection)
extracts every object's presence interval. New object types are added by *data /
config*, never by code.

Three steps, all feeding the OCEL 2.0 tables on :class:`~ts_shape.eventlog.model.EventLog`:

* :func:`object_intervals` — segment id signals into ``(oid, type, start, end)``
  presence intervals (an id is present until the next id on the same signal).
* :func:`detect_objects` — build the ``objects`` / ``o2o`` / ``object_changes``
  tables. ``part_of`` is asserted only along a declared ``hierarchy`` (it cannot
  be inferred from time alone); other overlaps are reported as ``co_occurs``.
* :func:`attach_objects` — link an existing event log's events to the detected
  objects by temporal containment, so the output of *every* event detector gains
  rich object references with no per-detector changes.

When you already hold the two lists — an **event list** and an **object list**
(intervals) — the relationship derivation is automated and needs no raw signals:

* :func:`link_events_to_objects` — all event-to-object (E2O) relations, by
  temporal containment and/or direct ``key_columns`` matching.
* :func:`derive_o2o` — all object-to-object (O2O) relations (declared
  ``part_of`` hierarchy or symmetric ``co_occurs``).
* :func:`relate` — assemble both, plus objects and ``object_changes``, into one
  validated :class:`EventLog` ready to ingest into an OCEL 2.0 database.

Presence semantics (shared by every function here)
--------------------------------------------------
* An id is present from its first sample until the **next value** on the same
  signal: the next id, or an empty / missing reading (``""``, whitespace,
  ``NaN`` / ``None`` / ``<NA>``), which means "no object". A gap in the data
  does not end an object; only a different value does.
* Presence is **half-open** ``[start, end)``: at a hand-over instant the
  incoming object owns the timestamp, never both. The last presence on a signal
  (and, in a user-supplied table, any presence no other object of its signal /
  type takes over from) is closed ``[start, end]``, its end being the last
  observation.
* The same id appearing again later is the **same object** (one row in
  ``objects``) with several presence intervals.
* All timestamps are UTC (OCEL 2.0); tz-aware input is converted, naive input
  is taken as UTC.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from ts_shape.utils.deprecation import caller_stacklevel

from ..errors import ColumnNotFoundError, DataQualityWarning
from ..utils.frames import require_columns, to_ns
from . import schema
from .concat import concat
from .model import EventLog

# Tidy intermediate columns (used between the three steps).
_OID = "oid"
_TYPE = "type"
_START = "start"
_END = "end"
_DURATION = "duration_seconds"
_UUID = "uuid"
_SOURCE = "source_uuid"

#: Value columns that can carry an object id.
_ID_VALUE_COLUMNS: tuple[str, ...] = ("value_string", "value_integer", "value_double")

#: Columns of the interval table that are not object attributes.
_RESERVED: tuple[str, ...] = (_OID, _TYPE, _START, _END, _DURATION, _UUID, _SOURCE)

_I64: TypeAlias = "npt.NDArray[np.int64]"
_BOOL: TypeAlias = "npt.NDArray[np.bool_]"
_NAT = np.iinfo(np.int64).min  # int64 view of NaT


def _empty_intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            _OID: pd.Series(dtype="string"),
            _TYPE: pd.Series(dtype="string"),
            _START: pd.Series(dtype="datetime64[ns, UTC]"),
            _END: pd.Series(dtype="datetime64[ns, UTC]"),
            _DURATION: pd.Series(dtype="float64"),
            _UUID: pd.Series(dtype="string"),
            _SOURCE: pd.Series(dtype="string"),
        }
    )


def _to_utc(values: object) -> pd.Series:
    """Coerce timestamps to a tz-aware UTC ``datetime64[ns]`` Series."""
    s = pd.to_datetime(pd.Series(values))
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    return s.dt.as_unit("ns")


def _ns(values: pd.Series) -> _I64:
    """int64 nanoseconds of a UTC datetime Series (NaT -> ``_NAT``)."""
    return to_ns(_to_utc(values))


def _from_ns(values: _I64) -> pd.Series:
    return pd.Series(
        np.asarray(values, dtype=np.int64).view("datetime64[ns]")
    ).dt.tz_localize("UTC")


def _id_codes(raw: pd.Series) -> tuple[_I64, pd.Index]:
    """Factorize an id column; empty / missing readings get code ``-1``.

    A float column whose values are all whole numbers (an integer id column
    that picked up ``NaN``) is read as integers, so ``1001`` never becomes
    ``"1001.0"``.
    """
    s = raw
    if pd.api.types.is_float_dtype(s.dtype):
        finite = s.dropna()
        if bool(np.all(np.mod(finite.to_numpy(dtype=float), 1) == 0)):
            s = s.astype("Int64")
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    codes = np.array(codes, dtype=np.int64)  # writable copy
    if len(uniques) and (
        pd.api.types.is_object_dtype(uniques.dtype)
        or pd.api.types.is_string_dtype(uniques.dtype)
    ):
        blank = np.array(
            [isinstance(v, str) and v.strip(" \t\r\n\x00") == "" for v in uniques],
            dtype=bool,
        )
        if blank.any():
            codes[blank[codes.clip(min=0)] & (codes >= 0)] = -1
    return codes, pd.Index(uniques)


def _rle(codes: _I64, first: _I64, last: _I64) -> tuple[_I64, _I64, _I64, _I64]:
    """Run-length encode ``codes`` (one entry per sample or per sub-segment).

    Returns ``(codes, start, end, first_index)`` per run. A run ends where the
    next run starts; the final run ends at its last observation.
    """
    n = len(codes)
    brk = np.ones(n, dtype=bool)
    brk[1:] = codes[1:] != codes[:-1]
    idx = np.flatnonzero(brk)
    starts = first[idx]
    last_idx = np.append(idx[1:] - 1, n - 1)
    ends = np.append(starts[1:], last[last_idx[-1:]])
    return codes[idx], starts, ends, idx


def _segments(
    sig: pd.DataFrame,
    *,
    value_column: str,
    time_column: str,
    min_duration: str | None,
) -> tuple[_I64, _I64, _I64, _I64, pd.Index]:
    """Presence runs of one id signal.

    Returns ``(codes, start_ns, end_ns, first_row, uniques)`` for the id runs
    only (empty / missing runs are boundaries, not objects). ``first_row`` is
    the positional row of each run's first sample in the time-sorted ``sig``.
    """
    times = _ns(sig[time_column])
    codes, uniques = _id_codes(sig[value_column].reset_index(drop=True))
    rows = np.arange(len(codes), dtype=np.int64)
    c, s, e, first = _rle(codes, times, times)
    first = rows[first]

    if min_duration is not None and len(c) > 1:
        # Debounce: an id run shorter than ``min_duration`` is a blip. It is
        # removed and its neighbours close over it (and merge when they carry
        # the same id). The final run is open-ended (its true duration is
        # unknown), and empty / missing runs are boundaries — neither is
        # dropped. Blips are removed shortest first and equal neighbours are
        # merged before the next check, so in A-X-A a short X is removed
        # and the merged A is judged on its full length.
        threshold = pd.Timedelta(min_duration).value
        while len(c) > 1:
            dur = e - s
            short = (c >= 0) & (dur < threshold)
            short[-1] = False
            if not short.any():
                break
            # Drop the short runs that are no longer than a short neighbour
            # (local minima); the shortest one always qualifies.
            big = np.iinfo(np.int64).max
            d = np.where(short, dur, big)
            left = np.append(big, d[:-1])
            right = np.append(d[1:], big)
            drop = short & (d < left) & (d <= right)
            keep = ~drop
            c, s, last, first = c[keep], s[keep], e[keep], first[keep]
            c, s, e, k = _rle(c, s, last)
            first = first[k]

    ids = c >= 0
    return c[ids], s[ids], e[ids], first[ids], uniques


def _id_text(value: object) -> str:
    """An object id as text; an integral float renders without ``.0``."""
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _render_oids(values: pd.Index, template: str, object_type: str) -> pd.Series:
    texts = [_id_text(v) for v in values]
    if template == "{value}":
        out = texts
    elif template == "{type}:{value}":
        out = [f"{object_type}:{v}" for v in texts]
    else:
        out = [template.format(value=v, type=object_type) for v in texts]
    return pd.Series(out, dtype="string")


@dataclass(frozen=True)
class ObjectSpec:
    """Declares that one signal carries the identity of one object type.

    Attributes:
        uuid: The signal (``uuid`` value) whose value column is the object id.
        object_type: OCEL object type, e.g. ``"batch"``, ``"coil"``, ``"serial"``,
            ``"recipe"``. Auto-registered via
            :func:`~ts_shape.eventlog.schema.register_object_type` if unknown.
        value_column: Column carrying the id (``"value_string"`` or
            ``"value_integer"``). Empty strings and missing values mean
            "no object present".
        min_duration: Optional minimum presence (e.g. ``"1min"``, measured
            start to next value); shorter blips are dropped and the presence
            around them closes over the gap. The last id on a signal is
            open-ended and never dropped.
        id_template: How to render the object id from the raw value. Supports
            ``{value}`` and ``{type}`` — default ``"{value}"``; use
            ``"{type}:{value}"`` to namespace ids across types.
        attributes: ``{attribute_name: signal_uuid}`` — the value in effect at
            each presence start (last non-missing sample at or before it) plus
            every change of that signal during the presence, emitted as
            ``object_changes``.
    """

    uuid: str
    object_type: str
    value_column: str = "value_string"
    min_duration: str | None = None
    id_template: str = "{value}"
    attributes: Mapping[str, str] = field(default_factory=dict)


def _check_frame(df: pd.DataFrame, uuid_column: str, time_column: str) -> None:
    require_columns(df, (uuid_column, time_column), where="object detection")


def _detect(
    df: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    uuid_column: str,
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interval table plus the in-presence attribute changes."""
    _check_frame(df, uuid_column, time_column)
    frames: list[pd.DataFrame] = []
    changes: list[pd.DataFrame] = []
    available = set(df[uuid_column].unique()) if not df.empty else set()
    times = df[time_column].dropna() if not df.empty else df[time_column]
    data_end_ns = int(np.max(_ns(times))) if len(times) else None
    for spec in specs:
        if not schema.is_known_object_type(spec.object_type):
            schema.register_object_type(spec.object_type)
        clash = set(spec.attributes) & set(_RESERVED)
        if clash:
            raise ValueError(
                f"ObjectSpec({spec.uuid!r}) attribute name(s) {sorted(clash)} "
                f"collide with reserved interval columns {list(_RESERVED)}"
            )
        if df.empty:
            continue
        if spec.value_column not in df.columns:
            raise ColumnNotFoundError(
                f"value column {spec.value_column!r} of ObjectSpec({spec.uuid!r}) "
                f"not found; available columns: {list(df.columns)}"
            )
        if spec.uuid not in available:
            warnings.warn(
                f"ObjectSpec signal {spec.uuid!r} is not in the frame; no "
                f"{spec.object_type!r} objects detected from it. Available "
                f"uuids: {sorted(map(str, available))[:20]}",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            continue

        sig = df[df[uuid_column] == spec.uuid]
        sig = sig[sig[time_column].notna()].sort_values(time_column, kind="stable")
        sig = sig.reset_index(drop=True)
        codes, start, end, first, uniques = _segments(
            sig,
            value_column=spec.value_column,
            time_column=time_column,
            min_duration=spec.min_duration,
        )
        if len(codes) == 0:
            continue
        # A signal that still carries an id at its last reading (change
        # encoded: the id is not re-sent) keeps that object present until the
        # end of the data, like the last segment of SegmentExtractor.
        last_value = sig[spec.value_column].iloc[-1]
        still_set = not (
            pd.isna(last_value)
            or (isinstance(last_value, str) and last_value.strip(" \t\r\n\x00") == "")
        )
        if still_set and data_end_ns is not None and end[-1] < data_end_ns:
            end = end.copy()
            end[-1] = data_end_ns

        oids = _render_oids(uniques, spec.id_template, spec.object_type)
        oid = oids.to_numpy()[codes]
        if _SOURCE in sig.columns:
            source = sig[_SOURCE].astype("string").to_numpy()[first]
        else:
            source = pd.array([pd.NA] * len(codes), dtype="string")
        out = pd.DataFrame(
            {
                _OID: pd.array(oid, dtype="string"),
                _TYPE: pd.array([spec.object_type] * len(codes), dtype="string"),
                _START: _from_ns(start),
                _END: _from_ns(end),
                _DURATION: (end - start) / 1e9,
                _UUID: pd.array([spec.uuid] * len(codes), dtype="string"),
                _SOURCE: pd.array(source, dtype="string"),
            }
        )
        is_last = np.zeros(len(codes), dtype=bool)
        is_last[-1] = True
        for attr_name, attr_uuid in spec.attributes.items():
            at_start, during = _capture_attribute(
                df,
                attr_uuid,
                start,
                end,
                is_last,
                uuid_column=uuid_column,
                time_column=time_column,
                value_column=spec.value_column,
            )
            out[attr_name] = at_start
            if len(during):
                pos = during["_pos"].to_numpy()
                changes.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_OID: out[_OID].to_numpy()[pos],
                            schema.OCEL_TYPE: spec.object_type,
                            schema.OCEL_FIELD: attr_name,
                            schema.OCEL_VALUE: during["_value"].to_numpy(),
                            schema.OCEL_TIMESTAMP: _from_ns(
                                during["_ts"].to_numpy()
                            ).to_numpy(),
                        }
                    )
                )
        frames.append(out)

    if not frames:
        intervals = _empty_intervals()
    else:
        intervals = pd.concat(frames, ignore_index=True)
    extra = (
        pd.concat(changes, ignore_index=True)
        if changes
        else schema.empty_object_changes()
    )
    return intervals, extra


def object_intervals(
    df: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    uuid_column: str = "uuid",
    time_column: str = "systime",
) -> pd.DataFrame:
    """Extract one presence interval per object presence from id signals.

    Returns a tidy frame with columns ``oid``, ``type``, ``start``, ``end``,
    ``duration_seconds``, ``uuid`` (the id signal), ``source_uuid`` (of the id
    signal, when the frame has that column) plus one column per captured
    attribute (its value at the presence start). Times are UTC.

    Each id signal is run-length encoded: an id is present from its first
    sample until the next value on the same signal (next id, or an empty /
    missing reading). A signal whose last reading is still an id keeps that
    object present until the last timestamp of the frame. An id that
    re-appears later yields another row with the same ``oid``. An empty frame
    or no matching signal gives an empty frame with the full column set.
    """
    intervals, _ = _detect(df, specs, uuid_column=uuid_column, time_column=time_column)
    return intervals


def _capture_attribute(
    df: pd.DataFrame,
    attr_uuid: str,
    starts: _I64,
    ends: _I64,
    is_last: _BOOL,
    *,
    uuid_column: str,
    time_column: str,
    value_column: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Attribute value at each presence start, and its changes during presence.

    The value at a start is the last non-missing sample at or before it
    (backward as-of). A change is a sample whose value differs from the
    previous non-missing one and that falls strictly after a presence start
    and before its end (at or before it, for the last presence).
    """
    empty_during = pd.DataFrame({"_pos": [], "_value": [], "_ts": []})
    sig = df[df[uuid_column] == attr_uuid]
    if sig.empty:
        return pd.Series([pd.NA] * len(starts), dtype="object"), empty_during
    # Pick the first populated value column for the attribute signal.
    col = next(
        (
            c
            for c in (value_column, "value_string", "value_double", "value_integer")
            if c in sig.columns and sig[c].notna().any()
        ),
        value_column,
    )
    if col not in sig.columns:
        return pd.Series([pd.NA] * len(starts), dtype="object"), empty_during
    sig = sig[sig[time_column].notna() & sig[col].notna()]
    sig = sig.sort_values(time_column, kind="stable")
    t = _ns(sig[time_column])
    vals = sig[col].to_numpy(dtype=object)
    if len(t) == 0:
        return pd.Series([pd.NA] * len(starts), dtype="object"), empty_during

    pos = np.searchsorted(t, starts, side="right") - 1
    at_start = np.where(pos >= 0, vals[pos.clip(min=0)], pd.NA)

    changed = np.ones(len(vals), dtype=bool)
    changed[1:] = vals[1:] != vals[:-1]
    ct, cv = t[changed], vals[changed]
    owner = np.searchsorted(starts, ct, side="right") - 1
    ok = owner >= 0
    o = owner.clip(min=0)
    inside = ok & (ct > starts[o]) & ((ct < ends[o]) | (is_last[o] & (ct == ends[o])))
    during = pd.DataFrame({"_pos": o[inside], "_value": cv[inside], "_ts": ct[inside]})
    return pd.Series(at_start, dtype="object"), during


def detect_objects(
    df: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    uuid_column: str = "uuid",
    time_column: str = "systime",
    hierarchy: Mapping[str, str] | None = None,
    infer_o2o: bool = True,
    validate: bool = True,
) -> EventLog:
    """Detect object instances and build their OCEL 2.0 tables.

    Produces an :class:`EventLog` with **no events** — only ``objects``, ``o2o``
    (object-to-object relations), and ``object_changes`` (presence lifecycle +
    captured attributes). Compose it with event-detector logs via
    :func:`~ts_shape.eventlog.concat.concat`.

    ``hierarchy`` maps a child object type to its parent type
    (e.g. ``{"serial": "batch", "batch": "work_order"}``). A compositional
    ``part_of`` relation is asserted **only** along these declared edges: each
    child presence is part of the parent presence it overlaps longest —
    because *part_of cannot be inferred from time alone* (a long-running tool
    temporally contains every batch without owning it). Without a declared
    hierarchy, overlapping objects of different types get the honest,
    symmetric ``co_occurs`` qualifier instead. Set ``infer_o2o=False`` to skip
    object-to-object relations entirely.

    Args:
        df: Long-format frame with the id signals.
        specs: Which signal carries which object type.
        uuid_column: Signal identifier column.
        time_column: Timestamp column.
        hierarchy: Child type -> parent type for ``part_of`` relations.
        infer_o2o: Derive object-to-object relations.
        validate: Check the result against the canonical schema.

    Returns:
        An :class:`EventLog` with ``objects``, ``o2o`` and ``object_changes``
        and no events.

    Raises:
        ColumnNotFoundError: When ``df`` lacks the uuid, time or a spec's
            value column.
    """
    intervals, extra = _detect(
        df, specs, uuid_column=uuid_column, time_column=time_column
    )
    log = EventLog(
        objects=_objects_table(intervals),
        o2o=_infer_o2o(intervals, hierarchy) if infer_o2o else schema.empty_o2o(),
        object_changes=_object_changes(intervals, extra),
    )
    if validate:
        schema.validate(log)
    return log


def attach_objects(
    event_log: EventLog,
    df: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    qualifiers: Mapping[str, str] | None = None,
    uuid_column: str = "uuid",
    time_column: str = "systime",
    hierarchy: Mapping[str, str] | None = None,
    infer_o2o: bool = True,
    validate: bool = True,
) -> EventLog:
    """Link an existing event log to detected objects by temporal containment.

    For every event, any object whose presence interval contains the event
    timestamp is linked with an event-to-object (E2O) relation. The detected
    objects, ``o2o`` and ``object_changes`` are merged in. This enriches the
    output of *any* event detector — no per-detector changes required. See
    :func:`detect_objects` for ``hierarchy`` (declared ``part_of`` edges).

    This is a convenience over :func:`relate`: it first segments ``df`` into an
    object-interval table via :func:`object_intervals`, then relates the events
    to it. When you already hold an interval/object table, call :func:`relate`
    directly and skip the raw signals.

    Args:
        event_log: The events to link.
        df: Long-format frame with the id signals.
        specs: Which signal carries which object type.
        qualifiers: Relation qualifier per object type.
        uuid_column: Signal identifier column.
        time_column: Timestamp column.
        hierarchy: Child type -> parent type for ``part_of`` relations.
        infer_o2o: Derive object-to-object relations.
        validate: Check the result against the canonical schema.

    Returns:
        ``event_log`` merged with the detected objects and the new relations.
    """
    intervals, extra = _detect(
        df, specs, uuid_column=uuid_column, time_column=time_column
    )
    return _relate(
        event_log,
        intervals,
        extra,
        hierarchy=hierarchy,
        qualifiers=qualifiers,
        key_columns=None,
        infer_o2o=infer_o2o,
        validate=validate,
    )


def relate(
    event_log: EventLog,
    intervals: pd.DataFrame,
    *,
    hierarchy: Mapping[str, str] | None = None,
    qualifiers: Mapping[str, str] | None = None,
    key_columns: Sequence[str] | None = None,
    infer_o2o: bool = True,
    validate: bool = True,
) -> EventLog:
    """Assemble a full OCEL 2.0 log from an event list and an object list.

    This is the automated relationship step: given an existing ``event_log``
    (with events) and an ``intervals`` table describing objects -- columns
    ``oid``, ``type`` and, for temporal linking, ``start`` / ``end`` (plus any
    attribute columns) -- it derives **all** relations and returns a validated
    :class:`EventLog` whose five tables are ready to ingest into an OCEL 2.0
    database:

    * ``objects`` from the distinct ``(oid, type)`` pairs,
    * ``relations`` (E2O) via :func:`link_events_to_objects`
      (temporal containment, plus optional ``key_columns`` matching),
    * ``o2o`` via :func:`derive_o2o` (declared ``part_of`` ``hierarchy`` or
      symmetric ``co_occurs``),
    * ``object_changes`` (lifecycle + captured attributes).

    A missing ``start`` / ``end`` (``NaT``) means the presence is unbounded on
    that side (e.g. an object that is still active). Every column other than
    ``oid / type / start / end / duration_seconds / uuid / source_uuid`` is an
    attribute, recorded at the presence start.

    Unlike :func:`attach_objects`, no raw signals or :class:`ObjectSpec` are
    needed -- you bring the two lists, ts-shape derives the relationships.

    Args:
        event_log: The events to link.
        intervals: Object table with ``oid``, ``type`` and optionally
            ``start`` / ``end`` and attribute columns.
        hierarchy: Child type -> parent type for ``part_of`` relations.
        qualifiers: Relation qualifier per object type.
        key_columns: Event columns that name an object id directly.
        infer_o2o: Derive object-to-object relations.
        validate: Check the result against the canonical schema.

    Returns:
        The validated five-table :class:`EventLog`.

    Raises:
        ColumnNotFoundError: When ``intervals`` lacks ``oid`` or ``type``.
    """
    return _relate(
        event_log,
        intervals,
        None,
        hierarchy=hierarchy,
        qualifiers=qualifiers,
        key_columns=key_columns,
        infer_o2o=infer_o2o,
        validate=validate,
    )


def _relate(
    event_log: EventLog,
    intervals: pd.DataFrame,
    extra_changes: pd.DataFrame | None,
    *,
    hierarchy: Mapping[str, str] | None,
    qualifiers: Mapping[str, str] | None,
    key_columns: Sequence[str] | None,
    infer_o2o: bool,
    validate: bool,
) -> EventLog:
    iv = _normalize_intervals(intervals)
    detected = EventLog(
        objects=_objects_table(iv),
        o2o=_infer_o2o(iv, hierarchy) if infer_o2o else schema.empty_o2o(),
        object_changes=_object_changes(iv, extra_changes),
    )
    merged = concat(event_log, detected)

    rels = link_events_to_objects(
        event_log.events, iv, qualifiers=qualifiers, key_columns=key_columns
    )
    if not rels.empty:
        merged.relations = (
            pd.concat([merged.relations, rels], ignore_index=True)
            .drop_duplicates(
                subset=[schema.OCEL_EID, schema.OCEL_OID, schema.OCEL_TYPE]
            )
            .reset_index(drop=True)
        )
    if validate:
        schema.validate(merged)
    return merged


def derive_o2o(
    intervals: pd.DataFrame, *, hierarchy: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Derive object-to-object (O2O) relations from an interval table.

    Public entry point over the overlap kernel: a declared ``hierarchy``
    (child type -> parent type) yields ``part_of`` from each child presence to
    the parent presence it overlaps longest; without one, overlapping objects
    of different types get the symmetric ``co_occurs`` qualifier (one row per
    pair, the type that sorts first in ``ocel:oid``).

    Args:
        intervals: Object table with ``oid``, ``type``, ``start``, ``end``.
        hierarchy: Child type -> parent type.

    Returns:
        The OCEL 2.0 ``o2o`` table, de-duplicated.
    """
    return _infer_o2o(_normalize_intervals(intervals), hierarchy)


def object_specs_from_metadata(
    metadata: pd.DataFrame,
    *,
    type_field: str = "object_type",
    value_column_field: str = "object_value_column",
    uuid_column: str = "uuid",
) -> list[ObjectSpec]:
    """Build :class:`ObjectSpec` list from per-signal metadata tags.

    Scans a metadata table (e.g. ``MetadataJsonLoader.to_df()``) for a column
    naming the object type per signal — so new object types are onboarded by
    tagging metadata, not by writing code. The uuid may be the index or a
    column; the type/value-column fields are matched directly or with a
    ``config_`` / ``config.`` prefix (mkdocs metadata flattening).

    Args:
        metadata: Metadata table, one row per signal.
        type_field: Field naming the object type of a signal; rows without
            one are skipped.
        value_column_field: Field naming the signal's id column (default
            ``value_string``); must be one of ``value_string``,
            ``value_integer`` or ``value_double``.
        uuid_column: Column (or index name) holding the signal uuid.

    Returns:
        One :class:`ObjectSpec` per tagged signal; empty when the table has
        no uuid or type field.

    Raises:
        ValueError: When a value-column entry is not an id column.
    """
    md = metadata.reset_index() if metadata.index.name == uuid_column else metadata

    def _col(name: str) -> str | None:
        for cand in (name, f"config_{name}", f"config.{name}"):
            if cand in md.columns:
                return cand
        return None

    uuid_col = uuid_column if uuid_column in md.columns else None
    type_col = _col(type_field)
    if uuid_col is None or type_col is None:
        return []
    valcol_col = _col(value_column_field)

    specs: list[ObjectSpec] = []
    for _, row in md.iterrows():
        otype = row.get(type_col)
        if otype is None or pd.isna(otype) or str(otype).strip() == "":
            continue
        value_column = "value_string"
        if valcol_col is not None and not pd.isna(row.get(valcol_col)):
            value_column = str(row[valcol_col])
        if value_column not in _ID_VALUE_COLUMNS:
            raise ValueError(
                f"signal {row[uuid_col]!r}: {valcol_col} is {value_column!r}; "
                f"an id signal's value column must be one of {_ID_VALUE_COLUMNS}"
            )
        specs.append(
            ObjectSpec(
                uuid=str(row[uuid_col]),
                object_type=str(otype),
                value_column=value_column,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _objects_table(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return schema.empty_objects()
    objs = (
        intervals[[_OID, _TYPE]]
        .drop_duplicates()
        .rename(columns={_OID: schema.OCEL_OID, _TYPE: schema.OCEL_TYPE})
        .reset_index(drop=True)
    )
    objs[schema.OCEL_OID] = objs[schema.OCEL_OID].astype("string")
    objs[schema.OCEL_TYPE] = objs[schema.OCEL_TYPE].astype("string")
    shared = objs[schema.OCEL_OID][objs[schema.OCEL_OID].duplicated()]
    if not shared.empty:
        warnings.warn(
            f"object id(s) {sorted(set(shared))[:5]} occur with more than one "
            "object type; OCEL 2.0 object ids must be unique across types — use "
            "id_template='{type}:{value}' to namespace them",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    return objs


def _object_changes(
    intervals: pd.DataFrame, extra: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Lifecycle + attribute ``object_changes`` derived from the interval table.

    ``lifecycle`` is ``active`` when an object becomes present and ``released``
    when it stops; overlapping or touching presences of the same object (e.g.
    the same serial handed from one station to the next) form one continuous
    presence. Attribute columns are inferred from the frame itself -- any
    column beyond the reserved interval columns -- and recorded at each
    presence start. ``extra`` adds in-presence attribute changes. Sorted by
    timestamp; one value per ``(oid, field, timestamp)``.
    """
    empty = schema.empty_object_changes()
    if intervals.empty or _START not in intervals.columns:
        return empty
    oid = intervals[_OID].astype("string").to_numpy()
    typ = intervals[_TYPE].astype("string").to_numpy()
    start = _ns(intervals[_START])
    end = (
        _ns(intervals[_END])
        if _END in intervals.columns
        else np.full(len(intervals), _NAT, dtype=np.int64)
    )
    parts: list[pd.DataFrame] = []

    # Lifecycle over the union of each object's presences.
    key, _ = pd.factorize(pd.MultiIndex.from_arrays([oid, typ]))
    s_open = np.where(start == _NAT, np.iinfo(np.int64).min + 1, start)
    e_open = np.where(end == _NAT, np.iinfo(np.int64).max, end)
    order = np.lexsort((s_open, key))
    k, s, e = key[order], s_open[order], e_open[order]
    runmax = pd.Series(e).groupby(k).cummax().to_numpy()
    prev_max = np.empty_like(runmax)
    prev_max[0] = runmax[0]
    prev_max[1:] = runmax[:-1]
    new = np.ones(len(k), dtype=bool)
    new[1:] = (k[1:] != k[:-1]) | (s[1:] > prev_max[1:])
    run = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    run_end = pd.Series(e).groupby(run).max().to_numpy()
    r_oid, r_typ = oid[order][first], typ[order][first]
    r_start = start[order][first]
    for label, ts, sel_oid, sel_typ in (
        ("active", r_start, r_oid, r_typ),
        ("released", run_end, r_oid, r_typ),
    ):
        valid = (ts != _NAT) & (ts != np.iinfo(np.int64).max)
        parts.append(
            pd.DataFrame(
                {
                    schema.OCEL_OID: sel_oid[valid],
                    schema.OCEL_TYPE: sel_typ[valid],
                    schema.OCEL_FIELD: "lifecycle",
                    schema.OCEL_VALUE: label,
                    schema.OCEL_TIMESTAMP: _from_ns(ts[valid]).to_numpy(),
                }
            )
        )

    attr_cols = [c for c in intervals.columns if c not in _RESERVED]
    for a in attr_cols:
        vals = intervals[a].to_numpy(dtype=object)
        ok = ~pd.isna(intervals[a]).to_numpy(dtype=bool) & (start != _NAT)
        if not ok.any():
            continue
        parts.append(
            pd.DataFrame(
                {
                    schema.OCEL_OID: oid[ok],
                    schema.OCEL_TYPE: typ[ok],
                    schema.OCEL_FIELD: a,
                    schema.OCEL_VALUE: vals[ok],
                    schema.OCEL_TIMESTAMP: _from_ns(start[ok]).to_numpy(),
                }
            )
        )
    if extra is not None and not extra.empty:
        parts.append(extra)

    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty
    out = pd.concat(parts, ignore_index=True)
    out[schema.OCEL_TIMESTAMP] = _to_utc(out[schema.OCEL_TIMESTAMP])
    out = (
        out.drop_duplicates(
            subset=[
                schema.OCEL_OID,
                schema.OCEL_TYPE,
                schema.OCEL_FIELD,
                schema.OCEL_TIMESTAMP,
            ]
        )
        .sort_values(schema.OCEL_TIMESTAMP, kind="stable")
        .reset_index(drop=True)
    )
    return out.astype(empty.dtypes.to_dict())


# ---------------------------------------------------------------------------
# Interval kernels (vectorized; int64 ns, UTC)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spans:
    """Finite int64 bounds plus whether the end instant belongs to the span."""

    start: _I64
    end: _I64
    closed: _BOOL


def _spans(iv: pd.DataFrame, lo: int, hi: int) -> _Spans:
    """Bounds of ``iv`` with open sides clamped to ``lo`` / ``hi``.

    A presence is half-open ``[start, end)`` when another presence of the same
    signal (``uuid`` column, when present) — else of the same type — starts at
    its end (a hand-over); otherwise it is closed ``[start, end]``.
    """
    s = _ns(iv[_START])
    e = _ns(iv[_END])
    open_end = e == _NAT
    s = np.where(s == _NAT, lo, s)
    e = np.where(open_end, hi, e)
    if _UUID in iv.columns and iv[_UUID].notna().all():
        scope = iv[_UUID].astype("string").to_numpy()
    else:
        scope = iv[_TYPE].astype("string").to_numpy()
    starts = pd.MultiIndex.from_arrays([scope, s])
    counts = pd.Series(1, index=starts).groupby(level=[0, 1]).sum()
    ends = pd.MultiIndex.from_arrays([scope, e])
    hits = np.array(counts.reindex(ends).fillna(0).to_numpy(), dtype=np.int64)
    hits -= (s == e).astype(np.int64)  # a point presence does not hand over to itself
    closed = (hits <= 0) | open_end
    return _Spans(s, e, closed)


def _bounds(*arrays: _I64) -> tuple[int, int]:
    finite = [a[a != _NAT] for a in arrays if len(a)]
    finite = [a for a in finite if len(a)]
    if not finite:
        return 0, 1
    lo = min(int(a.min()) for a in finite)
    hi = max(int(a.max()) for a in finite)
    return lo - 1, hi + 1


def _candidate_pairs(a: _Spans, b: _Spans) -> tuple[_I64, _I64]:
    """All ``(i, j)`` with ``b.start[j] <= a.end[i]`` and ``b.end[j] >= a.start[i]``.

    Spans of ``b`` are bucketed by length (powers of two) so a search window
    ``[a.start - max_len, a.end]`` on sorted starts stays tight: the work is
    proportional to the number of real overlaps, not ``len(a) * len(b)``.
    """
    out_a: list[_I64] = []
    out_b: list[_I64] = []
    if len(a.start) == 0 or len(b.start) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    length = (b.end - b.start).astype(np.float64)
    bucket = np.floor(np.log2(np.maximum(length, 0) + 1)).astype(np.int64)
    for k in np.unique(bucket):
        sel = np.flatnonzero(bucket == k)
        order = sel[np.argsort(b.start[sel], kind="stable")]
        bs = b.start[order]
        maxlen = int(np.max(b.end[order] - bs))
        lo = np.searchsorted(bs, a.start - maxlen, side="left")
        hi = np.searchsorted(bs, a.end, side="right")
        cnt = (hi - lo).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            continue
        ia = np.repeat(np.arange(len(a.start), dtype=np.int64), cnt)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        jb = order[np.repeat(lo, cnt) + offs]
        keep = b.end[jb] >= a.start[ia]
        out_a.append(ia[keep])
        out_b.append(jb[keep].astype(np.int64))
    if not out_a:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(out_a), np.concatenate(out_b)


def _intersections(a: _Spans, b: _Spans) -> tuple[_I64, _I64, _I64]:
    """Pairs whose presences overlap, with the overlap length.

    Two presences overlap when they share a stretch of time; merely touching
    end-to-start is not an overlap. A point presence (``start == end``)
    overlaps a presence that contains its instant.
    """
    ia, jb = _candidate_pairs(a, b)
    m = np.maximum(a.start[ia], b.start[jb])
    in_a = (m < a.end[ia]) | ((m == a.end[ia]) & a.closed[ia])
    in_b = (m < b.end[jb]) | ((m == b.end[jb]) & b.closed[jb])
    shared = np.minimum(a.end[ia], b.end[jb]) > m
    point = (a.start[ia] == a.end[ia]) | (b.start[jb] == b.end[jb])
    hit = in_a & in_b & (shared | point)
    ia, jb, m = ia[hit], jb[hit], m[hit]
    overlap = np.minimum(a.end[ia], b.end[jb]) - m
    return ia, jb, overlap


def _infer_o2o(
    intervals: pd.DataFrame, hierarchy: Mapping[str, str] | None
) -> pd.DataFrame:
    """Object-to-object relations from interval overlap between distinct types.

    With a declared ``hierarchy`` (child type -> parent type), each child
    presence gets ``part_of`` (child -> parent) to the parent presence it
    overlaps longest (ties: the earlier parent). Otherwise overlapping objects
    of different types get the honest symmetric ``co_occurs`` qualifier —
    ``part_of`` is never guessed from time alone.
    """
    if (
        intervals.empty
        or _START not in intervals.columns
        or _END not in intervals.columns
    ):
        return schema.empty_o2o()
    iv = intervals.reset_index(drop=True)
    lo, hi = _bounds(_ns(iv[_START]), _ns(iv[_END]))
    spans = _spans(iv, lo, hi)
    types = iv[_TYPE].astype("string").to_numpy()
    oids = iv[_OID].astype("string").to_numpy()

    def sub(mask: _BOOL) -> tuple[_I64, _Spans]:
        idx = np.flatnonzero(mask)
        return idx, _Spans(spans.start[idx], spans.end[idx], spans.closed[idx])

    frames: list[pd.DataFrame] = []
    if hierarchy:
        for child_type, parent_type in hierarchy.items():
            ci, cs = sub(types == child_type)
            pi, ps = sub(types == parent_type)
            a, b, overlap = _intersections(cs, ps)
            if len(a) == 0:
                continue
            pick = pd.DataFrame({"c": a, "ov": overlap, "ps": ps.start[b], "p": b})
            pick = pick.sort_values(
                ["c", "ov", "ps", "p"], ascending=[True, False, True, True]
            ).drop_duplicates("c")
            frames.append(
                pd.DataFrame(
                    {
                        schema.OCEL_OID: oids[ci[pick["c"].to_numpy()]],
                        schema.OCEL_OID2: oids[pi[pick["p"].to_numpy()]],
                        schema.OCEL_QUALIFIER: "part_of",
                    }
                )
            )
    else:
        uniq = sorted(set(types))
        for i, ta in enumerate(uniq):
            ai, as_ = sub(types == ta)
            for tb in uniq[i + 1 :]:
                bi, bs = sub(types == tb)
                a, b, _ = _intersections(as_, bs)
                if len(a) == 0:
                    continue
                frames.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_OID: oids[ai[a]],
                            schema.OCEL_OID2: oids[bi[b]],
                            schema.OCEL_QUALIFIER: "co_occurs",
                        }
                    )
                )
    if not frames:
        return schema.empty_o2o()
    out = pd.concat(frames, ignore_index=True)
    out = out[out[schema.OCEL_OID] != out[schema.OCEL_OID2]]
    if out.empty:
        return schema.empty_o2o()
    out = out.drop_duplicates().reset_index(drop=True)
    return out.astype(schema.empty_o2o().dtypes.to_dict())


def _normalize_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an object/interval table.

    Requires ``oid`` and ``type``; ``start`` / ``end`` are optional (needed only
    for temporal linking and ``o2o``). Ids/types are coerced to ``string`` and
    any present time bounds to tz-aware UTC, matching the OCEL columns. Rows
    without an ``oid`` are dropped; ``end`` before ``start`` raises.
    """
    missing = {_OID, _TYPE} - set(intervals.columns)
    if missing:
        raise ValueError(
            f"object/interval table is missing required column(s) {sorted(missing)}; "
            f"got {list(intervals.columns)}"
        )
    out = intervals.copy()
    out[_OID] = out[_OID].astype("string")
    out[_TYPE] = out[_TYPE].astype("string")
    out = out[out[_OID].notna() & out[_TYPE].notna()].reset_index(drop=True)
    if _START in out.columns:
        out[_START] = _to_utc(out[_START])
    if _END in out.columns:
        out[_END] = _to_utc(out[_END])
    if _START in out.columns and _END in out.columns:
        bad = out[out[_END] < out[_START]]
        if not bad.empty:
            row = bad.iloc[0]
            raise ValueError(
                f"object {row[_OID]!r} ends before it starts "
                f"({row[_START]} > {row[_END]})"
            )
    return out


def link_events_to_objects(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    qualifiers: Mapping[str, str] | None = None,
    key_columns: Sequence[str] | None = None,
    contain: bool = True,
) -> pd.DataFrame:
    """Derive all event-to-object (E2O) relations from two lists.

    Takes an OCEL ``events`` table and an object/interval table (``oid``,
    ``type`` and, for temporal linking, ``start`` / ``end``) and returns the
    OCEL 2.0 ``relations`` table -- with no access to the raw signals. Two
    complementary strategies are unioned and de-duplicated on
    ``(eid, oid, type)``:

    * **temporal containment** (``contain``, default on): link an event to every
      object whose presence contains the event timestamp. Presence is
      half-open ``[start, end)`` when another presence of the same signal
      (``uuid`` column) or, without one, of the same type starts at ``end`` —
      so an event at a hand-over instant belongs to the incoming object only —
      and closed ``[start, end]`` otherwise (the last presence). ``NaT``
      bounds are open. Requires ``start`` / ``end`` columns.
    * **key match** (opt-in via ``key_columns``): link an event to the object
      whose ``oid`` equals the value in one of the event's ``key_columns`` --
      for events that already name their object (e.g. a ``batch`` attribute).

    Args:
        events: OCEL events table with ``ocel:eid`` (and ``ocel:timestamp``
            for temporal linking).
        intervals: Object table with ``oid``, ``type`` and, for temporal
            linking, ``start`` / ``end``.
        qualifiers: Maps an object ``type`` to the relation qualifier to stamp
            (e.g. ``{"batch": "processed_in"}``); unmapped types get ``<NA>``.
        key_columns: Event columns whose value is an object id. Integral
            floats (an integer id column with gaps) match ``"1"``, not
            ``"1.0"``.
        contain: Link by temporal containment.

    Returns:
        The OCEL 2.0 ``relations`` table.

    Raises:
        ColumnNotFoundError: When ``events`` lacks ``ocel:eid`` (or
            ``ocel:timestamp`` with ``contain``) or ``intervals`` lacks
            ``oid`` / ``type``.
    """
    require_columns(events, (schema.OCEL_EID,), where="link_events_to_objects")
    if contain:
        require_columns(
            events, (schema.OCEL_TIMESTAMP,), where="link_events_to_objects"
        )
    if events.empty:
        return schema.empty_relations()
    iv = _normalize_intervals(intervals)
    if iv.empty:
        return schema.empty_relations()
    qual = dict(qualifiers or {})
    frames: list[pd.DataFrame] = []
    eids = events[schema.OCEL_EID].to_numpy()

    if contain and _START in iv.columns and _END in iv.columns:
        ts = _ns(events[schema.OCEL_TIMESTAMP])
        valid = np.flatnonzero(ts != _NAT)
        order = valid[np.argsort(ts[valid], kind="stable")]
        ts_sorted = ts[order]
        lo, hi = _bounds(ts, _ns(iv[_START]), _ns(iv[_END]))
        sp = _spans(iv, lo, hi)
        first = np.searchsorted(ts_sorted, sp.start, side="left")
        last = np.where(
            sp.closed,
            np.searchsorted(ts_sorted, sp.end, side="right"),
            np.searchsorted(ts_sorted, sp.end, side="left"),
        )
        cnt = np.maximum(last - first, 0).astype(np.int64)
        total = int(cnt.sum())
        if total:
            obj = np.repeat(np.arange(len(iv), dtype=np.int64), cnt)
            offs = np.arange(total, dtype=np.int64) - np.repeat(
                np.cumsum(cnt) - cnt, cnt
            )
            ev = order[np.repeat(first, cnt) + offs]
            frames.append(
                pd.DataFrame(
                    {
                        schema.OCEL_EID: eids[ev],
                        schema.OCEL_OID: iv[_OID].to_numpy()[obj],
                        schema.OCEL_TYPE: iv[_TYPE].to_numpy()[obj],
                    }
                )
            )

    if key_columns:
        oid_to_type = iv[[_OID, _TYPE]].drop_duplicates(_OID).set_index(_OID)[_TYPE]
        for col in [c for c in key_columns if c in events.columns]:
            vals = events[col]
            ok = vals.notna().to_numpy(dtype=bool)
            keys = pd.Series([_id_text(v) for v in vals[ok]], dtype="string")
            types = keys.map(oid_to_type)
            hit = types.notna().to_numpy(dtype=bool)
            if hit.any():
                frames.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_EID: eids[ok][hit],
                            schema.OCEL_OID: keys.to_numpy()[hit],
                            schema.OCEL_TYPE: types.to_numpy()[hit],
                        }
                    )
                )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return schema.empty_relations()
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=[schema.OCEL_EID, schema.OCEL_OID, schema.OCEL_TYPE])
        .reset_index(drop=True)
    )
    out[schema.OCEL_QUALIFIER] = out[schema.OCEL_TYPE].astype(object).map(qual)
    return out.astype(schema.empty_relations().dtypes.to_dict())
