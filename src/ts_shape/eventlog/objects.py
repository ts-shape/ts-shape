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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from . import schema
from .concat import concat
from .model import EventLog

# Tidy intermediate columns (used between the three steps).
_OID = "oid"
_TYPE = "type"
_START = "start"
_END = "end"


def _to_utc(values: object) -> pd.Series:
    """Coerce timestamps to a tz-aware UTC Series (OCEL columns are UTC)."""
    s = pd.to_datetime(pd.Series(values))
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def _segments(
    df: pd.DataFrame,
    segment_uuid: str,
    *,
    uuid_column: str,
    value_column: str,
    time_column: str,
    min_duration: str | None,
) -> pd.DataFrame:
    """Run-length encode one signal into constant-value segments.

    Returns ``value / start / end`` (observed first/last sample per segment).
    Unlike a naive ``value.ne(value.shift())`` the first sample always opens a
    segment, so neither the leading object nor a single-sample object is lost.
    """
    sig = df[df[uuid_column] == segment_uuid]
    cols = ["value", "start", "end"]
    if sig.empty:
        return pd.DataFrame(columns=cols)

    sig = sig.sort_values(time_column)
    times = _to_utc(sig[time_column]).reset_index(drop=True)
    vals = sig[value_column].ffill().reset_index(drop=True)
    changed = vals.ne(vals.shift())
    changed.iloc[0] = True  # the first sample always starts a segment
    groups = changed.cumsum().to_numpy()

    work = pd.DataFrame(
        {"value": vals.to_numpy(), "start": times, "end": times, "_g": groups}
    )
    agg = work.groupby("_g", sort=True).agg(
        value=("value", "first"), start=("start", "min"), end=("end", "max")
    )

    value = agg["value"]
    blank = value.map(lambda v: isinstance(v, str) and v.strip() == "")
    agg = agg[value.notna() & ~blank].reset_index(drop=True)

    if min_duration is not None and not agg.empty:
        keep = (agg["end"] - agg["start"]) >= pd.Timedelta(min_duration)
        agg = agg[keep].reset_index(drop=True)
    return agg


@dataclass(frozen=True)
class ObjectSpec:
    """Declares that one signal carries the identity of one object type.

    Attributes:
        uuid: The signal (``uuid`` value) whose value column is the object id.
        object_type: OCEL object type, e.g. ``"batch"``, ``"coil"``, ``"serial"``,
            ``"recipe"``. Auto-registered via
            :func:`~ts_shape.eventlog.schema.register_object_type` if unknown.
        value_column: Column carrying the id (``"value_string"`` or
            ``"value_integer"``).
        min_duration: Optional minimum presence (e.g. ``"1min"``); shorter blips
            are dropped.
        id_template: How to render the object id from the raw value. Supports
            ``{value}`` and ``{type}`` — default ``"{value}"``; use
            ``"{type}:{value}"`` to namespace ids across types.
        attributes: ``{attribute_name: signal_uuid}`` — values captured at each
            presence start (backward ``merge_asof``) and emitted as
            ``object_changes``.
    """

    uuid: str
    object_type: str
    value_column: str = "value_string"
    min_duration: str | None = None
    id_template: str = "{value}"
    attributes: Mapping[str, str] = field(default_factory=dict)


def object_intervals(
    df: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    uuid_column: str = "uuid",
    time_column: str = "systime",
) -> pd.DataFrame:
    """Extract one presence interval per object instance from id signals.

    Returns a tidy frame with columns ``oid``, ``type``, ``start``, ``end`` plus
    one column per captured attribute. Each id signal is run-length encoded into
    contiguous constant-value segments.
    """
    frames: list[pd.DataFrame] = []
    for spec in specs:
        if not schema.is_known_object_type(spec.object_type):
            schema.register_object_type(spec.object_type)

        segs = _segments(
            df,
            spec.uuid,
            uuid_column=uuid_column,
            value_column=spec.value_column,
            time_column=time_column,
            min_duration=spec.min_duration,
        )
        if segs.empty:
            continue

        starts = segs["start"].reset_index(drop=True)
        observed_ends = segs["end"].reset_index(drop=True)
        # An id is present until the *next* id on the same signal appears, so a
        # gap between segments (sparse sampling) does not orphan events that land
        # in it. The final segment keeps its last observed sample as the end.
        next_starts = starts.shift(-1)
        ends = next_starts.where(next_starts.notna(), observed_ends)
        out = pd.DataFrame(
            {
                _OID: [
                    spec.id_template.format(value=v, type=spec.object_type)
                    for v in segs["value"]
                ],
                _TYPE: spec.object_type,
                _START: starts,
                _END: ends,
            }
        )
        for attr_name, attr_uuid in spec.attributes.items():
            out[attr_name] = _capture_attribute(
                df, attr_uuid, out[_START], uuid_column, time_column, spec.value_column
            )
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=[_OID, _TYPE, _START, _END])
    result = pd.concat(frames, ignore_index=True)
    result[_OID] = result[_OID].astype("string")
    result[_TYPE] = result[_TYPE].astype("string")
    return result


def _capture_attribute(
    df: pd.DataFrame,
    attr_uuid: str,
    starts: pd.Series,
    uuid_column: str,
    time_column: str,
    value_column: str,
) -> pd.Series:
    """Value of ``attr_uuid`` in effect at each presence start (backward asof)."""
    sig = df[df[uuid_column] == attr_uuid]
    if sig.empty:
        return pd.Series([pd.NA] * len(starts))
    # Pick the first populated value column for the attribute signal.
    col = next(
        (
            c
            for c in (value_column, "value_string", "value_double", "value_integer")
            if c in sig.columns and sig[c].notna().any()
        ),
        value_column,
    )
    sig = sig[[time_column, col]].copy()
    sig[time_column] = _to_utc(sig[time_column])
    sig = sig.sort_values(time_column)
    left = pd.DataFrame({time_column: _to_utc(starts.values)})
    order = left[time_column].argsort()
    merged = pd.merge_asof(
        left.iloc[order.values], sig, on=time_column, direction="backward"
    )
    # Restore original (presence) order.
    return merged[col].iloc[order.argsort().values].reset_index(drop=True)


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
    ``part_of`` relation is asserted **only** along these declared edges, when a
    child's presence overlaps a parent's — because *part_of cannot be inferred
    from time alone* (a long-running tool temporally contains every batch
    without owning it). Without a declared hierarchy, overlapping objects of
    different types get the honest, symmetric ``co_occurs`` qualifier instead.
    Set ``infer_o2o=False`` to skip object-to-object relations entirely.
    """
    intervals = object_intervals(
        df, specs, uuid_column=uuid_column, time_column=time_column
    )
    log = EventLog(
        objects=_objects_table(intervals),
        o2o=_infer_o2o(intervals, hierarchy) if infer_o2o else schema.empty_o2o(),
        object_changes=_object_changes(intervals, specs),
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
    """
    intervals = object_intervals(
        df, specs, uuid_column=uuid_column, time_column=time_column
    )
    detected = detect_objects(
        df,
        specs,
        uuid_column=uuid_column,
        time_column=time_column,
        hierarchy=hierarchy,
        infer_o2o=infer_o2o,
        validate=False,
    )
    merged = concat(event_log, detected)

    rels = _contained_relations(event_log.events, intervals, qualifiers or {})
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
        if otype is None or pd.isna(otype):
            continue
        value_column = "value_string"
        if valcol_col is not None and not pd.isna(row.get(valcol_col)):
            value_column = str(row[valcol_col])
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
    return objs


def _object_changes(
    intervals: pd.DataFrame, specs: Sequence[ObjectSpec]
) -> pd.DataFrame:
    if intervals.empty:
        return schema.empty_object_changes()
    attr_cols = [a for spec in specs for a in spec.attributes if a in intervals.columns]
    rows: list[dict] = []
    for _, r in intervals.iterrows():
        # Lifecycle presence as a time-varying attribute.
        rows.append(_change_row(r, "lifecycle", "active", r[_START]))
        rows.append(_change_row(r, "lifecycle", "released", r[_END]))
        for a in attr_cols:
            val = r.get(a)
            if val is not None and not pd.isna(val):
                rows.append(_change_row(r, a, val, r[_START]))
    out = pd.DataFrame(rows)
    return out.astype(schema.empty_object_changes().dtypes.to_dict())


def _change_row(r: pd.Series, field_name: str, value: object, ts: object) -> dict:
    return {
        schema.OCEL_OID: r[_OID],
        schema.OCEL_TYPE: r[_TYPE],
        schema.OCEL_FIELD: field_name,
        schema.OCEL_VALUE: value,
        schema.OCEL_TIMESTAMP: pd.Timestamp(ts),
    }


def _infer_o2o(
    intervals: pd.DataFrame, hierarchy: Mapping[str, str] | None
) -> pd.DataFrame:
    """Object-to-object relations from interval overlap between distinct types.

    With a declared ``hierarchy`` (child type -> parent type), a child whose
    presence overlaps a parent's gets ``part_of`` (child -> parent). Otherwise
    overlapping objects of different types get the honest symmetric
    ``co_occurs`` qualifier — ``part_of`` is never guessed from time alone.
    """
    if intervals.empty:
        return schema.empty_o2o()
    return (
        _hierarchy_o2o(intervals, hierarchy)
        if hierarchy
        else _cooccurrence_o2o(intervals)
    )


def _hierarchy_o2o(
    intervals: pd.DataFrame, hierarchy: Mapping[str, str]
) -> pd.DataFrame:
    rows: list[dict] = []
    for child_type, parent_type in hierarchy.items():
        children = _non_degenerate(intervals[intervals[_TYPE] == child_type])
        parents = _non_degenerate(intervals[intervals[_TYPE] == parent_type])
        if children.empty or parents.empty:
            continue
        pidx = pd.IntervalIndex.from_arrays(
            parents[_START], parents[_END], closed="left"
        )
        for _, c in children.iterrows():
            c_int = pd.Interval(c[_START], c[_END], closed="left")
            for pos in _overlapping_positions(pidx, c_int):
                parent = parents.iloc[pos]
                if c[_OID] != parent[_OID]:
                    rows.append(_o2o_row(c[_OID], parent[_OID], "part_of"))
    return _o2o_frame(rows)


def _cooccurrence_o2o(intervals: pd.DataFrame) -> pd.DataFrame:
    types = sorted(pd.unique(intervals[_TYPE]))
    by_type = {t: _non_degenerate(intervals[intervals[_TYPE] == t]) for t in types}
    rows: list[dict] = []
    for i, ta in enumerate(types):
        for tb in types[i + 1 :]:
            b = by_type[tb]
            if b.empty or by_type[ta].empty:
                continue
            bidx = pd.IntervalIndex.from_arrays(b[_START], b[_END], closed="left")
            for _, a in by_type[ta].iterrows():
                a_int = pd.Interval(a[_START], a[_END], closed="left")
                for pos in _overlapping_positions(bidx, a_int):
                    brow = b.iloc[pos]
                    if a[_OID] != brow[_OID]:
                        rows.append(_o2o_row(a[_OID], brow[_OID], "co_occurs"))
    return _o2o_frame(rows)


def _non_degenerate(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-length presence intervals (start == end) — they overlap nothing
    meaningfully and are invalid for a half-open IntervalIndex."""
    return frame[frame[_START] < frame[_END]]


def _o2o_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return schema.empty_o2o()
    out = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return out.astype(schema.empty_o2o().dtypes.to_dict())


def _overlapping_positions(index: pd.IntervalIndex, interval: pd.Interval) -> list[int]:
    mask = index.overlaps(interval)
    return [i for i, hit in enumerate(mask) if hit]


def _o2o_row(oid: object, oid2: object, qualifier: str) -> dict:
    return {
        schema.OCEL_OID: oid,
        schema.OCEL_OID2: oid2,
        schema.OCEL_QUALIFIER: qualifier,
    }


def _contained_relations(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    qualifiers: Mapping[str, str],
) -> pd.DataFrame:
    """E2O relations: each event linked to objects whose interval contains it."""
    if events.empty or intervals.empty:
        return schema.empty_relations()
    ts = pd.to_datetime(events[schema.OCEL_TIMESTAMP])
    rows: list[dict] = []
    for _, obj in intervals.iterrows():
        hit = (ts >= obj[_START]) & (ts <= obj[_END])
        if not hit.any():
            continue
        for eid in events.loc[hit, schema.OCEL_EID]:
            rows.append(
                {
                    schema.OCEL_EID: eid,
                    schema.OCEL_OID: obj[_OID],
                    schema.OCEL_TYPE: obj[_TYPE],
                    schema.OCEL_QUALIFIER: qualifiers.get(obj[_TYPE]),
                }
            )
    if not rows:
        return schema.empty_relations()
    out = pd.DataFrame(rows)
    return out.astype(schema.empty_relations().dtypes.to_dict())
