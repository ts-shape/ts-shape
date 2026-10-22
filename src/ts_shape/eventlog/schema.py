"""Column-name constants and schema definitions for the ts-shape event log.

The canonical event-log uses OCEL 2.0 column names (`ocel:*`) with a few
ts-shape-prefixed columns for fields that have no OCEL counterpart but are
useful for downstream consumers (interval starts, durations, severity, etc.).
XES column names (`concept:name`, `time:timestamp`, `case:concept:name`,
`lifecycle:transition`, `org:resource`) are produced only by the flat
exporter; they do not appear in the canonical schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .model import EventLog


# ---- Events table -----------------------------------------------------------

OCEL_EID = "ocel:eid"
OCEL_ACTIVITY = "ocel:activity"
OCEL_TIMESTAMP = "ocel:timestamp"

TS_START_TIMESTAMP = "ts_shape:start_timestamp"
TS_DURATION_S = "ts_shape:duration_s"
TS_DETECTOR = "ts_shape:detector"
TS_PACK = "ts_shape:pack"
TS_SEVERITY = "ts_shape:severity"
TS_VALUE = "ts_shape:value"

EVENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    OCEL_EID,
    OCEL_ACTIVITY,
    OCEL_TIMESTAMP,
    TS_DETECTOR,
    TS_PACK,
)

EVENT_OPTIONAL_COLUMNS: tuple[str, ...] = (
    TS_START_TIMESTAMP,
    TS_DURATION_S,
    TS_SEVERITY,
    TS_VALUE,
)


# ---- Objects table ----------------------------------------------------------

OCEL_OID = "ocel:oid"
OCEL_TYPE = "ocel:type"

OBJECT_REQUIRED_COLUMNS: tuple[str, ...] = (OCEL_OID, OCEL_TYPE)


# ---- Event-to-object relations table (E2O) ---------------------------------

OCEL_QUALIFIER = "ocel:qualifier"

RELATION_REQUIRED_COLUMNS: tuple[str, ...] = (OCEL_EID, OCEL_OID, OCEL_TYPE)


# ---- Object-to-object relations table (O2O) --------------------------------
#
# OCEL 2.0 models qualified relations *between* objects (e.g. a ``part``
# belongs to a ``batch``; a ``sensor`` is mounted on an ``asset``). The
# qualifier names the role of the target object relative to the source.

OCEL_OID2 = "ocel:oid_2"

O2O_REQUIRED_COLUMNS: tuple[str, ...] = (OCEL_OID, OCEL_OID2, OCEL_QUALIFIER)


# ---- Object dynamic-attribute changes table --------------------------------
#
# OCEL 2.0 lets an object's attributes vary over time. Each row records that
# ``OCEL_FIELD`` of object ``OCEL_OID`` took ``OCEL_VALUE`` from ``OCEL_TIMESTAMP``
# onward. A purely static attribute is a single row with the log's base time.

OCEL_FIELD = "ocel:field"
OCEL_VALUE = "ocel:value"

OBJECT_CHANGE_REQUIRED_COLUMNS: tuple[str, ...] = (
    OCEL_OID,
    OCEL_TYPE,
    OCEL_FIELD,
    OCEL_VALUE,
    OCEL_TIMESTAMP,
)


# ---- XES export columns -----------------------------------------------------

XES_CASE = "case:concept:name"
XES_ACTIVITY = "concept:name"
XES_TIMESTAMP = "time:timestamp"
XES_LIFECYCLE = "lifecycle:transition"
XES_RESOURCE = "org:resource"
XES_INSTANCE = "concept:instance"


# ---- Object-type registry ---------------------------------------------------

STANDARD_OBJECT_TYPES: tuple[str, ...] = (
    "asset",
    "cycle",
    "batch",
    "lot",
    "material",
    "serial",
    "article",
    "part",
    "work_order",
    "shift",
    "operator",
    "tool",
    "recipe",
    "station",
    "signal",
    "sensor",
)
_OBJECT_TYPES: set[str] = set(STANDARD_OBJECT_TYPES)


# ---- Standard attribute extension ------------------------------------------
#
# Every detector method's LabelRule may declare a ``standard_attrs`` mapping
# from these fixed keys to either a detector-frame column name (string) or a literal
# scalar value. The adapter renames / broadcasts and coerces to the type
# below. Adding a new standard key requires updating this tuple AND the
# corresponding entry in ``STANDARD_ATTR_TYPES``.
STANDARD_ATTR_KEYS: tuple[str, ...] = (
    "ts_shape:method",
    "ts_shape:baseline",
    "ts_shape:threshold_low",
    "ts_shape:threshold_high",
    "ts_shape:deviation",
    "ts_shape:deviation_pct",
    "ts_shape:direction",
    "ts_shape:confidence",
    "ts_shape:sample_count",
    "ts_shape:outcome",
    "ts_shape:lifecycle_state",
    "ts_shape:lifecycle_pair_id",
)

STANDARD_ATTR_TYPES: dict[str, str] = {
    "ts_shape:method": "string",
    "ts_shape:baseline": "float64",
    "ts_shape:threshold_low": "float64",
    "ts_shape:threshold_high": "float64",
    "ts_shape:deviation": "float64",
    "ts_shape:deviation_pct": "float64",
    "ts_shape:direction": "string",
    "ts_shape:confidence": "float64",
    "ts_shape:sample_count": "Int64",
    "ts_shape:outcome": "string",
    "ts_shape:lifecycle_state": "string",
    "ts_shape:lifecycle_pair_id": "string",
}


def register_object_type(name: str) -> None:
    """Register a custom OCEL object type so adapters can emit it."""
    if not name or ":" in name:
        raise ValueError(f"invalid object type {name!r}")
    _OBJECT_TYPES.add(name)


def is_known_object_type(name: str) -> bool:
    """Check whether an OCEL object type is built in or has been registered.

    Args:
        name: Object type name to look up.

    Returns:
        True if ``name`` is a known object type, otherwise False.
    """
    return name in _OBJECT_TYPES


# ---- Empty frames -----------------------------------------------------------


def empty_events() -> pd.DataFrame:
    """Return an empty events table with the canonical columns and dtypes.

    Returns:
        A zero-row DataFrame with the expected events schema.
    """
    return pd.DataFrame(
        {
            OCEL_EID: pd.Series(dtype="string"),
            OCEL_ACTIVITY: pd.Series(dtype="string"),
            OCEL_TIMESTAMP: pd.Series(dtype="datetime64[ns, UTC]"),
            TS_START_TIMESTAMP: pd.Series(dtype="datetime64[ns, UTC]"),
            TS_DURATION_S: pd.Series(dtype="float64"),
            TS_DETECTOR: pd.Series(dtype="string"),
            TS_PACK: pd.Series(dtype="string"),
            TS_SEVERITY: pd.Series(dtype="string"),
            TS_VALUE: pd.Series(dtype="float64"),
        }
    )


def empty_objects() -> pd.DataFrame:
    """Return an empty objects table with the canonical columns and dtypes.

    Returns:
        A zero-row DataFrame with the expected objects schema.
    """
    return pd.DataFrame(
        {
            OCEL_OID: pd.Series(dtype="string"),
            OCEL_TYPE: pd.Series(dtype="string"),
        }
    )


def empty_relations() -> pd.DataFrame:
    """Return an empty event-to-object relations table with the canonical columns and dtypes.

    Returns:
        A zero-row DataFrame with the expected event-to-object relations schema.
    """
    return pd.DataFrame(
        {
            OCEL_EID: pd.Series(dtype="string"),
            OCEL_OID: pd.Series(dtype="string"),
            OCEL_TYPE: pd.Series(dtype="string"),
            OCEL_QUALIFIER: pd.Series(dtype="string"),
        }
    )


def empty_o2o() -> pd.DataFrame:
    """Return an empty object-to-object relations table with the canonical columns and dtypes.

    Returns:
        A zero-row DataFrame with the expected object-to-object relations schema.
    """
    return pd.DataFrame(
        {
            OCEL_OID: pd.Series(dtype="string"),
            OCEL_OID2: pd.Series(dtype="string"),
            OCEL_QUALIFIER: pd.Series(dtype="string"),
        }
    )


def empty_object_changes() -> pd.DataFrame:
    """Return an empty object changes table with the canonical columns and dtypes.

    Returns:
        A zero-row DataFrame with the expected object changes schema.
    """
    return pd.DataFrame(
        {
            OCEL_OID: pd.Series(dtype="string"),
            OCEL_TYPE: pd.Series(dtype="string"),
            OCEL_FIELD: pd.Series(dtype="string"),
            OCEL_VALUE: pd.Series(dtype="object"),
            OCEL_TIMESTAMP: pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


# ---- Validation -------------------------------------------------------------


def _arrow_strings(values: pd.Series) -> Any | None:
    """The Arrow array behind an Arrow-backed string Series, else None."""
    dtype = values.dtype
    if isinstance(dtype, pd.StringDtype) and dtype.storage == "pyarrow":
        return values.array.__arrow_array__()
    return None


def _isin(values: pd.Series, allowed: pd.Series) -> pd.Series:
    """``values.isin(allowed)``.

    Arrow-backed strings are matched by ``pyarrow.compute.is_in``; other
    columns via object dtype (Arrow-string ``Series.isin`` is slow).
    """
    left, right = _arrow_strings(values), _arrow_strings(allowed)
    if left is not None and right is not None:
        import pyarrow as pa
        import pyarrow.compute as pc

        value_set = pa.chunked_array(right).combine_chunks().cast(left.type)
        hit = pc.is_in(left, value_set=value_set)
        mask = np.asarray(hit.to_numpy(zero_copy_only=False), dtype=bool)
        return pd.Series(mask, index=values.index)
    return values.astype(object).isin(pd.Index(allowed.astype(object).unique()))


#: Sorted values compared at a time by :func:`_has_duplicates`.
_SORTED_CHUNK = 1 << 20


def _has_duplicates(values: pd.Series) -> bool:
    """``not values.is_unique``.

    Arrow-backed strings are sorted (an index per row) and neighbours
    compared a chunk at a time, instead of a hash table holding a copy of
    every value.
    """
    arr = _arrow_strings(values)
    if arr is None:
        return not values.is_unique
    import pyarrow.compute as pc

    if arr.null_count > 1:
        return True
    order = pc.sort_indices(arr)
    n = len(order)
    for a in range(0, n - 1, _SORTED_CHUNK):
        chunk = arr.take(order[a : a + _SORTED_CHUNK + 1])
        if pc.any(pc.equal(chunk[1:], chunk[:-1])).as_py():
            return True
    return False


def _unknown_pairs(relations: pd.DataFrame, objects: pd.DataFrame) -> np.ndarray:
    """Mask of relation rows whose ``(oid, type)`` is not in ``objects``.

    Matched one object type at a time (the oids of a type against the
    objects of that type), which needs no tuple per row.
    """
    rel_type, obj_type = relations[OCEL_TYPE], objects[OCEL_TYPE]
    if rel_type.isna().any() or obj_type.isna().any():
        known = pd.MultiIndex.from_arrays(
            [objects[OCEL_OID].astype(object), obj_type.astype(object)]
        )
        rel_pairs = pd.MultiIndex.from_arrays(
            [relations[OCEL_OID].astype(object), rel_type.astype(object)]
        )
        return np.asarray(~rel_pairs.isin(known), dtype=bool)
    unknown = np.zeros(len(relations), dtype=bool)
    rel_codes, rel_types = pd.factorize(rel_type)
    obj_type = obj_type.astype(object)
    for code, otype in enumerate(list(rel_types)):
        rows = rel_codes == code
        allowed = objects[OCEL_OID][(obj_type == otype).to_numpy()]
        unknown[rows] = ~_isin(relations[OCEL_OID][rows], allowed).to_numpy()
    return unknown


def validate(eventlog: EventLog) -> None:
    """Raise ``ValueError`` if the event log violates the canonical schema."""
    events, objects, relations = eventlog.events, eventlog.objects, eventlog.relations

    missing = [c for c in EVENT_REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table missing required columns: {missing}")

    if not events.empty and _has_duplicates(events[OCEL_EID]):
        dup = events[OCEL_EID][events[OCEL_EID].duplicated()].iloc[0]
        raise ValueError(f"duplicate {OCEL_EID}: {dup!r}")

    missing = [c for c in OBJECT_REQUIRED_COLUMNS if c not in objects.columns]
    if missing:
        raise ValueError(f"objects table missing required columns: {missing}")

    missing = [c for c in RELATION_REQUIRED_COLUMNS if c not in relations.columns]
    if missing:
        raise ValueError(f"relations table missing required columns: {missing}")

    if not relations.empty:
        bad = relations[OCEL_EID][~_isin(relations[OCEL_EID], events[OCEL_EID])]
        if not bad.empty:
            raise ValueError(f"relations reference unknown {OCEL_EID}: {bad.iloc[0]!r}")
        unknown_pairs = _unknown_pairs(relations, objects)
        if unknown_pairs.any():
            first = int(np.flatnonzero(unknown_pairs)[0])
            pair = (relations[OCEL_OID].iloc[first], relations[OCEL_TYPE].iloc[first])
            raise ValueError(
                f"relation references unknown object {pair!r}; "
                "every (oid, type) in relations must appear in objects"
            )

    o2o = eventlog.o2o
    missing = [c for c in O2O_REQUIRED_COLUMNS if c not in o2o.columns]
    if missing:
        raise ValueError(f"o2o table missing required columns: {missing}")
    if not o2o.empty:
        for col in (OCEL_OID, OCEL_OID2):
            unknown = o2o[col][~_isin(o2o[col], objects[OCEL_OID])]
            if not unknown.empty:
                raise ValueError(
                    f"o2o references unknown object id in {col!r}: "
                    f"{unknown.iloc[0]!r}; every oid must appear in objects"
                )

    changes = eventlog.object_changes
    missing = [c for c in OBJECT_CHANGE_REQUIRED_COLUMNS if c not in changes.columns]
    if missing:
        raise ValueError(f"object_changes table missing required columns: {missing}")
    if not changes.empty:
        unknown = changes[OCEL_OID][~_isin(changes[OCEL_OID], objects[OCEL_OID])]
        if not unknown.empty:
            raise ValueError(
                f"object_changes references unknown object id: {unknown.iloc[0]!r}; "
                "every oid must appear in objects"
            )
