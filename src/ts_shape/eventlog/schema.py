"""Column-name constants and schema definitions for the ts-shape event log.

The canonical event-log uses OCEL 2.0 column names (`ocel:*`) with a few
ts-shape-prefixed columns for fields that have no OCEL counterpart but are
useful for downstream consumers (interval starts, durations, severity, etc.).
XES column names (`concept:name`, `time:timestamp`, `case:concept:name`,
`lifecycle:transition`, `org:resource`) are produced only by the flat
exporter; they do not appear in the canonical schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


# ---- Event-to-object relations table ---------------------------------------

OCEL_QUALIFIER = "ocel:qualifier"

RELATION_REQUIRED_COLUMNS: tuple[str, ...] = (OCEL_EID, OCEL_OID, OCEL_TYPE)


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
# from these fixed keys to either a legacy column name (string) or a literal
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
    return name in _OBJECT_TYPES


# ---- Empty frames -----------------------------------------------------------


def empty_events() -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            OCEL_OID: pd.Series(dtype="string"),
            OCEL_TYPE: pd.Series(dtype="string"),
        }
    )


def empty_relations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            OCEL_EID: pd.Series(dtype="string"),
            OCEL_OID: pd.Series(dtype="string"),
            OCEL_TYPE: pd.Series(dtype="string"),
            OCEL_QUALIFIER: pd.Series(dtype="string"),
        }
    )


# ---- Validation -------------------------------------------------------------


def validate(eventlog: EventLog) -> None:
    """Raise ``ValueError`` if the event log violates the canonical schema."""
    events, objects, relations = eventlog.events, eventlog.objects, eventlog.relations

    missing = [c for c in EVENT_REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table missing required columns: {missing}")

    if not events.empty and events[OCEL_EID].is_unique is False:
        dup = events[OCEL_EID][events[OCEL_EID].duplicated()].iloc[0]
        raise ValueError(f"duplicate {OCEL_EID}: {dup!r}")

    missing = [c for c in OBJECT_REQUIRED_COLUMNS if c not in objects.columns]
    if missing:
        raise ValueError(f"objects table missing required columns: {missing}")

    missing = [c for c in RELATION_REQUIRED_COLUMNS if c not in relations.columns]
    if missing:
        raise ValueError(f"relations table missing required columns: {missing}")

    if not relations.empty:
        eids = set(events[OCEL_EID])
        bad = relations[OCEL_EID][~relations[OCEL_EID].isin(eids)]
        if not bad.empty:
            raise ValueError(f"relations reference unknown {OCEL_EID}: {bad.iloc[0]!r}")
        oids = set(zip(objects[OCEL_OID], objects[OCEL_TYPE]))
        rel_pairs = list(zip(relations[OCEL_OID], relations[OCEL_TYPE]))
        for pair in rel_pairs:
            if pair not in oids:
                raise ValueError(
                    f"relation references unknown object {pair!r}; "
                    "every (oid, type) in relations must appear in objects"
                )
