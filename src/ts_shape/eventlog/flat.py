"""XES-flat exporter — produces a pandas DataFrame with ``case:concept:name``,
``concept:name``, ``time:timestamp``, ``lifecycle:transition``,
``org:resource`` columns. ts-shape itself does **not** import pm4py; users
pass this DataFrame to ``pm4py.format_dataframe`` themselves.
"""

from __future__ import annotations

import pandas as pd

from . import schema
from .model import EventLog

# Object type whose linked oid names the resource (org:resource) that
# performed the activity. Falls back to absent when no such relation exists —
# org:resource is never fabricated from the case id.
_RESOURCE_OBJECT_TYPE = "operator"


def to_event_log_xes(
    eventlog: EventLog,
    *,
    case_object_type: str = "asset",
    lifecycle: str = "single",
) -> pd.DataFrame:
    """Flatten the event log into an XES-style DataFrame.

    A trace is built per object of ``case_object_type`` — each event linked
    to that object becomes one (or two) rows in the trace.

    ``lifecycle="single"`` — interval events become one row with
    ``lifecycle:transition="complete"``. ``time:timestamp`` is the
    interval-end; ``start_timestamp`` is exposed verbatim.

    ``lifecycle="two_row"`` — interval events expand into a ``start`` row
    (using ``ts_shape:start_timestamp``) and a ``complete`` row, paired by
    ``concept:instance``.
    """
    if not eventlog.has_objects:
        raise ValueError(
            "to_event_log_xes requires objects: this event log was produced by "
            "a detector with no object association. Use to_event_log_ocel() or "
            "supply objects to the adapter."
        )
    if lifecycle not in ("single", "two_row"):
        raise ValueError(f"invalid lifecycle {lifecycle!r}")

    rels = eventlog.relations
    rels = rels[rels[schema.OCEL_TYPE] == case_object_type]
    if rels.empty:
        raise ValueError(
            f"no objects of type {case_object_type!r} in event log; "
            f"available types: {eventlog.relations[schema.OCEL_TYPE].unique().tolist()}"
        )

    joined = rels.merge(eventlog.events, on=schema.OCEL_EID, how="inner")

    rename = {
        schema.OCEL_OID: schema.XES_CASE,
        schema.OCEL_ACTIVITY: schema.XES_ACTIVITY,
        schema.OCEL_TIMESTAMP: schema.XES_TIMESTAMP,
    }
    base = joined.rename(columns=rename)
    # org:resource is the operator that performed the activity, NOT the case
    # object. Populate it only from an actual operator relation; if the log has
    # no operators, the column is omitted rather than mislabelling the case id.
    resource = _resource_map(eventlog)
    if resource is not None:
        base[schema.XES_RESOURCE] = base[schema.OCEL_EID].map(resource).astype("string")
    base["start_timestamp"] = base[schema.TS_START_TIMESTAMP]

    if lifecycle == "single":
        base[schema.XES_LIFECYCLE] = "complete"
        base = base.sort_values([schema.XES_CASE, schema.XES_TIMESTAMP]).reset_index(
            drop=True
        )
        return _arrange_columns(base)

    # Two-row expansion.
    has_start = base[schema.TS_START_TIMESTAMP].notna()
    starts = base[has_start].copy()
    starts[schema.XES_TIMESTAMP] = starts[schema.TS_START_TIMESTAMP]
    starts[schema.XES_LIFECYCLE] = "start"
    completes = base.copy()
    completes[schema.XES_LIFECYCLE] = "complete"
    expanded = pd.concat([starts, completes], ignore_index=True)
    expanded[schema.XES_INSTANCE] = expanded[schema.OCEL_EID]
    expanded = expanded.sort_values(
        [schema.XES_CASE, schema.XES_TIMESTAMP, schema.XES_LIFECYCLE]
    ).reset_index(drop=True)
    return _arrange_columns(expanded)


def _resource_map(eventlog: EventLog) -> pd.Series | None:
    """Map ``ocel:eid`` -> resource oid from operator E2O relations.

    Returns ``None`` when the log has no operator relations, so the caller can
    omit ``org:resource`` entirely rather than fabricate it.
    """
    rels = eventlog.relations
    ops = rels[rels[schema.OCEL_TYPE] == _RESOURCE_OBJECT_TYPE]
    if ops.empty:
        return None
    # One resource per event; keep the first if several are linked.
    return ops.drop_duplicates(schema.OCEL_EID).set_index(schema.OCEL_EID)[
        schema.OCEL_OID
    ]


def _arrange_columns(df: pd.DataFrame) -> pd.DataFrame:
    head = [
        schema.XES_CASE,
        schema.XES_ACTIVITY,
        schema.XES_TIMESTAMP,
        schema.XES_LIFECYCLE,
        schema.XES_RESOURCE,
        "start_timestamp",
    ]
    head = [c for c in head if c in df.columns]
    rest = [c for c in df.columns if c not in head]
    return df[head + rest]
