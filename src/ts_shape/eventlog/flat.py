"""XES-flat exporter — produces a pandas DataFrame with ``case:concept:name``,
``concept:name``, ``time:timestamp``, ``lifecycle:transition``,
``org:resource`` columns. ts-shape itself does **not** import pm4py; users
pass this DataFrame to ``pm4py.format_dataframe`` themselves.
"""

from __future__ import annotations

import warnings

import pandas as pd

from . import schema
from .model import EventLog


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
    base[schema.XES_RESOURCE] = base.get(schema.XES_CASE)
    base["start_timestamp"] = base[schema.TS_START_TIMESTAMP]

    if lifecycle == "single":
        base[schema.XES_LIFECYCLE] = "complete"
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


def to_flat_df(*args, **kwargs) -> pd.DataFrame:
    """Deprecated alias for :func:`to_event_log_xes`."""
    warnings.warn(
        "to_flat_df is deprecated; use to_event_log_xes",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_event_log_xes(*args, **kwargs)
