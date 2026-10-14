"""OCEL 2.0–shaped exporter — returns the relational tables.

ts-shape does not write OCEL JSON or SQLite itself; users feed these
DataFrames to ``pm4py`` (or any other OCEL tool) directly. The column names
match the OCEL 2.0 spec verbatim. pm4py additionally expects the event
activity / timestamp on every E2O relation and a *wide* ``object_changes``
table (the new value in a column named after ``ocel:field``); pass
``pm4py=True`` to get exactly that layout, which maps straight onto pm4py's
``OCEL`` constructor and its OCEL 2.0 JSON / XML / SQLite writers::

    t = to_event_log_ocel(log, pm4py=True)
    ocel = pm4py.objects.ocel.obj.OCEL(
        events=t.events, objects=t.objects, relations=t.relations,
        o2o=t.o2o, object_changes=t.object_changes,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

from ..errors import DataQualityWarning
from . import schema
from .model import EventLog


@dataclass(frozen=True)
class OCEL2Tables:
    """The five OCEL 2.0 relational tables produced from an :class:`EventLog`."""

    events: pd.DataFrame
    objects: pd.DataFrame
    relations: pd.DataFrame
    o2o: pd.DataFrame
    object_changes: pd.DataFrame


def to_event_log_ocel(eventlog: EventLog, *, pm4py: bool = False) -> OCEL2Tables:
    """Return the OCEL 2.0 tables as an :class:`OCEL2Tables` bundle.

    The canonical schema already uses OCEL 2.0 column names, so by default
    this is a pass-through with defensive ``copy()`` calls to prevent caller
    mutation.

    With ``pm4py=True`` two tables are reshaped for pm4py's ``OCEL`` object:
    ``relations`` gains ``ocel:activity`` and ``ocel:timestamp`` of each
    event, and ``object_changes`` is widened — ``ocel:value`` is replaced by
    one column per distinct ``ocel:field`` holding the new value on that
    field's rows (a field named like a base column, e.g. ``ocel:type``,
    gets the column ``<name>_field`` and a
    :class:`~ts_shape.errors.DataQualityWarning`). The other tables are
    unchanged.

    Args:
        eventlog: The event log to export.
        pm4py: Reshape ``relations`` and ``object_changes`` for pm4py.

    Returns:
        The five tables as :class:`OCEL2Tables` (copies).
    """
    if pm4py:
        return OCEL2Tables(
            events=eventlog.events.copy(),
            objects=eventlog.objects.copy(),
            relations=_pm4py_relations(eventlog),
            o2o=eventlog.o2o.copy(),
            object_changes=_pm4py_object_changes(eventlog.object_changes),
        )
    return OCEL2Tables(
        events=eventlog.events.copy(),
        objects=eventlog.objects.copy(),
        relations=eventlog.relations.copy(),
        o2o=eventlog.o2o.copy(),
        object_changes=eventlog.object_changes.copy(),
    )


def _pm4py_relations(eventlog: EventLog) -> pd.DataFrame:
    cols = [schema.OCEL_EID, schema.OCEL_ACTIVITY, schema.OCEL_TIMESTAMP]
    info = eventlog.events[cols].drop_duplicates(schema.OCEL_EID)
    rel = eventlog.relations.drop(
        columns=[c for c in cols[1:] if c in eventlog.relations.columns]
    )
    out = rel.merge(info, on=schema.OCEL_EID, how="left")
    front = [
        schema.OCEL_EID,
        schema.OCEL_ACTIVITY,
        schema.OCEL_TIMESTAMP,
        schema.OCEL_OID,
        schema.OCEL_TYPE,
        schema.OCEL_QUALIFIER,
    ]
    order = [c for c in front if c in out.columns]
    return out[order + [c for c in out.columns if c not in order]]


def _pm4py_object_changes(changes: pd.DataFrame) -> pd.DataFrame:
    base = [schema.OCEL_OID, schema.OCEL_TYPE, schema.OCEL_TIMESTAMP, schema.OCEL_FIELD]
    out = changes[base].copy()
    if changes.empty:
        return out
    fields = changes[schema.OCEL_FIELD].astype(object)
    for name in pd.unique(fields):
        if pd.isna(name):
            continue
        column = str(name)
        if column in base:
            # A field named like a base column would overwrite it: keep both.
            column = f"{column}_field"
            warnings.warn(
                f"object_changes field {name!r} has the name of a base column; "
                f"its values are in column {column!r}",
                DataQualityWarning,
                stacklevel=3,
            )
        mask = (fields == name).to_numpy()
        col = pd.Series(pd.NA, index=changes.index, dtype=object)
        col[mask] = changes.loc[mask, schema.OCEL_VALUE]
        out[column] = col
    return out
