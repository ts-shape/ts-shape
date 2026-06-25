"""OCEL 2.0–shaped exporter — returns the relational tables.

ts-shape does not write OCEL JSON or SQLite itself; users feed these
DataFrames to ``pm4py`` (or any other OCEL tool) directly. The column names
match the OCEL 2.0 spec verbatim, so the result maps straight onto pm4py's
``OCEL`` constructor::

    t = to_event_log_ocel(log)
    ocel = pm4py.objects.ocel.obj.OCEL(
        events=t.events, objects=t.objects, relations=t.relations,
        o2o=t.o2o, object_changes=t.object_changes,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model import EventLog


@dataclass(frozen=True)
class OCEL2Tables:
    """The five OCEL 2.0 relational tables produced from an :class:`EventLog`."""

    events: pd.DataFrame
    objects: pd.DataFrame
    relations: pd.DataFrame
    o2o: pd.DataFrame
    object_changes: pd.DataFrame


def to_event_log_ocel(eventlog: EventLog) -> OCEL2Tables:
    """Return the OCEL 2.0 tables as an :class:`OCEL2Tables` bundle.

    The canonical schema already uses OCEL 2.0 column names, so this is a
    pass-through with defensive ``copy()`` calls to prevent caller mutation.
    """
    return OCEL2Tables(
        events=eventlog.events.copy(),
        objects=eventlog.objects.copy(),
        relations=eventlog.relations.copy(),
        o2o=eventlog.o2o.copy(),
        object_changes=eventlog.object_changes.copy(),
    )
