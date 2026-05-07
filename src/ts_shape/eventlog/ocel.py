"""OCEL 2.0–shaped exporter — returns the three relational tables.

ts-shape does not write OCEL JSON or SQLite itself; users feed these
DataFrames to ``pm4py.write_ocel2_json`` (or any other OCEL tool) directly.
The column names match the OCEL 2.0 spec verbatim.
"""
from __future__ import annotations

import pandas as pd

from .model import EventLog


def to_ocel_tables(eventlog: EventLog) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(events, objects, relations)`` DataFrames as-is.

    The canonical schema already uses OCEL 2.0 column names, so this is a
    pass-through with a defensive ``copy()`` to prevent caller mutation.
    """
    return (
        eventlog.events.copy(),
        eventlog.objects.copy(),
        eventlog.relations.copy(),
    )
