"""Concatenate multiple :class:`EventLog` instances into one."""
from __future__ import annotations

import pandas as pd

from . import schema
from .model import EventLog


def concat(*logs: EventLog) -> EventLog:
    if not logs:
        return EventLog()
    events = pd.concat([l.events for l in logs], ignore_index=True)
    objects = pd.concat([l.objects for l in logs], ignore_index=True)
    relations = pd.concat([l.relations for l in logs], ignore_index=True)

    if not objects.empty:
        objects = objects.drop_duplicates(
            subset=[schema.OCEL_OID, schema.OCEL_TYPE]
        ).reset_index(drop=True)

    if not events.empty:
        events = events.sort_values(schema.OCEL_TIMESTAMP).reset_index(drop=True)
        # Deduplicate identical event ids (e.g. same input passed twice).
        events = events.drop_duplicates(subset=[schema.OCEL_EID]).reset_index(drop=True)

    if not relations.empty:
        relations = relations.drop_duplicates(
            subset=[schema.OCEL_EID, schema.OCEL_OID, schema.OCEL_TYPE]
        ).reset_index(drop=True)

    return EventLog(events=events, objects=objects, relations=relations)
