"""Concatenate multiple :class:`EventLog` instances into one."""

from __future__ import annotations

import pandas as pd

from . import schema
from .model import EventLog


def concat(*logs: EventLog) -> EventLog:
    if not logs:
        return EventLog()
    events = pd.concat([log.events for log in logs], ignore_index=True)
    objects = pd.concat([log.objects for log in logs], ignore_index=True)
    relations = pd.concat([log.relations for log in logs], ignore_index=True)
    o2o = pd.concat([log.o2o for log in logs], ignore_index=True)
    object_changes = pd.concat([log.object_changes for log in logs], ignore_index=True)

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

    if not o2o.empty:
        o2o = o2o.drop_duplicates(
            subset=[schema.OCEL_OID, schema.OCEL_OID2, schema.OCEL_QUALIFIER]
        ).reset_index(drop=True)

    if not object_changes.empty:
        object_changes = object_changes.drop_duplicates(
            subset=[schema.OCEL_OID, schema.OCEL_FIELD, schema.OCEL_TIMESTAMP]
        ).reset_index(drop=True)

    return EventLog(
        events=events,
        objects=objects,
        relations=relations,
        o2o=o2o,
        object_changes=object_changes,
    )
