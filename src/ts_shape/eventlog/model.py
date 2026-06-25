"""The :class:`EventLog` dataclass — the OCEL 2.0 relational tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import schema


def _objects_in(objects: pd.DataFrame, used: set[tuple[str, str]]) -> pd.DataFrame:
    """Vectorized ``(oid, type)`` membership filter over the objects table."""
    if not used or objects.empty:
        return schema.empty_objects()
    index = pd.MultiIndex.from_frame(objects[[schema.OCEL_OID, schema.OCEL_TYPE]])
    return objects[index.isin(used)]


@dataclass
class EventLog:
    """Canonical ts-shape event log in OCEL 2.0 shape.

    Five relational tables, mirroring the OCEL 2.0 standard:

    * ``events`` — one row per detected event (+ event attributes),
    * ``objects`` — the objects events refer to (``ocel:oid`` / ``ocel:type``),
    * ``relations`` — event-to-object (E2O) relations with qualifiers,
    * ``o2o`` — object-to-object relations with qualifiers,
    * ``object_changes`` — time-varying object attribute values.

    ``objects`` / ``relations`` are empty when the source detector has no
    natural object association (e.g. a global cross-signal correlation
    statistic); ``o2o`` / ``object_changes`` are empty unless supplied. Use
    :attr:`has_objects` to check before calling :func:`to_event_log_xes`.
    """

    events: pd.DataFrame = field(default_factory=schema.empty_events)
    objects: pd.DataFrame = field(default_factory=schema.empty_objects)
    relations: pd.DataFrame = field(default_factory=schema.empty_relations)
    o2o: pd.DataFrame = field(default_factory=schema.empty_o2o)
    object_changes: pd.DataFrame = field(default_factory=schema.empty_object_changes)

    @property
    def has_objects(self) -> bool:
        return not self.objects.empty

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return (
            f"EventLog(events={len(self.events)}, objects={len(self.objects)}, "
            f"relations={len(self.relations)}, o2o={len(self.o2o)}, "
            f"object_changes={len(self.object_changes)})"
        )

    def _restrict_to_events(self, events: pd.DataFrame) -> EventLog:
        """Build a sub-log keeping only the objects/relations the events use."""
        eids = set(events[schema.OCEL_EID])
        relations = self.relations[self.relations[schema.OCEL_EID].isin(eids)]
        used = set(zip(relations[schema.OCEL_OID], relations[schema.OCEL_TYPE]))
        objects = _objects_in(self.objects, used)
        oids = set(objects[schema.OCEL_OID])
        o2o = self.o2o[
            self.o2o[schema.OCEL_OID].isin(oids) & self.o2o[schema.OCEL_OID2].isin(oids)
        ]
        changes = self.object_changes[self.object_changes[schema.OCEL_OID].isin(oids)]
        return EventLog(
            events.reset_index(drop=True),
            objects.reset_index(drop=True),
            relations.reset_index(drop=True),
            o2o.reset_index(drop=True),
            changes.reset_index(drop=True),
        )

    def filter_by_pack(self, pack: str) -> EventLog:
        return self._restrict_to_events(
            self.events[self.events[schema.TS_PACK] == pack]
        )

    def filter_by_object(self, oid: str, type_: str | None = None) -> EventLog:
        rel = self.relations
        mask = rel[schema.OCEL_OID] == oid
        if type_ is not None:
            mask &= rel[schema.OCEL_TYPE] == type_
        eids = set(rel[mask][schema.OCEL_EID])
        return self._restrict_to_events(
            self.events[self.events[schema.OCEL_EID].isin(eids)]
        )
