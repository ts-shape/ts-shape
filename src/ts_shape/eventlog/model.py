"""The :class:`EventLog` dataclass — three DataFrames in OCEL 2.0 shape."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import schema


@dataclass
class EventLog:
    """Canonical ts-shape event log: events + (optional) objects + relations.

    Objects/relations are empty when the source detector has no natural object
    association (e.g. a global cross-signal correlation statistic). Use
    :attr:`has_objects` to check before calling :meth:`to_flat_df`.
    """

    events: pd.DataFrame = field(default_factory=schema.empty_events)
    objects: pd.DataFrame = field(default_factory=schema.empty_objects)
    relations: pd.DataFrame = field(default_factory=schema.empty_relations)

    @property
    def has_objects(self) -> bool:
        return not self.objects.empty

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return (
            f"EventLog(events={len(self.events)}, "
            f"objects={len(self.objects)}, relations={len(self.relations)})"
        )

    def filter_by_pack(self, pack: str) -> "EventLog":
        events = self.events[self.events[schema.TS_PACK] == pack]
        return self._restrict(events)

    def filter_by_object(self, oid: str, type_: str | None = None) -> "EventLog":
        rel = self.relations
        mask = rel[schema.OCEL_OID] == oid
        if type_ is not None:
            mask &= rel[schema.OCEL_TYPE] == type_
        eids = set(rel[mask][schema.OCEL_EID])
        events = self.events[self.events[schema.OCEL_EID].isin(eids)]
        return self._restrict(events)

    def _restrict(self, events: pd.DataFrame) -> "EventLog":
        """Trim relations + objects to those reachable from ``events``."""
        eids = events[schema.OCEL_EID]
        relations = self.relations[self.relations[schema.OCEL_EID].isin(eids)]
        if relations.empty:
            objects = schema.empty_objects()
        else:
            keep_pairs = relations[
                [schema.OCEL_OID, schema.OCEL_TYPE]
            ].drop_duplicates()
            objects = self.objects.merge(
                keep_pairs, on=[schema.OCEL_OID, schema.OCEL_TYPE], how="inner",
            )
        return EventLog(
            events.reset_index(drop=True),
            objects.reset_index(drop=True),
            relations.reset_index(drop=True),
        )
