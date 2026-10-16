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

    Example:
        Built by :func:`~ts_shape.to_event_log` from a detector's output:

        >>> import ts_shape
        >>> df = ts_shape.make_timeseries(["temp"], n_outliers=3)
        >>> out = ts_shape.OutlierDetectionEvents(df, "value_double").detect_outliers_zscore()
        >>> log = ts_shape.to_event_log(
        ...     out, detector="OutlierDetectionEvents.detect_outliers_zscore")
        >>> len(log.events), log.has_objects
        (4, True)
        >>> log.events[["ocel:activity", "ocel:timestamp"]].head(2)
                    ocel:activity            ocel:timestamp
        0  quality.outlier.zscore 2025-01-01 01:34:30+00:00
        1  quality.outlier.zscore 2025-01-01 03:21:00+00:00
    """

    events: pd.DataFrame = field(default_factory=schema.empty_events)
    objects: pd.DataFrame = field(default_factory=schema.empty_objects)
    relations: pd.DataFrame = field(default_factory=schema.empty_relations)
    o2o: pd.DataFrame = field(default_factory=schema.empty_o2o)
    object_changes: pd.DataFrame = field(default_factory=schema.empty_object_changes)

    @property
    def has_objects(self) -> bool:
        """Whether the log contains at least one object in ``objects``."""
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
        isin = schema._isin
        relations = self.relations[
            isin(self.relations[schema.OCEL_EID], events[schema.OCEL_EID])
        ]
        used = set(zip(relations[schema.OCEL_OID], relations[schema.OCEL_TYPE]))
        objects = _objects_in(self.objects, used)
        oids = objects[schema.OCEL_OID]
        o2o = self.o2o[
            isin(self.o2o[schema.OCEL_OID], oids)
            & isin(self.o2o[schema.OCEL_OID2], oids)
        ]
        changes = self.object_changes[isin(self.object_changes[schema.OCEL_OID], oids)]
        return EventLog(
            events.reset_index(drop=True),
            objects.reset_index(drop=True),
            relations.reset_index(drop=True),
            o2o.reset_index(drop=True),
            changes.reset_index(drop=True),
        )

    def filter_by_pack(self, pack: str) -> EventLog:
        """Return a sub-log containing only the events emitted by one pack.

        Objects, relations, object-to-object links, and object changes are
        restricted to those referenced by the kept events.

        Args:
            pack: Pack name to match against the ``ts_shape:pack`` column.

        Returns:
            A new EventLog restricted to the matching events.
        """
        return self._restrict_to_events(
            self.events[self.events[schema.TS_PACK] == pack]
        )

    def filter_by_object(self, oid: str, type_: str | None = None) -> EventLog:
        """Return a sub-log containing only the events related to one object.

        Events are selected through the event-to-object ``relations`` table;
        the other tables are restricted to the objects those events use.

        Args:
            oid: Object identifier (``ocel:oid``) to match.
            type_: Optional object type (``ocel:type``) to match as well.

        Returns:
            A new EventLog restricted to the events related to the object.
        """
        rel = self.relations
        mask = rel[schema.OCEL_OID] == oid
        if type_ is not None:
            mask &= rel[schema.OCEL_TYPE] == type_
        eids = set(rel[mask][schema.OCEL_EID])
        return self._restrict_to_events(
            self.events[self.events[schema.OCEL_EID].isin(eids)]
        )
