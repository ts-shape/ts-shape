"""The :class:`EventLog` dataclass — the OCEL 2.0 relational tables."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

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
        changes = self.object_changes
        if schema.OCEL_TYPE in changes.columns:
            # Changes belong to an (oid, type) object: an oid kept under one
            # type does not keep the changes of the same oid of another type.
            kept = set(zip(objects[schema.OCEL_OID], objects[schema.OCEL_TYPE]))
            index = pd.MultiIndex.from_arrays(
                [
                    changes[schema.OCEL_OID].astype(object),
                    changes[schema.OCEL_TYPE].astype(object),
                ]
            )
            changes = changes[index.isin(list(kept))] if kept else changes.iloc[:0]
        else:
            changes = changes[isin(changes[schema.OCEL_OID], oids)]
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

    def filter_by_object_type(
        self,
        types: str | Iterable[str],
        *,
        keep_events: Literal["linked", "all"] = "linked",
    ) -> EventLog:
        """Return a sub-log containing only the objects of some types.

        ``objects`` keeps every object of ``types`` (also one no event links
        to); ``relations``, ``o2o`` (both ends) and ``object_changes`` are
        restricted to those objects, so no table references a dropped one.

        Args:
            types: Object type(s) (``ocel:type``) to keep.
            keep_events: ``"linked"`` keeps the events related to at least one
                kept object; ``"all"`` keeps every event (their relations to
                other objects are dropped).

        Returns:
            A new EventLog restricted to the objects of ``types``.

        Raises:
            ValueError: For an unknown ``keep_events``.
        """
        from .query import _filter_by_object_type

        return _filter_by_object_type(self, types, keep_events)

    def filter_by_time(self, start: object = None, end: object = None) -> EventLog:
        """Return a sub-log of the half-open time window ``[start, end)``.

        * ``events``: those whose ``ocel:timestamp`` is in the window.
        * ``objects``: those present in the window (a ``lifecycle``
          ``active`` -> ``released`` span of ``object_changes`` overlaps it),
          those without lifecycle rows (presence unknown), and every object a
          kept event relates to.
        * ``relations`` of the kept events; ``o2o`` with both ends kept.
        * ``object_changes`` of the kept objects inside the window, plus each
          field's last value before ``start`` (the state at the window start).

        Args:
            start: Window start (inclusive); ``None``: unbounded. Naive times
                are UTC.
            end: Window end (exclusive); ``None``: unbounded.

        Returns:
            A new EventLog restricted to the window.

        Raises:
            ValueError: When ``start`` is not before ``end``.
        """
        from .query import _filter_by_time

        return _filter_by_time(self, start, end)

    def where_used(
        self,
        oid: str,
        *,
        depth: int | None = None,
        start: object = None,
        end: object = None,
        qualifiers: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Everything ``oid`` went into; see :func:`~ts_shape.eventlog.where_used`.

        Args:
            oid: The object to start from.
            depth: Maximum number of steps; ``None``: no limit.
            start: Window start (inclusive).
            end: Window end (exclusive).
            qualifiers: The ``o2o`` qualifiers to follow (default: all but
                ``co_occurs``).

        Returns:
            ``oid``, ``object_type``, ``qualifier``, ``depth``, ``via`` (+ extra
            ``o2o`` columns).
        """
        from .query import where_used

        return where_used(
            self, oid, depth=depth, start=start, end=end, qualifiers=qualifiers
        )

    def where_from(
        self,
        oid: str,
        *,
        depth: int | None = None,
        start: object = None,
        end: object = None,
        qualifiers: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Everything ``oid`` came from; see :func:`~ts_shape.eventlog.where_from`.

        Args:
            oid: The object to start from.
            depth: Maximum number of steps; ``None``: no limit.
            start: Window start (inclusive).
            end: Window end (exclusive).
            qualifiers: The ``o2o`` qualifiers to follow (default: all but
                ``co_occurs``).

        Returns:
            ``oid``, ``object_type``, ``qualifier``, ``depth``, ``via`` (+ extra
            ``o2o`` columns).
        """
        from .query import where_from

        return where_from(
            self, oid, depth=depth, start=start, end=end, qualifiers=qualifiers
        )

    def lineage(
        self,
        oid: str,
        *,
        depth: int | None = None,
        start: object = None,
        end: object = None,
        qualifiers: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Both directions with a signed ``depth``; see :func:`~ts_shape.eventlog.lineage`.

        Args:
            oid: The object to start from.
            depth: Maximum number of steps per direction; ``None``: no limit.
            start: Window start (inclusive).
            end: Window end (exclusive).
            qualifiers: The ``o2o`` qualifiers to follow (default: all but
                ``co_occurs``).

        Returns:
            The :meth:`where_from` rows at negative depth, ``oid`` at ``0``,
            the :meth:`where_used` rows at positive depth.
        """
        from .query import lineage

        return lineage(
            self, oid, depth=depth, start=start, end=end, qualifiers=qualifiers
        )
