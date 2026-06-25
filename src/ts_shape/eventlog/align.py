"""Align the ``events`` tables of multiple :class:`EventLog` instances to a
shared column set.

Different detectors emit the same 9-column canonical core but different
*extra* columns (standard-attr extensions, ``<pack>:<col>`` passthroughs).
:func:`concat` already unions columns on merge, but when callers want to
append / stack the per-detector frames themselves they need every frame to
expose an identical column set first. :func:`align_columns` provides that.
"""

from __future__ import annotations

from . import schema
from .model import EventLog

# Canonical emit order of the core columns (mirrors ``adapters.adapt``).
_CORE_ORDER: tuple[str, ...] = (
    schema.OCEL_EID,
    schema.OCEL_ACTIVITY,
    schema.OCEL_TIMESTAMP,
    schema.TS_START_TIMESTAMP,
    schema.TS_DURATION_S,
    schema.TS_DETECTOR,
    schema.TS_PACK,
    schema.TS_SEVERITY,
    schema.TS_VALUE,
)
_CORE_SET = frozenset(_CORE_ORDER)


def align_columns(*logs: EventLog) -> list[EventLog]:
    """Reindex every log's ``events`` table to the union of their columns.

    Each returned :class:`EventLog` exposes an identical, identically-ordered
    set of event columns — the canonical core first (in emit order), then the
    remaining attribute columns sorted for determinism. Columns absent from a
    given log are added and filled with NA, so the frames can be appended or
    stacked directly without a column mismatch. ``objects`` and ``relations``
    already have fixed schemas and are returned unchanged.

    Returns a list parallel to the inputs. An empty call returns ``[]``.
    """
    if not logs:
        return []

    seen: list[str] = []
    seen_set: set[str] = set()
    for log in logs:
        for col in log.events.columns:
            if col not in seen_set:
                seen.append(col)
                seen_set.add(col)

    core = [c for c in _CORE_ORDER if c in seen_set]
    extras = sorted(c for c in seen if c not in _CORE_SET)
    union = core + extras

    return [
        EventLog(
            events=log.events.reindex(columns=union),
            objects=log.objects,
            relations=log.relations,
            o2o=log.o2o,
            object_changes=log.object_changes,
        )
        for log in logs
    ]
