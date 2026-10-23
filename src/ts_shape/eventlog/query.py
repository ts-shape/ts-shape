"""Genealogy (traceability) queries and coherent sub-logs of an :class:`EventLog`.

The object-to-object table ``o2o`` holds directed edges ``ocel:oid ->
ocel:oid_2``; ts-shape's ``part_of`` edges point from the child to the parent
(a serial is ``part_of`` its batch, a batch ``part_of`` its material lot).
On that graph:

* :func:`where_used` follows the edges **backwards**, from a parent to
  everything it went into: ``where_used(batch)`` returns the batch's serials,
  the handling units those serials were packed into, and so on.
* :func:`where_from` follows them **forwards**, from a child to everything
  it came from: ``where_from(serial)`` returns its batch, the batch's
  material, ...
* :func:`lineage` returns both as one frame with a signed ``depth``
  (negative: where the object came from, positive: where it was used, ``0``:
  the object itself).

Each traversal is a breadth-first search over an adjacency built once per
call (int codes, CSR arrays); a level is expanded with array operations, so
100k edges take milliseconds. Every object is reported at its shortest
distance from the start; cycles terminate because an object is visited once.

``o2o`` carries no validity time in ts-shape. A ``start`` / ``end`` window
therefore restricts the walk to objects **present** in the window, from the
``lifecycle`` rows of ``object_changes`` (``active`` -> ``released``); an
object without lifecycle rows (presence unknown, e.g. one imported from
another system) counts as present. Extra ``o2o`` columns, such as a validity
time a user-supplied table carries, are passed through on each row.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..errors import ObjectNotFoundError
from . import schema
from .objects import _ns

if TYPE_CHECKING:
    from .model import EventLog

#: Qualifier never followed by default: ``co_occurs`` is symmetric (time
#: overlap), not a genealogy relation.
_SYMMETRIC = "co_occurs"

_I64_MIN = np.iinfo(np.int64).min
_I64_MAX = np.iinfo(np.int64).max


# ---------------------------------------------------------------------------
# Time windows and presence
# ---------------------------------------------------------------------------


def _bound(value: object, default: int) -> int:
    """int64 UTC nanoseconds of a window bound (naive input is UTC)."""
    if value is None:
        return default
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return default
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.as_unit("ns").value)


def _window(start: object, end: object) -> tuple[int, int] | None:
    """``(lo, hi)`` ns of the half-open window ``[start, end)``; None: no window."""
    if start is None and end is None:
        return None
    lo, hi = _bound(start, _I64_MIN), _bound(end, _I64_MAX)
    if lo >= hi:
        raise ValueError(
            f"empty time window: start={start!r} is not before end={end!r}"
        )
    return lo, hi


def _presence(changes: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """``(oid, type, present)`` for every object with ``lifecycle`` rows.

    A presence runs from an ``active`` row to the next ``released`` row of the
    same object (open-ended when none follows; a ``released`` without a
    preceding ``active`` started before the log). ``present`` is whether one
    presence overlaps ``[lo, hi)``.
    """
    needed = (schema.OCEL_FIELD, schema.OCEL_VALUE, schema.OCEL_TIMESTAMP)
    if changes.empty or any(c not in changes.columns for c in needed):
        return pd.DataFrame({"oid": [], "type": [], "present": []})
    lc = changes[changes[schema.OCEL_FIELD] == "lifecycle"]
    if lc.empty:
        return pd.DataFrame({"oid": [], "type": [], "present": []})
    has_type = schema.OCEL_TYPE in lc.columns
    oid = lc[schema.OCEL_OID].astype(object).to_numpy()
    typ = (
        lc[schema.OCEL_TYPE].astype(object).to_numpy()
        if has_type
        else np.full(len(lc), None, dtype=object)
    )
    oid_code, _ = pd.factorize(oid, use_na_sentinel=False)
    typ_code, typ_uniq = pd.factorize(typ, use_na_sentinel=False)
    key, _ = pd.factorize(
        oid_code.astype(np.int64) * max(len(typ_uniq), 1) + typ_code, sort=False
    )
    first = np.zeros(int(key.max()) + 1, dtype=np.int64)
    first[key[::-1]] = np.arange(len(key))[::-1]
    ts = _ns(lc[schema.OCEL_TIMESTAMP].reset_index(drop=True))
    active = (lc[schema.OCEL_VALUE].astype(object) == "active").to_numpy(dtype=bool)
    # Within one timestamp an ``active`` sorts before its ``released``.
    order = np.lexsort((~active, ts, key))
    k, t, a = key[order], ts[order], active[order]
    same_next = np.zeros(len(k), dtype=bool)
    same_next[:-1] = k[1:] == k[:-1]
    # An ``active`` row runs to the next row of the object when that is a
    # ``released`` row, else it stays open.
    closes = np.zeros(len(k), dtype=bool)
    closes[:-1] = same_next[:-1] & ~a[1:]
    nxt_t = np.full(len(t), _I64_MAX, dtype=np.int64)
    nxt_t[:-1] = t[1:]
    # A ``released`` row with no ``active`` just before it started earlier.
    opened = np.zeros(len(k), dtype=bool)
    opened[1:] = (k[1:] == k[:-1]) & a[:-1]
    lone = ~a & ~opened
    s_start = np.concatenate([t[a], np.full(int(lone.sum()), _I64_MIN)])
    s_end = np.concatenate([np.where(closes[a], nxt_t[a], _I64_MAX), t[lone]])
    s_key = np.concatenate([k[a], k[lone]])
    # Touching the window at an end point is not an overlap.
    hit = (s_start < hi) & (s_end > lo)
    present = np.zeros(len(first), dtype=bool)
    present[s_key[hit]] = True
    return pd.DataFrame({"oid": oid[first], "type": typ[first], "present": present})


def _absent_oids(log: EventLog, window: tuple[int, int] | None) -> pd.Index:
    """Oids with lifecycle rows of which no presence overlaps ``window``."""
    if window is None:
        return pd.Index([], dtype=object)
    pres = _presence(log.object_changes, *window)
    with_lifecycle = pd.Index(pres["oid"].to_numpy(dtype=object)).unique()
    present = pres["oid"].to_numpy(dtype=object)[pres["present"].to_numpy(dtype=bool)]
    return with_lifecycle[~with_lifecycle.isin(present)]


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def _qualifier_mask(o2o: pd.DataFrame, qualifiers: object) -> np.ndarray:
    q = o2o[schema.OCEL_QUALIFIER].astype(object)
    if qualifiers is None:
        return np.asarray((q != _SYMMETRIC).to_numpy(dtype=bool))
    wanted = [qualifiers] if isinstance(qualifiers, str) else list(qualifiers)  # type: ignore[call-overload]
    return np.asarray(q.isin(wanted).to_numpy(dtype=bool))


class _Graph:
    """The ``o2o`` edges as int codes, with CSR adjacency per direction."""

    def __init__(
        self, log: EventLog, qualifiers: object, window: tuple[int, int] | None
    ):
        o2o = log.o2o
        objects = log.objects[log.objects[schema.OCEL_OID].notna()]
        keep = np.zeros(0, dtype=bool)
        if not o2o.empty:
            keep = _qualifier_mask(o2o, qualifiers) & np.asarray(
                (o2o[schema.OCEL_OID].notna() & o2o[schema.OCEL_OID2].notna()),
                dtype=bool,
            )
        self.edges = o2o[keep].reset_index(drop=True)
        qcodes, qnames = pd.factorize(
            self.edges[schema.OCEL_QUALIFIER].astype(object).to_numpy(),
            sort=True,
            use_na_sentinel=False,
        )
        self.qcodes = qcodes.astype(np.int64)
        self.qnames = np.asarray(qnames, dtype=object)
        child = self.edges[schema.OCEL_OID].astype(object).to_numpy()
        parent = self.edges[schema.OCEL_OID2].astype(object).to_numpy()
        obj_oids = objects[schema.OCEL_OID].astype(object).to_numpy()
        codes, uniques = pd.factorize(np.concatenate([obj_oids, child, parent]))
        self.names = pd.Index(uniques, dtype=object)
        n_obj, n_e = len(obj_oids), len(child)
        self.n = len(self.names)
        self.child = codes[n_obj : n_obj + n_e].astype(np.int64)
        self.parent = codes[n_obj + n_e :].astype(np.int64)
        # Type per code: the first type an oid has in ``objects``.
        types = np.full(self.n, None, dtype=object)
        if n_obj:
            first = pd.Series(np.arange(n_obj)).groupby(codes[:n_obj]).first()
            types[first.index.to_numpy()] = (
                objects[schema.OCEL_TYPE].astype(object).to_numpy()[first.to_numpy()]
            )
        self.types = types
        self.allowed = np.ones(self.n, dtype=bool)
        absent = _absent_oids(log, window)
        if len(absent):
            self.allowed[self.names.isin(absent)] = False
        self._csr: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def code(self, oid: object) -> int:
        loc = self.names.get_indexer([oid])[0]
        if loc < 0:
            raise ObjectNotFoundError(
                f"object {oid!r} is not in the event log (neither in objects nor in o2o)"
            )
        return int(loc)

    def adjacency(self, direction: str) -> tuple[np.ndarray, np.ndarray]:
        """``(indptr, edge order)``: the edges leaving each node, by node code."""
        if direction not in self._csr:
            src = self.parent if direction == "used" else self.child
            order = np.argsort(src, kind="stable")
            indptr = np.searchsorted(src[order], np.arange(self.n + 1))
            self._csr[direction] = (indptr, order)
        return self._csr[direction]

    def walk(self, root: int, direction: str, depth: int | None) -> pd.DataFrame:
        """Breadth-first walk from ``root``; one row per edge into a new node."""
        dst_of = self.child if direction == "used" else self.parent
        indptr, order = self.adjacency(direction)
        visited = np.zeros(self.n, dtype=bool)
        visited[root] = True
        frontier = np.array([root], dtype=np.int64)
        parts: list[tuple[np.ndarray, np.ndarray, int]] = []
        level = 0
        if not self.allowed[root]:
            frontier = frontier[:0]
        while len(frontier) and (depth is None or level < depth):
            level += 1
            lo, hi = indptr[frontier], indptr[frontier + 1]
            counts = hi - lo
            total = int(counts.sum())
            if total == 0:
                break
            offsets = np.repeat(lo - np.cumsum(counts) + counts, counts)
            idx = order[offsets + np.arange(total)]
            dst = dst_of[idx]
            ok = ~visited[dst] & self.allowed[dst]
            idx, dst = idx[ok], dst[ok]
            if len(idx) == 0:
                break
            parts.append((idx, dst, level))
            frontier = np.unique(dst)
            visited[frontier] = True
        return self._frame(parts, direction)

    def _frame(
        self, parts: list[tuple[np.ndarray, np.ndarray, int]], direction: str
    ) -> pd.DataFrame:
        extra = [c for c in self.edges.columns if c not in schema.O2O_REQUIRED_COLUMNS]
        if not parts:
            idx = np.zeros(0, dtype=np.int64)
            dst = idx
            lvl = idx
        else:
            idx = np.concatenate([p[0] for p in parts])
            dst = np.concatenate([p[1] for p in parts])
            lvl = np.concatenate(
                [np.full(len(p[0]), p[2], dtype=np.int64) for p in parts]
            )
        via = self.parent[idx] if direction == "used" else self.child[idx]
        q = self.qcodes[idx]
        # Order by depth, oid, via, qualifier (names ranked among the rows'
        # objects only) and drop repeated edges, all on int codes.
        involved = np.unique(np.concatenate([dst, via]))
        rank = np.zeros(self.n, dtype=np.int64)
        rank[involved[self.names[involved].argsort()]] = np.arange(len(involved))
        order = np.lexsort((q, rank[via], rank[dst], lvl))
        idx, dst, lvl, via, q = idx[order], dst[order], lvl[order], via[order], q[order]
        keep = np.ones(len(idx), dtype=bool)
        keep[1:] = (dst[1:] != dst[:-1]) | (via[1:] != via[:-1]) | (q[1:] != q[:-1])
        idx, dst, lvl, via, q = idx[keep], dst[keep], lvl[keep], via[keep], q[keep]
        names = self.names.to_numpy()
        out = pd.DataFrame(
            {
                "oid": pd.array(names[dst], dtype="string"),
                "object_type": pd.array(self.types[dst], dtype="string"),
                "qualifier": pd.array(self.qnames[q], dtype="string"),
                "depth": lvl,
                "via": pd.array(names[via], dtype="string"),
            }
        )
        for c in extra:
            out[c] = self.edges[c].iloc[idx].to_numpy()
        return out


def _check_depth(depth: int | None) -> None:
    if depth is not None and (
        isinstance(depth, bool) or int(depth) != depth or depth < 0
    ):
        raise ValueError(f"depth must be None or a non-negative integer; got {depth!r}")


def _traverse(
    log: EventLog,
    oid: object,
    direction: str,
    depth: int | None,
    start: object,
    end: object,
    qualifiers: object,
) -> pd.DataFrame:
    _check_depth(depth)
    graph = _Graph(log, qualifiers, _window(start, end))
    return graph.walk(graph.code(oid), direction, depth)


def where_used(
    log: EventLog,
    oid: str,
    *,
    depth: int | None = None,
    start: object = None,
    end: object = None,
    qualifiers: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Everything an object went into: its children, their children, ...

    Follows ``o2o`` edges from ``ocel:oid_2`` back to ``ocel:oid``: for
    ``part_of`` (child -> parent) ``where_used(batch)`` returns the batch's
    serials, then the handling units the serials are part of, and so on.

    Args:
        log: The event log.
        oid: The object to start from (an ``ocel:oid``).
        depth: Maximum number of steps; ``None`` follows the graph to the end.
        start: Window start (inclusive); naive times are UTC. With
            ``start`` / ``end`` only objects present in ``[start, end)``
            (``lifecycle`` rows of ``object_changes``) are reached; objects
            without lifecycle rows count as present.
        end: Window end (exclusive).
        qualifiers: The ``o2o`` qualifiers to follow; ``None`` follows every
            qualifier except the symmetric ``co_occurs``.

    Returns:
        One row per edge into a newly reached object, sorted by ``depth``,
        ``oid``, ``via``: ``oid``, ``object_type`` (from ``objects``; NA when
        the oid is only in ``o2o``), ``qualifier`` (of the edge), ``depth``
        (steps from ``oid``, 1 = direct), ``via`` (the object one step closer
        to the start), then any extra ``o2o`` columns of the edge. An object
        reached over two edges at its shortest depth (two parents) has two
        rows; the start object is not included.

    Raises:
        ObjectNotFoundError: When ``oid`` is neither in ``objects`` nor in
            ``o2o`` (a ``KeyError``).
        ValueError: For a negative ``depth`` or an empty window.

    Example:
        >>> import pandas as pd
        >>> from ts_shape.eventlog import EventLog, where_used
        >>> o2o = pd.DataFrame({"ocel:oid": ["S1", "S2", "HU1"],
        ...                     "ocel:oid_2": ["B1", "B1", "S1"],
        ...                     "ocel:qualifier": "part_of"})
        >>> where_used(EventLog(o2o=o2o), "B1")
           oid object_type qualifier  depth via
        0   S1        <NA>   part_of      1  B1
        1   S2        <NA>   part_of      1  B1
        2  HU1        <NA>   part_of      2  S1
    """
    return _traverse(log, oid, "used", depth, start, end, qualifiers)


def where_from(
    log: EventLog,
    oid: str,
    *,
    depth: int | None = None,
    start: object = None,
    end: object = None,
    qualifiers: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Everything an object came from: its parents, their parents, ...

    Follows ``o2o`` edges from ``ocel:oid`` to ``ocel:oid_2``: for ``part_of``
    (child -> parent) ``where_from(serial)`` returns the serial's batch (and
    handling unit), the batch's material, and so on.

    Args:
        log: The event log.
        oid: The object to start from (an ``ocel:oid``).
        depth: Maximum number of steps; ``None`` follows the graph to the end.
        start: Window start (inclusive), as in :func:`where_used`.
        end: Window end (exclusive).
        qualifiers: The ``o2o`` qualifiers to follow; ``None`` follows every
            qualifier except the symmetric ``co_occurs``.

    Returns:
        The columns of :func:`where_used`; ``via`` is the object one step
        closer to the start (the child of ``oid``'s row).

    Raises:
        ObjectNotFoundError: When ``oid`` is neither in ``objects`` nor in
            ``o2o`` (a ``KeyError``).
        ValueError: For a negative ``depth`` or an empty window.
    """
    return _traverse(log, oid, "from", depth, start, end, qualifiers)


def lineage(
    log: EventLog,
    oid: str,
    *,
    depth: int | None = None,
    start: object = None,
    end: object = None,
    qualifiers: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Both directions of an object's genealogy as one frame.

    Args:
        log: The event log.
        oid: The object to start from (an ``ocel:oid``).
        depth: Maximum number of steps in each direction; ``None``: no limit.
        start: Window start (inclusive), as in :func:`where_used`.
        end: Window end (exclusive).
        qualifiers: The ``o2o`` qualifiers to follow; ``None`` follows every
            qualifier except the symmetric ``co_occurs``.

    Returns:
        The columns of :func:`where_used` with a **signed** ``depth``: the
        :func:`where_from` rows with negative depth (``-1`` = direct parent),
        a row for ``oid`` itself at ``0`` (``qualifier`` and ``via`` NA), and
        the :func:`where_used` rows with positive depth. Sorted by ``depth``,
        ``oid``, ``via``. In a cycle an object can be both an ancestor and a
        descendant and then appears on both sides.

    Raises:
        ObjectNotFoundError: When ``oid`` is neither in ``objects`` nor in
            ``o2o`` (a ``KeyError``).
        ValueError: For a negative ``depth`` or an empty window.
    """
    _check_depth(depth)
    graph = _Graph(log, qualifiers, _window(start, end))
    root = graph.code(oid)
    up = graph.walk(root, "from", depth)
    down = graph.walk(root, "used", depth)
    up["depth"] = -up["depth"]
    parts = [up.iloc[::-1]]
    if graph.allowed[root]:
        self_row = down.iloc[:0].reindex([0])
        self_row["oid"] = pd.array([graph.names[root]], dtype="string")
        self_row["object_type"] = pd.array([graph.types[root]], dtype="string")
        self_row["depth"] = 0
        parts.append(self_row)
    parts.append(down)
    out = pd.concat(parts, ignore_index=True)
    out["depth"] = out["depth"].astype(np.int64)
    return out.sort_values(
        ["depth", "oid", "via", "qualifier"], kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sub-logs (EventLog.filter_by_object_type / filter_by_time)
# ---------------------------------------------------------------------------


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    """``(ocel:oid, ocel:type)`` of each row."""
    return pd.MultiIndex.from_arrays(
        [
            frame[schema.OCEL_OID].astype(object).to_numpy(),
            frame[schema.OCEL_TYPE].astype(object).to_numpy(),
        ]
    )


def _pairs_in(frame: pd.DataFrame, keys: pd.MultiIndex) -> np.ndarray:
    """Whether each row's ``(oid, type)`` is one of ``keys``."""
    if frame.empty or len(keys) == 0:
        return np.zeros(len(frame), dtype=bool)
    return np.asarray(_key_index(frame).isin(keys), dtype=bool)


def _sub_log(
    log: EventLog,
    events: pd.DataFrame,
    objects: pd.DataFrame,
    changes: pd.DataFrame,
) -> EventLog:
    """Assemble a sub-log whose relations / o2o / changes reference kept rows only."""
    from .model import EventLog

    isin = schema._isin
    kept = _key_index(objects)
    rel = log.relations
    rel = rel[isin(rel[schema.OCEL_EID], events[schema.OCEL_EID])]
    rel = rel[_pairs_in(rel, kept)]
    oids = objects[schema.OCEL_OID]
    o2o = log.o2o[
        isin(log.o2o[schema.OCEL_OID], oids) & isin(log.o2o[schema.OCEL_OID2], oids)
    ]
    if schema.OCEL_TYPE in changes.columns:
        # Changes belong to an (oid, type) object, as in filter_by_object.
        changes = changes[_pairs_in(changes, kept)]
    else:
        changes = changes[isin(changes[schema.OCEL_OID], oids)]
    return EventLog(
        events.reset_index(drop=True),
        objects.reset_index(drop=True),
        rel.reset_index(drop=True),
        o2o.reset_index(drop=True),
        changes.reset_index(drop=True),
    )


def _filter_by_object_type(
    log: EventLog,
    types: str | Iterable[str],
    keep_events: Literal["linked", "all"] = "linked",
) -> EventLog:
    if keep_events not in ("linked", "all"):
        raise ValueError(f"keep_events must be 'linked' or 'all'; got {keep_events!r}")
    wanted = [types] if isinstance(types, str) else list(types)
    objects = log.objects[log.objects[schema.OCEL_TYPE].astype(object).isin(wanted)]
    events = log.events
    if keep_events == "linked":
        rel = log.relations
        linked = rel[_pairs_in(rel, _key_index(objects))]
        events = events[schema._isin(events[schema.OCEL_EID], linked[schema.OCEL_EID])]
    return _sub_log(log, events, objects, log.object_changes)


def _filter_by_time(log: EventLog, start: object, end: object) -> EventLog:
    window = _window(start, end)
    if window is None:
        return _sub_log(log, log.events, log.objects, log.object_changes)
    lo, hi = window
    ev = log.events
    ts = _ns(ev[schema.OCEL_TIMESTAMP].reset_index(drop=True))
    events = ev[(ts >= lo) & (ts < hi)]

    # Objects: present in the window, presence unknown, or used by a kept event.
    objects = log.objects
    pres = _presence(log.object_changes, lo, hi)
    keep = np.ones(len(objects), dtype=bool)
    if not pres.empty:
        lifecycle = pd.MultiIndex.from_arrays(
            [pres["oid"].to_numpy(dtype=object), pres["type"].to_numpy(dtype=object)]
        )
        present = lifecycle[pres["present"].to_numpy(dtype=bool)]
        keep = ~_pairs_in(objects, lifecycle) | _pairs_in(objects, present)
    rel = log.relations
    rel = rel[schema._isin(rel[schema.OCEL_EID], events[schema.OCEL_EID])]
    if not rel.empty:
        keep |= _pairs_in(objects, _key_index(rel))
    objects = objects[keep]

    # Changes inside the window, plus each field's last value before it (the
    # state at the window start).
    ch = log.object_changes
    if not ch.empty:
        cts = _ns(ch[schema.OCEL_TIMESTAMP].reset_index(drop=True))
        inside = (cts >= lo) & (cts < hi)
        before = np.flatnonzero(cts < lo)
        last = np.zeros(len(ch), dtype=bool)
        if len(before):
            group = [
                c
                for c in (schema.OCEL_OID, schema.OCEL_TYPE, schema.OCEL_FIELD)
                if c in ch.columns
            ]
            codes = [
                pd.factorize(ch[c].to_numpy(dtype=object)[before])[0] for c in group
            ]
            # Latest row per (oid, type, field): sort by key then time.
            order = np.lexsort((cts[before], *codes[::-1]))
            ordered = [c[order] for c in codes]
            differs = np.zeros(len(order), dtype=bool)
            differs[-1] = True
            for c in ordered:
                differs[:-1] |= c[1:] != c[:-1]
            last[before[order[differs]]] = True
        ch = ch[inside | last]
    return _sub_log(log, events, objects, ch)
