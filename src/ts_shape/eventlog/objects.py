"""Object detection — turn identifier-bearing timeseries signals into OCEL 2.0
objects, relations, and time-varying attributes.

The event packs answer *"what happened?"*. This module answers *"to which
**things**?"* — the batches, serials, coils, recipes, tools, materials, drums,
bundles, families, customer part-numbers (and any other object out there) that
events refer to.

It is deliberately **one generic, declarative layer**, not another family of
detector classes: you describe which signals carry which object type with an
:class:`ObjectSpec` (or discover them from metadata), and a run-length kernel
(the same value-change segmentation that powers batch/traceability detection)
extracts every object's presence interval. New object types are added by *data /
config*, never by code.

Three steps, all feeding the OCEL 2.0 tables on :class:`~ts_shape.eventlog.model.EventLog`:

* :func:`object_intervals` — segment id signals into ``(oid, type, start, end)``
  presence intervals (an id is present until the next id on the same signal).
* :func:`detect_objects` — build the ``objects`` / ``o2o`` / ``object_changes``
  tables. ``part_of`` is asserted only along a declared ``hierarchy`` of object
  types or id signals (it cannot be inferred from time alone); usage links
  (``made_from`` / ``made_with`` / ``made_by``) along declared ``uses``; the
  symmetric ``co_occurs`` for every other overlap is opt-in
  (``infer_o2o="co_occurs"``).
* :func:`attach_objects` — link an existing event log's events to the detected
  objects by temporal overlap, optionally within a ``scope`` (e.g. a line), so
  the output of *every* event detector gains rich object references with no
  per-detector changes.

When you already hold the two lists — an **event list** and an **object list**
(intervals) — the relationship derivation is automated and needs no raw signals:

* :func:`link_events_to_objects` — all event-to-object (E2O) relations, by
  temporal overlap and/or direct ``key_columns`` matching.
* :func:`derive_o2o` — all object-to-object (O2O) relations (declared
  ``part_of`` hierarchy, optionally the symmetric ``co_occurs``).
* :func:`relate` — assemble both, plus objects and ``object_changes``, into one
  validated :class:`EventLog` ready to ingest into an OCEL 2.0 database.

Presence semantics (shared by every function here)
--------------------------------------------------
* An id is present from its first sample until the **next value** on the same
  signal: the next id, or an empty / missing reading (``""``, whitespace,
  ``NaN`` / ``None`` / ``<NA>``), which means "no object". A gap in the data
  does not end an object; only a different value does.
* A misread (``NOREAD``, ``#N/A``, ``0``, ``65535``, ``"????????"``, a
  truncated id, ...; see :attr:`ObjectSpec.exclude`) is not an id: between two
  readings of the same id it is a gap (the id is held across it), anywhere
  else it is an empty reading.
* Presence is **half-open** ``[start, end)``: at a hand-over instant the
  incoming object owns the timestamp, never both. The last presence on a signal
  (and, in a user-supplied table, any presence no other object of its signal /
  type takes over from) is closed ``[start, end]``, its end being the last
  observation.
* The same id appearing again later is the **same object** (one row in
  ``objects``) with several presence intervals.
* With a change flag (``is_delta``, see :attr:`ObjectSpec.delta_column`) on
  a signal with both flagged and unflagged rows, a flagged reading that
  repeats the id before it is a re-setup and starts a new presence of the
  same object.
* With a cycle signal (:attr:`ObjectSpec.cycle_uuid`) a presence spans the
  machine cycle ``[rise, fall)`` its first reading falls in.
* All timestamps are UTC (OCEL 2.0); tz-aware input is converted, naive input
  is taken as UTC.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from ts_shape.utils.deprecation import (
    caller_stacklevel,
    renamed_kwargs,
)

from ..errors import ColumnNotFoundError, DataQualityWarning
from ..utils.frames import require_columns, to_ns
from ..utils.id_codes import is_misread
from . import schema
from .concat import concat
from .model import EventLog

# Tidy intermediate columns (used between the three steps).
_OID = "oid"
_TYPE = "type"
_START = "start"
_END = "end"
_DURATION = "duration_seconds"
_UUID = "uuid"
_SOURCE = "source_uuid"

#: Value columns that can carry an object id.
_ID_VALUE_COLUMNS: tuple[str, ...] = ("value_string", "value_integer", "value_double")

#: Columns of the interval table that are not object attributes.
_RESERVED: tuple[str, ...] = (_OID, _TYPE, _START, _END, _DURATION, _UUID, _SOURCE)

_I64: TypeAlias = "npt.NDArray[np.int64]"
_BOOL: TypeAlias = "npt.NDArray[np.bool_]"
_NAT = np.iinfo(np.int64).min  # int64 view of NaT


def _empty_intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            _OID: pd.Series(dtype="string"),
            _TYPE: pd.Series(dtype="string"),
            _START: pd.Series(dtype="datetime64[ns, UTC]"),
            _END: pd.Series(dtype="datetime64[ns, UTC]"),
            _DURATION: pd.Series(dtype="float64"),
            _UUID: pd.Series(dtype="string"),
            _SOURCE: pd.Series(dtype="string"),
        }
    )


def _to_utc(values: object) -> pd.Series:
    """Coerce timestamps to a tz-aware UTC ``datetime64[ns]`` Series."""
    s = pd.to_datetime(pd.Series(values))
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    return s.dt.as_unit("ns")


def _ns(values: pd.Series) -> _I64:
    """int64 nanoseconds of a UTC datetime Series (NaT -> ``_NAT``)."""
    return to_ns(_to_utc(values))


def _from_ns(values: _I64) -> pd.Series:
    return pd.Series(
        np.asarray(values, dtype=np.int64).view("datetime64[ns]")
    ).dt.tz_localize("UTC")


def _id_codes(raw: pd.Series) -> tuple[_I64, pd.Index]:
    """Factorize an id column; empty / missing readings get code ``-1``.

    A float column whose values are all whole numbers (an integer id column
    that picked up ``NaN``) is read as integers, so ``1001`` never becomes
    ``"1001.0"``.
    """
    s = raw
    if pd.api.types.is_float_dtype(s.dtype):
        finite = s.dropna()
        if bool(np.all(np.mod(finite.to_numpy(dtype=float), 1) == 0)):
            s = s.astype("Int64")
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    codes = np.array(codes, dtype=np.int64)  # writable copy
    if len(uniques) and (
        pd.api.types.is_object_dtype(uniques.dtype)
        or pd.api.types.is_string_dtype(uniques.dtype)
    ):
        blank = np.array(
            [isinstance(v, str) and v.strip(" \t\r\n\x00") == "" for v in uniques],
            dtype=bool,
        )
        if blank.any():
            codes[blank[codes.clip(min=0)] & (codes >= 0)] = -1
    return codes, pd.Index(uniques)


def _rle(
    codes: _I64, first: _I64, last: _I64, forced: _BOOL | None = None
) -> tuple[_I64, _I64, _I64, _I64]:
    """Run-length encode ``codes`` (one entry per sample or per sub-segment).

    Returns ``(codes, start, end, first_index)`` per run. A run ends where the
    next run starts; the final run ends at its last observation. ``forced``
    marks entries that start a new run although the code repeats (a
    same-value re-setup).
    """
    n = len(codes)
    brk = np.ones(n, dtype=bool)
    brk[1:] = codes[1:] != codes[:-1]
    if forced is not None:
        brk[1:] |= forced[1:]
    idx = np.flatnonzero(brk)
    starts = first[idx]
    last_idx = np.append(idx[1:] - 1, n - 1)
    ends = np.append(starts[1:], last[last_idx[-1:]])
    return codes[idx], starts, ends, idx


def _fill_excluded(codes: _I64, excluded: _BOOL) -> _I64:
    """Replace excluded readings: a run of them between two readings of the
    same id takes that id (a gap in the data), anywhere else it becomes an
    empty reading (code ``-1``, the presence ends there)."""
    bad = excluded & (codes >= 0)
    if not bad.any():
        return codes
    n = len(codes)
    pos = np.arange(n, dtype=np.int64)
    prev = np.maximum.accumulate(np.where(bad, -1, pos))
    nxt = np.minimum.accumulate(np.where(bad, n, pos)[::-1])[::-1]
    prev_code = np.where(prev >= 0, codes[prev.clip(min=0)], -1)
    next_code = np.where(nxt < n, codes[nxt.clip(max=n - 1)], -1)
    fill = (prev_code == next_code) & (prev_code >= 0)
    out = codes.copy()
    out[bad] = np.where(fill, prev_code, -1)[bad]
    return out


def _segments(
    times: _I64,
    values: pd.Series,
    *,
    min_duration: str | None,
    excluded: _BOOL | None = None,
    delta: _BOOL | None = None,
) -> tuple[_I64, _I64, _I64, _I64, pd.Index, bool]:
    """Presence runs of one id signal (``times`` sorted, ``values`` aligned).

    Returns ``(codes, start_ns, end_ns, first_row, uniques, still_set)`` for
    the id runs only (empty / missing runs are boundaries, not objects).
    ``first_row`` is the positional row of each run's first sample;
    ``still_set`` tells whether the last reading is an id. ``excluded`` marks
    readings that are not ids (see :func:`_fill_excluded`). ``delta`` marks
    readings where the id was set: one that repeats the reading just before
    it (both ids, neither excluded) starts a new run.
    """
    codes, uniques = _id_codes(values.reset_index(drop=True))
    forced: _BOOL | None = None
    if delta is not None and len(codes) > 1:
        forced = np.zeros(len(codes), dtype=bool)
        same = (codes[1:] == codes[:-1]) & (codes[1:] >= 0)
        if excluded is not None:
            same &= ~excluded[1:] & ~excluded[:-1]
        forced[1:] = delta[1:] & same
        if not forced.any():
            forced = None
    if excluded is not None:
        codes = _fill_excluded(codes, excluded)
    still_set = bool(len(codes)) and bool(codes[-1] >= 0)
    rows = np.arange(len(codes), dtype=np.int64)
    c, s, e, first = _rle(codes, times, times, forced)
    run_forced = forced[first] if forced is not None else None
    first = rows[first]

    if min_duration is not None and len(c) > 1:
        # Debounce: an id run shorter than ``min_duration`` is a blip. It is
        # removed and its neighbours close over it (and merge when they carry
        # the same id). The final run is open-ended (its true duration is
        # unknown), and empty / missing runs are boundaries — neither is
        # dropped. Blips are removed shortest first and equal neighbours are
        # merged before the next check, so in A-X-A a short X is removed
        # and the merged A is judged on its full length.
        threshold = pd.Timedelta(min_duration).value
        while len(c) > 1:
            dur = e - s
            short = (c >= 0) & (dur < threshold)
            short[-1] = False
            if not short.any():
                break
            # Drop the short runs that are no longer than a short neighbour
            # (local minima); the shortest one always qualifies.
            big = np.iinfo(np.int64).max
            d = np.where(short, dur, big)
            left = np.append(big, d[:-1])
            right = np.append(d[1:], big)
            drop = short & (d < left) & (d <= right)
            keep = ~drop
            c, s, last, first = c[keep], s[keep], e[keep], first[keep]
            if run_forced is not None:
                run_forced = run_forced[keep]
            c, s, e, k = _rle(c, s, last, run_forced)
            first = first[k]
            if run_forced is not None:
                run_forced = run_forced[k]

    ids = c >= 0
    return c[ids], s[ids], e[ids], first[ids], uniques, still_set


def _id_text(value: object) -> str:
    """An object id as text; an integral float renders without ``.0``."""
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _render_oids(values: pd.Index, template: str, object_type: str) -> pd.Series:
    texts = [_id_text(v) for v in values]
    if template == "{value}":
        out = texts
    elif template == "{type}:{value}":
        out = [f"{object_type}:{v}" for v in texts]
    else:
        out = [template.format(value=v, type=object_type) for v in texts]
    return pd.Series(out, dtype="string")


#: :attr:`ObjectSpec.exclude` default: drop the readings
#: :func:`~ts_shape.utils.id_codes.is_misread` flags.
MISREADS = "misreads"

#: What :attr:`ObjectSpec.exclude` accepts.
Exclude: TypeAlias = "str | Collection[object] | Callable[[pd.Series], object] | None"


@dataclass(frozen=True)
class ObjectSpec:
    """Declares that one signal carries the identity of one object type.

    Attributes:
        uuid: The signal (``uuid`` value) whose value column is the object id.
        object_type: OCEL object type, e.g. ``"batch"``, ``"coil"``, ``"serial"``,
            ``"recipe"``. Auto-registered via
            :func:`~ts_shape.eventlog.schema.register_object_type` if unknown.
        value_column: Column carrying the id (``"value_string"`` or
            ``"value_integer"``). Empty strings and missing values mean
            "no object present".
        min_duration: Optional minimum presence (e.g. ``"1min"``, measured
            start to next value); shorter blips are dropped and the presence
            around them closes over the gap. The last id on a signal is
            open-ended and never dropped.
        id_template: How to render the object id from the raw value. Supports
            ``{value}`` and ``{type}`` — default ``"{value}"``; use
            ``"{type}:{value}"`` to namespace ids across types.
        attributes: ``{attribute_name: signal_uuid}`` — the value in effect at
            each presence start (last non-missing sample at or before it) plus
            every change of that signal during the presence, emitted as
            ``object_changes``.
        exclude: Readings that are not ids (scanner no-reads, PLC defaults).
            ``"misreads"`` (default) applies
            :func:`ts_shape.utils.id_codes.is_misread` -- ``NOREAD``,
            ``#N/A``, ``NULL``, ``ERR``, ``0``, ``-1``, ``32767``, ``65535``,
            ``"????????"``, fillers such as ``"XXXXXXXX"``, truncated reads.
            A collection of values excludes exactly those values (compared
            as text, so ``0`` matches ``"0"``); a callable takes the signal's
            id values in time order and returns a boolean mask; ``None``
            excludes nothing. An excluded reading never becomes an object and
            never splits a presence: between two readings of the same id it
            is a gap in the data (the id is held across it); anywhere else it
            is an empty reading (the presence before it ends there).
        delta_column: Column flagging the rows where the id was *set* (the
            historian's ``is_delta``: ``True`` when the operator scans or
            sets up the id, ``False`` on the periodic repeats). When the
            column exists and the signal has both ``True`` and ``False``
            rows (hold + repeat: the flag tells changes from snapshots), a
            flagged row whose value equals the reading before it starts a
            new presence (a re-setup with the same material number). On a
            signal whose rows are all ``True`` (a change-only store, where
            an unchanged row is a re-send) or all ``False`` / missing, the
            flag is not used. A value
            change starts a new presence with or without the flag. A flag
            right after a misread or an empty reading adds nothing (the
            change it marks is the one back from the misread), so
            ``A, NOREAD, A`` stays one presence. A missing flag never starts
            a presence and is not a ``False`` row. ``None`` ignores the
            column (value changes only).
        cycle_uuid: A bool signal whose ``True`` stretches are the machine
            cycles (e.g. "door closed"). Each presence is assigned to the
            cycle ``[rise, fall)`` its first reading (the scan) falls in and
            then spans exactly that cycle: the serial scanned mid-cycle is
            the part made in that cycle. Several presences of one id in one
            cycle are one presence. A presence that starts outside every
            cycle keeps its own span (scan to next change), with one
            :class:`~ts_shape.errors.DataQualityWarning` per call. A cycle
            still ``True`` at the cycle signal's last reading ends at the
            last timestamp of the frame. ``None`` (default): no cycles.
        cycle_value_column: Value column of ``cycle_uuid`` (default
            ``"value_bool"``; numbers count as ``True`` when non-zero, text
            when it reads ``true`` / ``1`` / ``on`` / ``yes``).
    """

    uuid: str
    object_type: str
    value_column: str = "value_string"
    min_duration: str | None = None
    id_template: str = "{value}"
    attributes: Mapping[str, str] = field(default_factory=dict)
    exclude: Exclude = MISREADS
    delta_column: str | None = "is_delta"
    cycle_uuid: str | None = None
    cycle_value_column: str = "value_bool"


def _exclusion_mask(spec: ObjectSpec, values: pd.Series) -> _BOOL | None:
    """Boolean mask of the readings ``spec.exclude`` removes (``None``: none)."""
    rule = spec.exclude
    if rule is None:
        return None
    if isinstance(rule, str):
        if rule != MISREADS:
            raise ValueError(
                f"ObjectSpec({spec.uuid!r}).exclude={rule!r}: the only text value "
                f"is {MISREADS!r}; pass a list to exclude single values"
            )
        return is_misread(values)
    if callable(rule):
        mask = np.asarray(rule(values), dtype=bool)
        if mask.shape != (len(values),):
            raise ValueError(
                f"ObjectSpec({spec.uuid!r}).exclude returned a mask of shape "
                f"{mask.shape}; expected ({len(values)},)"
            )
        return mask
    texts = {_id_text(v).strip() for v in rule}
    na = values.isna().to_numpy(dtype=bool)
    return np.array(
        [
            not m and _id_text(v).strip() in texts
            for v, m in zip(values, na, strict=True)
        ],
        dtype=bool,
    )


_TRUE_TEXT = frozenset({"true", "1", "on", "yes", "t", "y"})


def _truthy(values: pd.Series) -> _BOOL:
    """``True`` where a flag / bool reading is set (missing -> ``False``)."""
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    codes = np.asarray(codes, dtype=np.int64)
    truth = np.array(
        [
            v.strip().lower() in _TRUE_TEXT if isinstance(v, str) else bool(v)
            for v in uniques
        ],
        dtype=bool,
    )
    if len(truth) == 0:
        return np.zeros(len(codes), dtype=bool)
    return np.asarray((codes >= 0) & truth[codes.clip(min=0)], dtype=bool)


def _delta_mask(spec: ObjectSpec, df: pd.DataFrame, rows: _I64) -> _BOOL | None:
    """``spec.delta_column`` of the signal's rows (``None``: no flag in use).

    The flag tells changes from snapshots only on a signal that has both
    ``True`` and ``False`` rows (hold + repeat). A signal whose rows are all
    ``True`` (a change-encoded store: an unchanged ``True`` row is a
    re-send) or all ``False`` / missing gives ``None``.
    """
    col = spec.delta_column
    if col is None or col not in df.columns:
        return None
    flags = df[col].iloc[rows]
    mask = _truthy(flags)
    if not mask.any():
        return None
    has_false = bool((flags.notna().to_numpy(dtype=bool) & ~mask).any())
    return mask if has_false else None


def _cycles(
    signals: _Signals, spec: ObjectSpec, data_end_ns: int
) -> tuple[_I64, _I64, _BOOL]:
    """``(rise, fall, open_end)`` of the ``True`` stretches of the cycle signal.

    Missing readings are dropped (the state is held). A first reading that
    is ``True`` starts a cycle there; a cycle still ``True`` at the last
    reading ends at ``data_end_ns`` and is flagged ``open_end``.
    """
    rows = signals.rows(spec.cycle_uuid)
    vals = signals.df[spec.cycle_value_column].iloc[rows]
    ok = vals.notna().to_numpy(dtype=bool)
    t = signals.times[rows][ok]
    b = _truthy(vals[ok])
    prev = np.r_[False, b[:-1]]
    rise = t[b & ~prev]
    fall = t[~b & prev]
    open_end = np.zeros(len(rise), dtype=bool)
    if len(fall) < len(rise):
        fall = np.append(fall, max(data_end_ns, int(rise[-1])))
        open_end[-1] = True
    return rise.astype(np.int64), fall.astype(np.int64), open_end


def _assign_cycles(
    codes: _I64,
    start: _I64,
    end: _I64,
    first: _I64,
    cycles: tuple[_I64, _I64, _BOOL],
) -> tuple[_I64, _I64, _I64, _I64, int, int]:
    """Presences spanning the cycle their first reading falls in.

    Returns the new ``(codes, start, end, first)`` sorted by start, the
    number of presences outside every cycle, and the number of cycles that
    hold more than one id.
    """
    rise, fall, open_end = cycles
    n = len(codes)
    k = np.searchsorted(rise, start, side="right") - 1
    kc = k.clip(min=0)
    inside = (k >= 0) & ((start < fall[kc]) | (open_end[kc] & (start <= fall[kc])))
    if len(rise) == 0:
        inside = np.zeros(n, dtype=bool)
    cyc = np.where(inside, k, -1 - np.arange(n, dtype=np.int64))
    keep = ~pd.DataFrame({"c": codes, "k": cyc}).duplicated().to_numpy(dtype=bool)
    new_start = np.where(inside, rise[kc], start)
    new_end = np.where(inside, fall[kc], end)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(new_start[idx], kind="stable")]
    in_k = cyc[idx][inside[idx]]
    multi = int((np.bincount(in_k)[np.unique(in_k)] > 1).sum()) if len(in_k) else 0
    outside = int((~inside).sum())
    return codes[idx], new_start[idx], new_end[idx], first[idx], outside, multi


def _check_frame(df: pd.DataFrame, uuid_column: str, time_column: str) -> None:
    require_columns(df, (uuid_column, time_column), where="object detection")


class _Signals:
    """The frame grouped by signal once: time-sorted rows per uuid."""

    def __init__(self, df: pd.DataFrame, uuid_column: str, time_column: str):
        self.df = df
        times = _ns(df[time_column]) if len(df) else np.empty(0, dtype=np.int64)
        codes, uniques = pd.factorize(df[uuid_column], use_na_sentinel=True)
        codes = np.asarray(codes, dtype=np.int64)
        valid = np.flatnonzero((times != _NAT) & (codes >= 0))
        order = valid[np.lexsort((times[valid], codes[valid]))]
        self.times = times
        self.order = order
        self.bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
        self.code_of = {u: i for i, u in enumerate(uniques)}
        finite = times[times != _NAT]
        self.data_end_ns: int | None = int(finite.max()) if len(finite) else None

    def __contains__(self, uuid: object) -> bool:
        return uuid in self.code_of

    def available(self) -> list[object]:
        return list(self.code_of)

    def rows(self, uuid: object) -> _I64:
        """Positional rows of ``uuid`` with a timestamp, in time order."""
        code = self.code_of.get(uuid)
        if code is None:
            return np.empty(0, dtype=np.int64)
        return self.order[self.bounds[code] : self.bounds[code + 1]]


def _detect(
    df: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    uuid_column: str,
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interval table plus the in-presence attribute changes.

    The frame is grouped by signal once (:class:`_Signals`); every spec then
    reads its time-sorted rows by position.
    """
    _check_frame(df, uuid_column, time_column)
    frames: list[pd.DataFrame] = []
    changes: list[pd.DataFrame] = []
    signals = _Signals(df, uuid_column, time_column) if not df.empty else None
    data_end_ns = signals.data_end_ns if signals is not None else None
    for spec in specs:
        if not schema.is_known_object_type(spec.object_type):
            schema.register_object_type(spec.object_type)
        clash = set(spec.attributes) & set(_RESERVED)
        if clash:
            raise ValueError(
                f"ObjectSpec({spec.uuid!r}) attribute name(s) {sorted(clash)} "
                f"collide with reserved interval columns {list(_RESERVED)}"
            )
        if signals is None:
            continue
        if spec.value_column not in df.columns:
            raise ColumnNotFoundError(
                f"value column {spec.value_column!r} of ObjectSpec({spec.uuid!r}) "
                f"not found; available columns: {list(df.columns)}"
            )
        if spec.uuid not in signals:
            warnings.warn(
                f"ObjectSpec signal {spec.uuid!r} is not in the frame; no "
                f"{spec.object_type!r} objects detected from it. Available "
                f"uuids: {sorted(map(str, signals.available()))[:20]}",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            continue

        cycles = None
        if spec.cycle_uuid is not None:
            if spec.cycle_uuid not in signals:
                raise ValueError(
                    f"ObjectSpec({spec.uuid!r}).cycle_uuid={spec.cycle_uuid!r} "
                    "is not a signal of the frame"
                )
            if spec.cycle_value_column not in df.columns:
                raise ColumnNotFoundError(
                    f"cycle value column {spec.cycle_value_column!r} of "
                    f"ObjectSpec({spec.uuid!r}) not found; available columns: "
                    f"{list(df.columns)}"
                )
            cycles = _cycles(signals, spec, data_end_ns or 0)

        rows = signals.rows(spec.uuid)
        if len(rows) == 0:
            continue
        values = df[spec.value_column].iloc[rows].reset_index(drop=True)
        codes, start, end, first, uniques, still_set = _segments(
            signals.times[rows],
            values,
            min_duration=spec.min_duration,
            excluded=_exclusion_mask(spec, values),
            delta=_delta_mask(spec, df, rows),
        )
        if len(codes) == 0:
            continue
        # A signal that still carries an id at its last reading (change
        # encoded: the id is not re-sent) keeps that object present until the
        # end of the data, like the last segment of SegmentExtractor.
        if still_set and data_end_ns is not None and end[-1] < data_end_ns:
            end = end.copy()
            end[-1] = data_end_ns
        if cycles is not None:
            codes, start, end, first, outside, multi = _assign_cycles(
                codes, start, end, first, cycles
            )
            if outside:
                warnings.warn(
                    f"ObjectSpec({spec.uuid!r}): {outside} presence(s) start "
                    f"outside every cycle of {spec.cycle_uuid!r}; they keep "
                    "their own span (reading to next change)",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )
            if multi:
                warnings.warn(
                    f"ObjectSpec({spec.uuid!r}): {multi} cycle(s) of "
                    f"{spec.cycle_uuid!r} hold more than one id; each id "
                    "gets the whole cycle",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )

        oids = _render_oids(uniques, spec.id_template, spec.object_type)
        oid = oids.to_numpy()[codes]
        if _SOURCE in df.columns:
            source = df[_SOURCE].iloc[rows[first]].astype("string").to_numpy()
        else:
            source = pd.array([pd.NA] * len(codes), dtype="string")
        out = pd.DataFrame(
            {
                _OID: pd.array(oid, dtype="string"),
                _TYPE: pd.array([spec.object_type] * len(codes), dtype="string"),
                _START: _from_ns(start),
                _END: _from_ns(end),
                _DURATION: (end - start) / 1e9,
                _UUID: pd.array([spec.uuid] * len(codes), dtype="string"),
                _SOURCE: pd.array(source, dtype="string"),
            }
        )
        is_last = np.zeros(len(codes), dtype=bool)
        is_last[-1] = True
        for attr_name, attr_uuid in spec.attributes.items():
            at_start, during = _capture_attribute(
                signals,
                attr_uuid,
                start,
                end,
                is_last,
                value_column=spec.value_column,
            )
            out[attr_name] = at_start
            if len(during):
                pos = during["_pos"].to_numpy()
                changes.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_OID: out[_OID].to_numpy()[pos],
                            schema.OCEL_TYPE: spec.object_type,
                            schema.OCEL_FIELD: attr_name,
                            schema.OCEL_VALUE: during["_value"].to_numpy(),
                            schema.OCEL_TIMESTAMP: _from_ns(
                                during["_ts"].to_numpy()
                            ).to_numpy(),
                        }
                    )
                )
        frames.append(out)

    if not frames:
        intervals = _empty_intervals()
    else:
        intervals = pd.concat(frames, ignore_index=True)
    extra = (
        pd.concat(changes, ignore_index=True)
        if changes
        else schema.empty_object_changes()
    )
    return intervals, extra


@renamed_kwargs(df="dataframe")
def object_intervals(
    dataframe: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    uuid_column: str = "uuid",
    time_column: str = "systime",
) -> pd.DataFrame:
    """Extract one presence interval per object presence from id signals.

    Returns a tidy frame with columns ``oid``, ``type``, ``start``, ``end``,
    ``duration_seconds``, ``uuid`` (the id signal), ``source_uuid`` (of the id
    signal, when the frame has that column) plus one column per captured
    attribute (its value at the presence start). Times are UTC.

    Each id signal is run-length encoded: an id is present from its first
    sample until the next value on the same signal (next id, or an empty /
    missing reading). A signal whose last reading is still an id keeps that
    object present until the last timestamp of the frame. An id that
    re-appears later yields another row with the same ``oid``. Readings
    excluded by :attr:`ObjectSpec.exclude` (misreads by default) are never
    objects and do not split a presence. A reading flagged in
    :attr:`ObjectSpec.delta_column` (``is_delta``) that repeats the id before
    it starts a new presence (a re-setup). With :attr:`ObjectSpec.cycle_uuid`
    each presence spans the cycle its first reading falls in, sorted by
    start. An empty frame or no matching signal gives an empty frame with
    the full column set.

    Raises:
        ValueError: When a spec's ``cycle_uuid`` is not in the frame.
    """
    intervals, _ = _detect(
        dataframe, specs, uuid_column=uuid_column, time_column=time_column
    )
    return intervals


def _capture_attribute(
    signals: _Signals,
    attr_uuid: str,
    starts: _I64,
    ends: _I64,
    is_last: _BOOL,
    *,
    value_column: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Attribute value at each presence start, and its changes during presence.

    The value at a start is the last non-missing sample at or before it
    (backward as-of). A change is a sample whose value differs from the
    previous non-missing one and that falls strictly after a presence start
    and before its end (at or before it, for the last presence).
    """
    empty_during = pd.DataFrame({"_pos": [], "_value": [], "_ts": []})
    rows = signals.rows(attr_uuid)
    if len(rows) == 0:
        return pd.Series([pd.NA] * len(starts), dtype="object"), empty_during
    sig = signals.df.iloc[rows]
    # Pick the first populated value column for the attribute signal.
    col = next(
        (
            c
            for c in (value_column, "value_string", "value_double", "value_integer")
            if c in sig.columns and sig[c].notna().any()
        ),
        value_column,
    )
    if col not in sig.columns:
        return pd.Series([pd.NA] * len(starts), dtype="object"), empty_during
    keep = sig[col].notna().to_numpy(dtype=bool)
    t = signals.times[rows[keep]]
    vals = sig[col].to_numpy(dtype=object)[keep]
    if len(t) == 0:
        return pd.Series([pd.NA] * len(starts), dtype="object"), empty_during

    pos = np.searchsorted(t, starts, side="right") - 1
    at_start = np.where(pos >= 0, vals[pos.clip(min=0)], pd.NA)

    changed = np.ones(len(vals), dtype=bool)
    changed[1:] = vals[1:] != vals[:-1]
    ct, cv = t[changed], vals[changed]
    owner = np.searchsorted(starts, ct, side="right") - 1
    ok = owner >= 0
    o = owner.clip(min=0)
    inside = ok & (ct > starts[o]) & ((ct < ends[o]) | (is_last[o] & (ct == ends[o])))
    during = pd.DataFrame({"_pos": o[inside], "_value": cv[inside], "_ts": ct[inside]})
    return pd.Series(at_start, dtype="object"), during


#: ``hierarchy`` of :func:`detect_objects`: child -> parent(s), each an object
#: type or an id-signal uuid.
Hierarchy: TypeAlias = "Mapping[str, str | Sequence[str]]"
#: ``uses`` of :func:`detect_objects`: child -> {parent: qualifier}, each an
#: id-signal uuid or an object type (e.g. ``{"serial_sig": {"mat_sig":
#: "made_from", "tool_sig": "made_with"}}``).
Uses: TypeAlias = "Mapping[str, Mapping[str, str]]"
#: ``infer_o2o`` of :func:`detect_objects`.
InferO2O: TypeAlias = "bool | Literal['hierarchy', 'co_occurs']"
#: ``scope`` of :func:`link_events_to_objects`: signal uuid -> scope label.
Scope: TypeAlias = "Mapping[str, object]"

#: Default ``fanout_threshold``: an event normally refers to one object of a
#: type at a time.
FANOUT_THRESHOLD = 1


def _o2o_mode(infer_o2o: object) -> tuple[bool, bool]:
    """``(derive o2o, include co_occurs)`` from an ``infer_o2o`` value."""
    if infer_o2o is True or infer_o2o == "hierarchy":
        return True, False
    if infer_o2o == "co_occurs":
        return True, True
    if infer_o2o is False or infer_o2o is None:
        return False, False
    raise ValueError(
        f"infer_o2o={infer_o2o!r}; use True / 'hierarchy' (declared part_of "
        "edges), 'co_occurs' (also every overlap of two types) or False"
    )


@renamed_kwargs(df="dataframe")
def detect_objects(
    dataframe: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    uuid_column: str = "uuid",
    time_column: str = "systime",
    hierarchy: Hierarchy | None = None,
    infer_o2o: InferO2O = True,
    validate: bool = True,
    uses: Uses | None = None,
) -> EventLog:
    """Detect object instances and build their OCEL 2.0 tables.

    Produces an :class:`EventLog` with **no events** — only ``objects``, ``o2o``
    (object-to-object relations), and ``object_changes`` (presence lifecycle +
    captured attributes). Compose it with event-detector logs via
    :func:`~ts_shape.eventlog.concat.concat`.

    ``hierarchy`` declares the compositional ``part_of`` relations, because
    *part_of cannot be inferred from time alone* (a long-running tool
    temporally contains every batch without owning it). Each key is a child
    and each value a parent or a list of parents; a key or value is an
    **id-signal uuid** when it names one of the specs' signals, else an
    **object type**:

    * ``{"serial": "batch", "batch": "work_order"}`` -- type level: each
      serial presence is part of the batch presence it overlaps longest.
    * ``{"L1:serial": "L1:batch", "L2:serial": "L2:batch"}`` -- signal
      level: a serial of line 1 is part of a line-1 batch only, so two lines
      stay apart.
    * ``{"L1:serial": ["L1:batch", "L1:hu"]}`` -- several parents: one
      ``part_of`` per parent (a serial in a batch *and* a handling unit).

    Each child presence gets one ``part_of`` per declared parent: to the
    parent presence it overlaps longest (ties: the earlier parent).

    ``uses`` declares usage (bill-of-materials) links: ``{child: {parent:
    qualifier}}``, named like ``hierarchy`` entries, e.g. ``{"cell:serial":
    {"line:material": "made_from", "line:tool": "made_with", "line:operator":
    "made_by"}}`` (any qualifier text works). Each child presence is linked
    to *every* parent presence it overlaps (half-open; touching is no
    overlap), so every serial made while a material is held is ``made_from``
    it, also when the material never changes, and a serial whose cycle spans
    a material change is ``made_from`` both. One row per ``(child, parent,
    qualifier)``. :func:`~ts_shape.eventlog.where_used` and
    :func:`~ts_shape.eventlog.where_from` follow these qualifiers.

    ``infer_o2o``: ``True`` / ``"hierarchy"`` (default) emits the declared
    ``part_of`` edges only -- no hierarchy, no ``o2o`` rows.
    ``"co_occurs"`` also relates every two overlapping objects of different
    types with the symmetric ``co_occurs`` qualifier (pairs already related
    by ``part_of`` or ``uses`` excepted); on a plant this can be hundreds
    of thousands of rows. ``False`` skips object-to-object relations (also
    ``uses``).

    Args:
        dataframe: Long-format frame with the id signals.
        specs: Which signal carries which object type.
        uuid_column: Signal identifier column.
        time_column: Timestamp column.
        hierarchy: Child -> parent(s) for ``part_of`` relations, by object
            type or id-signal uuid.
        infer_o2o: Which object-to-object relations to derive.
        validate: Check the result against the canonical schema.
        uses: Child -> ``{parent: qualifier}`` usage links, by id-signal uuid
            or object type.

    Returns:
        An :class:`EventLog` with ``objects``, ``o2o`` and ``object_changes``
        and no events.

    Raises:
        ColumnNotFoundError: When ``dataframe`` lacks the uuid, time or a spec's
            value column (or a spec's ``cycle_value_column``).
        ValueError: For an unknown ``infer_o2o`` value, a ``cycle_uuid`` that
            is not in the frame, or an empty ``uses`` qualifier.
        TypeError: When a ``uses`` entry is not a ``{parent: qualifier}``
            mapping.
    """
    derive, co_occurs = _o2o_mode(infer_o2o)
    intervals, extra = _detect(
        dataframe, specs, uuid_column=uuid_column, time_column=time_column
    )
    log = EventLog(
        objects=_objects_table(intervals),
        o2o=(
            _infer_o2o(intervals, hierarchy, co_occurs=co_occurs, uses=uses)
            if derive
            else schema.empty_o2o()
        ),
        object_changes=_object_changes(intervals, extra),
    )
    if validate:
        schema.validate(log)
    return log


@renamed_kwargs(df="dataframe")
def attach_objects(
    event_log: EventLog,
    dataframe: pd.DataFrame,
    specs: Sequence[ObjectSpec],
    *,
    qualifiers: Mapping[str, str] | None = None,
    uuid_column: str = "uuid",
    time_column: str = "systime",
    hierarchy: Hierarchy | None = None,
    infer_o2o: InferO2O = True,
    scope: Scope | None = None,
    fanout_threshold: int | None = FANOUT_THRESHOLD,
    validate: bool = True,
    uses: Uses | None = None,
) -> EventLog:
    """Link an existing event log to detected objects by temporal overlap.

    Every event is linked with an event-to-object (E2O) relation to each
    object whose presence overlaps the event (see
    :func:`link_events_to_objects`, which also explains ``scope`` and
    ``fanout_threshold``). The detected objects, ``o2o`` and
    ``object_changes`` are merged in. This enriches the output of *any*
    event detector — no per-detector changes required. See
    :func:`detect_objects` for ``hierarchy`` and ``infer_o2o``.

    This is a convenience over :func:`relate`: it first segments ``dataframe`` into an
    object-interval table via :func:`object_intervals`, then relates the events
    to it. When you already hold an interval/object table, call :func:`relate`
    directly and skip the raw signals.

    Args:
        event_log: The events to link.
        dataframe: Long-format frame with the id signals. Naive timestamps
            are read as UTC, like :func:`~ts_shape.eventlog.to_event_log`
            reads the detector frames.
        specs: Which signal carries which object type.
        qualifiers: Relation qualifier per object type.
        uuid_column: Signal identifier column.
        time_column: Timestamp column.
        hierarchy: Child -> parent(s) for ``part_of`` relations.
        infer_o2o: Which object-to-object relations to derive.
        scope: ``{signal_uuid: scope_label}`` (e.g. the line of each id and
            event source signal); an event links only to objects of its own
            scope.
        fanout_threshold: Without ``scope``, warn when an event links to
            more than this many objects of one type present at the same
            time; ``None`` disables the check.
        validate: Check the result against the canonical schema.
        uses: Child -> ``{parent: qualifier}`` usage links (see
            :func:`detect_objects`).

    Returns:
        ``event_log`` merged with the detected objects and the new relations.
    """
    intervals, extra = _detect(
        dataframe, specs, uuid_column=uuid_column, time_column=time_column
    )
    return _relate(
        event_log,
        intervals,
        extra,
        hierarchy=hierarchy,
        qualifiers=qualifiers,
        key_columns=None,
        infer_o2o=infer_o2o,
        scope=scope,
        fanout_threshold=fanout_threshold,
        validate=validate,
        uses=uses,
    )


def relate(
    event_log: EventLog,
    intervals: pd.DataFrame,
    *,
    hierarchy: Hierarchy | None = None,
    qualifiers: Mapping[str, str] | None = None,
    key_columns: Sequence[str] | None = None,
    infer_o2o: InferO2O = True,
    scope: Scope | None = None,
    fanout_threshold: int | None = FANOUT_THRESHOLD,
    validate: bool = True,
    uses: Uses | None = None,
) -> EventLog:
    """Assemble a full OCEL 2.0 log from an event list and an object list.

    This is the automated relationship step: given an existing ``event_log``
    (with events) and an ``intervals`` table describing objects -- columns
    ``oid``, ``type`` and, for temporal linking, ``start`` / ``end`` (plus any
    attribute columns) -- it derives **all** relations and returns a validated
    :class:`EventLog` whose five tables are ready to ingest into an OCEL 2.0
    database:

    * ``objects`` from the distinct ``(oid, type)`` pairs,
    * ``relations`` (E2O) via :func:`link_events_to_objects`
      (temporal overlap, optionally within a ``scope``, plus optional
      ``key_columns`` matching),
    * ``o2o`` via :func:`derive_o2o` (declared ``part_of`` ``hierarchy``;
      ``co_occurs`` on request),
    * ``object_changes`` (lifecycle + captured attributes).

    A missing ``start`` / ``end`` (``NaT``) means the presence is unbounded on
    that side (e.g. an object that is still active). Every column other than
    ``oid / type / start / end / duration_seconds / uuid / source_uuid`` is an
    attribute, recorded at the presence start. A ``uuid`` column (the id
    signal) enables signal-level ``hierarchy`` entries and ``scope``.

    Unlike :func:`attach_objects`, no raw signals or :class:`ObjectSpec` are
    needed -- you bring the two lists, ts-shape derives the relationships.

    Args:
        event_log: The events to link.
        intervals: Object table with ``oid``, ``type`` and optionally
            ``start`` / ``end``, ``uuid`` and attribute columns.
        hierarchy: Child -> parent(s) for ``part_of`` relations, by object
            type or id-signal uuid (see :func:`detect_objects`).
        qualifiers: Relation qualifier per object type.
        key_columns: Event columns that name an object id directly.
        infer_o2o: Which object-to-object relations to derive.
        scope: ``{signal_uuid: scope_label}``; see
            :func:`link_events_to_objects`.
        fanout_threshold: See :func:`link_events_to_objects`.
        validate: Check the result against the canonical schema.
        uses: Child -> ``{parent: qualifier}`` usage links (see
            :func:`detect_objects`).

    Returns:
        The validated five-table :class:`EventLog`.

    Raises:
        ColumnNotFoundError: When ``intervals`` lacks ``oid`` or ``type``.
    """
    if _START in intervals.columns:
        _warn_tz_mismatch(
            event_log.events.get(schema.OCEL_TIMESTAMP), intervals[_START]
        )
    return _relate(
        event_log,
        intervals,
        None,
        hierarchy=hierarchy,
        qualifiers=qualifiers,
        key_columns=key_columns,
        infer_o2o=infer_o2o,
        scope=scope,
        fanout_threshold=fanout_threshold,
        validate=validate,
        uses=uses,
    )


def _relate(
    event_log: EventLog,
    intervals: pd.DataFrame,
    extra_changes: pd.DataFrame | None,
    *,
    hierarchy: Hierarchy | None,
    qualifiers: Mapping[str, str] | None,
    key_columns: Sequence[str] | None,
    infer_o2o: InferO2O,
    scope: Scope | None,
    fanout_threshold: int | None,
    validate: bool,
    uses: Uses | None = None,
) -> EventLog:
    derive, co_occurs = _o2o_mode(infer_o2o)
    iv = _normalize_intervals(intervals)
    detected = EventLog(
        objects=_objects_table(iv),
        o2o=(
            _infer_o2o(iv, hierarchy, co_occurs=co_occurs, uses=uses)
            if derive
            else schema.empty_o2o()
        ),
        object_changes=_object_changes(iv, extra_changes),
    )
    merged = concat(event_log, detected)

    rels = _link(
        event_log.events,
        iv,
        qualifiers=qualifiers,
        key_columns=key_columns,
        contain=True,
        scope=scope,
        fanout_threshold=fanout_threshold,
        sources=(
            _event_sources(event_log.events, event_log.relations)
            if scope is not None
            else None
        ),
    )
    if not rels.empty:
        merged.relations = (
            pd.concat([merged.relations, rels], ignore_index=True)
            .drop_duplicates(
                subset=[schema.OCEL_EID, schema.OCEL_OID, schema.OCEL_TYPE]
            )
            .reset_index(drop=True)
        )
    if validate:
        schema.validate(merged)
    return merged


def derive_o2o(
    intervals: pd.DataFrame,
    *,
    hierarchy: Hierarchy | None = None,
    co_occurs: bool = False,
    uses: Uses | None = None,
) -> pd.DataFrame:
    """Derive object-to-object (O2O) relations from an interval table.

    Public entry point over the overlap kernel: a declared ``hierarchy``
    (child -> parent or list of parents, each an object type or an id-signal
    ``uuid``; see :func:`detect_objects`) yields ``part_of`` from each child
    presence to the parent presence it overlaps longest. ``co_occurs=True``
    also relates every two overlapping objects of different types with the
    symmetric ``co_occurs`` qualifier (one row per pair, the type that sorts
    first in ``ocel:oid``; pairs related by ``part_of`` or ``uses``
    excepted). ``uses`` (child -> ``{parent: qualifier}``) links each child
    presence to every parent presence it overlaps (see
    :func:`detect_objects`).

    Args:
        intervals: Object table with ``oid``, ``type``, ``start``, ``end``
            (and ``uuid`` for signal-level hierarchy entries).
        hierarchy: Child -> parent(s).
        co_occurs: Also emit ``co_occurs`` for every overlap of two types.
        uses: Child -> ``{parent: qualifier}`` usage links.

    Returns:
        The OCEL 2.0 ``o2o`` table, de-duplicated.
    """
    return _infer_o2o(
        _normalize_intervals(intervals), hierarchy, co_occurs=co_occurs, uses=uses
    )


def object_specs_from_metadata(
    metadata: pd.DataFrame,
    *,
    type_field: str = "object_type",
    value_column_field: str = "object_value_column",
    uuid_column: str = "uuid",
) -> list[ObjectSpec]:
    """Build :class:`ObjectSpec` list from per-signal metadata tags.

    Scans a metadata table (e.g. ``MetadataJsonLoader.to_df()``) for a column
    naming the object type per signal — so new object types are onboarded by
    tagging metadata, not by writing code. The uuid may be the index or a
    column; the type/value-column fields are matched directly or with a
    ``config_`` / ``config.`` prefix (mkdocs metadata flattening).

    Args:
        metadata: Metadata table, one row per signal.
        type_field: Field naming the object type of a signal; rows without
            one are skipped.
        value_column_field: Field naming the signal's id column (default
            ``value_string``); must be one of ``value_string``,
            ``value_integer`` or ``value_double``.
        uuid_column: Column (or index name) holding the signal uuid.

    Returns:
        One :class:`ObjectSpec` per tagged signal; empty when the table has
        no uuid or type field.

    Raises:
        ValueError: When a value-column entry is not an id column.
    """
    md = metadata.reset_index() if metadata.index.name == uuid_column else metadata

    def _col(name: str) -> str | None:
        for cand in (name, f"config_{name}", f"config.{name}"):
            if cand in md.columns:
                return cand
        return None

    uuid_col = uuid_column if uuid_column in md.columns else None
    type_col = _col(type_field)
    if uuid_col is None or type_col is None:
        return []
    valcol_col = _col(value_column_field)

    specs: list[ObjectSpec] = []
    for _, row in md.iterrows():
        otype = row.get(type_col)
        if otype is None or pd.isna(otype) or str(otype).strip() == "":
            continue
        value_column = "value_string"
        if valcol_col is not None and not pd.isna(row.get(valcol_col)):
            value_column = str(row[valcol_col])
        if value_column not in _ID_VALUE_COLUMNS:
            raise ValueError(
                f"signal {row[uuid_col]!r}: {valcol_col} is {value_column!r}; "
                f"an id signal's value column must be one of {_ID_VALUE_COLUMNS}"
            )
        specs.append(
            ObjectSpec(
                uuid=str(row[uuid_col]),
                object_type=str(otype),
                value_column=value_column,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _objects_table(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return schema.empty_objects()
    objs = (
        intervals[[_OID, _TYPE]]
        .drop_duplicates()
        .rename(columns={_OID: schema.OCEL_OID, _TYPE: schema.OCEL_TYPE})
        .reset_index(drop=True)
    )
    objs[schema.OCEL_OID] = objs[schema.OCEL_OID].astype("string")
    objs[schema.OCEL_TYPE] = objs[schema.OCEL_TYPE].astype("string")
    shared = objs[schema.OCEL_OID][objs[schema.OCEL_OID].duplicated()]
    if not shared.empty:
        warnings.warn(
            f"object id(s) {sorted(set(shared))[:5]} occur with more than one "
            "object type; OCEL 2.0 object ids must be unique across types — use "
            "id_template='{type}:{value}' to namespace them",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    return objs


def _presence_station(intervals: pd.DataFrame) -> pd.Series:
    """Where each presence is: its ``source_uuid``, else its id signal ``uuid``.

    ``<NA>`` when the table has neither (a user-supplied object list).
    """
    station = pd.Series(pd.NA, index=intervals.index, dtype="string")
    for column in (_SOURCE, _UUID):
        if column in intervals.columns:
            station = station.fillna(intervals[column].astype("string"))
    return station


def _object_changes(
    intervals: pd.DataFrame, extra: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Lifecycle + attribute ``object_changes`` derived from the interval table.

    ``lifecycle`` is ``active`` when an object becomes present and ``released``
    when it stops. Overlapping presences of the same object form one
    continuous presence, and so do touching ones at the same station. A
    hand-over to another station at the same instant (the serial leaves
    station A as it enters station B) starts a new presence: the incoming
    presence owns that instant, so the lifecycle reads ``active`` there and
    the ``released`` of the outgoing one is not recorded (one value per
    instant). The station is :func:`_presence_station`. Attribute columns are inferred from the frame itself -- any
    column beyond the reserved interval columns -- and recorded at each
    presence start. ``extra`` adds in-presence attribute changes. Sorted by
    timestamp; one value per ``(oid, field, timestamp)``.
    """
    empty = schema.empty_object_changes()
    if intervals.empty or _START not in intervals.columns:
        return empty
    oid = intervals[_OID].astype("string").to_numpy()
    typ = intervals[_TYPE].astype("string").to_numpy()
    start = _ns(intervals[_START])
    end = (
        _ns(intervals[_END])
        if _END in intervals.columns
        else np.full(len(intervals), _NAT, dtype=np.int64)
    )
    parts: list[pd.DataFrame] = []

    # Lifecycle over the union of each object's presences.
    key, _ = pd.factorize(pd.MultiIndex.from_arrays([oid, typ]))
    s_open = np.where(start == _NAT, np.iinfo(np.int64).min + 1, start)
    e_open = np.where(end == _NAT, np.iinfo(np.int64).max, end)
    order = np.lexsort((s_open, key))
    k, s, e = key[order], s_open[order], e_open[order]
    station = pd.factorize(_presence_station(intervals), use_na_sentinel=False)[0]
    station = station[order]
    runmax = pd.Series(e).groupby(k).cummax().to_numpy()
    prev_max = np.empty_like(runmax)
    prev_max[0] = runmax[0]
    prev_max[1:] = runmax[:-1]
    new = np.ones(len(k), dtype=bool)
    handover = (s[1:] == prev_max[1:]) & (station[1:] != station[:-1])
    new[1:] = (k[1:] != k[:-1]) | (s[1:] > prev_max[1:]) | handover
    run = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    run_end = pd.Series(e).groupby(run).max().to_numpy()
    r_oid, r_typ = oid[order][first], typ[order][first]
    r_start = start[order][first]
    for label, ts, sel_oid, sel_typ in (
        ("active", r_start, r_oid, r_typ),
        ("released", run_end, r_oid, r_typ),
    ):
        valid = (ts != _NAT) & (ts != np.iinfo(np.int64).max)
        parts.append(
            pd.DataFrame(
                {
                    schema.OCEL_OID: sel_oid[valid],
                    schema.OCEL_TYPE: sel_typ[valid],
                    schema.OCEL_FIELD: "lifecycle",
                    schema.OCEL_VALUE: label,
                    schema.OCEL_TIMESTAMP: _from_ns(ts[valid]).to_numpy(),
                }
            )
        )

    attr_cols = [c for c in intervals.columns if c not in _RESERVED]
    for a in attr_cols:
        vals = intervals[a].to_numpy(dtype=object)
        ok = ~pd.isna(intervals[a]).to_numpy(dtype=bool) & (start != _NAT)
        if not ok.any():
            continue
        parts.append(
            pd.DataFrame(
                {
                    schema.OCEL_OID: oid[ok],
                    schema.OCEL_TYPE: typ[ok],
                    schema.OCEL_FIELD: a,
                    schema.OCEL_VALUE: vals[ok],
                    schema.OCEL_TIMESTAMP: _from_ns(start[ok]).to_numpy(),
                }
            )
        )
    if extra is not None and not extra.empty:
        parts.append(extra)

    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty
    out = pd.concat(parts, ignore_index=True)
    out[schema.OCEL_TIMESTAMP] = _to_utc(out[schema.OCEL_TIMESTAMP])
    out = (
        out.drop_duplicates(
            subset=[
                schema.OCEL_OID,
                schema.OCEL_TYPE,
                schema.OCEL_FIELD,
                schema.OCEL_TIMESTAMP,
            ]
        )
        .sort_values(schema.OCEL_TIMESTAMP, kind="stable")
        .reset_index(drop=True)
    )
    return out.astype(empty.dtypes.to_dict())


# ---------------------------------------------------------------------------
# Interval kernels (vectorized; int64 ns, UTC)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spans:
    """Finite int64 bounds plus whether the end instant belongs to the span."""

    start: _I64
    end: _I64
    closed: _BOOL


def _spans(iv: pd.DataFrame, lo: int, hi: int) -> _Spans:
    """Bounds of ``iv`` with open sides clamped to ``lo`` / ``hi``.

    A presence is half-open ``[start, end)`` when another presence of the same
    signal (``uuid`` column, when present) — else of the same type — starts at
    its end (a hand-over); otherwise it is closed ``[start, end]``.
    """
    s = _ns(iv[_START])
    e = _ns(iv[_END])
    open_end = e == _NAT
    s = np.where(s == _NAT, lo, s)
    e = np.where(open_end, hi, e)
    if _UUID in iv.columns and iv[_UUID].notna().all():
        scope = iv[_UUID].astype("string").to_numpy()
    else:
        scope = iv[_TYPE].astype("string").to_numpy()
    starts = pd.MultiIndex.from_arrays([scope, s])
    counts = pd.Series(1, index=starts).groupby(level=[0, 1]).sum()
    ends = pd.MultiIndex.from_arrays([scope, e])
    hits = np.array(counts.reindex(ends).fillna(0).to_numpy(), dtype=np.int64)
    hits -= (s == e).astype(np.int64)  # a point presence does not hand over to itself
    closed = (hits <= 0) | open_end
    return _Spans(s, e, closed)


def _bounds(*arrays: _I64) -> tuple[int, int]:
    finite = [a[a != _NAT] for a in arrays if len(a)]
    finite = [a for a in finite if len(a)]
    if not finite:
        return 0, 1
    lo = min(int(a.min()) for a in finite)
    hi = max(int(a.max()) for a in finite)
    return lo - 1, hi + 1


def _candidate_pairs(a: _Spans, b: _Spans) -> tuple[_I64, _I64]:
    """All ``(i, j)`` with ``b.start[j] <= a.end[i]`` and ``b.end[j] >= a.start[i]``.

    Spans of ``b`` are bucketed by length (powers of two) so a search window
    ``[a.start - max_len, a.end]`` on sorted starts stays tight: the work is
    proportional to the number of real overlaps, not ``len(a) * len(b)``.
    """
    out_a: list[_I64] = []
    out_b: list[_I64] = []
    if len(a.start) == 0 or len(b.start) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    length = (b.end - b.start).astype(np.float64)
    bucket = np.floor(np.log2(np.maximum(length, 0) + 1)).astype(np.int64)
    for k in np.unique(bucket):
        sel = np.flatnonzero(bucket == k)
        order = sel[np.argsort(b.start[sel], kind="stable")]
        bs = b.start[order]
        maxlen = int(np.max(b.end[order] - bs))
        lo = np.searchsorted(bs, a.start - maxlen, side="left")
        hi = np.searchsorted(bs, a.end, side="right")
        cnt = (hi - lo).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            continue
        ia = np.repeat(np.arange(len(a.start), dtype=np.int64), cnt)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        jb = order[np.repeat(lo, cnt) + offs]
        keep = b.end[jb] >= a.start[ia]
        out_a.append(ia[keep])
        out_b.append(jb[keep].astype(np.int64))
    if not out_a:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(out_a), np.concatenate(out_b)


def _intersections(a: _Spans, b: _Spans) -> tuple[_I64, _I64, _I64]:
    """Pairs whose presences overlap, with the overlap length.

    Two presences overlap when they share a stretch of time; merely touching
    end-to-start is not an overlap. A point presence (``start == end``)
    overlaps a presence that contains its instant.
    """
    ia, jb = _candidate_pairs(a, b)
    m = np.maximum(a.start[ia], b.start[jb])
    in_a = (m < a.end[ia]) | ((m == a.end[ia]) & a.closed[ia])
    in_b = (m < b.end[jb]) | ((m == b.end[jb]) & b.closed[jb])
    shared = np.minimum(a.end[ia], b.end[jb]) > m
    point = (a.start[ia] == a.end[ia]) | (b.start[jb] == b.end[jb])
    hit = in_a & in_b & (shared | point)
    ia, jb, m = ia[hit], jb[hit], m[hit]
    overlap = np.minimum(a.end[ia], b.end[jb]) - m
    return ia, jb, overlap


def _check_uses(uses: Uses | None) -> Mapping[str, Mapping[str, str]]:
    """``uses`` as given, after checking its shape (child -> {parent: q})."""
    if uses is None:
        return {}
    if not isinstance(uses, Mapping):
        raise TypeError(
            f"uses must map a child signal to {{parent_signal: qualifier}}; got "
            f"{type(uses).__name__}"
        )
    for child, parents in uses.items():
        if not isinstance(parents, Mapping):
            raise TypeError(
                f"uses[{child!r}] must map parent signals to a qualifier, e.g. "
                f'{{"material_signal": "made_from"}}; got {parents!r}'
            )
        for parent, qualifier in parents.items():
            if not isinstance(qualifier, str) or not qualifier:
                raise ValueError(
                    f"uses[{child!r}][{parent!r}]: the qualifier must be a "
                    f"non-empty text such as 'made_from'; got {qualifier!r}"
                )
    return uses


def _infer_o2o(
    intervals: pd.DataFrame,
    hierarchy: Hierarchy | None,
    *,
    co_occurs: bool = False,
    uses: Uses | None = None,
) -> pd.DataFrame:
    """Object-to-object relations from interval overlap.

    Along each declared ``hierarchy`` edge (child -> parent, each an id-signal
    uuid when the ``uuid`` column has it, else an object type) every child
    presence gets ``part_of`` (child -> parent) to the parent presence it
    overlaps longest (ties: the earlier parent). With ``co_occurs`` every
    two overlapping objects of different types also get the symmetric
    ``co_occurs`` qualifier, unless ``part_of`` or a ``uses`` link already
    relates them — ``part_of`` is never guessed from time alone. Along each
    ``uses`` entry every child presence is linked to every parent presence
    it overlaps, with the entry's qualifier.
    """
    if (
        intervals.empty
        or _START not in intervals.columns
        or _END not in intervals.columns
    ):
        return schema.empty_o2o()
    iv = intervals.reset_index(drop=True)
    lo, hi = _bounds(_ns(iv[_START]), _ns(iv[_END]))
    spans = _spans(iv, lo, hi)
    type_s = iv[_TYPE].astype("string")
    types = type_s.to_numpy()
    oids = iv[_OID].astype("string").to_numpy()
    uuid_s = iv[_UUID].astype("string") if _UUID in iv.columns else None
    uuid_set = set(uuid_s.dropna()) if uuid_s is not None else set()

    def rows_of(name: str) -> _BOOL:
        col = uuid_s if (uuid_s is not None and name in uuid_set) else type_s
        return np.asarray(col.eq(name).fillna(False).to_numpy(dtype=bool))

    def sub(mask: _BOOL) -> tuple[_I64, _Spans]:
        idx = np.flatnonzero(mask)
        return idx, _Spans(spans.start[idx], spans.end[idx], spans.closed[idx])

    frames: list[pd.DataFrame] = []
    for child, parents in (hierarchy or {}).items():
        plist = [parents] if isinstance(parents, str) else list(parents)
        ci, cs = sub(rows_of(child))
        if len(ci) == 0:
            continue
        for parent in plist:
            if parent == child:
                continue
            pi, ps = sub(rows_of(parent))
            a, b, overlap = _intersections(cs, ps)
            if len(a) == 0:
                continue
            pick = pd.DataFrame({"c": a, "ov": overlap, "ps": ps.start[b], "p": b})
            pick = pick.sort_values(
                ["c", "ov", "ps", "p"], ascending=[True, False, True, True]
            ).drop_duplicates("c")
            frames.append(
                pd.DataFrame(
                    {
                        schema.OCEL_OID: oids[ci[pick["c"].to_numpy()]],
                        schema.OCEL_OID2: oids[pi[pick["p"].to_numpy()]],
                        schema.OCEL_QUALIFIER: "part_of",
                    }
                )
            )
    for child, used in _check_uses(uses).items():
        ci, cs = sub(rows_of(child))
        if len(ci) == 0:
            continue
        for parent, qualifier in used.items():
            if parent == child:
                continue
            pi, ps = sub(rows_of(parent))
            a, b, _ = _intersections(cs, ps)
            if len(a) == 0:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        schema.OCEL_OID: oids[ci[a]],
                        schema.OCEL_OID2: oids[pi[b]],
                        schema.OCEL_QUALIFIER: str(qualifier),
                    }
                )
            )
    if co_occurs:
        co: list[pd.DataFrame] = []
        uniq = sorted(set(types))
        for i, ta in enumerate(uniq):
            ai, as_ = sub(types == ta)
            for tb in uniq[i + 1 :]:
                bi, bs = sub(types == tb)
                a, b, _ = _intersections(as_, bs)
                if len(a) == 0:
                    continue
                co.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_OID: oids[ai[a]],
                            schema.OCEL_OID2: oids[bi[b]],
                            schema.OCEL_QUALIFIER: "co_occurs",
                        }
                    )
                )
        if co and frames:
            part = pd.concat(frames, ignore_index=True)
            related = set(
                zip(part[schema.OCEL_OID], part[schema.OCEL_OID2], strict=True)
            )
            related |= {(b, a) for a, b in related}
            co = [
                f[
                    [
                        (a, b) not in related
                        for a, b in zip(
                            f[schema.OCEL_OID], f[schema.OCEL_OID2], strict=True
                        )
                    ]
                ]
                for f in co
            ]
        frames.extend(co)
    if not frames:
        return schema.empty_o2o()
    out = pd.concat(frames, ignore_index=True)
    out = out[out[schema.OCEL_OID] != out[schema.OCEL_OID2]]
    if out.empty:
        return schema.empty_o2o()
    out = out.drop_duplicates().reset_index(drop=True)
    return out.astype(schema.empty_o2o().dtypes.to_dict())


def _normalize_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an object/interval table.

    Requires ``oid`` and ``type``; ``start`` / ``end`` are optional (needed only
    for temporal linking and ``o2o``). Ids/types are coerced to ``string`` and
    any present time bounds to tz-aware UTC, matching the OCEL columns. Rows
    without an ``oid`` are dropped; ``end`` before ``start`` raises.
    """
    missing = {_OID, _TYPE} - set(intervals.columns)
    if missing:
        raise ValueError(
            f"object/interval table is missing required column(s) {sorted(missing)}; "
            f"got {list(intervals.columns)}"
        )
    out = intervals.copy()
    out[_OID] = out[_OID].astype("string")
    out[_TYPE] = out[_TYPE].astype("string")
    out = out[out[_OID].notna() & out[_TYPE].notna()].reset_index(drop=True)
    if _START in out.columns:
        out[_START] = _to_utc(out[_START])
    if _END in out.columns:
        out[_END] = _to_utc(out[_END])
    if _START in out.columns and _END in out.columns:
        bad = out[out[_END] < out[_START]]
        if not bad.empty:
            row = bad.iloc[0]
            raise ValueError(
                f"object {row[_OID]!r} ends before it starts "
                f"({row[_START]} > {row[_END]})"
            )
    return out


def _tz_kind(values: object) -> str | None:
    """``"naive"`` / ``"aware"`` for a datetime Series, else ``None``."""
    if not isinstance(values, pd.Series):
        return None
    dtype = values.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "aware"
    if pd.api.types.is_datetime64_dtype(dtype):
        return "naive"
    return None


def _warn_tz_mismatch(event_times: object, object_times: object) -> None:
    """Warn when one side is tz-naive and the other tz-aware."""
    ev, ob = _tz_kind(event_times), _tz_kind(object_times)
    if ev is None or ob is None or ev == ob:
        return
    naive = "event" if ev == "naive" else "object"
    warnings.warn(
        f"the {naive} timestamps are tz-naive and the others tz-aware; the naive "
        "ones are read as UTC. If they are local wall-clock times, localize "
        "them first (Series.dt.tz_localize) or the links are shifted by the "
        "UTC offset.",
        DataQualityWarning,
        stacklevel=caller_stacklevel(),
    )


def _event_sources(
    events: pd.DataFrame, relations: pd.DataFrame | None = None
) -> npt.NDArray[np.object_]:
    """The source signal uuid of each event (``None`` when unknown).

    Read from a ``source_uuid`` column or a pack's ``<pack>:source_uuid``
    attribute (the first one set), else from the event's ``asset`` relation.
    """
    out = pd.Series([None] * len(events), index=events.index, dtype="object")
    cols = [
        c
        for c in events.columns
        if isinstance(c, str) and (c == _SOURCE or c.endswith(":" + _SOURCE))
    ]
    for c in cols:
        vals = events[c]
        fill = out.isna() & vals.notna()
        out[fill] = vals[fill].astype(str)
    if relations is not None and out.isna().any() and not relations.empty:
        assets = relations[relations[schema.OCEL_TYPE] == "asset"]
        by_eid = assets.drop_duplicates(schema.OCEL_EID).set_index(schema.OCEL_EID)[
            schema.OCEL_OID
        ]
        fill = out.isna()
        out[fill] = events.loc[fill, schema.OCEL_EID].map(by_eid)
    return np.array(
        [None if pd.isna(v) else str(v) for v in out.to_numpy()], dtype=object
    )


def _scope_labels(
    keys: Sequence[npt.NDArray[np.object_]], scope: Scope
) -> npt.NDArray[np.object_]:
    """Scope label per row: the first of ``keys`` that ``scope`` maps."""
    n = len(keys[0]) if keys else 0
    lookup = {k: str(v) for k, v in scope.items() if v is not None}
    out = pd.Series([None] * n, dtype=object)
    for k in keys:
        mapped = pd.Series(k, dtype=object).map(lookup)
        fill = (out.isna() & mapped.notna()).to_numpy(dtype=bool)
        out[fill] = mapped[fill]
    return np.array([None if pd.isna(v) else v for v in out], dtype=object)


def _warn_fanout(
    eids: npt.NDArray[np.object_],
    ev: _I64,
    obj: _I64,
    ev_spans: _Spans,
    obj_spans: _Spans,
    types: npt.NDArray[np.object_],
    oids: npt.NDArray[np.object_],
    threshold: int,
) -> None:
    """Warn when an event links more than ``threshold`` objects of one type
    present at the same time (a sweep over each (event, type) group)."""
    if len(ev) <= threshold:
        return
    tcode, tuniq = pd.factorize(pd.Series(types[obj], dtype=object))
    group = ev * (len(tuniq) + 1) + np.asarray(tcode, dtype=np.int64)
    cs = np.maximum(ev_spans.start[ev], obj_spans.start[obj])
    ce = np.minimum(ev_spans.end[ev], obj_spans.end[obj])
    positive = ce > cs
    g = np.concatenate([group, group])
    t = np.concatenate([cs, ce])
    d = np.concatenate([np.ones(len(cs)), -np.ones(len(cs))]).astype(np.int64)
    # at one instant: ends of real stretches first, then starts, then the
    # ends of zero-length overlaps (points count while present)
    o = np.concatenate([np.ones(len(cs)), np.where(positive, 0, 2)]).astype(np.int64)
    order = np.lexsort((o, t, g))
    cum = np.cumsum(d[order])
    gs = g[order]
    first = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    peak = np.maximum.reduceat(cum, first)
    over = peak > threshold
    if not over.any():
        return
    bad_groups = gs[first][over]
    n_events = len(np.unique(bad_groups // (len(tuniq) + 1)))
    worst = int(np.argmax(np.where(over, peak, -1)))
    wg = gs[first][worst]
    we = int(wg // (len(tuniq) + 1))
    wt = str(tuniq[int(wg % (len(tuniq) + 1))])
    sample = sorted({str(x) for x in oids[obj[group == wg]]})[:5]
    warnings.warn(
        f"{n_events} event(s) link to more than {threshold} object(s) of one type "
        f"present at the same time (worst: event {eids[we]!r} -> {int(peak[worst])} "
        f"{wt!r} objects at once, e.g. {sample}). The objects come from several id "
        "signals (e.g. two lines); pass scope={signal_uuid: line} so an event "
        "links only to the objects of its own line, or fanout_threshold=None to "
        "silence this check.",
        DataQualityWarning,
        stacklevel=caller_stacklevel(),
    )


def _event_spans(events: pd.DataFrame) -> tuple[_I64, _Spans]:
    """Valid event rows and their ``[start, end)`` spans (points closed)."""
    end = _ns(events[schema.OCEL_TIMESTAMP])
    if schema.TS_START_TIMESTAMP in events.columns:
        start = _ns(events[schema.TS_START_TIMESTAMP])
    else:
        start = np.full(len(end), _NAT, dtype=np.int64)
    start = np.where((start == _NAT) | (start > end), end, start)
    valid = np.flatnonzero(end != _NAT)
    s, e = start[valid], end[valid]
    return valid, _Spans(s, e, s == e)


def _link(
    events: pd.DataFrame,
    iv: pd.DataFrame,
    *,
    qualifiers: Mapping[str, str] | None,
    key_columns: Sequence[str] | None,
    contain: bool,
    scope: Scope | None,
    fanout_threshold: int | None,
    sources: npt.NDArray[np.object_] | None,
) -> pd.DataFrame:
    """E2O relations of normalized ``iv`` (see :func:`link_events_to_objects`)."""
    if events.empty or iv.empty:
        return schema.empty_relations()
    qual = dict(qualifiers or {})
    frames: list[pd.DataFrame] = []
    eids = events[schema.OCEL_EID].to_numpy()
    oids = iv[_OID].to_numpy()
    types = iv[_TYPE].to_numpy()

    if contain and _START in iv.columns and _END in iv.columns:
        end_ns = _ns(events[schema.OCEL_TIMESTAMP])
        start_ns = (
            _ns(events[schema.TS_START_TIMESTAMP])
            if schema.TS_START_TIMESTAMP in events.columns
            else end_ns
        )
        lo, hi = _bounds(end_ns, start_ns, _ns(iv[_START]), _ns(iv[_END]))
        sp = _spans(iv, lo, hi)
        rows, es = _event_spans(events)
        groups: list[tuple[_I64, _I64]] = []
        if scope is None:
            groups.append(
                (
                    np.arange(len(rows), dtype=np.int64),
                    np.arange(len(iv), dtype=np.int64),
                )
            )
        else:
            src = sources if sources is not None else _event_sources(events)
            ev_label = _scope_labels([src[rows]], scope)
            obj_keys = (
                [iv[_UUID].astype(object).to_numpy()] if _UUID in iv.columns else []
            )
            if _SOURCE in iv.columns:
                obj_keys.append(iv[_SOURCE].astype(object).to_numpy())
            obj_label = (
                _scope_labels(obj_keys, scope)
                if obj_keys
                else np.full(len(iv), None, dtype=object)
            )
            _warn_unscoped(ev_label, src[rows], obj_label, iv)
            for label in sorted({x for x in ev_label if x is not None}):
                e_sel = np.flatnonzero(ev_label == label)
                o_sel = np.flatnonzero(obj_label == label)
                if len(e_sel) and len(o_sel):
                    groups.append((e_sel.astype(np.int64), o_sel.astype(np.int64)))
        ev_parts: list[_I64] = []
        obj_parts: list[_I64] = []
        for e_sel, o_sel in groups:
            a, b, _ = _intersections(
                _Spans(es.start[e_sel], es.end[e_sel], es.closed[e_sel]),
                _Spans(sp.start[o_sel], sp.end[o_sel], sp.closed[o_sel]),
            )
            ev_parts.append(e_sel[a])
            obj_parts.append(o_sel[b])
        if ev_parts:
            ev = np.concatenate(ev_parts)
            obj = np.concatenate(obj_parts)
            if scope is None and fanout_threshold is not None and len(ev):
                _warn_fanout(eids[rows], ev, obj, es, sp, types, oids, fanout_threshold)
            if len(ev):
                order = np.lexsort((obj, ev))
                ev, obj = ev[order], obj[order]
                frames.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_EID: eids[rows[ev]],
                            schema.OCEL_OID: oids[obj],
                            schema.OCEL_TYPE: types[obj],
                        }
                    )
                )

    if key_columns:
        oid_to_type = iv[[_OID, _TYPE]].drop_duplicates(_OID).set_index(_OID)[_TYPE]
        for col in [c for c in key_columns if c in events.columns]:
            vals = events[col]
            ok = vals.notna().to_numpy(dtype=bool)
            keys = pd.Series([_id_text(v) for v in vals[ok]], dtype="string")
            ktypes = keys.map(oid_to_type)
            hit = ktypes.notna().to_numpy(dtype=bool)
            if hit.any():
                frames.append(
                    pd.DataFrame(
                        {
                            schema.OCEL_EID: eids[ok][hit],
                            schema.OCEL_OID: keys.to_numpy()[hit],
                            schema.OCEL_TYPE: ktypes.to_numpy()[hit],
                        }
                    )
                )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return schema.empty_relations()
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=[schema.OCEL_EID, schema.OCEL_OID, schema.OCEL_TYPE])
        .reset_index(drop=True)
    )
    out[schema.OCEL_QUALIFIER] = out[schema.OCEL_TYPE].astype(object).map(qual)
    return out.astype(schema.empty_relations().dtypes.to_dict())


def _warn_unscoped(
    ev_label: npt.NDArray[np.object_],
    ev_source: npt.NDArray[np.object_],
    obj_label: npt.NDArray[np.object_],
    iv: pd.DataFrame,
) -> None:
    """Warn about events / objects ``scope`` gives no label (never linked)."""
    ev_missing = np.array([x is None for x in ev_label], dtype=bool)
    if ev_missing.any():
        srcs = sorted({str(x) for x in ev_source[ev_missing] if x is not None})
        warnings.warn(
            f"{int(ev_missing.sum())} event(s) have no scope and are linked to no "
            "object by time: their source signal "
            + (f"(e.g. {srcs[:5]}) is not in scope" if srcs else "is unknown")
            + ". Add their source signals to scope.",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    obj_missing = np.array([x is None for x in obj_label], dtype=bool)
    if obj_missing.any():
        sigs = (
            sorted(set(iv.loc[obj_missing, _UUID].dropna().astype(str)))
            if _UUID in iv.columns
            else []
        )
        warnings.warn(
            f"{int(obj_missing.sum())} object presence(s) have no scope and are "
            "linked to no event by time"
            + (f": id signal(s) {sigs[:5]} are not in scope" if sigs else "")
            + ". Add the id signals to scope.",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )


def link_events_to_objects(
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    qualifiers: Mapping[str, str] | None = None,
    key_columns: Sequence[str] | None = None,
    contain: bool = True,
    scope: Scope | None = None,
    fanout_threshold: int | None = FANOUT_THRESHOLD,
) -> pd.DataFrame:
    """Derive all event-to-object (E2O) relations from two lists.

    Takes an OCEL ``events`` table and an object/interval table (``oid``,
    ``type`` and, for temporal linking, ``start`` / ``end``) and returns the
    OCEL 2.0 ``relations`` table -- with no access to the raw signals. Two
    complementary strategies are unioned and de-duplicated on
    ``(eid, oid, type)``:

    * **temporal overlap** (``contain``, default on): link an event to every
      object whose presence overlaps the event. An interval event spans
      ``ts_shape:start_timestamp`` .. ``ocel:timestamp`` (half-open) and
      links to every object present during a stretch of it -- also one that
      left before the event ended, and every object of a hand-over inside
      it; merely touching at an end point is no overlap. A point event (no
      start, or start == end) links to the objects whose presence contains
      its timestamp. Presence is half-open ``[start, end)`` when another
      presence of the same signal (``uuid`` column) or, without one, of the
      same type starts at ``end`` -- so a point event at a hand-over instant
      belongs to the incoming object only -- and closed ``[start, end]``
      otherwise (the last presence). ``NaT`` bounds are open. Requires
      ``start`` / ``end`` columns.
    * **key match** (opt-in via ``key_columns``): link an event to the object
      whose ``oid`` equals the value in one of the event's ``key_columns`` --
      for events that already name their object (e.g. a ``batch`` attribute).
      ``scope`` does not apply.

    **Scope.** Without ``scope`` an event links to the objects of *every* id
    signal, so a machine-state event of line 1 also gets the serials and
    batches of line 2. ``scope={signal_uuid: scope_label}`` labels the id
    signals (the ``uuid`` column of ``intervals``, else ``source_uuid``) and
    the event source signals (a ``source_uuid`` / ``<pack>:source_uuid``
    column of ``events``); an event then links only to objects with the same
    label. Events and objects without a label are linked to nothing by time,
    with a :class:`~ts_shape.errors.DataQualityWarning`. Without ``scope``,
    a :class:`~ts_shape.errors.DataQualityWarning` suggests one when an
    event links to more than ``fanout_threshold`` objects of one type
    present at the same time (one signal carries one object at a time, so
    this means several signals of that type, e.g. two lines).

    Naive timestamps are read as UTC. When one table is tz-naive and the
    other tz-aware a :class:`~ts_shape.errors.DataQualityWarning` says so
    (local wall-clock times would be linked shifted by the UTC offset).

    Args:
        events: OCEL events table with ``ocel:eid`` (and ``ocel:timestamp``,
            optionally ``ts_shape:start_timestamp``, for temporal linking).
        intervals: Object table with ``oid``, ``type`` and, for temporal
            linking, ``start`` / ``end`` (``uuid`` / ``source_uuid`` for
            ``scope``).
        qualifiers: Maps an object ``type`` to the relation qualifier to stamp
            (e.g. ``{"batch": "processed_in"}``); unmapped types get ``<NA>``.
        key_columns: Event columns whose value is an object id. Integral
            floats (an integer id column with gaps) match ``"1"``, not
            ``"1.0"``.
        contain: Link by temporal overlap.
        scope: ``{signal_uuid: scope_label}`` for id and event source
            signals; links only within one label.
        fanout_threshold: Without ``scope``, warn when an event links to
            more than this many objects of one type present at the same
            time (default 1); ``None`` disables the check.

    Returns:
        The OCEL 2.0 ``relations`` table.

    Raises:
        ColumnNotFoundError: When ``events`` lacks ``ocel:eid`` (or
            ``ocel:timestamp`` with ``contain``) or ``intervals`` lacks
            ``oid`` / ``type``.
    """
    require_columns(events, (schema.OCEL_EID,), where="link_events_to_objects")
    if contain:
        require_columns(
            events, (schema.OCEL_TIMESTAMP,), where="link_events_to_objects"
        )
        if _START in intervals.columns:
            _warn_tz_mismatch(events[schema.OCEL_TIMESTAMP], intervals[_START])
    if events.empty:
        return schema.empty_relations()
    iv = _normalize_intervals(intervals)
    return _link(
        events,
        iv,
        qualifiers=qualifiers,
        key_columns=key_columns,
        contain=contain,
        scope=scope,
        fanout_threshold=fanout_threshold,
        sources=None,
    )
