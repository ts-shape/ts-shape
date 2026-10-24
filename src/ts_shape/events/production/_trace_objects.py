"""Shared helpers of the traceability classes for the object model.

``to_intervals()`` of :class:`ValueTraceabilityEvents`,
:class:`RoutingTraceabilityEvents` and :class:`MultiProcessTraceabilityEvents`
returns the interval table of :func:`ts_shape.eventlog.object_intervals`
(plus a ``station`` column), and their timelines take
``interval_end="last_sample" | "next_change"``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

#: ``interval_end`` of the traceability timelines.
IntervalEnd: TypeAlias = Literal["last_sample", "next_change"]

INTERVAL_ENDS: tuple[str, ...] = ("last_sample", "next_change")

#: ``exclude`` of the traceability classes (as :attr:`ObjectSpec.exclude`).
Exclude: TypeAlias = "str | Collection[object] | Callable[[pd.Series], object] | None"


def exclude_readings(values: pd.Series, exclude: Exclude, uuid: str) -> pd.Series:
    """``values`` (one id signal, time order) with excluded readings replaced.

    ``exclude`` is read like :attr:`ts_shape.eventlog.ObjectSpec.exclude`
    (``"misreads"``, a collection of values, a callable, ``None``). A run of
    excluded readings between two readings of the same id takes that id (a
    gap in the data); anywhere else it becomes a missing reading (the visit
    before it ends there).
    """
    if exclude is None or values.empty:
        return values
    # Lazy: ``ts_shape.eventlog`` imports the detector packs.
    from ts_shape.eventlog.objects import ObjectSpec, _exclusion_mask

    s = values.reset_index(drop=True)
    bad = _exclusion_mask(ObjectSpec(uuid, "serial", exclude=exclude), s)
    if bad is None or not bad.any():
        return values
    codes = np.asarray(pd.factorize(s, use_na_sentinel=True)[0], dtype=np.int64)
    n = len(s)
    pos = np.arange(n, dtype=np.int64)
    prev = np.maximum.accumulate(np.where(bad, -1, pos))
    nxt = np.minimum.accumulate(np.where(bad, n, pos)[::-1])[::-1]
    prev_code = np.where(prev >= 0, codes[prev.clip(min=0)], -1)
    next_code = np.where(nxt < n, codes[nxt.clip(max=n - 1)], -1)
    fill = bad & (prev_code == next_code) & (prev_code >= 0)
    out = s.copy()
    if fill.any():
        out.iloc[np.flatnonzero(fill)] = s.iloc[prev[fill]].to_numpy()
    out = out.mask(pd.Series(bad & ~fill))
    out.index = values.index
    return out


#: Columns of ``to_intervals()``: those of
#: :func:`ts_shape.eventlog.object_intervals`, then ``station``.
INTERVAL_COLUMNS: tuple[str, ...] = (
    "oid",
    "type",
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
    "station",
)


def check_interval_end(interval_end: object, where: str) -> None:
    """Raise ``ValueError`` for an unknown ``interval_end``."""
    if interval_end not in INTERVAL_ENDS:
        raise ValueError(
            f"{where}: interval_end={interval_end!r}; use 'last_sample' (an "
            "interval ends at its last sample) or 'next_change' (it ends at the "
            "next different reading, like object_intervals)"
        )


def next_change_ends(
    run_start_times: Any, data_end: Any, *, held: np.ndarray | None = None
) -> pd.Series:
    """``end`` of each run under ``interval_end="next_change"``.

    ``run_start_times`` are the start times of *every* run of one signal in
    time order (empty readings included). A run ends where the next run
    starts; the last run is held until ``data_end``. Where ``held[k]`` is
    False, run ``k`` keeps no next-change end (``NaT``; the caller keeps its
    last sample).
    """
    starts = pd.Series(pd.array(run_start_times))
    ends = starts.shift(-1)
    if len(ends):
        ends.iloc[-1] = data_end
    if held is not None:
        ends = ends.where(pd.Series(held, index=ends.index))
    return ends


def _to_utc(values: pd.Series) -> pd.Series:
    """Timestamps as tz-aware UTC ``datetime64[ns]`` (naive read as UTC)."""
    s = pd.to_datetime(values.reset_index(drop=True))
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    return s.dt.as_unit("ns")


def id_text(value: object) -> str:
    """An object id as text; an integral float renders without ``.0``."""
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def empty_intervals() -> pd.DataFrame:
    """``to_intervals()`` result without rows (the documented dtypes)."""
    return pd.DataFrame(
        {
            "oid": pd.Series(dtype="string"),
            "type": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
            "duration_seconds": pd.Series(dtype="float64"),
            "uuid": pd.Series(dtype="string"),
            "source_uuid": pd.Series(dtype="string"),
            "station": pd.Series(dtype="string"),
        }
    )


def intervals_frame(
    *,
    oid: pd.Series,
    object_type: str,
    start: pd.Series,
    end: pd.Series,
    uuid: Any,
    source_uuid: Any,
    station: pd.Series,
) -> pd.DataFrame:
    """The ``to_intervals()`` table of a timeline (one row per visit)."""
    n = len(oid)
    if n == 0:
        return empty_intervals()
    s, e = _to_utc(start), _to_utc(end)
    s_ns = s.array.asi8
    e_ns = e.array.asi8

    def _strings(value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            return value.reset_index(drop=True).astype("string")
        return pd.Series([value] * n, dtype="string")

    return pd.DataFrame(
        {
            "oid": pd.Series([id_text(v) for v in oid.tolist()], dtype="string"),
            "type": pd.Series([object_type] * n, dtype="string"),
            "start": s,
            "end": e,
            "duration_seconds": (e_ns - s_ns) / 1e9,
            "uuid": _strings(uuid),
            "source_uuid": _strings(source_uuid),
            "station": _strings(station),
        }
    )
