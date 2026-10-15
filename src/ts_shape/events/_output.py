"""Canonical event-output column schema and helpers for ts-shape detectors.

Every public DataFrame-returning method on a detector class under
``ts_shape.events.*`` MUST emit one of three canonical shapes:

* ``point``     -- a single timestamp per row, in the ``systime`` column.
* ``interval``  -- explicit ``start`` / ``end`` columns plus
  ``duration_seconds``.
* ``summary``   -- a windowed aggregate; same time columns as ``interval``.

Use :func:`empty_event_df` whenever a method has no rows to return, and
:func:`finalize_point_df`, :func:`finalize_interval_df`, or
:func:`finalize_summary_df` to attach standard identity columns and ensure
canonical column ordering before returning. This guarantees consumers see
the same column names regardless of detector pack or method.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column-name constants
# ---------------------------------------------------------------------------

COL_SYSTIME = "systime"
COL_START = "start"
COL_END = "end"
COL_DURATION_S = "duration_seconds"

COL_UUID = "uuid"
COL_SOURCE_UUID = "source_uuid"

COL_SEVERITY = "severity"
COL_SEVERITY_LEVEL = "severity_level"

#: The shared severity vocabulary of ``severity_level``.
SEVERITY_LEVELS: tuple[str, ...] = ("info", "warn", "critical")

#: How detector-specific severity labels map onto :data:`SEVERITY_LEVELS`.
#: Detectors whose labels mean something else pass their own overrides to
#: :func:`add_severity_level`.
DEFAULT_SEVERITY_LEVEL: dict[str, str] = {
    "info": "info",
    "ok": "info",
    "low": "info",
    "warn": "warn",
    "warning": "warn",
    "medium": "warn",
    "moderate": "warn",
    "high": "critical",
    "major": "critical",
    "severe": "critical",
    "critical": "critical",
}


def add_severity_level(
    df: pd.DataFrame, overrides: dict[str, str] | None = None
) -> pd.DataFrame:
    """Add ``severity_level`` (info / warn / critical) after ``severity``.

    Each detector keeps its own ``severity`` labels; ``severity_level`` maps
    them onto one vocabulary so results of different detectors can be
    filtered and ranked together. Labels without a mapping (e.g. ``"none"``)
    give a missing value. Frames without a ``severity`` column are returned
    unchanged.

    Args:
        df: A detector result.
        overrides: Detector-specific label mappings that replace the
            defaults in :data:`DEFAULT_SEVERITY_LEVEL`.

    Returns:
        ``df`` with the ``severity_level`` column.
    """
    if COL_SEVERITY not in df.columns:
        return df
    mapping = {**DEFAULT_SEVERITY_LEVEL, **(overrides or {})}
    levels = df[COL_SEVERITY].map(
        lambda v: mapping.get(v) if isinstance(v, str) else None
    )
    out = df.drop(columns=[COL_SEVERITY_LEVEL], errors="ignore")
    out.insert(
        list(out.columns).index(COL_SEVERITY) + 1,
        COL_SEVERITY_LEVEL,
        levels.astype(object),
    )
    return out


# ---------------------------------------------------------------------------
# Required-column tuples per shape
# ---------------------------------------------------------------------------

POINT_SCHEMA: tuple[str, ...] = (COL_SYSTIME, COL_UUID, COL_SOURCE_UUID)
INTERVAL_SCHEMA: tuple[str, ...] = (
    COL_START,
    COL_END,
    COL_DURATION_S,
    COL_UUID,
    COL_SOURCE_UUID,
)
SUMMARY_SCHEMA: tuple[str, ...] = (COL_START, COL_END, COL_DURATION_S)


Shape = Literal["point", "interval", "summary"]


def _schema_for(shape: Shape) -> tuple[str, ...]:
    if shape == "point":
        return POINT_SCHEMA
    if shape == "interval":
        return INTERVAL_SCHEMA
    if shape == "summary":
        return SUMMARY_SCHEMA
    raise ValueError(f"Unknown event shape: {shape!r}")


def validate_event_output(df: pd.DataFrame, shape: Shape) -> pd.DataFrame:
    """Assert that ``df`` conforms to the canonical schema for ``shape``.

    A lightweight contract check for detector output: confirms the required
    columns for the given shape are present. Use it in tests or at the end of a
    custom detector to guarantee downstream consumers (event-log normalizers,
    OCEL/XES exporters) see the expected columns.

    Args:
        df: The detector output to validate.
        shape: One of ``"point"``, ``"interval"`` or ``"summary"``.

    Returns:
        The same ``df`` unchanged, so the call can wrap a return value.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``shape`` is unknown or required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    required = _schema_for(shape)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{shape!r} event output is missing required column(s) {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    return df


def empty_event_df(shape: Shape, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical columns for ``shape``.

    ``extra_cols`` are appended after the required columns and may include
    optional standard columns (``severity``, ``value``, ``is_delta``) plus
    detector-specific columns. Duplicates are removed while preserving order.
    """
    cols: list[str] = list(_schema_for(shape))
    for col in extra_cols:
        if col not in cols:
            cols.append(col)
    return pd.DataFrame(columns=cols)


def drop_duplicate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Drop event rows identical in every column, keeping the first.

    Duplicated input samples (same timestamp, uuid and value) otherwise
    produce the same event twice. Rows that differ in any column are kept.
    """
    if len(df) < 2:
        return df
    try:
        dup = df.duplicated()
    except TypeError:  # unhashable cells (lists, dicts)
        dup = df.astype(str).duplicated()
    return df[~dup.to_numpy()] if bool(dup.any()) else df


def _reorder(df: pd.DataFrame, leading: Iterable[str]) -> pd.DataFrame:
    """Move ``leading`` columns to the front, keep remaining order."""
    leading_present = [c for c in leading if c in df.columns]
    rest = [c for c in df.columns if c not in leading_present]
    return df[leading_present + rest]


def finalize_point_df(
    df: pd.DataFrame,
    *,
    uuid: str | None,
    source_uuid: str | None,
    time_col: str = COL_SYSTIME,
) -> pd.DataFrame:
    """Attach identity columns and canonical column order to a point-event df.

    If ``time_col`` differs from ``systime`` the column is renamed.
    Existing ``uuid`` / ``source_uuid`` columns are preserved if present;
    otherwise the supplied scalars are broadcast.
    """
    out = df.copy()
    if time_col != COL_SYSTIME and time_col in out.columns:
        out = out.rename(columns={time_col: COL_SYSTIME})
    if COL_UUID not in out.columns and uuid is not None:
        out[COL_UUID] = uuid
    if COL_SOURCE_UUID not in out.columns and source_uuid is not None:
        out[COL_SOURCE_UUID] = source_uuid
    return _reorder(out, POINT_SCHEMA)


def finalize_interval_df(
    df: pd.DataFrame,
    *,
    uuid: str | None,
    source_uuid: str | None,
) -> pd.DataFrame:
    """Attach identity columns, compute ``duration_seconds``, and canonicalize order.

    The DataFrame must already contain ``start`` and ``end`` (datetime) columns.
    """
    out = df.copy()
    if COL_START not in out.columns or COL_END not in out.columns:
        raise ValueError(
            f"interval frame requires {COL_START!r} and {COL_END!r} columns; "
            f"got {list(out.columns)}"
        )
    out[COL_START] = pd.to_datetime(out[COL_START])
    out[COL_END] = pd.to_datetime(out[COL_END])
    if COL_DURATION_S not in out.columns:
        out[COL_DURATION_S] = (out[COL_END] - out[COL_START]).dt.total_seconds()
    if COL_UUID not in out.columns and uuid is not None:
        out[COL_UUID] = uuid
    if COL_SOURCE_UUID not in out.columns and source_uuid is not None:
        out[COL_SOURCE_UUID] = source_uuid
    return _reorder(out, INTERVAL_SCHEMA)


def finalize_summary_df(
    df: pd.DataFrame,
    *,
    uuid: str | None = None,
    source_uuid: str | None = None,
) -> pd.DataFrame:
    """Attach optional identity columns and canonicalize a summary/window df.

    The DataFrame must already contain ``start`` and ``end`` (datetime) columns.
    ``uuid`` / ``source_uuid`` are optional for summary rows; pass ``None`` to
    omit when the aggregate spans multiple sources.
    """
    out = df.copy()
    if COL_START not in out.columns or COL_END not in out.columns:
        raise ValueError(
            f"summary frame requires {COL_START!r} and {COL_END!r} columns; "
            f"got {list(out.columns)}"
        )
    out[COL_START] = pd.to_datetime(out[COL_START])
    out[COL_END] = pd.to_datetime(out[COL_END])
    if COL_DURATION_S not in out.columns:
        out[COL_DURATION_S] = (out[COL_END] - out[COL_START]).dt.total_seconds()
    if COL_UUID not in out.columns and uuid is not None:
        out[COL_UUID] = uuid
    if COL_SOURCE_UUID not in out.columns and source_uuid is not None:
        out[COL_SOURCE_UUID] = source_uuid
    leading = SUMMARY_SCHEMA + (COL_UUID, COL_SOURCE_UUID)
    return _reorder(out, leading)
