"""Public ``to_event_log`` entry point — looks up the adapter and runs it."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

import pandas as pd

from . import adapters, schema, taxonomy
from .model import EventLog

TableInput: TypeAlias = pd.DataFrame | Iterable[Mapping[str, object]] | None


#: Columns a user-supplied ``o2o`` / ``object_changes`` table must have; the
#: other canonical columns are optional and filled with ``<NA>``.
_REQUIRED = {
    "o2o": (schema.OCEL_OID, schema.OCEL_OID2),
    "object_changes": (
        schema.OCEL_OID,
        schema.OCEL_TYPE,
        schema.OCEL_FIELD,
        schema.OCEL_TIMESTAMP,
    ),
}


def _coerce_table(data: TableInput, empty: pd.DataFrame, name: str) -> pd.DataFrame:
    """Coerce a DataFrame or iterable-of-mappings onto a canonical table.

    Reindexes to the canonical columns (extra columns dropped, optional
    missing ones filled with NA) and aligns dtypes, so callers can pass plain
    dicts. Naive timestamps are read as UTC.

    Raises:
        ValueError: When a required column is missing.
    """
    if data is None:
        return empty
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(list(data))
    if df.empty and len(df.columns) == 0:
        return empty
    missing = [c for c in _REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} table is missing columns {missing}; got {list(df.columns)}. "
            f"Expected columns: {list(empty.columns)}"
        )
    if df.empty:
        return empty
    df = df.reindex(columns=empty.columns)
    for col, dtype in empty.dtypes.items():
        if isinstance(dtype, pd.DatetimeTZDtype):
            times = pd.to_datetime(df[col])
            df[col] = (
                times.dt.tz_localize("UTC")
                if times.dt.tz is None
                else times.dt.tz_convert("UTC")
            )
    return df.astype(empty.dtypes.to_dict()).reset_index(drop=True)


def to_event_log(
    df: pd.DataFrame,
    *,
    detector: str,
    objects: Mapping[str, object] | None = None,
    qualifiers: Mapping[str, str] | None = None,
    o2o: TableInput = None,
    object_changes: TableInput = None,
    validate: bool = True,
    timestamp: object | None = None,
) -> EventLog:
    """Normalize a detector frame (a detector's output DataFrame) into the
    canonical event log.

    Timestamps: tz-aware times are converted to UTC; **naive times are taken
    to be UTC** (localize them first if they are local wall-clock times).
    Interval rows that end before they start are dropped with a
    :class:`~ts_shape.errors.DataQualityWarning`.

    Event ids (``ocel:eid``) are content-based: a hash of the detector, the
    row's ``source_uuid``, its timestamps, the activity and every other
    column of the row. They are reproducible across runs, independent of
    row order, and the same detector on two machines does not collide.

    Args:
        df: The detector frame.
        detector: ``"ClassName.method_name"`` -- the key used in the taxonomy
            registry and the value written to ``ts_shape:detector``.
            Deprecated names resolve to the current ones.
        objects: Binds OCEL object types to, per type:

            * a column name of ``df`` (e.g. ``{"asset": "source_uuid"}``);
            * any other string or scalar -- a literal broadcast to every row
              (``{"line": "L1"}``); ``None`` binds no object;
            * a callable taking a row dict and returning an oid;
            * a ``pd.Series`` aligned with ``df`` by index label (a length or
              label mismatch raises ``ValueError``).

            Types listed in the rule's ``produces_objects`` are also
            auto-extracted from standard columns (``source_uuid -> asset``)
            when not bound explicitly.
        qualifiers: Relation qualifier per object type.
        o2o: OCEL 2.0 object-to-object relations, a DataFrame or iterable of
            dict rows with ``ocel:oid`` / ``ocel:oid2`` (``ocel:qualifier``
            optional); see :func:`~ts_shape.eventlog.schema.empty_o2o`.
        object_changes: Time-varying object attributes with ``ocel:oid``,
            ``ocel:type``, ``ocel:field``, ``ocel:timestamp`` (``ocel:value``
            optional); see
            :func:`~ts_shape.eventlog.schema.empty_object_changes`. Naive
            timestamps are read as UTC.
        validate: Check the result against the canonical schema.
        timestamp: Timestamp of the events of a ``static`` shape detector
            (which has no time column). Default: the Unix epoch
            (``1970-01-01T00:00Z``), so results are reproducible.

    Returns:
        The canonical :class:`~ts_shape.eventlog.model.EventLog`.

    Raises:
        ValueError: For a malformed ``detector`` name, a misaligned Series
            binding or a table missing required columns.
        KeyError: When the detector has no taxonomy entry, or the frame lacks
            the time columns of its shape.
    """
    if "." not in detector:
        raise ValueError(f"detector must be 'ClassName.method_name', got {detector!r}")
    class_name, method_name = taxonomy.resolve(*detector.split(".", 1))
    detector = f"{class_name}.{method_name}"  # current names in the output

    rule = taxonomy.get(class_name, method_name)
    if rule is None:
        raise KeyError(
            f"no taxonomy entry for {detector!r}. "
            "Add a LabelRule to ts_shape.eventlog.taxonomy.REGISTRY."
        )

    override = adapters.get_override(class_name, method_name)
    if override is not None:
        log = override(
            df, rule=rule, detector=detector, objects=objects, qualifiers=qualifiers
        )
    else:
        log = adapters.adapt(
            df,
            rule=rule,
            detector=detector,
            objects=objects,
            qualifiers=qualifiers,
            timestamp=timestamp,
        )

    log.o2o = _coerce_table(o2o, schema.empty_o2o(), "o2o")
    log.object_changes = _coerce_table(
        object_changes, schema.empty_object_changes(), "object_changes"
    )

    if validate:
        schema.validate(log)
    return log
