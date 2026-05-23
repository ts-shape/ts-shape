"""Shape-driven adapters that convert legacy detector DataFrames into the
canonical :class:`~ts_shape.eventlog.model.EventLog`.

There is one adapter implementation per *shape* (point, interval, summary,
static). The per-method specifics live in
:mod:`ts_shape.eventlog.taxonomy` (the ``REGISTRY``). A custom adapter for
a specific ``(class, method)`` pair can be registered with
``@register_adapter(...)`` to override the shape-driven default — useful
for the few detectors whose legacy schema does not fit any standard shape.
"""

from __future__ import annotations

import uuid as _uuid
from collections.abc import Callable, Mapping

import pandas as pd

from . import schema
from ..events._output import COL_END, COL_START, COL_SYSTIME
from .model import EventLog
from .taxonomy import LabelRule, render_activity

# ----------------------------------------------------------------------------
# Registry of optional per-method overrides
# ----------------------------------------------------------------------------

AdapterFn = Callable[..., EventLog]
_OVERRIDES: dict[tuple[str, str], AdapterFn] = {}


def register_adapter(
    class_name: str, method_name: str
) -> Callable[[AdapterFn], AdapterFn]:
    """Register a custom adapter for a specific ``(class, method)``.

    The function will be called with ``(legacy_df, *, rule, detector,
    objects, qualifiers)``.
    """

    def deco(fn: AdapterFn) -> AdapterFn:
        _OVERRIDES[(class_name, method_name)] = fn
        return fn

    return deco


def get_override(class_name: str, method_name: str) -> AdapterFn | None:
    return _OVERRIDES.get((class_name, method_name))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

_NAMESPACE = _uuid.UUID("8a4d3f1c-9b2e-4d3a-8f1e-c0ffee123456")


def _eid(detector: str, ts: pd.Timestamp, key: str) -> str:
    return "e-" + str(_uuid.uuid5(_NAMESPACE, f"{detector}|{ts.isoformat()}|{key}"))


def _require_columns(
    df: pd.DataFrame, cols: tuple[str, ...], shape: str, detector: str
) -> None:
    """Raise KeyError if any canonical time column for ``shape`` is missing.

    Post-#62 every detector is guaranteed to emit these columns. A missing
    column means the caller fed a malformed frame, not a legacy schema.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"detector {detector!r} produced shape {shape!r} but is missing "
            f"canonical column(s) {missing}; got {list(df.columns)}"
        )


def _to_utc_ts(value: object) -> pd.Timestamp | pd._libs.tslibs.nattype.NaTType:
    if value is None:
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        return pd.NaT
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _classify_severity(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v >= 4.5:
        return "critical"
    if v >= 3.0:
        return "warn"
    return "info"


def _build_attrs_columns(
    legacy: pd.DataFrame,
    *,
    pack: str,
    drop: set[str],
) -> dict[str, pd.Series]:
    """Return a dict of ``"<pack>:<col>" -> Series`` for non-canonical columns."""
    out: dict[str, pd.Series] = {}
    for col in legacy.columns:
        if col in drop:
            continue
        out[f"{pack}:{col}"] = legacy[col].reset_index(drop=True)
    return out


def _resolve_objects(
    legacy: pd.DataFrame,
    rule: LabelRule,
    user_objects: Mapping[str, object] | None,
) -> dict[str, pd.Series]:
    """Produce ``object_type -> Series-of-oids`` aligned with legacy rows.

    Auto-extracts types declared in ``rule.produces_objects`` from the
    legacy DataFrame's standard columns (e.g. ``source_uuid -> asset``).
    Caller-supplied bindings via ``user_objects`` are always honored —
    object types not declared by the adapter are treated as contextual
    annotations the caller knows about (e.g. "this outlier happened
    during batch B-2026-117").
    """
    declared = set(rule.produces_objects)
    bindings: dict[str, pd.Series] = {}

    def _bind(otype: str, value: object) -> None:
        if isinstance(value, str):
            if value not in legacy.columns:
                return  # silently skip; column not present
            bindings[otype] = legacy[value].reset_index(drop=True).astype("string")
        elif callable(value):
            bindings[otype] = pd.Series(
                [value(r) for r in legacy.to_dict("records")], dtype="string"
            )
        elif isinstance(value, pd.Series):
            bindings[otype] = value.reset_index(drop=True).astype("string")
        else:
            # Scalar broadcast.
            bindings[otype] = pd.Series([str(value)] * len(legacy), dtype="string")

    if user_objects:
        for otype, val in user_objects.items():
            _bind(otype, val)

    # Defaults: asset = source_uuid if column exists.
    if "asset" in declared and "asset" not in bindings:
        if "source_uuid" in legacy.columns:
            bindings["asset"] = (
                legacy["source_uuid"].reset_index(drop=True).astype("string")
            )

    # Drop empty / all-NaN bindings.
    return {k: v for k, v in bindings.items() if not v.isna().all()}


def _to_relations(
    eids: pd.Series,
    object_bindings: Mapping[str, pd.Series],
    qualifiers: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not object_bindings:
        return schema.empty_objects(), schema.empty_relations()

    qualifiers = qualifiers or {}
    rel_frames: list[pd.DataFrame] = []
    obj_pairs: list[tuple[str, str]] = []
    for otype, oids in object_bindings.items():
        oids = oids.reset_index(drop=True)
        mask = oids.notna() & (oids.astype(str) != "") & (oids.astype(str) != "<NA>")
        if not mask.any():
            continue
        rel_frames.append(
            pd.DataFrame(
                {
                    schema.OCEL_EID: eids[mask].reset_index(drop=True).astype("string"),
                    schema.OCEL_OID: oids[mask].astype("string").reset_index(drop=True),
                    schema.OCEL_TYPE: pd.Series(
                        [otype] * int(mask.sum()), dtype="string"
                    ),
                    schema.OCEL_QUALIFIER: pd.Series(
                        [qualifiers.get(otype)] * int(mask.sum()), dtype="string"
                    ),
                }
            )
        )
        for o in oids[mask].astype(str).unique():
            obj_pairs.append((o, otype))

    relations = (
        pd.concat(rel_frames, ignore_index=True)
        if rel_frames
        else schema.empty_relations()
    )
    if obj_pairs:
        obj_df = pd.DataFrame(obj_pairs, columns=[schema.OCEL_OID, schema.OCEL_TYPE])
        obj_df = obj_df.drop_duplicates().reset_index(drop=True)
        obj_df[schema.OCEL_OID] = obj_df[schema.OCEL_OID].astype("string")
        obj_df[schema.OCEL_TYPE] = obj_df[schema.OCEL_TYPE].astype("string")
    else:
        obj_df = schema.empty_objects()
    return obj_df, relations


_RESERVED = {"start", "end", "systime", "uuid", "source_uuid"}


def _value_series(legacy: pd.DataFrame, rule: LabelRule) -> pd.Series:
    if rule.value_field and rule.value_field in legacy.columns:
        return pd.to_numeric(legacy[rule.value_field], errors="coerce")
    for c in ("value", "value_double", "value_integer", "ts_shape:value"):
        if c in legacy.columns:
            return pd.to_numeric(legacy[c], errors="coerce")
    return pd.Series([float("nan")] * len(legacy), dtype="float64")


def _severity_series(legacy: pd.DataFrame, rule: LabelRule) -> pd.Series:
    if rule.severity_field and rule.severity_field in legacy.columns:
        col = legacy[rule.severity_field]
        return col.map(_classify_severity).astype("string")
    if "severity" in legacy.columns:
        return legacy["severity"].map(_classify_severity).astype("string")
    return pd.Series([pd.NA] * len(legacy), dtype="string")


def _apply_standard_attrs(
    legacy: pd.DataFrame,
    rule: LabelRule,
    n: int,
) -> tuple[dict[str, pd.Series], set[str]]:
    """Resolve LabelRule.standard_attrs into typed columns.

    Returns ``(extra_columns, source_columns_consumed)``. The caller adds
    ``source_columns_consumed`` to the drop set so the legacy column does
    not also appear under its ``<pack>:<col>`` prefix.
    """
    extras: dict[str, pd.Series] = {}
    consumed: set[str] = set()
    if not rule.standard_attrs:
        return extras, consumed

    for std_key, source in rule.standard_attrs.items():
        target_dtype = schema.STANDARD_ATTR_TYPES.get(std_key, "string")

        if source is None:
            continue
        if isinstance(source, str) and source in legacy.columns:
            # String matching a legacy column → rename / coerce.
            series = legacy[source].reset_index(drop=True)
            consumed.add(source)
        else:
            # Literal scalar (or string that doesn't match a column,
            # which is the common case for ts_shape:method = "zscore"
            # and similar enum-like values). Broadcast to every row.
            series = pd.Series([source] * n)

        if target_dtype == "Int64":
            extras[std_key] = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif target_dtype == "float64":
            extras[std_key] = pd.to_numeric(series, errors="coerce").astype("float64")
        else:
            extras[std_key] = series.astype("string")

    return extras, consumed


# ----------------------------------------------------------------------------
# The shape-driven adapter
# ----------------------------------------------------------------------------


def adapt(
    legacy: pd.DataFrame,
    *,
    rule: LabelRule,
    detector: str,
    objects: Mapping[str, object] | None = None,
    qualifiers: Mapping[str, str] | None = None,
) -> EventLog:
    """Convert one detector's output into a canonical :class:`EventLog`."""
    if legacy is None or len(legacy) == 0:
        return EventLog()

    legacy = legacy.reset_index(drop=True)
    n = len(legacy)

    # Resolve timestamps based on shape. Detectors are required (post-#62) to
    # emit canonical time columns: ``systime`` for point, ``start``/``end``
    # for interval and summary. Missing columns indicate a malformed frame —
    # we fail fast rather than silently substituting ``utcnow()``.
    if rule.shape == "interval":
        _require_columns(legacy, (COL_START, COL_END), rule.shape, detector)
        ts_end = legacy[COL_END].apply(_to_utc_ts)
        ts_start = legacy[COL_START].apply(_to_utc_ts)
        duration = (ts_end - ts_start).dt.total_seconds()
        time_cols_used = {COL_START, COL_END}
    elif rule.shape == "point":
        _require_columns(legacy, (COL_SYSTIME,), rule.shape, detector)
        ts_end = legacy[COL_SYSTIME].apply(_to_utc_ts)
        ts_start = pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
        duration = pd.Series([float("nan")] * n, dtype="float64")
        time_cols_used = {COL_SYSTIME}
    elif rule.shape == "summary":
        _require_columns(legacy, (COL_START, COL_END), rule.shape, detector)
        ts_end = legacy[COL_END].apply(_to_utc_ts)
        ts_start = legacy[COL_START].apply(_to_utc_ts)
        duration = (ts_end - ts_start).dt.total_seconds()
        time_cols_used = {COL_START, COL_END}
    elif rule.shape == "static":
        # No natural time. Use a single fixed reference (now-UTC) for all rows.
        now = (
            pd.Timestamp.utcnow().tz_convert("UTC")
            if pd.Timestamp.utcnow().tz
            else pd.Timestamp.utcnow().tz_localize("UTC")
        )
        ts_end = pd.Series([now] * n)
        ts_start = pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
        duration = pd.Series([float("nan")] * n, dtype="float64")
        time_cols_used = set()
    else:
        raise ValueError(f"unknown shape {rule.shape!r}")

    # Render activity name (template substitution per row).
    activities = legacy.apply(lambda r: render_activity(rule, r), axis=1).astype(
        "string"
    )

    # Build canonical event ids.
    eids = pd.Series(
        [_eid(detector, ts_end.iloc[i], f"{i}|{activities.iloc[i]}") for i in range(n)],
        dtype="string",
    )

    severity = _severity_series(legacy, rule)
    value = _value_series(legacy, rule)

    # Resolve standard attribute extension (ts_shape:method, baseline,
    # threshold_low/high, etc). Source columns get added to ``drop`` so
    # they don't double-up under their ``<pack>:<col>`` prefix.
    standard_extras, std_consumed = _apply_standard_attrs(legacy, rule, n)

    # Drop time columns + columns redundantly captured elsewhere from attrs.
    drop = set(time_cols_used) | set(rule.drop_fields) | std_consumed
    if rule.severity_field:
        drop.add(rule.severity_field)
    if rule.value_field:
        drop.add(rule.value_field)
    # Don't dump huge object columns (`is_delta` etc are kept as attrs).
    attrs = _build_attrs_columns(legacy, pack=rule.pack, drop=drop)

    events = pd.DataFrame(
        {
            schema.OCEL_EID: eids,
            schema.OCEL_ACTIVITY: activities,
            schema.OCEL_TIMESTAMP: ts_end.astype("datetime64[ns, UTC]"),
            schema.TS_START_TIMESTAMP: ts_start.astype("datetime64[ns, UTC]"),
            schema.TS_DURATION_S: duration.astype("float64"),
            schema.TS_DETECTOR: pd.Series([detector] * n, dtype="string"),
            schema.TS_PACK: pd.Series([rule.pack] * n, dtype="string"),
            schema.TS_SEVERITY: severity.astype("string"),
            schema.TS_VALUE: value.astype("float64"),
            **standard_extras,
            **attrs,
        }
    )

    object_bindings = _resolve_objects(legacy, rule, objects)
    objects_df, relations_df = _to_relations(eids, object_bindings, qualifiers)

    return EventLog(events=events, objects=objects_df, relations=relations_df)
