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

import dataclasses
import re
import uuid as _uuid
import warnings
from typing import Callable, Mapping, Sequence

import pandas as pd

from . import schema
from .model import EventLog
from .taxonomy import LabelRule, render_activity, template_fields


# ----------------------------------------------------------------------------
# Registry of optional per-method overrides
# ----------------------------------------------------------------------------

AdapterFn = Callable[..., EventLog]
_OVERRIDES: dict[tuple[str, str], AdapterFn] = {}


def register_adapter(class_name: str, method_name: str) -> Callable[[AdapterFn], AdapterFn]:
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
"""Deterministic UUIDv5 namespace for ts-shape event ids — pinning this
value keeps eids stable across re-runs."""


def _to_utc_series(s: pd.Series) -> pd.Series:
    """Vectorised tz-aware UTC conversion. Naive timestamps are assumed UTC."""
    out = pd.to_datetime(s, errors="coerce", utc=True)
    return out.astype("datetime64[ns, UTC]")


def _pick_time_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    strict: bool = False,
) -> str | None:
    """Find the first column matching ``candidates``.

    ``strict=False`` (default) falls back to the first datetime-like
    column when no candidate matches — useful for the point/summary
    primary timestamp probe, where any datetime column is plausible.

    ``strict=True`` returns ``None`` when no named candidate matches —
    used for the interval start/end probes so a missing pair triggers a
    fallback to point shape rather than silently picking some other
    datetime column.
    """
    for c in candidates:
        if c in df.columns:
            return c
    if strict:
        return None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


_POINT_CANDIDATES = (
    "systime", "timestamp", "time", "event_time", "ts", "datetime",
    "window_start", "window_end", "period_start", "period_end", "date",
)

_VALID_SEVERITY = {"info", "warn", "critical"}
_LOOKS_LIKE_COLUMN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _classify_severity_series(s: pd.Series) -> pd.Series:
    """Vectorised numeric → ``info``/``warn``/``critical`` bucket."""
    numeric = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out = out.mask(numeric < 3.0, "info")
    out = out.mask((numeric >= 3.0) & (numeric < 4.5), "warn")
    out = out.mask(numeric >= 4.5, "critical")
    return out


def _passthrough_severity(s: pd.Series) -> pd.Series:
    """Validate a literal ``severity`` column. Values outside the canonical
    ``info`` / ``warn`` / ``critical`` set become ``<NA>`` with a warning.
    """
    string_s = s.astype("string")
    valid_mask = string_s.isin(_VALID_SEVERITY) | string_s.isna()
    if not bool(valid_mask.all()):
        bad = sorted(set(string_s[~valid_mask].dropna().tolist()))
        warnings.warn(
            f"severity column contains values outside info/warn/critical "
            f"({bad}); coercing to <NA>",
            stacklevel=2,
        )
        string_s = string_s.where(valid_mask, pd.NA)
    return string_s


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
    n = len(legacy)

    def _bind(otype: str, value: object) -> None:
        if isinstance(value, str):
            if value in legacy.columns:
                bindings[otype] = legacy[value].reset_index(drop=True).astype("string")
            else:
                # String literal broadcast (e.g. ``objects={"shift": "A"}``).
                bindings[otype] = pd.Series([value] * n, dtype="string")
        elif callable(value):
            bindings[otype] = pd.Series(
                [value(r) for r in legacy.to_dict("records")], dtype="string"
            )
        elif isinstance(value, pd.Series):
            bindings[otype] = value.reset_index(drop=True).astype("string")
        else:
            # Non-string scalar broadcast.
            bindings[otype] = pd.Series([str(value)] * n, dtype="string")

    if user_objects:
        for otype, val in user_objects.items():
            _bind(otype, val)

    # Defaults: asset = source_uuid if column exists.
    if "asset" in declared and "asset" not in bindings:
        if "source_uuid" in legacy.columns:
            bindings["asset"] = legacy["source_uuid"].reset_index(drop=True).astype("string")

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
        oids = oids.reset_index(drop=True).astype("string")
        mask = oids.notna() & (oids != "")
        if not bool(mask.any()):
            continue
        n_keep = int(mask.sum())
        rel_frames.append(pd.DataFrame({
            schema.OCEL_EID: eids[mask].reset_index(drop=True).astype("string"),
            schema.OCEL_OID: oids[mask].reset_index(drop=True),
            schema.OCEL_TYPE: pd.Series([otype] * n_keep, dtype="string"),
            schema.OCEL_QUALIFIER: pd.Series(
                [qualifiers.get(otype)] * n_keep, dtype="string"
            ),
        }))
        for o in oids[mask].dropna().unique().tolist():
            obj_pairs.append((str(o), otype))

    relations = (pd.concat(rel_frames, ignore_index=True)
                 if rel_frames else schema.empty_relations())
    if obj_pairs:
        obj_df = pd.DataFrame(obj_pairs, columns=[schema.OCEL_OID, schema.OCEL_TYPE])
        obj_df = obj_df.drop_duplicates().reset_index(drop=True)
        obj_df[schema.OCEL_OID] = obj_df[schema.OCEL_OID].astype("string")
        obj_df[schema.OCEL_TYPE] = obj_df[schema.OCEL_TYPE].astype("string")
    else:
        obj_df = schema.empty_objects()
    return obj_df, relations


def _value_series(legacy: pd.DataFrame, rule: LabelRule) -> pd.Series:
    if rule.value_field and rule.value_field in legacy.columns:
        return pd.to_numeric(legacy[rule.value_field], errors="coerce")
    for c in ("value", "value_double", "value_integer"):
        if c in legacy.columns:
            return pd.to_numeric(legacy[c], errors="coerce")
    return pd.Series([float("nan")] * len(legacy), dtype="float64")


def _severity_series(legacy: pd.DataFrame, rule: LabelRule) -> pd.Series:
    if rule.severity_field and rule.severity_field in legacy.columns:
        return _classify_severity_series(legacy[rule.severity_field])
    if "severity" in legacy.columns:
        return _passthrough_severity(legacy["severity"])
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
            if (
                isinstance(source, str)
                and target_dtype != "string"
                and _LOOKS_LIKE_COLUMN_NAME.match(source)
            ):
                # Numeric-typed standard attr received an identifier-like
                # string that doesn't match any legacy column. Almost
                # certainly a typo, since literal numerics for these keys
                # would be ints/floats, not strings.
                warnings.warn(
                    f"standard_attrs[{std_key!r}] = {source!r} looks like a "
                    f"column name but no column with that name exists; "
                    f"broadcasting as a literal value (likely a typo)",
                    stacklevel=3,
                )
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

def _resolve_timestamps(
    legacy: pd.DataFrame, rule: LabelRule, n: int,
) -> tuple[pd.Series, pd.Series, pd.Series, set[str], LabelRule | None]:
    """Return ``(ts_end, ts_start, duration_s, consumed_cols, fallback_rule)``.

    ``fallback_rule`` is non-None when the rule was ``interval`` but the
    legacy DataFrame lacks start/end columns — caller should re-dispatch
    using that point-shaped rule.
    """
    if rule.shape == "interval":
        start_col = _pick_time_col(legacy,
                                   ("start", "window_start", "period_start"),
                                   strict=True)
        end_col = _pick_time_col(legacy,
                                 ("end", "window_end", "period_end"),
                                 strict=True)
        if start_col is None or end_col is None:
            return (
                pd.Series(dtype="datetime64[ns, UTC]"),
                pd.Series(dtype="datetime64[ns, UTC]"),
                pd.Series(dtype="float64"),
                set(),
                dataclasses.replace(rule, shape="point"),
            )
        ts_end = _to_utc_series(legacy[end_col])
        ts_start = _to_utc_series(legacy[start_col])
        duration = (ts_end - ts_start).dt.total_seconds()
        return ts_end, ts_start, duration, {start_col, end_col}, None

    if rule.shape in {"point", "summary"}:
        time_col = _pick_time_col(legacy, _POINT_CANDIDATES)
        consumed: set[str] = set()
        if time_col is None:
            ts_end = pd.Series([pd.Timestamp.now(tz="UTC")] * n,
                               dtype="datetime64[ns, UTC]")
        else:
            ts_end = _to_utc_series(legacy[time_col])
            consumed.add(time_col)

        start_col = _pick_time_col(
            legacy, ("window_start", "period_start", "start"), strict=True,
        )
        if rule.shape == "summary" and start_col is not None and start_col != time_col:
            ts_start = _to_utc_series(legacy[start_col])
            consumed.add(start_col)
            duration = (ts_end - ts_start).dt.total_seconds()
        else:
            ts_start = pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
            duration = pd.Series([float("nan")] * n, dtype="float64")
        return ts_end, ts_start, duration, consumed, None

    if rule.shape == "static":
        now = pd.Timestamp.now(tz="UTC")
        ts_end = pd.Series([now] * n, dtype="datetime64[ns, UTC]")
        ts_start = pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
        duration = pd.Series([float("nan")] * n, dtype="float64")
        return ts_end, ts_start, duration, set(), None

    raise ValueError(f"unknown shape {rule.shape!r}")


def _render_activities(legacy: pd.DataFrame, rule: LabelRule, n: int) -> pd.Series:
    """Vectorised activity rendering. Literal templates broadcast; templated
    ones use vectorised string concatenation across the relevant columns.
    """
    fields = template_fields(rule.template)
    if not fields:
        return pd.Series([rule.template] * n, dtype="string")

    # Split the template once into (literal, field_name) chunks: literals
    # alternate with field substitutions.
    chunks: list[tuple[str, str | None]] = []
    pos = 0
    for m in _TEMPLATE_FIELD_RE.finditer(rule.template):
        if m.start() > pos:
            chunks.append((rule.template[pos:m.start()], None))
        chunks.append(("", m.group(1)))  # placeholder
        pos = m.end()
    if pos < len(rule.template):
        chunks.append((rule.template[pos:], None))

    out = pd.Series([""] * n, dtype="string")
    for literal, field_name in chunks:
        if field_name is None:
            out = out.str.cat(pd.Series([literal] * n, dtype="string"))
            continue
        if field_name in legacy.columns:
            col = legacy[field_name].astype("string")
            col = col.where(col.notna(), "unknown")
        else:
            col = pd.Series(["unknown"] * n, dtype="string")
        out = out.str.cat(col)
    return out


_TEMPLATE_FIELD_RE = re.compile(r"\{([^{}]+)\}")


def _build_eids(detector: str, ts_end: pd.Series, activities: pd.Series) -> pd.Series:
    """Generate stable UUIDv5 event ids — vectorised input formatting,
    only the ``uuid5`` call itself runs in Python.
    """
    n = len(ts_end)
    if n == 0:
        return pd.Series(dtype="string")
    ts_iso = ts_end.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    keys = (
        pd.Series([f"{detector}|"] * n, dtype="string")
        .str.cat(ts_iso.astype("string"))
        .str.cat(pd.Series([f"|{i}|" for i in range(n)], dtype="string"))
        .str.cat(activities.astype("string"))
    )
    eids = ["e-" + str(_uuid.uuid5(_NAMESPACE, k)) for k in keys.tolist()]
    return pd.Series(eids, dtype="string")


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

    ts_end, ts_start, duration, time_cols_used, fallback_rule = _resolve_timestamps(
        legacy, rule, n,
    )
    if fallback_rule is not None:
        return adapt(legacy, rule=fallback_rule, detector=detector,
                     objects=objects, qualifiers=qualifiers)

    activities = _render_activities(legacy, rule, n)
    eids = _build_eids(detector, ts_end, activities)

    severity = _severity_series(legacy, rule)
    value = _value_series(legacy, rule)

    standard_extras, std_consumed = _apply_standard_attrs(legacy, rule, n)

    drop = set(time_cols_used) | set(rule.drop_fields) | std_consumed
    if rule.severity_field:
        drop.add(rule.severity_field)
    if rule.value_field:
        drop.add(rule.value_field)
    attrs = _build_attrs_columns(legacy, pack=rule.pack, drop=drop)

    events = pd.DataFrame({
        schema.OCEL_EID: eids,
        schema.OCEL_ACTIVITY: activities,
        schema.OCEL_TIMESTAMP: ts_end,
        schema.TS_START_TIMESTAMP: ts_start,
        schema.TS_DURATION_S: duration.astype("float64"),
        schema.TS_DETECTOR: pd.Series([detector] * n, dtype="string"),
        schema.TS_PACK: pd.Series([rule.pack] * n, dtype="string"),
        schema.TS_SEVERITY: severity.astype("string"),
        schema.TS_VALUE: value.astype("float64"),
        **standard_extras,
        **attrs,
    })

    object_bindings = _resolve_objects(legacy, rule, objects)
    objects_df, relations_df = _to_relations(eids, object_bindings, qualifiers)

    return EventLog(events=events, objects=objects_df, relations=relations_df)
