"""Shape-driven adapters that convert detector frames (the DataFrames
detectors return) into the canonical :class:`~ts_shape.eventlog.model.EventLog`.

There is one adapter implementation per *shape* (point, interval, summary,
static). The per-method specifics live in
:mod:`ts_shape.eventlog.taxonomy` (the ``REGISTRY``). A custom adapter for
a specific ``(class, method)`` pair can be registered with
``@register_adapter(...)`` to override the shape-driven default — useful
for the few detectors whose frame layout does not fit any standard shape.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import uuid as _uuid
import warnings
from collections.abc import Callable, Mapping
from typing import SupportsFloat

import numpy as np
import pandas as pd

from ts_shape.utils.deprecation import caller_stacklevel

from ..errors import ColumnNotFoundError, DataQualityWarning
from ..events._output import COL_END, COL_START, COL_SYSTIME
from ..utils.coerce import to_numeric_checked
from ..utils.deprecation import renamed_kwargs
from . import schema
from .model import EventLog
from .taxonomy import (
    LabelRule,
    _is_missing,
    render_activity,
    resolve,
    template_fields,
)

# ----------------------------------------------------------------------------
# Registry of optional per-method overrides
# ----------------------------------------------------------------------------

AdapterFn = Callable[..., EventLog]
_OVERRIDES: dict[tuple[str, str], AdapterFn] = {}


def register_adapter(
    class_name: str, method_name: str
) -> Callable[[AdapterFn], AdapterFn]:
    """Register a custom adapter for a specific ``(class, method)``.

    The function will be called with ``(detector_frame, *, rule, detector,
    objects, qualifiers)``.
    """

    def deco(fn: AdapterFn) -> AdapterFn:
        _OVERRIDES[resolve(class_name, method_name)] = fn
        return fn

    return deco


def get_override(class_name: str, method_name: str) -> AdapterFn | None:
    """Return the custom adapter registered for ``(class, method)``.

    Deprecated class and method names resolve to their current names.

    Args:
        class_name: Detector class name, e.g. ``"MachineStateEvents"``.
        method_name: Method name on that class.

    Returns:
        The registered adapter function, or None when the shape-driven
        default applies.
    """
    return _OVERRIDES.get(resolve(class_name, method_name))


def has_override(class_name: str, method_name: str) -> bool:
    """Return True iff a custom adapter is registered for ``(class, method)``."""
    return resolve(class_name, method_name) in _OVERRIDES


def clear_overrides(
    class_name: str | None = None,
    method_name: str | None = None,
) -> None:
    """Remove registered override(s).

    * No arguments: clear every registration (typical test teardown).
    * Both arguments: clear that specific ``(class, method)`` only.
    * Class name only: clear every override on that class.

    Deprecated class and method names (e.g. ``CycleTimeTracking``) resolve
    to their current names, like :func:`register_adapter`. Has no effect if
    the targeted entries are already absent.

    Args:
        class_name: Detector class name; None clears every registration.
        method_name: Method name; None clears every method of ``class_name``.

    Raises:
        ValueError: If ``method_name`` is given without ``class_name``.
    """
    if class_name is None and method_name is None:
        _OVERRIDES.clear()
        return
    if class_name is not None and method_name is not None:
        _OVERRIDES.pop(resolve(class_name, method_name), None)
        return
    if class_name is not None:
        current = resolve(class_name, "")[0]
        for key in [k for k in _OVERRIDES if k[0] == current]:
            _OVERRIDES.pop(key, None)
        return
    raise ValueError("pass class_name when method_name is given")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

_NAMESPACE = _uuid.UUID("8a4d3f1c-9b2e-4d3a-8f1e-c0ffee123456")

#: Timestamp of ``static`` shape events when ``to_event_log`` gets no
#: ``timestamp``: the Unix epoch (UTC). A fixed value keeps event ids and
#: timestamps reproducible across runs.
STATIC_TIMESTAMP = pd.Timestamp("1970-01-01T00:00:00", tz="UTC")


def _stable(value: object) -> str:
    """Render one cell for the event-id hash, the same for every dtype.

    ``1``, ``1.0`` and ``Int64(1)`` render alike, missing values render as an
    empty marker and timestamps are rendered in UTC, so the id depends on the
    content of a row only -- not on its dtype or its position.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return "\x00"
    if isinstance(value, bool | np.bool_):
        return "True" if value else "False"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        f = float(value)
        if np.isnan(f):
            return "\x00"
        return str(int(f)) if f.is_integer() and abs(f) < 2**63 else repr(f)
    if isinstance(value, pd.Timestamp | np.datetime64 | _dt.datetime):
        ts = pd.Timestamp(value)
        if ts is pd.NaT:
            return "\x00"
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return str(ts.isoformat())
    if isinstance(value, pd.Timedelta | np.timedelta64 | _dt.timedelta):
        td = pd.Timedelta(value)
        return "\x00" if td is pd.NaT else str(td.value)
    return str(value)


_UNITS_PER_SECOND = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}


def _stable_datetimes(values: pd.Series) -> list[str] | None:
    """:func:`_stable` of every cell of a datetime64 column, vectorized.

    ``Timestamp.isoformat()`` in UTC: no fraction for whole seconds, else
    six digits (nine with a sub-microsecond part), then ``+00:00``. Returns
    None when the column falls outside what is rendered here (non-standard
    unit, years outside 1000-9999); the caller then renders cell by cell.
    """
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        values = values.dt.tz_convert("UTC").dt.tz_localize(None)
    arr = values.to_numpy()
    unit = np.datetime_data(arr.dtype)[0]
    per_second = _UNITS_PER_SECOND.get(unit)
    if per_second is None:
        return None
    nat = np.isnat(arr)
    ticks = arr.view("i8")
    seconds = np.floor_divide(ticks, per_second)
    frac_ns = (ticks - seconds * per_second) * (10**9 // per_second)
    whole = np.where(nat, 0, seconds).astype("datetime64[s]")
    years = whole.astype("datetime64[Y]").astype(np.int64) + 1970
    if len(arr) and ((years < 1000) | (years > 9999))[~nat].any():
        return None
    base = np.datetime_as_string(whole, unit="s").tolist()
    micro = (frac_ns // 1000).tolist()
    nano = (frac_ns % 1000).tolist()
    out = []
    for b, is_nat, us, ns in zip(base, nat.tolist(), micro, nano, strict=True):
        if is_nat:
            out.append("\x00")
        elif ns:
            out.append(f"{b}.{us:06d}{ns:03d}+00:00")
        elif us:
            out.append(f"{b}.{us:06d}+00:00")
        else:
            out.append(f"{b}+00:00")
    return out


def _stable_column(values: pd.Series) -> list[str]:
    """``[_stable(v) for v in values.tolist()]``, vectorized per dtype.

    Columns whose rendering is not reproduced here exactly (mixed objects,
    extension types other than strings, timedeltas) are rendered cell by
    cell with :func:`_stable`, so every column renders exactly as before.
    """
    dtype = values.dtype
    if isinstance(dtype, np.dtype):
        if dtype.kind == "b":
            return ["True" if v else "False" for v in values.to_numpy().tolist()]
        if dtype.kind in "iu":
            out: list[str] = values.to_numpy().astype(str).tolist()
            return out
        if dtype.kind == "f":
            arr = values.to_numpy(dtype=np.float64)
            nan = np.isnan(arr)
            with np.errstate(invalid="ignore"):
                integral = np.isfinite(arr) & (arr == np.floor(arr))
                integral &= np.abs(arr) < 2.0**63
            as_int = np.where(integral, arr, 0.0).astype(np.int64).astype(str)
            return [
                "\x00" if is_nan else (i if is_int else repr(f))
                for f, is_nan, is_int, i in zip(
                    arr.tolist(),
                    nan.tolist(),
                    integral.tolist(),
                    as_int.tolist(),
                    strict=True,
                )
            ]
    if dtype.kind == "M":
        rendered = _stable_datetimes(values)
        if rendered is not None:
            return rendered
    elif dtype == np.dtype(object) or isinstance(dtype, pd.StringDtype):
        if pd.api.types.infer_dtype(values, skipna=True) in ("string", "empty"):
            # Text and missing markers only: str(v) is v itself.
            missing = values.isna().to_numpy().tolist()
            return [
                "\x00" if m else str(v)
                for v, m in zip(values.tolist(), missing, strict=True)
            ]
    return [_stable(v) for v in values.tolist()]


def _uuid5_strings(names: list[str]) -> list[str]:
    """``str(uuid.uuid5(_NAMESPACE, name))`` for each name, without building
    ``UUID`` objects (the same SHA-1 digest, version and variant bits)."""
    prefix = _NAMESPACE.bytes
    sha1 = hashlib.sha1
    variant = "89ab89ab89ab89ab"  # (nibble & 0x3) | 0x8, indexed by nibble
    out = []
    for name in names:
        h = sha1(prefix + name.encode("utf-8")).hexdigest()
        out.append(
            f"{h[:8]}-{h[8:12]}-5{h[13:16]}-"
            f"{variant[int(h[16], 16)]}{h[17:20]}-{h[20:32]}"
        )
    return out


def _content_eids(
    detector: str,
    frame: pd.DataFrame,
    ts_start: pd.Series,
    ts_end: pd.Series,
    activities: pd.Series,
    time_cols: set[str],
) -> pd.Series:
    """Deterministic, content-based event ids.

    The id hashes the detector, the row's ``source_uuid``, its start / end
    timestamps, the activity and every other column of the row (values and
    payload). It does not depend on the row's position, so the same
    detector on two machines never collides, re-running gives the same ids,
    and concatenating identical logs in any row order deduplicates. Rows that
    are identical in every column get an occurrence suffix.

    Every cell is rendered with :func:`_stable` (one column at a time, see
    :func:`_stable_column`); the hash input of a row is
    ``detector|source|start|end|activity|<col>=<value>|...`` with the payload
    columns in sorted order, and the id is ``"e-" + uuid5(namespace, key)``.
    """
    n = len(frame)
    payload_cols = sorted(str(c) for c in frame.columns if c not in time_cols)
    rendered = [[f"{c}={v}" for v in _stable_column(frame[c])] for c in payload_cols]
    source = (
        _stable_column(frame["source_uuid"])
        if "source_uuid" in frame.columns
        else ["\x00"] * n
    )
    starts = _stable_column(ts_start)
    ends = _stable_column(ts_end)
    acts = [str(a) for a in activities.tolist()]
    head = detector + "|"
    keys = [
        head + "|".join(parts) for parts in zip(source, starts, ends, acts, *rendered)
    ]
    seen: dict[str, int] = {}
    for i, key in enumerate(keys):
        k = seen.get(key, 0)
        seen[key] = k + 1
        if k:
            keys[i] = f"{key}|#{k}"
    return pd.Series(["e-" + u for u in _uuid5_strings(keys)], dtype="string")


class MissingCanonicalColumn(ColumnNotFoundError, KeyError):
    """A detector frame lacks a time column of its shape.

    A :class:`~ts_shape.errors.ColumnNotFoundError` like every missing column,
    and a ``KeyError`` for code written against earlier releases.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _require_columns(
    df: pd.DataFrame, cols: tuple[str, ...], shape: str, detector: str
) -> None:
    """Raise if any canonical time column for ``shape`` is missing.

    Post-#62 every detector is guaranteed to emit these columns. A missing
    column means the caller fed a malformed frame.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MissingCanonicalColumn(
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


def _to_utc_column(values: pd.Series) -> pd.Series:
    """``values.apply(_to_utc_ts)``, vectorized for datetime64 columns.

    Naive times are localized to UTC, aware ones converted (same dtype and
    unit as the element-wise path). Other columns (text, objects) and an
    all-NaT column go through :func:`_to_utc_ts` element by element.
    """
    dtype = values.dtype
    if dtype.kind == "M" and values.notna().any():
        if isinstance(dtype, pd.DatetimeTZDtype):
            return values.dt.tz_convert("UTC")
        return values.dt.tz_localize("UTC")
    return values.apply(_to_utc_ts)


def _template_parts(template: str) -> list[tuple[bool, str]]:
    """The scan of :func:`render_activity`: ``(is_field, text)`` in order."""
    parts: list[tuple[bool, str]] = []
    i = 0
    while i < len(template):
        j = template.find("{", i)
        if j == -1:
            parts.append((False, template[i:]))
            break
        parts.append((False, template[i:j]))
        k = template.find("}", j + 1)
        if k == -1:
            parts.append((False, template[j:]))
            break
        parts.append((True, template[j + 1 : k]))
        i = k + 1
    return parts


def _render_activities(rule: LabelRule, frame: pd.DataFrame) -> pd.Series:
    """``frame.apply(lambda r: render_activity(rule, r), axis=1)`` as strings,
    rendered one template field (column) at a time.

    A row handed to ``apply`` holds the frame's values cast to one common
    dtype (``frame.values``): an integer next to a float column renders as
    ``5.0``. The same array is used here, so the text is identical. Frames
    whose rows would be extension arrays or datetimes, or with duplicate
    column names, take the row-wise path.
    """
    template = rule.template
    n = len(frame)
    if "{" not in template:
        return pd.Series([template] * n, index=frame.index, dtype="string")
    parts = _template_parts(template)
    fields = {name for is_field, name in parts if is_field}
    values = None
    if any(f in frame.columns for f in fields):
        row_wise = not frame.columns.is_unique or isinstance(
            frame.iloc[0].dtype, pd.api.extensions.ExtensionDtype
        )
        values = None if row_wise else frame.values
        if values is None or values.dtype.kind in "mM":
            return frame.apply(lambda r: render_activity(rule, r), axis=1).astype(
                "string"
            )
    rendered: dict[str, list[str]] = {}
    for name in fields:
        if values is None or name not in frame.columns:
            rendered[name] = ["unknown"] * n
            continue
        column = values[:, frame.columns.get_loc(name)]
        rendered[name] = [
            "unknown" if v is None or _is_missing(v) else str(v) for v in column
        ]
    pieces = [rendered[name] if is_field else [name] * n for is_field, name in parts]
    return pd.Series(
        ["".join(row) for row in zip(*pieces, strict=True)],
        index=frame.index,
        dtype="string",
    )


def _classify_severity(value: object) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        # A boolean flag ("is severe"): True -> warn, False -> info.
        return "warn" if value else "info"
    if not isinstance(value, SupportsFloat):
        return None
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
    frame: pd.DataFrame,
    *,
    pack: str,
    drop: set[str],
) -> dict[str, pd.Series]:
    """Return a dict of ``"<pack>:<col>" -> Series`` for non-canonical columns."""
    out: dict[str, pd.Series] = {}
    for col in frame.columns:
        if col in drop:
            continue
        out[f"{pack}:{col}"] = frame[col].reset_index(drop=True)
    return out


def _resolve_objects(
    frame: pd.DataFrame,
    rule: LabelRule,
    user_objects: Mapping[str, object] | None,
    index: pd.Index,
) -> dict[str, pd.Series]:
    """Produce ``object_type -> Series-of-oids`` aligned with the frame rows.

    Auto-extracts types declared in ``rule.produces_objects`` from the
    detector frame's standard columns (e.g. ``source_uuid -> asset``).
    Caller-supplied bindings via ``user_objects`` are always honored —
    object types not declared by the adapter are treated as contextual
    annotations the caller knows about (e.g. "this outlier happened
    during batch B-2026-117").

    ``frame`` has a fresh RangeIndex; ``index`` is the caller's original
    index, used to align a Series binding by label.
    """
    declared = set(rule.produces_objects)
    bindings: dict[str, pd.Series] = {}

    def _bind(otype: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, str) and value in frame.columns:
            bindings[otype] = frame[value].reset_index(drop=True).astype("string")
        elif callable(value):
            bindings[otype] = pd.Series(
                [value(r) for r in frame.to_dict("records")], dtype="string"
            )
        elif isinstance(value, pd.Series):
            bindings[otype] = _align_series(otype, value, index)
        else:
            # Scalar broadcast (a string that is not a column name included).
            bindings[otype] = pd.Series([str(value)] * len(frame), dtype="string")

    if user_objects:
        for otype, val in user_objects.items():
            _bind(otype, val)

    # Defaults: asset = source_uuid if column exists.
    if "asset" in declared and "asset" not in bindings:
        if "source_uuid" in frame.columns:
            bindings["asset"] = (
                frame["source_uuid"].reset_index(drop=True).astype("string")
            )

    # Drop empty / all-NaN bindings.
    return {k: v for k, v in bindings.items() if not v.isna().all()}


def _align_series(otype: str, value: pd.Series, index: pd.Index) -> pd.Series:
    """A Series binding, aligned with the detector frame by index label."""
    if value.index.equals(index):
        return value.reset_index(drop=True).astype("string")
    if len(value) != len(index):
        raise ValueError(
            f"objects[{otype!r}]: Series has {len(value)} rows, the detector "
            f"frame has {len(index)}; pass a Series with the frame's index"
        )
    if not value.index.is_unique or not index.is_unique:
        raise ValueError(
            f"objects[{otype!r}]: Series index does not match the detector "
            "frame's index and one of them has duplicate labels"
        )
    missing = index.difference(value.index)
    if len(missing):
        raise ValueError(
            f"objects[{otype!r}]: Series index lacks {len(missing)} label(s) of "
            f"the detector frame, e.g. {list(missing[:3])}; pass a Series with "
            "the frame's index"
        )
    return value.reindex(index).reset_index(drop=True).astype("string")


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


def _value_series(frame: pd.DataFrame, rule: LabelRule) -> pd.Series:
    if rule.value_field and rule.value_field in frame.columns:
        return to_numeric_checked(
            frame[rule.value_field], what=f"value column {rule.value_field!r}"
        )
    for c in ("value", "value_double", "value_integer", "ts_shape:value"):
        if c in frame.columns:
            return to_numeric_checked(frame[c], what=f"value column {c!r}")
    return pd.Series([float("nan")] * len(frame), dtype="float64")


_SEVERITY_VOCABULARY = ("info", "warn", "critical")
# Detector vocabularies mapped onto the canonical three levels.
_SEVERITY_ALIASES = {
    "warning": "warn",
    "low": "info",
    "minor": "info",
    "medium": "warn",
    "major": "warn",
    "high": "critical",
    "alarm": "critical",
    "error": "critical",
}


def _severity_series(frame: pd.DataFrame, rule: LabelRule) -> pd.Series:
    if rule.severity_field and rule.severity_field in frame.columns:
        col = frame[rule.severity_field]
    elif "severity" in frame.columns:
        col = frame["severity"]
    else:
        return pd.Series([pd.NA] * len(frame), dtype="string")
    out = col.map(_classify_severity).astype("string").str.lower()
    out = out.replace(_SEVERITY_ALIASES)
    unknown = out.notna() & ~out.isin(_SEVERITY_VOCABULARY)
    if unknown.any():
        bad = sorted(set(out[unknown].astype(str)))[:5]
        warnings.warn(
            "severity column contains values outside "
            f"{_SEVERITY_VOCABULARY}: {bad}; they are set to <NA>",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
        out = out.mask(unknown)
    return out


def _apply_standard_attrs(
    frame: pd.DataFrame,
    rule: LabelRule,
    n: int,
) -> tuple[dict[str, pd.Series], set[str]]:
    """Resolve LabelRule.standard_attrs into typed columns.

    Returns ``(extra_columns, source_columns_consumed)``. The caller adds
    ``source_columns_consumed`` to the drop set so the source column does
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
        numeric_target = target_dtype in ("Int64", "float64")
        if numeric_target and isinstance(source, str) and source not in frame.columns:
            # A numeric attribute named after a column this detector frame
            # does not carry (optional output): there is no value to attach.
            continue
        if isinstance(source, str) and source in frame.columns:
            # String matching a frame column → rename / coerce.
            series = frame[source].reset_index(drop=True)
            consumed.add(source)
        else:
            # Literal scalar (or string that doesn't match a column,
            # which is the common case for ts_shape:method = "zscore"
            # and similar enum-like values). Broadcast to every row.
            series = pd.Series([source] * n)

        if numeric_target:
            numeric = to_numeric_checked(series, what=f"attribute {std_key!r}")
            extras[std_key] = numeric.astype(target_dtype)
        else:
            extras[std_key] = series.astype("string")

    return extras, consumed


# ----------------------------------------------------------------------------
# The shape-driven adapter
# ----------------------------------------------------------------------------


@renamed_kwargs(legacy="frame")
def adapt(
    frame: pd.DataFrame,
    *,
    rule: LabelRule,
    detector: str,
    objects: Mapping[str, object] | None = None,
    qualifiers: Mapping[str, str] | None = None,
    timestamp: object | None = None,
) -> EventLog:
    """Convert one detector's output into a canonical :class:`EventLog`.

    Args:
        frame: The detector frame.
        rule: The label rule of the detector method.
        detector: ``"ClassName.method_name"``, written to ``ts_shape:detector``.
        objects: Object bindings (see :func:`~ts_shape.eventlog.to_event_log`).
        qualifiers: Relation qualifier per object type.
        timestamp: Timestamp of ``static`` shape events; default
            :data:`STATIC_TIMESTAMP` (the Unix epoch, UTC).

    Returns:
        The event log; an empty one for an empty frame.
    """
    if frame is None or len(frame) == 0:
        return EventLog()

    original_index = frame.index
    frame = frame.reset_index(drop=True)
    n = len(frame)

    # Bind objects on the full frame, so a Series binding aligns with the
    # caller's index before any row is dropped.
    object_bindings = _resolve_objects(frame, rule, objects, original_index)

    # Resolve timestamps based on shape. Detectors are required (post-#62) to
    # emit canonical time columns: ``systime`` for point, ``start``/``end``
    # for interval and summary. Missing columns indicate a malformed frame —
    # we fail fast rather than silently substituting ``utcnow()``. Naive
    # timestamps are read as UTC.
    if rule.shape in ("interval", "summary"):
        _require_columns(frame, (COL_START, COL_END), rule.shape, detector)
        ts_end = _to_utc_column(frame[COL_END])
        ts_start = _to_utc_column(frame[COL_START])
        time_cols_used = {COL_START, COL_END}
        backwards = (ts_end < ts_start).fillna(False).astype(bool)
        if backwards.any():
            warnings.warn(
                f"{detector}: {int(backwards.sum())} row(s) end before they "
                "start; they are dropped from the event log",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            keep = ~backwards.to_numpy()
            frame = frame.loc[keep].reset_index(drop=True)
            ts_end = ts_end.loc[keep].reset_index(drop=True)
            ts_start = ts_start.loc[keep].reset_index(drop=True)
            object_bindings = {
                k: v.loc[keep].reset_index(drop=True)
                for k, v in object_bindings.items()
            }
            n = len(frame)
            if n == 0:
                return EventLog()
        duration = (
            pd.to_datetime(ts_end, utc=True) - pd.to_datetime(ts_start, utc=True)
        ).dt.total_seconds()
    elif rule.shape == "point":
        _require_columns(frame, (COL_SYSTIME,), rule.shape, detector)
        ts_end = _to_utc_column(frame[COL_SYSTIME])
        ts_start = pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
        duration = pd.Series([float("nan")] * n, dtype="float64")
        time_cols_used = {COL_SYSTIME}
    elif rule.shape == "static":
        # No natural time: every row carries one explicit timestamp.
        ts = _to_utc_ts(timestamp) if timestamp is not None else STATIC_TIMESTAMP
        ts_end = pd.Series([ts] * n)
        ts_start = pd.Series([pd.NaT] * n, dtype="datetime64[ns, UTC]")
        duration = pd.Series([float("nan")] * n, dtype="float64")
        time_cols_used = set()
    else:
        raise ValueError(f"unknown shape {rule.shape!r}")

    # Render activity name (template substitution per row). A missing value
    # in a template field renders as "unknown".
    nan_fields = {
        f: int(frame[f].isna().sum())
        for f in template_fields(rule.template)
        if f in frame.columns and frame[f].isna().any()
    }
    if nan_fields:
        warnings.warn(
            f"{detector}: activity template {rule.template!r} has missing "
            f"values in {nan_fields}; they render as 'unknown'",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    activities = _render_activities(rule, frame)

    # Content-based event ids: same row content -> same id, independent of
    # row position.
    eids = _content_eids(detector, frame, ts_start, ts_end, activities, time_cols_used)

    severity = _severity_series(frame, rule)
    value = _value_series(frame, rule)

    # Resolve standard attribute extension (ts_shape:method, baseline,
    # threshold_low/high, etc). Source columns get added to ``drop`` so
    # they don't double-up under their ``<pack>:<col>`` prefix.
    standard_extras, std_consumed = _apply_standard_attrs(frame, rule, n)

    # Drop time columns + columns redundantly captured elsewhere from attrs.
    drop = set(time_cols_used) | set(rule.drop_fields) | std_consumed
    if rule.severity_field:
        drop.add(rule.severity_field)
    if rule.value_field:
        drop.add(rule.value_field)
    # Don't dump huge object columns (`is_delta` etc are kept as attrs).
    attrs = _build_attrs_columns(frame, pack=rule.pack, drop=drop)

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

    # Trim *empty* variable-extra columns. The canonical core
    # (EVENT_REQUIRED_COLUMNS ∪ EVENT_OPTIONAL_COLUMNS) is always retained so
    # every event log exposes an identical core schema — this keeps appends /
    # ``concat`` consistent and ``schema.validate`` / ``filter_by_pack``
    # working. Standard-attr extensions and ``<pack>:<col>`` passthroughs are
    # per-detector extras; an all-null one carries no information, so drop it.
    keep_always = set(schema.EVENT_REQUIRED_COLUMNS) | set(
        schema.EVENT_OPTIONAL_COLUMNS
    )
    drop_empty = [
        c for c in events.columns if c not in keep_always and events[c].isna().all()
    ]
    if drop_empty:
        events = events.drop(columns=drop_empty)

    object_bindings = {k: v for k, v in object_bindings.items() if not v.isna().all()}
    objects_df, relations_df = _to_relations(eids, object_bindings, qualifiers)

    return EventLog(events=events, objects=objects_df, relations=relations_df)
