"""Public ``to_event_log`` entry point — looks up the adapter and runs it."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from . import adapters, schema, taxonomy
from .model import EventLog


def to_event_log(
    df: pd.DataFrame,
    *,
    detector: str,
    objects: Mapping[str, object] | None = None,
    qualifiers: Mapping[str, str] | None = None,
    validate: bool = True,
) -> EventLog:
    """Normalize a legacy detector DataFrame into the canonical event log.

    ``detector`` is ``"ClassName.method_name"`` — the same key used in the
    taxonomy registry and the value written to ``ts_shape:detector``.

    ``objects`` binds OCEL object types to either:

    * a string column name in ``df`` (e.g. ``{"asset": "source_uuid"}``),
    * a callable taking a row dict and returning an oid,
    * a ``pd.Series`` aligned with ``df`` rows,
    * a scalar broadcast to every row.

    Caller-supplied bindings are always honored. Types listed in the
    adapter's ``LabelRule.produces_objects`` are *also* auto-extracted from
    standard legacy columns (e.g. ``source_uuid -> asset``) when no explicit
    binding is given.
    """
    if "." not in detector:
        raise ValueError(f"detector must be 'ClassName.method_name', got {detector!r}")
    class_name, method_name = detector.split(".", 1)

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
            df, rule=rule, detector=detector, objects=objects, qualifiers=qualifiers
        )

    if validate:
        schema.validate(log)
    return log
