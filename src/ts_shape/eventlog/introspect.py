"""Introspection helpers — answer "what does this method emit?" without
forcing readers to crawl the registry, archetypes table, and override
table by hand.
"""
from __future__ import annotations

from typing import Any

from . import adapters
from .archetypes import ARCHETYPE_BY_METHOD
from .taxonomy import REGISTRY


def describe(detector: str) -> dict[str, Any]:
    """Return everything ts-shape knows about a detector method.

    ``detector`` is ``"ClassName.method_name"`` — the same key accepted by
    :func:`~ts_shape.eventlog.to_event_log`.

    Returns a dict with:

    * ``class_name`` / ``method_name`` — split form.
    * ``pack``, ``shape``, ``activity_template``, ``produces_objects``,
      ``severity_field``, ``value_field``, ``drop_fields``,
      ``standard_attrs`` — fields of the registered :class:`LabelRule`.
    * ``archetype`` — the conceptual archetype (one of ``threshold``,
      ``interval``, ``aggregate``, ``outcome``, ``static``, ``trace``,
      ``forecast``, ``correlation``).
    * ``has_override`` — True if a custom adapter is registered.

    Raises ``KeyError`` if ``detector`` is not in the registry.
    """
    if "." not in detector:
        raise ValueError(
            f"detector must be 'ClassName.method_name', got {detector!r}"
        )
    class_name, method_name = detector.split(".", 1)
    key = (class_name, method_name)

    rule = REGISTRY.get(key)
    if rule is None:
        raise KeyError(
            f"no taxonomy entry for {detector!r}; "
            "add a LabelRule in ts_shape.eventlog.taxonomy.REGISTRY"
        )

    return {
        "class_name": class_name,
        "method_name": method_name,
        "pack": rule.pack,
        "shape": rule.shape,
        "archetype": ARCHETYPE_BY_METHOD.get(key),
        "activity_template": rule.template,
        "produces_objects": rule.produces_objects,
        "severity_field": rule.severity_field,
        "value_field": rule.value_field,
        "drop_fields": rule.drop_fields,
        "standard_attrs": dict(rule.standard_attrs),
        "has_override": adapters.has_override(class_name, method_name),
    }
