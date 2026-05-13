"""Registration helpers + YAML/dict loaders for lambda rules.

A lambda rule lives in two places once registered:

* :data:`ts_shape.eventlog.taxonomy.REGISTRY` — the
  :class:`~ts_shape.eventlog.taxonomy.LabelRule` mapping that
  :func:`~ts_shape.eventlog.to_event_log` consults.
* :data:`ts_shape.eventlog.archetypes.ARCHETYPE_BY_METHOD` — the
  archetype classification that the coverage tests use to enforce
  ``standard_attrs`` completeness.

Both mutations happen through :func:`register_lambda_rule`. The
coverage test ``test_no_orphan_registry_entries`` is configured to
exempt entries whose class name starts with ``Lambda``, so registered
lambda rules do not need a real Python class on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..archetypes import ARCHETYPE_BY_METHOD
from ..schema import STANDARD_ATTR_KEYS
from ..taxonomy import REGISTRY, LabelRule
from .detector import LambdaDetector
from .spec import RuleSpec, TriggerSpec


# Required ``standard_attrs`` keys per archetype, mirroring the contract
# enforced by ``tests/eventlog/test_adapter_coverage.py``. Kept here so
# registration fails fast with a clear message instead of waiting for the
# coverage test to find the gap later.
_REQUIRED_KEYS_BY_ARCHETYPE: dict[str, frozenset[str]] = {
    "threshold": frozenset({"ts_shape:method", "ts_shape:direction"}),
    "interval": frozenset({"ts_shape:lifecycle_state"}),
    "aggregate": frozenset({"ts_shape:sample_count"}),
    "outcome": frozenset({"ts_shape:outcome"}),
    "static": frozenset({"ts_shape:method"}),
    "trace": frozenset({"ts_shape:lifecycle_state", "ts_shape:direction"}),
    "forecast": frozenset({"ts_shape:method", "ts_shape:confidence"}),
    "correlation": frozenset({"ts_shape:method"}),
}


def _validate_standard_attrs(spec: RuleSpec) -> None:
    allowed = set(STANDARD_ATTR_KEYS)
    bad = [k for k in spec.standard_attrs if k not in allowed]
    if bad:
        raise ValueError(
            f"lambda rule {spec.class_name}.{spec.method_name}: "
            f"standard_attrs uses unknown keys {bad}; "
            f"must be a subset of {sorted(allowed)}"
        )
    required = _REQUIRED_KEYS_BY_ARCHETYPE[spec.archetype]
    missing = required - set(spec.standard_attrs)
    if missing:
        raise ValueError(
            f"lambda rule {spec.class_name}.{spec.method_name} "
            f"(archetype={spec.archetype!r}) is missing required "
            f"standard_attrs keys: {sorted(missing)}"
        )


def register_lambda_rule(spec: RuleSpec) -> LambdaDetector:
    """Register ``spec`` in REGISTRY + ARCHETYPE_BY_METHOD; return runnable detector.

    Raises :class:`ValueError` if the spec is malformed, the
    ``standard_attrs`` mapping uses unknown keys, the archetype's
    required keys are missing, or a rule with the same
    ``(class_name, method_name)`` is already registered.
    """
    _validate_standard_attrs(spec)
    key = (spec.class_name, spec.method_name)
    if key in REGISTRY:
        raise ValueError(
            f"a rule for {spec.class_name}.{spec.method_name} is already registered"
        )

    REGISTRY[key] = LabelRule(
        template=spec.template,
        pack=spec.pack,
        shape=spec.shape,
        produces_objects=spec.produces_objects,
        severity_field=spec.severity_field,
        value_field=spec.value_field,
        standard_attrs=dict(spec.standard_attrs),
    )
    ARCHETYPE_BY_METHOD[key] = spec.archetype
    return LambdaDetector(spec)


def unregister_lambda_rule(class_name: str, method_name: str) -> None:
    """Remove a rule previously installed by :func:`register_lambda_rule`.

    Used by tests to keep the global REGISTRY clean between cases. No-op
    if the rule was not registered.
    """
    key = (class_name, method_name)
    REGISTRY.pop(key, None)
    ARCHETYPE_BY_METHOD.pop(key, None)


def _spec_from_dict(entry: Mapping[str, Any]) -> RuleSpec:
    trigger_raw = dict(entry.get("trigger", {}))
    trigger = TriggerSpec(
        expression=str(trigger_raw["expression"]),
        min_duration_s=(float(trigger_raw["min_duration_s"])
                        if trigger_raw.get("min_duration_s") is not None
                        else None),
        group_by=tuple(trigger_raw.get("group_by", ())),
    )
    return RuleSpec(
        id=str(entry["id"]),
        class_name=str(entry["class_name"]),
        method_name=str(entry["method_name"]),
        pack=str(entry["pack"]),
        shape=str(entry["shape"]),
        archetype=str(entry["archetype"]),
        template=str(entry["template"]),
        trigger=trigger,
        produces_objects=tuple(entry.get("produces_objects", ("asset",))),
        severity_field=entry.get("severity_field"),
        value_field=entry.get("value_field"),
        standard_attrs=dict(entry.get("standard_attrs") or {}),
    )


def load_dicts(entries: Iterable[Mapping[str, Any]]) -> list[LambdaDetector]:
    """Compile + register an iterable of dict rules."""
    return [register_lambda_rule(_spec_from_dict(e)) for e in entries]


def load_yaml(path: str | Path) -> list[LambdaDetector]:
    """Load and register every rule under a YAML file's ``rules:`` key.

    YAML is imported lazily so :mod:`ts_shape.eventlog` does not gain a
    hard runtime dependency on PyYAML for users who only use built-in
    detectors. Install ``pyyaml`` to enable this loader.
    """
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "load_yaml requires PyYAML. Install with `pip install pyyaml` "
            "or use load_dicts(...) with a hand-built list."
        ) from exc
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping) or "rules" not in raw:
        raise ValueError(
            f"YAML file {path!s} must define a top-level 'rules:' list"
        )
    return load_dicts(raw["rules"])
