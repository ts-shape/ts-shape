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

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ...loader._utils import missing_optional
from ..archetypes import ARCHETYPE_BY_METHOD
from ..schema import STANDARD_ATTR_KEYS
from ..taxonomy import REGISTRY, LabelRule, resolve
from .detector import LambdaDetector
from .expression import compile_expression
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

    The trigger expression is compiled here, so a bad expression fails at
    registration rather than at the first evaluation.

    Args:
        spec: The rule to register.

    Returns:
        The runnable :class:`LambdaDetector`.

    Raises:
        ValueError: If the spec is malformed, the expression does not compile
            (:class:`UnsafeExpression`), the ``standard_attrs`` mapping uses
            unknown keys, the archetype's required keys are missing, or a rule
            with the same ``(class_name, method_name)`` is already registered.
    """
    compile_expression(spec.trigger.expression)
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
    if the rule was not registered. Deprecated class / method names resolve
    to their current names, as in the registry lookup.

    Args:
        class_name: The rule's class name.
        method_name: The rule's method name.
    """
    key = resolve(class_name, method_name)
    REGISTRY.pop(key, None)
    ARCHETYPE_BY_METHOD.pop(key, None)


_REQUIRED_FIELDS = (
    "id",
    "class_name",
    "method_name",
    "pack",
    "shape",
    "archetype",
    "template",
    "trigger",
)


def _spec_from_dict(entry: Mapping[str, Any], position: int) -> RuleSpec:
    if not isinstance(entry, Mapping):
        raise ValueError(f"rule #{position} must be a mapping, got {type(entry)}")
    where = f"rule #{position} ({entry.get('id', '?')!r})"
    missing = [k for k in _REQUIRED_FIELDS if k not in entry]
    if missing:
        raise ValueError(f"{where} is missing required field(s) {missing}")
    trigger_raw = entry["trigger"]
    if not isinstance(trigger_raw, Mapping) or "expression" not in trigger_raw:
        raise ValueError(
            f"{where}: 'trigger' must be a mapping with an 'expression', "
            f"got {trigger_raw!r}"
        )
    group_by = trigger_raw.get("group_by") or ()
    if isinstance(group_by, str):
        group_by = (group_by,)
    produces = entry.get("produces_objects", ("asset",))
    if isinstance(produces, str):
        produces = (produces,)
    trigger = TriggerSpec(
        expression=str(trigger_raw["expression"]),
        min_duration_s=(
            float(trigger_raw["min_duration_s"])
            if trigger_raw.get("min_duration_s") is not None
            else None
        ),
        group_by=tuple(str(g) for g in group_by),
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
        produces_objects=tuple(produces),
        severity_field=entry.get("severity_field"),
        value_field=entry.get("value_field"),
        standard_attrs=dict(entry.get("standard_attrs") or {}),
    )


def load_dicts(entries: Iterable[Mapping[str, Any]]) -> list[LambdaDetector]:
    """Compile and register an iterable of dict rules, all or nothing.

    Every rule is validated and its trigger compiled before any is
    registered; if one fails, the rules registered by this call are removed
    again and the error is raised.

    Args:
        entries: Rule mappings with the :class:`RuleSpec` fields; ``trigger``
            is a mapping with ``expression`` (and optional ``min_duration_s``,
            ``group_by`` -- a column name or a list of them).

    Returns:
        One :class:`LambdaDetector` per rule, in input order.

    Raises:
        ValueError: For a malformed rule, a duplicate registration or an
            expression that does not compile (:class:`UnsafeExpression`).
    """
    specs = [_spec_from_dict(e, i) for i, e in enumerate(entries)]
    for spec in specs:
        compile_expression(spec.trigger.expression)
        _validate_standard_attrs(spec)
    keys = [(s.class_name, s.method_name) for s in specs]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"rules registered twice in one batch: {sorted(dup)}")
    done: list[RuleSpec] = []
    try:
        for spec in specs:
            done.append(spec)
            register_lambda_rule(spec)
    except Exception:
        for spec in done[:-1]:
            unregister_lambda_rule(spec.class_name, spec.method_name)
        raise
    return [LambdaDetector(spec) for spec in specs]


def load_yaml(path: str | Path) -> list[LambdaDetector]:
    """Load and register every rule under a YAML file's ``rules:`` key.

    YAML is imported lazily so :mod:`ts_shape.eventlog` does not gain a
    hard runtime dependency on PyYAML for users who only use built-in
    detectors. Install ``ts-shape[yaml]`` to enable this loader, or use
    :func:`load_dicts` with a hand-built list.

    Args:
        path: Path to a YAML file with a top-level ``rules:`` list.

    Returns:
        One :class:`LambdaDetector` per rule (see :func:`load_dicts`).

    Raises:
        ImportError: When PyYAML is not installed.
        ValueError: When the file has no ``rules:`` list or a rule is invalid.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - import error path
        raise missing_optional("pyyaml", "yaml") from exc
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping) or "rules" not in raw:
        raise ValueError(f"YAML file {path!s} must define a top-level 'rules:' list")
    return load_dicts(raw["rules"])
