"""Coverage test: every public DataFrame-returning method on every detector
class under ``ts_shape.events.*`` must have a taxonomy entry. CI fails if a
new detector method ships without a registered LabelRule.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Iterator

import pandas as pd
import pytest

import ts_shape.events as events_pkg
from ts_shape.eventlog.schema import STANDARD_ATTR_KEYS
from ts_shape.eventlog.taxonomy import REGISTRY

# Required standard_attrs keys per archetype. Each method classified in
# ``ARCHETYPE_BY_METHOD`` (see archetypes.py) must populate at least these
# keys in its LabelRule.standard_attrs mapping.
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


# All 264 methods have been populated; enforcement is on.
_ENFORCE_REQUIRED_KEYS = True


# Methods inherited from utils.base.Base that are not detector outputs.
_BASE_METHODS = {"get_dataframe"}


def _walk_detector_methods() -> Iterator[tuple[str, str]]:
    for m in pkgutil.walk_packages(events_pkg.__path__, prefix="ts_shape.events."):
        if m.ispkg:
            continue
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for cname, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != m.name:
                continue
            for mname, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
                if mname.startswith("_") or mname in _BASE_METHODS:
                    continue
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                ann = sig.return_annotation
                if ann is pd.DataFrame or "DataFrame" in str(ann):
                    yield (cname, mname)


def test_every_detector_method_has_label_rule():
    discovered = set(_walk_detector_methods())
    missing = sorted(discovered - set(REGISTRY))
    assert not missing, (
        f"{len(missing)} detector method(s) lack a taxonomy entry — "
        f"add a LabelRule for each in ts_shape.eventlog.taxonomy.REGISTRY:\n  "
        + "\n  ".join(f"{c}.{m}" for c, m in missing[:30])
    )


def test_no_orphan_registry_entries():
    """Every registry entry should match a real detector method.

    Lambda-rule entries (class name starting with ``Lambda``) are exempt:
    they are registered dynamically by
    :func:`ts_shape.eventlog.register_lambda_rule` and intentionally have
    no Python class on disk.
    """
    discovered = set(_walk_detector_methods())
    orphans = sorted(
        k for k in set(REGISTRY) - discovered if not k[0].startswith("Lambda")
    )
    assert (
        not orphans
    ), "Registry entries with no corresponding detector method:\n  " + "\n  ".join(
        f"{c}.{m}" for c, m in orphans[:30]
    )


def test_every_label_rule_has_valid_pack():
    valid = {
        "quality",
        "production",
        "engineering",
        "maintenance",
        "supplychain",
        "energy",
        "correlation",
    }
    bad = [(k, r.pack) for k, r in REGISTRY.items() if r.pack not in valid]
    assert not bad, f"invalid pack on entries: {bad[:10]}"


def test_every_label_rule_has_valid_shape():
    valid = {"point", "interval", "summary", "static"}
    bad = [(k, r.shape) for k, r in REGISTRY.items() if r.shape not in valid]
    assert not bad, f"invalid shape on entries: {bad[:10]}"


def test_standard_attrs_use_known_keys():
    """Every key in any LabelRule.standard_attrs must be a registered
    standard key. Forbids ad-hoc invention outside the fixed vocabulary.
    """
    allowed = set(STANDARD_ATTR_KEYS)
    bad: list[tuple[tuple[str, str], str]] = []
    for key, rule in REGISTRY.items():
        for attr_key in rule.standard_attrs:
            if attr_key not in allowed:
                bad.append((key, attr_key))
    assert not bad, (
        "LabelRule entries use unknown standard_attrs keys (must be one of "
        f"{sorted(allowed)}):\n  "
        + "\n  ".join(f"{c}.{m}: {k}" for (c, m), k in bad[:20])
    )


def test_archetype_assignment_is_complete():
    """Every method in REGISTRY must be classified in ARCHETYPE_BY_METHOD."""
    from ts_shape.eventlog.archetypes import ARCHETYPE_BY_METHOD

    missing = sorted(set(REGISTRY) - set(ARCHETYPE_BY_METHOD))
    assert not missing, (
        f"{len(missing)} method(s) without an archetype classification — "
        "add to ts_shape.eventlog.archetypes.ARCHETYPE_BY_METHOD:\n  "
        + "\n  ".join(f"{c}.{m}" for c, m in missing[:30])
    )
    extra = sorted(set(ARCHETYPE_BY_METHOD) - set(REGISTRY))
    assert (
        not extra
    ), "ARCHETYPE_BY_METHOD has entries that are not in REGISTRY:\n  " + "\n  ".join(
        f"{c}.{m}" for c, m in extra[:30]
    )


def test_archetype_values_are_valid():
    from ts_shape.eventlog.archetypes import ARCHETYPE_BY_METHOD

    valid = set(_REQUIRED_KEYS_BY_ARCHETYPE)
    bad = [(k, v) for k, v in ARCHETYPE_BY_METHOD.items() if v not in valid]
    assert (
        not bad
    ), f"invalid archetype labels (must be one of {sorted(valid)}):\n  " + "\n  ".join(
        f"{c}.{m} = {a!r}" for (c, m), a in bad[:20]
    )


def test_required_standard_attrs_per_archetype():
    """For each method, the standard_attrs mapping contains at least the
    keys required by the method's archetype.

    Until Phase 8 finishes the population, this test is lenient (only
    reports missing keys). Flip ``_ENFORCE_REQUIRED_KEYS`` once every pack
    has been populated.
    """
    from ts_shape.eventlog.archetypes import ARCHETYPE_BY_METHOD

    missing: list[tuple[tuple[str, str], str, frozenset[str]]] = []
    for key, archetype in ARCHETYPE_BY_METHOD.items():
        rule = REGISTRY.get(key)
        if rule is None:
            continue
        required = _REQUIRED_KEYS_BY_ARCHETYPE[archetype]
        absent = required - set(rule.standard_attrs)
        if absent:
            missing.append((key, archetype, frozenset(absent)))

    if _ENFORCE_REQUIRED_KEYS:
        assert not missing, (
            f"{len(missing)} method(s) missing required standard_attrs keys "
            "for their archetype:\n  "
            + "\n  ".join(
                f"{c}.{m} ({arch}) missing {sorted(keys)}"
                for (c, m), arch, keys in missing[:30]
            )
        )
