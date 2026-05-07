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
from ts_shape.eventlog.taxonomy import REGISTRY


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
    """Every registry entry should match a real detector method."""
    discovered = set(_walk_detector_methods())
    orphans = sorted(set(REGISTRY) - discovered)
    assert not orphans, (
        "Registry entries with no corresponding detector method:\n  "
        + "\n  ".join(f"{c}.{m}" for c, m in orphans[:30])
    )


def test_every_label_rule_has_valid_pack():
    valid = {"quality", "production", "engineering",
             "maintenance", "supplychain", "energy", "correlation"}
    bad = [(k, r.pack) for k, r in REGISTRY.items() if r.pack not in valid]
    assert not bad, f"invalid pack on entries: {bad[:10]}"


def test_every_label_rule_has_valid_shape():
    valid = {"point", "interval", "summary", "static"}
    bad = [(k, r.shape) for k, r in REGISTRY.items() if r.shape not in valid]
    assert not bad, f"invalid shape on entries: {bad[:10]}"
