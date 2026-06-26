"""Guards the top-level lazy re-export surface in ts_shape/__init__.py.

These tests catch the most likely drift: a name listed in ``__all__``
whose ``_LAZY`` target is wrong, or a new detector added to a pack
without a matching top-level entry.
"""

from __future__ import annotations

import importlib

import ts_shape


def test_all_names_are_importable():
    """Every name in ts_shape.__all__ resolves without error."""
    assert ts_shape.__all__, "__all__ should not be empty"
    for name in ts_shape.__all__:
        obj = getattr(ts_shape, name)
        assert obj is not None, name


def test_headline_classes_reachable_from_top_level():
    """A few well-known classes are importable straight from ts_shape."""
    from ts_shape import (  # noqa: F401
        MachineStateEvents,
        OEECalculator,
        OutlierDetectionEvents,
        to_event_log,
    )


def test_deep_imports_still_work():
    """The lazy re-exports are additive — deep imports are unchanged."""
    mod = importlib.import_module("ts_shape.events.quality.outlier_detection")
    assert hasattr(mod, "OutlierDetectionEvents")
    # The top-level alias is the very same object, not a copy.
    assert ts_shape.OutlierDetectionEvents is mod.OutlierDetectionEvents


def test_unknown_attribute_raises_attributeerror():
    try:
        ts_shape.ThisDoesNotExist  # noqa: B018
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError for unknown attribute")


def test_every_registered_detector_is_top_level_exported():
    """Every detector class in taxonomy.REGISTRY that lives in the
    physical source tree must be re-exported at the top level.

    This is the drift guard: add a detector + a REGISTRY entry and
    forget the ts_shape/__init__.py ``_LAZY`` row, and this fails.
    Lambda-rule detectors are dynamic (no source module) and exempt.
    """
    from ts_shape.eventlog.taxonomy import REGISTRY

    exported = set(ts_shape.__all__)
    missing = set()
    for class_name, _method in REGISTRY:
        if class_name in exported:
            continue
        # Exempt dynamically-registered (lambda-rule) detectors: they
        # have no importable module, so they cannot be re-exported.
        try:
            importlib.import_module("ts_shape.events")
        except Exception:  # pragma: no cover
            pass
        missing.add(class_name)

    # Resolve which of the "missing" actually have a real source module
    # (and therefore should have been exported).
    truly_missing = sorted(name for name in missing if _has_source_class(name))
    assert not truly_missing, (
        "detectors in REGISTRY missing from ts_shape.__all__: " f"{truly_missing}"
    )


def _has_source_class(class_name: str) -> bool:
    """True if ``class_name`` is defined by an importable ts_shape module."""
    import pkgutil

    import ts_shape.events as events_pkg

    for mod_info in pkgutil.walk_packages(
        events_pkg.__path__, prefix="ts_shape.events."
    ):
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception:
            continue
        obj = getattr(mod, class_name, None)
        if obj is not None and getattr(obj, "__module__", None) == mod_info.name:
            return True
    return False
