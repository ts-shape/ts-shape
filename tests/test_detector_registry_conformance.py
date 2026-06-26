"""All-detector schema-conformance guard.

``ts_shape.eventlog.taxonomy.REGISTRY`` is the authoritative contract that maps
every ``(DetectorClass, method)`` to the canonical event *shape* its output
takes (``point`` / ``interval`` / ``summary`` / ``static``). The event-log
normalizer (:func:`ts_shape.eventlog.normalizer.to_event_log`) relies on that
mapping, so drift between the registry and the real classes silently breaks
OCEL/XES export.

These tests pin the whole detector surface at once:

* every registered class is importable from the top-level package,
* every registered method actually exists on its class,
* every declared shape is a canonical one, and the temporal shapes line up
  with the column schemas in ``ts_shape.events._output``.

Add a detector method or rename one without updating the registry (or vice
versa) and exactly the offending parametrized case fails -- naming the class
and method -- instead of a far-away normalizer error at runtime.
"""

from __future__ import annotations

import pytest

import ts_shape
from ts_shape.events._output import (
    INTERVAL_SCHEMA,
    POINT_SCHEMA,
    SUMMARY_SCHEMA,
    _schema_for,
    validate_event_output,
)
from ts_shape.eventlog.taxonomy import REGISTRY

# Canonical shapes. The first three have an explicit column schema in
# ``_output.py``; ``static`` is the non-temporal shape (e.g. driver rankings,
# capability tables) that carries no start/end and is intentionally exempt from
# the temporal column schemas.
TEMPORAL_SHAPES = {"point", "interval", "summary"}
KNOWN_SHAPES = TEMPORAL_SHAPES | {"static"}

# Sorted, stable list of registry entries with readable ids like
# "OEECalculator.calculate_oee".
_ENTRIES = sorted(REGISTRY.items(), key=lambda kv: (kv[0][0], kv[0][1]))
_IDS = [f"{cls}.{method}" for (cls, method), _ in _ENTRIES]


def test_registry_is_populated():
    assert REGISTRY, "taxonomy REGISTRY should not be empty"
    # Guard against accidental shrinkage of the detector surface.
    assert len({cls for (cls, _method) in REGISTRY}) >= 70


@pytest.mark.parametrize(("key", "rule"), _ENTRIES, ids=_IDS)
def test_registered_class_is_top_level_exported(key, rule):
    cls_name, _method = key
    cls = getattr(ts_shape, cls_name, None)
    assert cls is not None, (
        f"{cls_name} is in REGISTRY but is not re-exported from ts_shape. "
        "Add a row to the _LAZY map in ts_shape/__init__.py."
    )


@pytest.mark.parametrize(("key", "rule"), _ENTRIES, ids=_IDS)
def test_registered_method_exists_on_class(key, rule):
    cls_name, method = key
    cls = getattr(ts_shape, cls_name)
    attr = getattr(cls, method, None)
    assert callable(attr), (
        f"REGISTRY declares {cls_name}.{method} but that method does not exist "
        "(renamed or removed?). Update REGISTRY or the class."
    )


@pytest.mark.parametrize(("key", "rule"), _ENTRIES, ids=_IDS)
def test_registered_shape_is_canonical(key, rule):
    cls_name, method = key
    assert rule.shape in KNOWN_SHAPES, (
        f"{cls_name}.{method} declares unknown shape {rule.shape!r}; "
        f"expected one of {sorted(KNOWN_SHAPES)}."
    )


@pytest.mark.parametrize("shape", sorted(TEMPORAL_SHAPES))
def test_temporal_shapes_have_a_column_schema(shape):
    schema = _schema_for(shape)
    assert schema, f"temporal shape {shape!r} must define required columns"


def test_temporal_schema_constants_are_distinct():
    # Sanity-check the canonical schemas so the guard above is meaningful.
    assert POINT_SCHEMA != INTERVAL_SCHEMA != SUMMARY_SCHEMA


def test_live_detector_output_matches_its_registered_shape():
    """End-to-end teeth: a real detector run conforms to its REGISTRY shape.

    The metadata guards above prove the registry and classes agree; this proves
    a detector's *actual* output columns match the canonical schema declared for
    it -- closing the loop between contract and behaviour.
    """
    shape = REGISTRY[("OutlierDetectionEvents", "detect_outliers_zscore")].shape
    df = ts_shape.make_timeseries(["sensor:temp"], n_points=500, n_outliers=6)
    detector = ts_shape.OutlierDetectionEvents(df, value_column="value_double")
    out = detector.detect_outliers_zscore(threshold=3.0)
    # Raises if the declared shape's required columns are absent.
    validate_event_output(out, shape)
