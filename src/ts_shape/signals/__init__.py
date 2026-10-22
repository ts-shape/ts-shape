"""Signal semantics: specs, inference and spec-driven primitives.

Describe each signal once -- what kind of value it carries, how it is
encoded, how often it is sampled and when a gap means "no data" -- and let
the primitives handle counters, intervals, time weighting and alignment
correctly::

    from ts_shape.signals import infer_specs

    specs = infer_specs(df, overrides={"line1:parts": {"unit": "pcs"}})
    specs.to_frame()

:func:`identify_objects` reports which signals carry object identifiers
(serials, batches, orders, carriers, tools, ...) and their type, from the
values alone.
"""

from ts_shape.signals.identify import (
    IDENTIFIER_TYPES,
    ObjectReport,
    identify_objects,
    register_identifier_pattern,
    register_identifier_type,
)
from ts_shape.signals.infer import infer_spec, infer_specs
from ts_shape.signals.spec import ENCODINGS, KINDS, SignalSpec, SignalSpecs

__all__ = [
    "ENCODINGS",
    "IDENTIFIER_TYPES",
    "KINDS",
    "ObjectReport",
    "SignalSpec",
    "SignalSpecs",
    "identify_objects",
    "infer_spec",
    "infer_specs",
    "register_identifier_pattern",
    "register_identifier_type",
]
