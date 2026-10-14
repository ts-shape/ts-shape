"""Signal semantics: specs, inference and spec-driven primitives.

Describe each signal once -- what kind of value it carries, how it is
encoded, how often it is sampled and when a gap means "no data" -- and let
the primitives handle counters, intervals, time weighting and alignment
correctly::

    from ts_shape.signals import infer_specs

    specs = infer_specs(df, overrides={"line1:parts": {"unit": "pcs"}})
    specs.to_frame()
"""

from ts_shape.signals.infer import infer_spec, infer_specs
from ts_shape.signals.spec import ENCODINGS, KINDS, SignalSpec, SignalSpecs

__all__ = [
    "ENCODINGS",
    "KINDS",
    "SignalSpec",
    "SignalSpecs",
    "infer_spec",
    "infer_specs",
]
