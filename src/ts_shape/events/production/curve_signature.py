"""Moved to :mod:`ts_shape.events.profiling.curves` (November 2026).

Kept so old imports keep working: every name is read from the new module,
with a DeprecationWarning. Removed in the next major release.
"""

from ts_shape.utils.deprecation import moved_module

__getattr__, __dir__ = moved_module(
    "ts_shape.events.production.curve_signature", "ts_shape.events.profiling.curves"
)
