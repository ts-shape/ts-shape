"""Moved to :mod:`ts_shape.events.profiling.runner` (November 2026).

Kept so old imports keep working: every name is read from the new module,
with a DeprecationWarning. Removed in the next major release.
"""

from ts_shape.utils.deprecation import moved_module

__getattr__, __dir__ = moved_module(
    "ts_shape.cycle_profile", "ts_shape.events.profiling.runner"
)

if __name__ == "__main__":  # pragma: no cover - python -m ts_shape.cycle_profile
    import warnings

    from ts_shape.events.profiling.runner import main

    warnings.warn(
        "python -m ts_shape.cycle_profile moved to python -m ts_shape.events.profiling",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main())
