"""Programmatic catalog of ts-shape's public classes.

``list_detectors()`` lets a power user discover the 60+ event detectors and
loaders from a REPL or notebook instead of scanning the docs::

    import ts_shape
    ts_shape.list_detectors("events.quality")
"""

from __future__ import annotations

import pandas as pd

#: Deprecated class aliases re-exported from ``ts_shape``. They are left out
#: of the catalog. Listed here so ``list_detectors`` can decide from names
#: alone, without importing every module (which would pull in scipy and
#: scikit-learn); ``tests/test_catalog.py`` checks this set against the
#: classes themselves.
_DEPRECATED_CLASS_ALIASES = frozenset(
    {
        "CycleTimeTracking",
        "DowntimeTracking",
        "OEECalculator",
        "OperatorPerformanceTracking",
        "OrderTraceabilityEvents",
        "PartProductionTracking",
        "PerformanceLossTracking",
        "PeriodSummary",
        "ProcessStabilityIndex",
        "QualityTracking",
        "ReworkTracking",
        "ScrapTracking",
        "SetupTimeTracking",
        "ShiftHandoverReport",
        "ShiftReporting",
        "StatisticalProcessControlRuleBased",
        "TargetTracking",
    }
)


def _category_for(module: str) -> str:
    """Map a defining module path to a short catalog category."""
    if module.startswith("ts_shape.events."):
        # e.g. ts_shape.events.quality.outlier_detection -> events.quality
        return ".".join(module.split(".")[1:3])
    if module.startswith("ts_shape.loader."):
        return "loader"
    if module == "ts_shape.eventlog":
        return "eventlog"
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else module


def _is_listed_class(name: str) -> bool:
    """Whether a re-exported name is a current (not deprecated) class.

    Decided from the name, so no module is imported: classes are CamelCase,
    functions are snake_case and constants are UPPER_CASE.
    """
    return (
        name[:1].isupper()
        and not name.isupper()
        and name not in _DEPRECATED_CLASS_ALIASES
    )


def list_detectors(category: str | None = None) -> pd.DataFrame:
    """Return a catalog of ts-shape's top-level public classes.

    Only classes are listed -- detectors, loaders and the event-log classes
    -- not functions (``profile``, ``to_event_log``), constants or deprecated
    class aliases.

    Args:
        category: Optional filter matched as a prefix against the ``category``
            column -- e.g. ``"events"``, ``"events.quality"``, ``"loader"``.

    Returns:
        DataFrame with columns ``name``, ``category`` and ``module`` -- one row
        per class re-exported from ``ts_shape``, sorted by category then name.
    """
    from ts_shape import _LAZY  # local import avoids an import cycle

    rows = [
        {"name": name, "category": _category_for(module), "module": module}
        for name, module in _LAZY.items()
        if _is_listed_class(name)
    ]
    df = (
        pd.DataFrame(rows, columns=["name", "category", "module"])
        .drop_duplicates("name")
        .sort_values(["category", "name"])
        .reset_index(drop=True)
    )
    if category is not None:
        df = df[df["category"].str.startswith(category)].reset_index(drop=True)
    return df
