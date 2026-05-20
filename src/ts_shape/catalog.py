"""Programmatic catalog of ts-shape's public classes.

``list_detectors()`` lets a power user discover the 60+ event detectors and
loaders from a REPL or notebook instead of scanning the docs::

    import ts_shape
    ts_shape.list_detectors("events.quality")
"""

from __future__ import annotations

from typing import Optional

import pandas as pd  # type: ignore


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


def list_detectors(category: Optional[str] = None) -> pd.DataFrame:
    """Return a catalog of ts-shape's top-level public classes.

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
