"""Context

Utilities for enriching DataFrames with contextual information and mappings.

Classes:
- ValueMapper: Map categorical codes to readable values from external files.
  - map_values: Merge and replace a target column using a CSV/JSON mapping table.
"""

import importlib as _importlib
from typing import TYPE_CHECKING

# Explicit imports for type checkers and IDEs only (so ``from ts_shape import X``
# is typed as X, not object); at runtime __getattr__ below imports lazily.
# tests/typing/test_lazy_exports.py keeps this block in sync with the map.
if TYPE_CHECKING:
    from ts_shape.context.value_mapping import ValueMapper as ValueMapper

# Imported on first access, like the classes of the other packs.
_CLASSES = {
    "ValueMapper": "ts_shape.context.value_mapping",
}

__all__ = sorted(_CLASSES)


def __getattr__(name: str) -> object:
    if name in _CLASSES:
        return getattr(_importlib.import_module(_CLASSES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASSES))
