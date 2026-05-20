"""
Custom warnings and exceptions for ts-shape.

Follows the pandas/scikit-learn pattern:
- ``warnings.warn()`` for user-facing feedback (visible by default).
- ``logging`` for internal diagnostics (silent unless configured).

Users can filter specific categories, e.g.::

    import warnings
    from ts_shape.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
"""


class TsShapeWarning(UserWarning):
    """Base warning for all ts-shape warnings."""


class PerformanceWarning(TsShapeWarning):
    """Warn when an operation may be slow due to data size or shape."""


class DataQualityWarning(TsShapeWarning):
    """Warn about potential data quality issues (gaps, duplicates, NaNs)."""


class ColumnNotFoundError(ValueError):
    """Raised when a required column is missing from the DataFrame.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working, while callers that want to react specifically to a missing
    column can catch this narrower type.
    """
