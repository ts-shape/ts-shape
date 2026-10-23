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


class LoaderConfigWarning(TsShapeWarning):
    """Warn when a loader returns no data, likely due to misconfiguration."""


class ColumnNotFoundError(ValueError):
    """Raised when a required column is missing from the DataFrame.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working, while callers that want to react specifically to a missing
    column can catch this narrower type.
    """


class LoaderError(Exception):
    """Raised when a loader cannot read from its configured source.

    Covers misconfiguration (missing path, absent credentials) and I/O that
    fails after the configured number of retries. Callers can catch this single
    type regardless of the underlying backend (local, S3, Azure, database).
    """


class NotChunkableError(ValueError):
    """Raised when a detector method cannot run slice by slice.

    :func:`ts_shape.run_chunked` and :meth:`ts_shape.Pipeline.run_stream`
    only run methods whose chunked result is known to equal the in-memory
    result (:data:`ts_shape.stream.SUPPORT`). Methods that need the whole
    signal at once -- whole-signal statistics, a model fit, a baseline taken
    from the first window -- raise this error. Subclasses ``ValueError`` so
    existing ``except ValueError`` handlers keep working.
    """


class ObjectNotFoundError(KeyError):
    """Raised when an object id is not in an event log.

    Raised by the genealogy queries (:func:`ts_shape.eventlog.where_used`,
    :func:`~ts_shape.eventlog.where_from`, :func:`~ts_shape.eventlog.lineage`)
    for an ``ocel:oid`` that is neither in ``objects`` nor in ``o2o``.
    Subclasses ``KeyError`` so lookups by id can be caught the usual way;
    the message is printed without the quotes ``KeyError`` adds.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""
