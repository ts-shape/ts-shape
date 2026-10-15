"""Shared helpers for ts-shape loaders.

Loaders pull data from local disk, object stores (S3/Azure), databases and
REST APIs. The two cross-cutting concerns are:

* **input validation** -- fail fast with a clear :class:`~ts_shape.errors.LoaderError`
  instead of silently returning an empty DataFrame when a source is missing or
  misconfigured.
* **transient-failure handling** -- network reads against object stores and
  APIs occasionally fail; :func:`retry_call` retries them with exponential
  backoff before giving up.

These helpers are intentionally backend-agnostic so every loader can reuse the
same behavior and error type.
"""

from __future__ import annotations

import functools
import logging
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar

import pandas as pd

from ts_shape.errors import DataQualityWarning, LoaderError
from ts_shape.utils.deprecation import caller_stacklevel

logger = logging.getLogger(__name__)

T = TypeVar("T")


#: Columns of a long-format signal frame, in their usual order.
SIGNAL_COLUMNS: tuple[str, ...] = (
    "systime",
    "uuid",
    "value_bool",
    "value_integer",
    "value_double",
    "value_string",
    "is_delta",
)


def empty_signal_frame(tz: str | None = None) -> pd.DataFrame:
    """A zero-row long-format frame with the standard columns and dtypes.

    Value columns use the nullable dtypes ``boolean`` / ``Int64`` /
    ``float64`` / ``object``, as :func:`ts_shape.datasets.make_timeseries`.

    Args:
        tz: Timezone of ``systime``; None for naive.

    Returns:
        The empty frame.
    """
    time_dtype = f"datetime64[ns, {tz}]" if tz else "datetime64[ns]"
    return pd.DataFrame(
        {
            "systime": pd.Series(dtype=time_dtype),
            "uuid": pd.Series(dtype=object),
            "value_bool": pd.Series(dtype="boolean"),
            "value_integer": pd.Series(dtype="Int64"),
            "value_double": pd.Series(dtype="float64"),
            "value_string": pd.Series(dtype=object),
            "is_delta": pd.Series(dtype="bool"),
        }
    )


def validate_local_path(path: str | Path, *, must_be_dir: bool = False) -> Path:
    """Validate that a local filesystem path exists.

    Args:
        path: The path to check.
        must_be_dir: When ``True``, also require the path to be a directory.

    Returns:
        The path as a :class:`pathlib.Path`.

    Raises:
        LoaderError: If the path does not exist, or ``must_be_dir`` is set and
            the path is not a directory.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise LoaderError(f"Path does not exist: {resolved}")
    if must_be_dir and not resolved.is_dir():
        raise LoaderError(f"Path is not a directory: {resolved}")
    return resolved


def require_config(config: dict, keys: list[str], *, name: str = "config") -> None:
    """Validate that ``config`` contains every key in ``keys``.

    Args:
        config: The configuration mapping to check.
        keys: Required keys.
        name: Label used in the error message (e.g. ``"s3_config"``).

    Raises:
        LoaderError: If any key is missing, listing all of the missing keys.
    """
    missing = [k for k in keys if k not in config]
    if missing:
        raise LoaderError(
            f"{name} is missing required key(s): {missing}. "
            f"Provided keys: {sorted(config)}"
        )


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    initial_delay: float = 0.5,
    backoff: float = 2.0,
    retry_exceptions: tuple[type[BaseException], ...] = (OSError,),
    exclude: tuple[type[BaseException], ...] = (),
    description: str = "loader call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` and retry on transient failures with exponential backoff.

    Args:
        func: A zero-argument callable performing the I/O.
        attempts: Maximum number of attempts (``>= 1``).
        initial_delay: Seconds to wait before the second attempt.
        backoff: Multiplier applied to the delay after each failed attempt.
        retry_exceptions: Exception types that trigger a retry.
        exclude: Exception types that are re-raised immediately even when they
            are a subclass of ``retry_exceptions`` (e.g. ``FileNotFoundError``,
            which means "no data" rather than a transient fault).
        description: Human-readable label used in log messages.
        sleep: Sleep function; injectable so tests run without real delays.

    Returns:
        Whatever ``func`` returns on the first successful attempt.

    Raises:
        ValueError: If ``attempts`` is less than 1.
        Exception: Re-raises the last exception once all attempts are exhausted,
            or immediately for any exception listed in ``exclude``.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    delay = initial_delay
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exclude:
            raise
        except retry_exceptions as exc:
            last_exc = exc
            if attempt == attempts:
                break
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.2gs.",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            sleep(delay)
            delay *= backoff

    # All attempts exhausted.
    assert last_exc is not None  # for type-checkers; loop guarantees it
    raise LoaderError(
        f"{description} failed after {attempts} attempt(s): {last_exc}"
    ) from last_exc


def missing_optional(package: str, extra: str) -> ImportError:
    """Return the ImportError raised when an optional backend is missing.

    Use it where an optional package is imported lazily::

        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:
            raise missing_optional("azure-storage-blob", "azure") from exc
    """
    return ImportError(
        f"This feature needs the optional package {package!r}. Install it with "
        f"`pip install ts-shape[{extra}]` (or `pip install {package}`)."
    )


def require_optional(obj: object, package: str, extra: str) -> None:
    """Raise a clear ImportError when an optional backend is not installed.

    ts-shape's core needs only pandas, numpy and scipy; each loader backend
    lives in an extra (``pip install ts-shape[extra]``). ``obj`` is the
    module-level name the optional import bound (``None`` when the import
    failed).
    """
    if obj is None:
        raise missing_optional(package, extra)


def parse_utc_times(values: pd.Series) -> pd.Series:
    """Parse timestamps to tz-aware UTC.

    Strings with an offset (``2024-01-01T00:00:00+01:00``) are converted to
    UTC; strings without one are taken as UTC. A column that mixes both
    (or mixes ISO layouts) is parsed value by value.

    Args:
        values: Timestamps as text, numbers or datetimes.

    Returns:
        ``datetime64[ns, UTC]`` values.
    """
    try:
        return pd.to_datetime(values, utc=True)
    except (TypeError, ValueError):
        return pd.to_datetime(values, utc=True, format="mixed")


def same_tz_bounds(
    start: str | pd.Timestamp, end: str | pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse two range bounds so they can be compared.

    When one bound is tz-aware and the other naive, the naive one is read in
    the timezone of the aware one.

    Args:
        start: Range start (anything ``pd.Timestamp`` accepts).
        end: Range end.

    Returns:
        ``(start, end)`` as Timestamps with the same timezone handling.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None and end.tzinfo is not None:
        start = start.tz_localize(end.tz)
    elif end.tzinfo is None and start.tzinfo is not None:
        end = end.tz_localize(start.tz)
    return start, end


def utc_folder_hours(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freq: str = "h",
) -> pd.DatetimeIndex:
    """Folder slots covering ``[start, end]`` for a layout written in UTC.

    The start is floored to ``freq`` (``09:30 .. 11:00`` gives ``09, 10, 11``).
    tz-aware bounds are converted to UTC first; naive bounds are taken as
    UTC. The result is naive, ready for ``strftime``.

    Args:
        start: Range start.
        end: Range end, inclusive.
        freq: Folder granularity (default hourly).

    Returns:
        Naive UTC timestamps, one per folder; empty when start is after end.
    """
    start, end = same_tz_bounds(start, end)
    start, end = utc_naive(start), utc_naive(end)
    if start > end:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start.floor(freq), end=end, freq=freq)


def utc_naive(timestamp: str | pd.Timestamp) -> pd.Timestamp:
    """``timestamp`` as naive UTC, to compare with ``YYYY/MM/DD/HH`` folder times.

    A tz-aware value is converted to UTC; a naive value is taken as UTC (the
    timezone the folder layouts are written in).
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _align_bound(bound: pd.Timestamp, column: pd.Series) -> pd.Timestamp:
    """Match ``bound``'s timezone-awareness to a datetime ``column``."""
    col_tz = getattr(column.dt, "tz", None)
    if col_tz is not None and bound.tz is None:
        return bound.tz_localize(col_tz)
    if col_tz is None and bound.tz is not None:
        return bound.tz_convert("UTC").tz_localize(None)
    return bound


def trim_to_time_range(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    *,
    time_column: str = "systime",
) -> pd.DataFrame:
    """Keep the rows whose ``time_column`` lies in ``[start, end]`` (both inclusive).

    Frames without ``time_column`` (e.g. because a column projection left it
    out) are returned unchanged. A naive bound is interpreted in the column's
    timezone; an aware bound is compared against a naive column in UTC.

    Args:
        df: Frame to trim.
        start: Inclusive lower bound; None leaves the start open.
        end: Inclusive upper bound; None leaves the end open.
        time_column: Name of the timestamp column to compare.

    Returns:
        The trimmed frame with a fresh ``RangeIndex``, or ``df`` itself when
        there is nothing to trim.
    """
    if df.empty or time_column not in df.columns:
        return df
    if start is None and end is None:
        return df
    times = pd.to_datetime(df[time_column])
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= times >= _align_bound(pd.to_datetime(start), times)
    if end is not None:
        mask &= times <= _align_bound(pd.to_datetime(end), times)
    return df.loc[mask].reset_index(drop=True)


#: How a loader that reads many objects treats one it cannot read: ``"warn"``
#: skips it and emits a :class:`~ts_shape.errors.DataQualityWarning` naming
#: every skipped object; ``"raise"`` raises :class:`~ts_shape.errors.LoaderError`.
ErrorsMode = Literal["warn", "raise"]


def check_errors_mode(errors: str) -> None:
    """Raise ``ValueError`` unless ``errors`` is ``"warn"`` or ``"raise"``."""
    if errors not in ("warn", "raise"):
        raise ValueError(f"errors must be 'warn' or 'raise'; got {errors!r}")


@functools.cache
def read_errors() -> tuple[type[Exception], ...]:
    """Exceptions a single object read can raise without aborting a bulk load.

    * ``OSError`` -- local and network I/O (``pyarrow``'s ``ArrowIOError`` is
      one).
    * ``ValueError`` -- unparseable content (``pyarrow``'s ``ArrowInvalid``,
      ``pandas``' ``ParserError`` / ``EmptyDataError``, ``UnicodeDecodeError``).
    * ``pyarrow.ArrowException`` -- the remaining parquet engine errors.
    * ``azure.core.exceptions.AzureError`` -- missing blob, permission,
      throttling and transport errors from Azure Blob Storage.

    Packages that are not installed are left out.
    """
    found: list[type[Exception]] = [OSError, ValueError]
    try:
        import pyarrow
    except ImportError:  # pragma: no cover - pyarrow ships with the loaders
        pass
    else:
        found.append(pyarrow.ArrowException)
    try:
        from azure.core.exceptions import AzureError
    except ImportError:  # pragma: no cover - only without the [azure] extra
        pass
    else:
        found.append(AzureError)
    return tuple(found)


def is_not_found(exc: BaseException) -> bool:
    """True when ``exc`` says the object does not exist (file or blob)."""
    if isinstance(exc, FileNotFoundError):
        return True
    try:
        from azure.core.exceptions import ResourceNotFoundError
    except ImportError:  # pragma: no cover - only without the [azure] extra
        return False
    return isinstance(exc, ResourceNotFoundError)


def describe_failures(failures: Sequence[tuple[str, BaseException]], what: str) -> str:
    """One-line summary naming every failed object and the first error."""
    names = [name for name, _ in failures]
    shown = ", ".join(names[:20])
    if len(names) > 20:
        shown += f" and {len(names) - 20} more"
    first = failures[0][1]
    return (
        f"{len(failures)} {what} could not be read and were skipped: {shown}. "
        f"First error: {type(first).__name__}: {first}"
    )


def report_failures(
    failures: Sequence[tuple[str, BaseException]],
    *,
    what: str,
    errors: ErrorsMode,
) -> None:
    """Warn about (or raise for) objects a bulk load could not read.

    Args:
        failures: ``(object name, exception)`` pairs; nothing happens when empty.
        what: Plural noun for the objects, e.g. ``"parquet blob(s)"``.
        errors: ``"warn"`` emits one :class:`~ts_shape.errors.DataQualityWarning`
            naming every failed object; ``"raise"`` raises
            :class:`~ts_shape.errors.LoaderError` from the first error.
    """
    if not failures:
        return
    message = describe_failures(failures, what)
    if errors == "raise":
        raise LoaderError(message) from failures[0][1]
    warnings.warn(message, DataQualityWarning, stacklevel=caller_stacklevel())
