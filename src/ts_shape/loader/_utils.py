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

import datetime
import functools
import logging
import re
import time
import urllib.parse
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pandas as pd

from ts_shape.errors import DataQualityWarning, LoaderError
from ts_shape.utils.deprecation import caller_stacklevel

if TYPE_CHECKING:
    from sqlalchemy import URL
    from sqlalchemy.engine import Engine

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


#: Default DB-API driver of the PostgreSQL loaders (the ``postgres`` extra).
POSTGRES_DRIVER = "psycopg2"


def postgres_url(
    *,
    username: str,
    password: str,
    host: str,
    database: str,
    port: int | str = 5432,
    driver: str = POSTGRES_DRIVER,
) -> URL:
    """Connection URL of a PostgreSQL / TimescaleDB database.

    Built with :meth:`sqlalchemy.URL.create`, so a password with ``@``, ``:``,
    ``/`` or ``%`` needs no escaping, and with an explicit driver: SQLAlchemy
    2.1 changed the default PostgreSQL driver from psycopg2 to psycopg 3.

    Args:
        username: Database user.
        password: Database password, as is (not URL-encoded).
        host: Host name or address.
        database: Database name.
        port: Port. Default 5432.
        driver: DB-API driver name. Default ``"psycopg2"`` (the ``postgres``
            extra); ``"psycopg"`` and ``"pg8000"`` also work when installed.

    Returns:
        The URL. Its ``str()`` / ``repr()`` mask the password.
    """
    try:
        from sqlalchemy import URL
    except ImportError as exc:
        raise missing_optional("sqlalchemy", "sql") from exc

    return URL.create(
        f"postgresql+{driver}",
        username=username,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )


def create_postgres_engine(
    create_engine: Callable[..., Engine], url: URL, **kwargs: Any
) -> Engine:
    """``create_engine(url)``, naming the extra to install when the driver is missing.

    Args:
        create_engine: :func:`sqlalchemy.create_engine` (passed in so tests can
            replace the loader module's name).
        url: The connection URL, see :func:`postgres_url`.
        **kwargs: Passed to ``create_engine``.

    Raises:
        ImportError: The DB-API driver is not installed; for psycopg2 the
            message names ``pip install ts-shape[postgres]``.
    """
    try:
        return create_engine(url, **kwargs)
    except ImportError as exc:
        if url.get_driver_name() == POSTGRES_DRIVER:
            raise missing_optional("psycopg2-binary", "postgres") from exc
        raise


#: Shown in a loader's ``repr`` in place of a secret.
REDACTED = "***"


def redact_url(url: str) -> str:
    """``url`` with its password and query-string values replaced by ``***``.

    Query values often carry a SAS signature or an API key, and
    ``user:password@`` a password, so neither appears in a ``repr``.
    ``"https://u:pw@host/path?sig=abc"`` gives
    ``"https://u:***@host/path?sig=***"``.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return REDACTED
    netloc = parts.netloc
    if parts.password is not None:
        userinfo, _, hostport = netloc.rpartition("@")
        user = userinfo.split(":", 1)[0]
        netloc = f"{user}:{REDACTED}@{hostport}"
    query = "&".join(
        f"{key}={REDACTED}" if sep else key
        for key, sep, _ in (
            item.partition("=") for item in parts.query.split("&") if item
        )
    )
    return urllib.parse.urlunsplit(
        (parts.scheme, netloc, parts.path, query, parts.fragment)
    )


def redact_mapping(mapping: dict[str, Any] | None) -> dict[str, str]:
    """``mapping`` with every value replaced by ``***`` (headers, configs)."""
    return {key: REDACTED for key in (mapping or {})}


def loader_repr(obj: object, **fields: Any) -> str:
    """``ClassName(field=value, ...)`` for a loader's ``__repr__``.

    Callers pass only what is safe to show: never a password, key, token,
    connection string or SAS URL (use :func:`redact_url` for URLs).
    """
    shown = ", ".join(f"{name}={value!r}" for name, value in fields.items())
    return f"{type(obj).__name__}({shown})"


def azure_container_repr(obj: Any) -> str:
    """``repr`` of an Azure Blob loader: account, container and prefix only.

    The connection string, SAS token or credential the loader was created
    with is not shown.
    """
    client = obj.container_client
    return loader_repr(
        obj,
        account_name=getattr(client, "account_name", None),
        container_name=getattr(client, "container_name", None),
        prefix=obj.prefix,
    )


#: Unit of epoch numbers, see :func:`parse_utc_times`.
EpochUnit = Literal["s", "ms", "us", "ns"]

#: Epoch unit inferred from the largest absolute value, see :func:`epoch_unit`.
_EPOCH_UNIT_LIMITS: tuple[tuple[float, EpochUnit], ...] = (
    (1e11, "s"),
    (1e14, "ms"),
    (1e17, "us"),
)


def epoch_unit(values: pd.Series) -> EpochUnit:
    """Unit of epoch numbers, inferred from their magnitude.

    The largest absolute value decides: below ``1e11`` it is seconds, below
    ``1e14`` milliseconds, below ``1e17`` microseconds, otherwise
    nanoseconds. (Seconds up to ``1e11`` reach the year 5138; the current
    time in ms is about ``1.7e12``, in us ``1.7e15``, in ns ``1.7e18``.)

    Args:
        values: Numeric epoch values; nulls are ignored.

    Returns:
        ``"s"``, ``"ms"``, ``"us"`` or ``"ns"``; ``"s"`` when all are null.
    """
    peak = pd.to_numeric(values, errors="coerce").abs().max()
    if pd.isna(peak):
        return "s"
    for limit, unit in _EPOCH_UNIT_LIMITS:
        if peak < limit:
            return unit
    return "ns"


def parse_utc_times(values: pd.Series, unit: EpochUnit | None = None) -> pd.Series:
    """Parse timestamps to tz-aware UTC.

    Strings with an offset (``2024-01-01T00:00:00+01:00``) are converted to
    UTC; strings without one are taken as UTC. A column that mixes both
    (or mixes ISO layouts) is parsed value by value.

    Numbers are epoch times. Their unit is ``unit``, or, when None (the
    default), inferred from their magnitude (see :func:`epoch_unit`):
    ``1704067200``, ``1704067200000`` and ``1704067200000000000`` all give
    ``2024-01-01 00:00 UTC``.

    Args:
        values: Timestamps as text, numbers or datetimes.
        unit: Epoch unit of numeric values (``"s"``, ``"ms"``, ``"us"``,
            ``"ns"``); None infers it. Ignored for text and datetimes.

    Returns:
        ``datetime64[ns, UTC]`` values.
    """
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return pd.to_datetime(values, utc=True, unit=unit or epoch_unit(values))
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


def _utc_bound(bound: str | pd.Timestamp) -> pd.Timestamp:
    """``bound`` as a UTC Timestamp; a naive bound is taken as UTC."""
    ts = pd.Timestamp(bound)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _column_times(column: pd.Series) -> pd.Series:
    """A datetime view of ``column``; text and numbers are parsed as UTC."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return column
    return parse_utc_times(column)


def trim_to_time_range(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    *,
    time_column: str = "systime",
) -> pd.DataFrame:
    """Keep the rows whose ``time_column`` lies in ``[start, end]`` (both inclusive).

    Frames without ``time_column`` (e.g. because a column projection left it
    out) are returned unchanged.

    Bounds are read as the folder selection reads them: a naive bound is UTC
    (or, when the other bound is tz-aware, in that bound's timezone, as
    :func:`same_tz_bounds`), whatever the column's timezone. A tz-aware
    column is compared in its own timezone, which gives the same instants; a
    naive column is taken as UTC. A text or numeric column is parsed with
    :func:`parse_utc_times`, so mixed UTC offsets and epoch numbers work.

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
    if start is not None and end is not None:
        start, end = same_tz_bounds(start, end)
    times = _column_times(df[time_column])
    col_tz = getattr(times.dt, "tz", None)

    def _aligned(bound: str | pd.Timestamp) -> pd.Timestamp:
        utc = _utc_bound(bound)
        return utc if col_tz is not None else utc.tz_localize(None)

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= times >= _aligned(start)
    if end is not None:
        mask &= times <= _aligned(end)
    return df.loc[mask].reset_index(drop=True)


_DATE_ONLY = re.compile(r"\d{4}([-/]?)\d{2}\1\d{2}")


def is_date_only(bound: object) -> bool:
    """True for a calendar date with no time of day.

    ``"2024-01-31"``, ``"2024/01/31"``, ``"20240131"`` and a
    :class:`datetime.date` (not a :class:`datetime.datetime`) are dates;
    ``"2024-01-31 00:00"`` and Timestamps are not.
    """
    if isinstance(bound, datetime.date) and not isinstance(bound, datetime.datetime):
        return True
    return isinstance(bound, str) and _DATE_ONLY.fullmatch(bound.strip()) is not None


def trim_to_day_range(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    *,
    time_column: str = "systime",
) -> pd.DataFrame:
    """:func:`trim_to_time_range` for loaders whose folders are whole days.

    The same, except that an ``end`` given as a date only (see
    :func:`is_date_only`) keeps that whole day: ``("2024-01-01",
    "2024-01-01")`` keeps every row of 1 January (UTC).
    """
    if end is not None and is_date_only(end):
        end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    return trim_to_time_range(df, start, end, time_column=time_column)


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
