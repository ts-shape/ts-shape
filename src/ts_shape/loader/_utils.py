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

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd

from ts_shape.errors import LoaderError

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


def require_optional(obj: object, package: str, extra: str) -> None:
    """Raise a clear ImportError when an optional backend is not installed.

    ts-shape's core needs only pandas, numpy and scipy; each loader backend
    lives in an extra (``pip install ts-shape[extra]``).
    """
    if obj is None:
        raise ImportError(
            f"This loader needs the optional package {package!r}. Install it with "
            f"`pip install ts-shape[{extra}]` (or `pip install {package}`)."
        )


def hourly_slots(
    start_timestamp: str | pd.Timestamp, end_timestamp: str | pd.Timestamp
) -> pd.DatetimeIndex:
    """Return every hour that overlaps the closed range ``[start, end]``.

    ``start`` is floored to the hour, so a range starting mid-hour still
    includes the hour that contains it, and the hour that contains ``end`` is
    always included: ``09:30 .. 11:00`` yields ``09:00, 10:00, 11:00``.

    Args:
        start_timestamp: Range start (anything ``pd.to_datetime`` accepts).
        end_timestamp: Range end, inclusive (anything ``pd.to_datetime`` accepts).

    Returns:
        Hour-aligned timestamps, one per hour folder to visit; empty when
        ``start`` is after ``end``.
    """
    start = pd.to_datetime(start_timestamp)
    end = pd.to_datetime(end_timestamp)
    if start > end:
        return pd.DatetimeIndex([], tz=start.tz)
    return pd.date_range(start=start.floor("h"), end=end.floor("h"), freq="h")


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
    start_timestamp: str | pd.Timestamp | None,
    end_timestamp: str | pd.Timestamp | None,
    *,
    time_column: str = "systime",
) -> pd.DataFrame:
    """Keep the rows whose ``time_column`` lies in ``[start, end]`` (both inclusive).

    Frames without ``time_column`` (e.g. because a column projection left it
    out) are returned unchanged. A naive bound is interpreted in the column's
    timezone; an aware bound is compared against a naive column in UTC.

    Args:
        df: Frame to trim.
        start_timestamp: Inclusive lower bound; None leaves the start open.
        end_timestamp: Inclusive upper bound; None leaves the end open.
        time_column: Name of the timestamp column to compare.

    Returns:
        The trimmed frame with a fresh ``RangeIndex``, or ``df`` itself when
        there is nothing to trim.
    """
    if df.empty or time_column not in df.columns:
        return df
    if start_timestamp is None and end_timestamp is None:
        return df
    times = pd.to_datetime(df[time_column])
    mask = pd.Series(True, index=df.index)
    if start_timestamp is not None:
        mask &= times >= _align_bound(pd.to_datetime(start_timestamp), times)
    if end_timestamp is not None:
        mask &= times <= _align_bound(pd.to_datetime(end_timestamp), times)
    return df.loc[mask].reset_index(drop=True)
