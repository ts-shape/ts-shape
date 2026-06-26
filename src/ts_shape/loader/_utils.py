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
same behaviour and error type.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

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
