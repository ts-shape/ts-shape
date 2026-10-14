"""Consistent input checks for detector calls (wired in by ``Base``).

Every public method of a detector in ``ts_shape.events`` is wrapped so that
its signal arguments are checked the same way, whichever pack it is in:

* a signal uuid (any argument named ``*_uuid`` / ``*_uuids``, except
  ``event_uuid``) that does not occur in a non-empty frame emits a
  :class:`~ts_shape.errors.DataQualityWarning` naming the uuids that do --
  instead of silently returning an empty result;
* a frame without the uuid column raises
  :class:`~ts_shape.errors.ColumnNotFoundError` (a ``ValueError``) instead of
  a bare ``KeyError`` deep inside the method.

The checks never change a result.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning

#: Arguments that name the output, not an input signal.
_NOT_SIGNALS = {"event_uuid", "uuid_column"}

_CHECKING = "_ts_shape_checking"
# Set on every wrapped method so ``Base`` can tell which methods are checked.
_CHECKED_MARKER = "__ts_shape_checked__"


def _mark_checked(wrapper: Callable[..., Any]) -> None:
    """Flag ``wrapper`` as signal-checked (read back with ``getattr``)."""
    setattr(wrapper, _CHECKED_MARKER, True)


def _signal_values(name: str, value: Any) -> list[str]:
    if name in _NOT_SIGNALS or not (name.endswith("_uuid") or name.endswith("_uuids")):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(k) for k in value]
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if isinstance(v, str)]
    return []


def _frame_uuids(obj: Any) -> tuple[pd.DataFrame | None, str]:
    frame = getattr(obj, "dataframe", None)
    column = getattr(obj, "uuid_column", None) or getattr(obj, "_uuid_column", "uuid")
    return (frame if isinstance(frame, pd.DataFrame) else None), str(column)


def check_signals(obj: Any, fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    """Warn about signal uuids that are not in ``obj.dataframe``."""
    frame, column = _frame_uuids(obj)
    if frame is None or frame.empty:
        return
    try:
        bound = inspect.signature(fn).bind_partial(obj, *args, **kwargs)
    except TypeError:  # deprecated keyword spelling: checked after renaming
        return
    requested = [
        (name, uuid)
        for name, value in bound.arguments.items()
        for uuid in _signal_values(name, value)
    ]
    if not requested:
        return
    if column not in frame.columns:
        raise ColumnNotFoundError(
            f"{type(obj).__name__}: the data has no {column!r} column to find "
            f"signal {requested[0][1]!r} in. Columns: {list(frame.columns)}"
        )
    present = set(frame[column].dropna().astype(str).unique())
    missing = [(n, u) for n, u in requested if u not in present]
    if missing:
        shown = sorted(present)[:10]
        more = f" (+{len(present) - 10} more)" if len(present) > 10 else ""
        details = ", ".join(f"{n}={u!r}" for n, u in missing)
        warnings.warn(
            f"{type(obj).__name__}: signal {details} not found in the data; "
            f"the result will be empty. Available uuids: {shown}{more}",
            DataQualityWarning,
            stacklevel=4,
        )


def checked(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a detector method with :func:`check_signals` (outermost call only)."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, _CHECKING, False):  # a detector method calling another
            return fn(self, *args, **kwargs)
        check_signals(self, fn, args, kwargs)
        object.__setattr__(self, _CHECKING, True)
        try:
            return fn(self, *args, **kwargs)
        finally:
            object.__setattr__(self, _CHECKING, False)

    _mark_checked(wrapper)
    return wrapper


def checked_init(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a detector constructor: column check before, uuid check after."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            bound = inspect.signature(fn).bind_partial(self, *args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None:
            frame = bound.arguments.get("dataframe")
            column = bound.arguments.get("uuid_column", "uuid")
            wants = any(_signal_values(n, v) for n, v in bound.arguments.items())
            if (
                wants
                and isinstance(frame, pd.DataFrame)
                and not frame.empty
                and isinstance(column, str)
                and column not in frame.columns
            ):
                raise ColumnNotFoundError(
                    f"{type(self).__name__}: the data has no {column!r} column. "
                    f"Columns: {list(frame.columns)}"
                )
        fn(self, *args, **kwargs)
        if type(self).__init__ is wrapper or not getattr(self, _CHECKING, False):
            check_signals(self, fn, args, kwargs)

    _mark_checked(wrapper)
    return wrapper
