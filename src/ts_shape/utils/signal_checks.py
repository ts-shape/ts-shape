"""Consistent input checks for detector calls (wired in by ``Base``).

Every public method of a detector in ``ts_shape.events`` is wrapped so that
its signal arguments are checked the same way, whichever pack it is in:

* a signal uuid (any argument named ``*_uuid`` / ``*_uuids``, except
  ``event_uuid``) that does not occur in a non-empty frame emits a
  :class:`~ts_shape.errors.DataQualityWarning` naming the uuids that do --
  instead of silently returning an empty result;
* a frame without the uuid column raises
  :class:`~ts_shape.errors.ColumnNotFoundError` (a ``ValueError``) instead of
  a bare ``KeyError`` deep inside the method;
* a column argument (``value_column``, ``value_column_<role>``,
  ``<role>_column``) that names a column the frame does not have raises
  :class:`~ts_shape.errors.ColumnNotFoundError` listing the columns, when the
  method looks it up.

The checks never change a result.
"""

from __future__ import annotations

import functools
import inspect
import warnings
import weakref
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel
from ts_shape.utils.normalize import unique_text

#: Arguments that name the output, not an input signal.
_NOT_SIGNALS = {"event_uuid", "uuid_column"}

_CHECKING = "_ts_shape_checking"
# The uuids of ``obj.dataframe``, kept by ``Base`` (which already computed
# them) so a method call does not scan the uuid column again.
_UUIDS = "_ts_shape_uuids"
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


def remember_uuids(obj: Any, frame: pd.DataFrame, column: str, uuids: set[str]) -> None:
    """Record ``uuids`` (as text) as the values of ``frame[column]``.

    Used while ``obj.dataframe`` is still ``frame`` (the same object, with
    the same length); a detector that replaces its frame gets its uuids
    computed again.
    """
    try:
        ref = weakref.ref(frame)
    except TypeError:
        return
    try:
        object.__setattr__(obj, _UUIDS, (ref, column, len(frame), frozenset(uuids)))
    except (AttributeError, TypeError):
        pass


def _present_uuids(obj: Any, frame: pd.DataFrame, column: str) -> set[str]:
    """``set(frame[column].dropna().astype(str).unique())``, cached per frame."""
    cached = getattr(obj, _UUIDS, None)
    if cached is not None:
        ref, cached_column, length, uuids = cached
        if ref() is frame and cached_column == column and length == len(frame):
            return set(uuids)
    uuids = unique_text(frame[column])
    remember_uuids(obj, frame, column, uuids)
    return set(uuids)


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
    present = _present_uuids(obj, frame, column)
    # A method that also takes its own frame (production_df, report_df ...)
    # may name signals that live there.
    for value in bound.arguments.values():
        if value is not frame and isinstance(value, pd.DataFrame):
            if column in value.columns:
                present |= unique_text(value[column])
    missing = [(n, u) for n, u in requested if u not in present]
    if missing:
        shown = sorted(present)[:10]
        more = f" (+{len(present) - 10} more)" if len(present) > 10 else ""
        details = ", ".join(f"{n}={u!r}" for n, u in missing)
        warnings.warn(
            f"{type(obj).__name__}: signal {details} not found in the data; "
            f"the result will be empty. Available uuids: {shown}{more}",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )


def _column_arguments(obj: Any, bound: Mapping[str, Any]) -> set[str]:
    """Column names the caller passed or the detector stored."""
    named = {**vars(obj), **bound} if hasattr(obj, "__dict__") else dict(bound)
    return {
        v
        for n, v in named.items()
        if isinstance(v, str)
        and isinstance(n, str)
        and "column" in n
        and not n.startswith("_ts_shape")
    }


def _missing_column(
    obj: Any, frame: Any, bound: Mapping[str, Any], err: KeyError
) -> ColumnNotFoundError | None:
    """A ColumnNotFoundError for ``err`` if it is a missing column argument."""
    key = err.args[0] if len(err.args) == 1 else None
    if not isinstance(key, str) or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    # pandas reports ``df["x"]`` as KeyError('x') and ``df[["x"]]`` as
    # KeyError("['x'] not in index").
    absent = sorted(c for c in _column_arguments(obj, bound) if c not in frame.columns)
    named = [c for c in absent if key == c or repr(c) in key]
    if not named:
        return None
    return ColumnNotFoundError(
        f"{type(obj).__name__}: the data has no {named[0]!r} column. "
        f"Columns: {list(frame.columns)}"
    )


def _bound_arguments(
    fn: Callable[..., Any], obj: Any, args: tuple, kwargs: dict
) -> dict[str, Any]:
    try:
        bound = inspect.signature(fn).bind_partial(obj, *args, **kwargs)
    except TypeError:
        return dict(kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _with_severity_level(obj: Any, method: str, result: Any) -> Any:
    """Add ``severity_level`` to a result that has categorical ``severity``.

    A detector class can set ``_severity_levels`` to label overrides keyed by
    method name (``"*"`` for every method), or to ``None`` when its
    ``severity`` is a numeric score.
    """
    if not isinstance(result, pd.DataFrame) or "severity" not in result.columns:
        return result
    table = getattr(type(obj), "_severity_levels", {})
    if table is None or pd.api.types.is_numeric_dtype(result["severity"]):
        return result
    from ts_shape.events._output import add_severity_level

    overrides = {**table.get("*", {}), **table.get(method, {})}
    return add_severity_level(result, overrides)


def checked(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a detector method with :func:`check_signals` (outermost call only)."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, _CHECKING, False):  # a detector method calling another
            return fn(self, *args, **kwargs)
        check_signals(self, fn, args, kwargs)
        object.__setattr__(self, _CHECKING, True)
        try:
            return _with_severity_level(self, fn.__name__, fn(self, *args, **kwargs))
        except KeyError as err:
            bound = _bound_arguments(fn, self, args, kwargs)
            frame, _ = _frame_uuids(self)
            error = _missing_column(self, frame, bound, err)
            if error is None:
                raise
            raise error from err
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
        try:
            fn(self, *args, **kwargs)
        except KeyError as err:
            arguments = _bound_arguments(fn, self, args, kwargs)
            error = _missing_column(self, arguments.get("dataframe"), arguments, err)
            if error is None:
                raise
            raise error from err
        if type(self).__init__ is wrapper or not getattr(self, _CHECKING, False):
            check_signals(self, fn, args, kwargs)

    _mark_checked(wrapper)
    return wrapper
