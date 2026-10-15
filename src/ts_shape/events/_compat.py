"""Shared helpers for consistent detector outputs.

* :func:`warn_dict_return` -- the deprecation warning for ``as_frame=False``
  (the legacy dict / list return of a summary method).
* :func:`state_mask` -- match a run-state value on either a string state
  signal (``value_string``) or a boolean run signal (``value_bool``).
* :func:`summary_bounds` -- ``start`` / ``end`` / ``duration_seconds`` of
  the data a one-row summary was computed from.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.utils.deprecation import REMOVAL, caller_stacklevel

__all__ = ["warn_dict_return", "state_mask", "uses_bool_state", "summary_bounds"]

_BOOL_COLUMN = "value_bool"
_STRING_COLUMN = "value_string"


def warn_dict_return(qualname: str, *, kind: str = "dict") -> None:
    """Warn that ``qualname(as_frame=False)`` (a dict/list return) is deprecated.

    The warning names the replacement and the removal release
    (:data:`ts_shape.utils.deprecation.REMOVAL`) and is attributed to the
    caller's line.
    """
    warnings.warn(
        f"{qualname}(as_frame=False) is deprecated: the {kind} return will be "
        f"removed in {REMOVAL}; use the default DataFrame return",
        DeprecationWarning,
        stacklevel=caller_stacklevel(),
    )


def uses_bool_state(data: pd.DataFrame, value_column: str, value: Any) -> bool:
    """True when ``value`` should be matched on the boolean run signal.

    That is the case when ``value`` itself is a bool, or when the state rows
    carry no value in ``value_column`` but do carry ``value_bool``.
    """
    if isinstance(value, (bool, np.bool_)):
        return True
    if _BOOL_COLUMN not in data.columns or not data[_BOOL_COLUMN].notna().any():
        return False
    return value_column not in data.columns or not data[value_column].notna().any()


def state_mask(
    data: pd.DataFrame,
    value_column: str,
    value: Any,
    *,
    bool_value: bool,
) -> pd.Series:
    """Rows of ``data`` whose state equals ``value``.

    A string state signal is compared as ``data[value_column] == value``.
    A boolean run signal is used instead when ``value`` is a bool (it is then
    matched directly) or when the rows carry ``value_bool`` but no
    ``value_column`` value; the bool is then compared with ``bool_value``
    (``True`` for "running", ``False`` for "stopped"). With a bool ``value``
    and the default ``value_column="value_string"`` the ``value_bool`` column
    is read; any other ``value_column`` is read as given. Missing booleans
    match nothing.
    """
    if not uses_bool_state(data, value_column, value):
        return data[value_column] == value
    if isinstance(value, (bool, np.bool_)):
        column = _BOOL_COLUMN if value_column == _STRING_COLUMN else value_column
        target = bool(value)
    else:
        column, target = _BOOL_COLUMN, bool_value
    if column not in data.columns:
        return pd.Series(False, index=data.index)
    raw = data[column]
    known = raw.notna()
    as_bool = raw.where(known, False).astype(bool)
    return known & (as_bool == target)


def summary_bounds(times: pd.Series) -> tuple[Any, Any, float]:
    """``(start, end, duration_seconds)`` spanned by ``times``.

    ``(NaT, NaT, 0.0)`` when there are no timestamps. The bounds keep the
    data's timezone.
    """
    times = pd.to_datetime(times.dropna()) if len(times) else times
    if len(times) == 0:
        return pd.NaT, pd.NaT, 0.0
    start, end = times.min(), times.max()
    return start, end, float((end - start).total_seconds())
