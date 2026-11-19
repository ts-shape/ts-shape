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

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.utils.coerce import bool_or_false, run_mask
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
    if isinstance(value, (bool, np.bool_)) or value_column == _BOOL_COLUMN:
        return True
    if _BOOL_COLUMN not in data.columns or not data[_BOOL_COLUMN].notna().any():
        return False
    return value_column not in data.columns or not data[value_column].notna().any()


def _is_state_set(value: Any) -> bool:
    """Whether ``value`` lists several state values (a set, list or tuple)."""
    return isinstance(value, (set, frozenset, list, tuple))


def _warn_unmatched_state(values: pd.Series, value: Any) -> None:
    """Warn when a text state value is compared with a numeric state signal.

    That is a mapping problem (a mode signal such as 1/2/5 read with
    ``"Running"``), not a state that just did not occur in the window, which
    is normal and not reported.
    """
    known = values.dropna()
    if known.empty or not isinstance(value, str):
        return
    numbers = pd.to_numeric(known, errors="coerce")
    if not numbers.notna().all():
        return  # a text state signal: a value that does not occur is fine
    shown = sorted(map(str, pd.unique(known.to_numpy())))[:6]
    warnings.warn(
        f"the text state value {value!r} is compared with a numeric state signal "
        f"(values {shown}), so it never matches. Pass the numbers that mean the "
        "state as a set, e.g. running_value={2} or stopped_value={1, 5}.",
        DataQualityWarning,
        stacklevel=caller_stacklevel(),
    )


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

    Raises:
        ColumnNotFoundError: When ``value_column`` is neither the default
            ``"value_string"`` nor ``"value_bool"`` and ``data`` has no such
            column (a misspelled column is not silently replaced by
            ``value_bool``).
    """
    if value_column not in data.columns and value_column not in (
        _STRING_COLUMN,
        _BOOL_COLUMN,
    ):
        raise ColumnNotFoundError(
            f"the data has no {value_column!r} column for the state signal. "
            f"Columns: {list(data.columns)}"
        )
    if _is_state_set(value):
        values = list(value)
        bools = [v for v in values if isinstance(v, (bool, np.bool_))]
        texts = [v for v in values if not isinstance(v, (bool, np.bool_))]
        if texts and not uses_bool_state(data, value_column, texts[0]):
            if value_column not in data.columns:
                return pd.Series(False, index=data.index)
            mask = run_mask(data[value_column], run_states=values, warn=False)
            return mask.fillna(False).astype(bool)
        # A set on a boolean run signal: like a scalar (a bool in the set is
        # matched directly, text falls back to bool_value).
        targets = {bool(v) for v in bools} if bools else {bool_value}
        if _BOOL_COLUMN not in data.columns:
            return pd.Series(False, index=data.index)
        raw = data[_BOOL_COLUMN]
        return raw.notna() & bool_or_false(raw).isin(targets)
    if not uses_bool_state(data, value_column, value):
        _warn_unmatched_state(data[value_column], value)
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
    return known & (bool_or_false(raw) == target)


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
