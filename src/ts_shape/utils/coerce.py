"""Numeric coercion that reports what it could not convert.

``pd.to_numeric(..., errors="coerce")`` turns every value it cannot parse
into ``NaN`` without a trace, so a column of ``"12,5"`` strings or a stray
``"n/a"`` silently drops out of every downstream calculation.
:func:`to_numeric_checked` performs the same coercion and emits a
:class:`~ts_shape.errors.DataQualityWarning` naming how many non-null values
were not numeric.

:func:`bool_or_false` reads a run/state column as plain ``bool`` with
missing values as ``False``, without relying on pandas' deprecated silent
downcasting of ``object`` columns (``.fillna(False).astype(bool)``).

:func:`run_mask` is the one convention every run/state detector uses to read
a run signal: a ``bool``, a mode number or a text state, mapped with
``run_states`` (the values that mean running) or ``value_range`` (inclusive
numeric bounds). Without a mapping, a signal that is not a plain 0/1 or
bool emits a :class:`~ts_shape.errors.DataQualityWarning`.
"""

import warnings
from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel

__all__ = ["bool_or_false", "run_bool", "run_mask", "to_numeric_checked"]


def to_numeric_checked(values: pd.Series, *, what: str) -> pd.Series:
    """Coerce ``values`` to numbers, warning when non-null values become NaN.

    Args:
        values: Values to coerce.
        what: Short description of the values for the warning message,
            e.g. ``"counter column 'value_integer'"``.

    Returns:
        A numeric Series aligned with ``values``. Values that are not numeric
        are ``NaN``.

    Warns:
        DataQualityWarning: When at least one non-null value is not numeric.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    lost = int((numeric.isna() & values.notna()).sum())
    if lost:
        warnings.warn(
            f"{lost} non-null value(s) in {what} are not numeric and were "
            "treated as missing (NaN).",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    return numeric


def bool_or_false(values: pd.Series) -> pd.Series:
    """``values`` as a NumPy ``bool`` Series; missing values are ``False``.

    Known values are cast with ``bool`` (as ``.astype(bool)`` does), so the
    result equals ``values.fillna(False).astype(bool)`` for every dtype, but
    an ``object`` column is never filled first, which pandas 2.2 reports as
    a deprecated downcast.

    Args:
        values: State values (``bool``, nullable ``boolean``, numbers or
            ``object``).

    Returns:
        A ``bool`` Series with the index and name of ``values``.
    """
    known = values.notna().to_numpy(dtype=bool)
    out = np.zeros(len(values), dtype=bool)
    if known.any():
        out[known] = values[known].astype(bool).to_numpy(dtype=bool)
    return pd.Series(out, index=values.index, name=values.name)


#: How many distinct values are shown in the run-state warning.
_SHOWN_STATES = 6


def _unmapped_states(known: pd.Series) -> list[Any] | None:
    """The distinct values of ``known`` when a bool cast would misread them.

    A bool or 0/1 signal reads correctly; text, or numbers other than 0 and
    1 (a mode such as 1 manual, 2 automatic, 5 setup), do not.
    """
    if known.empty or pd.api.types.is_bool_dtype(known):
        return None
    if pd.api.types.is_numeric_dtype(known):
        levels = pd.unique(known.to_numpy())
        if set(np.asarray(levels, dtype=float).tolist()) <= {0.0, 1.0}:
            return None
        shown = sorted(float(v) for v in levels.tolist())[:_SHOWN_STATES]
        return [int(v) if v.is_integer() else v for v in shown]
    if known.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        return None
    numbers = pd.to_numeric(known, errors="coerce")
    if numbers.notna().all() and set(numbers.unique().tolist()) <= {0.0, 1.0}:
        return None
    return sorted(map(str, pd.unique(known.to_numpy())))[:_SHOWN_STATES]


def run_mask(
    values: pd.Series,
    *,
    run_states: Collection[Any] | None = None,
    value_range: tuple[float | None, float | None] | None = None,
    what: str | None = None,
    warn: bool = True,
) -> pd.Series:
    """Read a run signal as a nullable ``boolean`` Series (``True`` = running).

    Args:
        values: The run signal's values (bool, numbers or text).
        run_states: Values that mean running (e.g. ``{2}`` for a mode signal
            where 2 is automatic, or ``{"RUN", "AUTO"}``); every other known
            value is not running. Numbers match across int and float
            (``2`` matches ``2.0``).
        value_range: ``(lower, upper)`` inclusive bounds; a value inside
            counts as running (either bound may be ``None``). Used when
            ``run_states`` is ``None``.
        what: Description of the signal for the warning (e.g. ``"OEEEvents:
            run signal 'mode'"``).
        warn: Emit the warning for an unmapped signal (default ``True``).

    Returns:
        A ``boolean`` Series aligned with ``values``; a missing value is
        ``pd.NA`` (unknown).

    Warns:
        DataQualityWarning: When neither ``run_states`` nor ``value_range``
            is given and the values are text or numbers other than 0/1: the
            bool cast then counts every non-zero number and every non-empty
            text (``"Stopped"`` too) as running.
    """
    known = values.notna()
    if run_states is not None:
        states = list(run_states)
        numeric = [
            s
            for s in states
            if isinstance(s, (int, float, np.number))
            and not isinstance(s, (bool, np.bool_))
        ]
        state = values.isin(states).to_numpy(dtype=bool, copy=True)
        if numeric:
            as_numbers = pd.to_numeric(values, errors="coerce")
            state |= as_numbers.isin([float(s) for s in numeric]).to_numpy(dtype=bool)
        return (
            pd.Series(state, index=values.index, name=values.name)
            .astype("boolean")
            .where(known, pd.NA)
        )
    if value_range is not None:
        lower, upper = value_range
        numbers = pd.to_numeric(values, errors="coerce")
        mask = pd.Series(True, index=values.index, name=values.name)
        if lower is not None:
            mask &= (numbers >= lower).fillna(False).astype(bool)
        if upper is not None:
            mask &= (numbers <= upper).fillna(False).astype(bool)
        return mask.astype("boolean").where(known, pd.NA)
    if warn:
        shown = _unmapped_states(values[known])
        if shown is not None:
            label = what or "run signal"
            warnings.warn(
                f"{label} holds the values {shown}, not a bool; read as a bool, "
                "every non-zero number and every non-empty text counts as "
                "running. Pass run_states={...} with the values that mean "
                "running (or value_range=(low, high)).",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
    return bool_or_false(values).astype("boolean").where(known, pd.NA)


def run_bool(
    values: pd.Series,
    *,
    run_states: Collection[Any] | None = None,
    value_range: tuple[float | None, float | None] | None = None,
    what: str | None = None,
    warn: bool = True,
) -> pd.Series:
    """:func:`run_mask` as a NumPy ``bool`` Series; unknown values are ``False``."""
    mask = run_mask(
        values, run_states=run_states, value_range=value_range, what=what, warn=warn
    )
    return mask.fillna(False).astype(bool)
