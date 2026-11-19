"""Apply a datapoint workbook's linear scaling (opt-in).

A datapoint export can carry ``scaling_factor`` and ``scaling_offset`` per
datapoint: the engineering value is ``raw * factor + offset``. The library
never applies them on its own, because exports often carry defaults that are
not a real scaling (an offset of 1 on every datapoint, text datapoints
included). :func:`apply_scaling` applies them when asked to, to numeric
values only, and reports a scaling set on datapoints whose values are text
or bool.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel

__all__ = ["apply_scaling"]

_NUMERIC = ("value_double", "value_integer")


def _scaling_table(metadata: Any) -> pd.DataFrame:
    """``scaling_factor`` / ``scaling_offset`` indexed by uuid (text)."""
    if hasattr(metadata, "scaling"):
        table = metadata.scaling()
    else:
        frame = metadata if "uuid" in metadata.columns else metadata.reset_index()
        table = pd.DataFrame(index=frame["uuid"].astype(str).to_numpy())
        for column, default in (("scaling_factor", 1.0), ("scaling_offset", 0.0)):
            source = frame[column] if column in frame.columns else None
            table[column] = (
                pd.to_numeric(source, errors="coerce").fillna(default).to_numpy()
                if source is not None
                else default
            )
    table.index = table.index.astype(str)
    return table


def apply_scaling(
    dataframe: pd.DataFrame,
    metadata: Any,
    *,
    uuid_column: str = "uuid",
    target_column: str = "value_double",
) -> pd.DataFrame:
    """``raw * scaling_factor + scaling_offset`` for the numeric datapoints.

    Args:
        dataframe: Long-format frame (``uuid``, ``value_double`` /
            ``value_integer`` ...).
        metadata: A :class:`~ts_shape.MetadataJsonLoader` (its
            :meth:`~ts_shape.MetadataJsonLoader.scaling`) or a table with
            ``uuid``, ``scaling_factor`` and ``scaling_offset``.
        uuid_column: The signal column of ``dataframe``.
        target_column: Where the scaled numbers are written (default
            ``value_double``; integer values become floats there).

    Returns:
        A copy of ``dataframe`` with the scaled values. Rows of datapoints
        without a scaling (factor 1, offset 0), and text / bool values, are
        unchanged.

    Warns:
        DataQualityWarning: When a scaling is set on datapoints whose values
            are text or bool (an export default, not a scaling; not applied).
    """
    out = dataframe.copy()
    if out.empty or uuid_column not in out.columns:
        return out
    table = _scaling_table(metadata)
    scaled = table[(table["scaling_factor"] != 1.0) | (table["scaling_offset"] != 0.0)]
    if scaled.empty:
        return out
    keys = out[uuid_column].astype(str)
    numeric_cols = [c for c in _NUMERIC if c in out.columns]
    has_number = (
        out[numeric_cols].notna().any(axis=1)
        if numeric_cols
        else pd.Series(False, index=out.index)
    )
    rows = keys.isin(scaled.index) & has_number
    non_numeric = sorted(set(keys[keys.isin(scaled.index)]) - set(keys[rows]), key=str)
    if non_numeric:
        warnings.warn(
            f"apply_scaling: a scaling is set on {len(non_numeric)} datapoint(s) "
            f"whose values are text or bool (e.g. {non_numeric[:5]}); that is an "
            "export default, not a scaling, and is not applied. Check the "
            "offsets of the numeric datapoints too.",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    if not rows.any():
        return out
    raw = pd.Series(np.nan, index=out.index, dtype="float64")
    for column in reversed(numeric_cols):
        raw = raw.where(raw.notna(), pd.to_numeric(out[column], errors="coerce"))
    factor = keys.map(scaled["scaling_factor"]).astype(float)
    offset = keys.map(scaled["scaling_offset"]).astype(float)
    values = raw * factor + offset
    if target_column not in out.columns or out[target_column].dtype != "float64":
        out[target_column] = pd.to_numeric(
            out.get(target_column, pd.Series(np.nan, index=out.index)),
            errors="coerce",
        ).astype("float64")
    out.loc[rows, target_column] = values[rows]
    for column in numeric_cols:
        if column != target_column:
            out.loc[rows, column] = pd.NA if out[column].dtype != "float64" else np.nan
    return out
