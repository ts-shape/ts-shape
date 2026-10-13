"""Infer :class:`~ts_shape.signals.spec.SignalSpec` from data.

Inference is a starting point, not an oracle: every inferred field records
*why* it was chosen in ``spec.notes``, and anything the data cannot tell
(units, a counter that never reset yet, the real outage threshold of a
change-encoded signal) should be set with :meth:`SignalSpecs.override`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd  # type: ignore

from ts_shape.signals.spec import SignalSpec, SignalSpecs

__all__ = ["infer_specs", "infer_spec"]

_VALUE_COLUMNS = ("value_double", "value_integer", "value_bool", "value_string")

#: A sampled signal's value is held for at most this many nominal periods.
MAX_GAP_PERIODS = 5
#: Integer signals with at most this many distinct values may be states.
STATE_MAX_CARDINALITY = 16


def _value_column(group: pd.DataFrame) -> str | None:
    present = [c for c in _VALUE_COLUMNS if c in group.columns]
    counts = {c: int(group[c].notna().sum()) for c in present}
    best = max(counts, key=lambda c: counts[c], default=None)
    return best if best is not None and counts[best] > 0 else None


def _interval_stats(times: pd.Series) -> tuple[pd.Timedelta | None, float]:
    """Median positive interval and its coefficient of variation."""
    diffs = times.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None, 0.0
    seconds = diffs.dt.total_seconds()
    mean = float(seconds.mean())
    cv = float(seconds.std(ddof=0) / mean) if mean > 0 else 0.0
    return pd.Timedelta(seconds=float(seconds.median())), cv


def _is_counter(values: pd.Series) -> tuple[bool, str]:
    diffs = values.diff().dropna()
    if len(diffs) < 3:
        return False, "too few samples to recognise a counter"
    rising = float((diffs > 0).mean())
    drops = diffs[diffs < 0]
    if rising < 0.2:
        return False, "values do not increase often enough for a counter"
    if len(drops) > max(1, 0.01 * len(diffs)):
        return False, f"{len(drops)} decreases: not monotonic"
    # Every decrease must look like a reset (falls to under half the level).
    before = values.shift(1)[drops.index]
    if (values[drops.index] > 0.5 * before).any():
        return False, "a decrease is too small to be a counter reset"
    reason = "monotonic non-decreasing"
    if len(drops):
        reason += f" with {len(drops)} reset(s)"
    return True, reason


def infer_spec(
    group: pd.DataFrame, uuid: str, time_column: str = "systime"
) -> SignalSpec:
    """Infer the spec of one signal from its rows."""
    notes: dict[str, str] = {}
    g = group.sort_values(time_column)
    times = pd.to_datetime(g[time_column]).dropna()
    tz = str(times.dt.tz) if getattr(times.dt, "tz", None) is not None else None
    period, cv = _interval_stats(times)

    value_column = _value_column(g) or "value_double"
    values = g[value_column].dropna() if value_column in g.columns else pd.Series()
    notes["value_column"] = "most populated value_* column"

    kind = "continuous"
    if value_column in ("value_bool", "value_string"):
        kind = "state"
        notes["kind"] = f"{value_column} carries discrete values"
    elif len(values) >= 3:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        is_counter, why = _is_counter(numeric)
        integral = bool(np.all(np.isclose(numeric, np.round(numeric))))
        if is_counter:
            kind = "counter"
            notes["kind"] = why
        elif integral and numeric.nunique() <= STATE_MAX_CARDINALITY:
            kind = "state"
            notes["kind"] = f"integer codes, {numeric.nunique()} distinct values"
        else:
            notes["kind"] = f"numeric level ({why})"
    else:
        notes["kind"] = "too few samples; assumed continuous"

    repeats = (
        float((values.shift() == values).iloc[1:].mean()) if len(values) > 1 else 0.0
    )
    if kind in ("state", "counter"):
        change_encoded = len(values) >= 3 and repeats < 0.05 and cv > 0.5
        notes["encoding"] = f"{repeats:.0%} repeated values, interval CV {cv:.2f}"
    else:
        change_encoded = cv > 1.0
        notes["encoding"] = f"interval CV {cv:.2f} (deadband/irregular if > 1)"
    encoding = "change_encoded" if change_encoded else "sampled"

    if encoding == "sampled" and period is not None:
        max_gap = period * MAX_GAP_PERIODS
        notes["max_gap"] = f"{MAX_GAP_PERIODS} x nominal period"
    else:
        max_gap = None
        notes["max_gap"] = (
            "not inferred: a change-encoded value legitimately holds for long "
            "spans; set max_gap to your logging-outage threshold"
        )

    return SignalSpec(
        uuid=uuid,
        kind=kind,  # type: ignore[arg-type]
        encoding=encoding,  # type: ignore[arg-type]
        value_column=value_column,
        nominal_period=period,
        max_gap=max_gap,
        tz=tz,
        notes=notes,
    )


def infer_specs(
    dataframe: pd.DataFrame,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    uuid_column: str = "uuid",
    time_column: str = "systime",
) -> SignalSpecs:
    """Infer a :class:`SignalSpec` for every uuid in a long-format frame.

    Args:
        dataframe: Long-format data (``systime``, ``uuid``, ``value_*``).
        overrides: ``{uuid: {field: value}}`` applied after inference --
            the place for what the data cannot tell (units, outage limits).
        uuid_column: Column identifying the signal.
        time_column: Timestamp column.

    Returns:
        :class:`SignalSpecs` keyed by uuid; each spec's ``notes`` explain the
        inferred fields.
    """
    specs = {
        str(uuid): infer_spec(group, str(uuid), time_column)
        for uuid, group in dataframe.groupby(uuid_column, sort=True)
    }
    result = SignalSpecs(specs)
    return result.override(overrides) if overrides else result
