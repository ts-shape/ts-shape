"""Infer :class:`~ts_shape.signals.spec.SignalSpec` from data.

Inference is a starting point, not an oracle: every inferred field records
*why* it was chosen in ``spec.notes``, and anything the data cannot tell
(units, a counter that never reset yet, the real outage threshold of a
change-encoded signal) should be set with :meth:`SignalSpecs.override`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.signals.spec import Encoding, Kind, SignalSpec, SignalSpecs
from ts_shape.utils.deprecation import caller_stacklevel
from ts_shape.utils.frames import require_columns
from ts_shape.utils.normalize import _is_extension, _plain

__all__ = ["infer_specs", "infer_spec", "parse_times", "signal_groups"]

_VALUE_COLUMNS = ("value_double", "value_integer", "value_bool", "value_string")

#: A sampled signal's value is held for at most this many nominal periods.
MAX_GAP_PERIODS = 5
#: Integer signals with at most this many distinct values may be states.
STATE_MAX_CARDINALITY = 16
#: An interval within this fraction of the median interval is "regular".
REGULAR_TOLERANCE = 0.10
#: A signal is sampled when at least this share of its intervals is regular.
REGULAR_SHARE = 0.8
#: A counter must increase at least this many times.
COUNTER_MIN_INCREASES = 3


def most_populated_value_column(group: pd.DataFrame) -> str | None:
    """The ``value_*`` column with the most values in ``group``, or None.

    Ties go to the first of ``value_double``, ``value_integer``,
    ``value_bool``, ``value_string``; None when no value column has a value.
    Also used by the detectors' value-column check
    (:func:`ts_shape.utils.signal_checks.warn_if_values_elsewhere`).
    """
    present = [c for c in _VALUE_COLUMNS if c in group.columns]
    counts = {c: int(group[c].notna().sum()) for c in present}
    best = max(counts, key=lambda c: counts[c], default=None)
    return best if best is not None and counts[best] > 0 else None


def _interval_stats(times: pd.Series) -> tuple[pd.Timedelta | None, float, float]:
    """Median positive interval, its coefficient of variation, regular share.

    The regular share is the fraction of intervals within
    :data:`REGULAR_TOLERANCE` of the median. Unlike the CV it is robust: a
    few long logging gaps in a regularly sampled signal barely change it.
    """
    diffs = times.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None, 0.0, 1.0
    seconds = diffs.dt.total_seconds()
    mean = float(seconds.mean())
    cv = float(seconds.std(ddof=0) / mean) if mean > 0 else 0.0
    median = float(seconds.median())
    regular = float(((seconds - median).abs() <= REGULAR_TOLERANCE * median).mean())
    return pd.Timedelta(seconds=median), cv, regular


def _is_counter(values: pd.Series) -> tuple[bool, str]:
    """Whether ``values`` look like a cumulative counter, and why.

    A counter never decreases except at a reset (a fall to under half the
    level, at most 1 % of the steps) and increases at least
    :data:`COUNTER_MIN_INCREASES` times -- however rarely, so a slow counter
    (one part every ten readings) is a counter too. A signal with only two
    distinct values (an on/off state) is never a counter. A slow series
    (increasing at under 20 % of the readings) of at most
    :data:`STATE_MAX_CARDINALITY` integer levels is ambiguous with a step
    state and is left to the state rule.
    """
    diffs = values.diff().dropna()
    if len(diffs) < 3:
        return False, "too few samples to recognise a counter"
    increases = int((diffs > 0).sum())
    rising = increases / len(diffs)
    drops = diffs[diffs < 0]
    if increases < COUNTER_MIN_INCREASES:
        return False, f"{increases} increase(s): too few for a counter"
    if values.nunique() <= 2:
        return False, "two distinct values: a two-level state, not a counter"
    if len(drops) > max(1, 0.01 * len(diffs)):
        return False, f"{len(drops)} decreases: not monotonic"
    # Every decrease must look like a reset (falls to under half the level).
    before = values.shift(1)[drops.index]
    if (values[drops.index] > 0.5 * before).any():
        return False, "a decrease is too small to be a counter reset"
    if rising < 0.2:
        integral = bool(np.all(np.isclose(values, np.round(values))))
        if integral and values.nunique() <= STATE_MAX_CARDINALITY:
            return False, (
                f"slow rise over {values.nunique()} integer levels: ambiguous "
                f"with a step state"
            )
    reason = "monotonic non-decreasing"
    if len(drops):
        reason += f" with {len(drops)} reset(s)"
    return True, reason


def parse_times(values: pd.Series) -> pd.Series:
    """A time column as ``datetime64``; unparseable values become ``NaT``.

    Text with different UTC offsets (``+01:00`` next to ``+02:00``) is
    converted to UTC with a :class:`~ts_shape.errors.DataQualityWarning`.

    Args:
        values: The time column.

    Returns:
        The parsed column.
    """
    if pd.api.types.is_datetime64_any_dtype(values.dtype) and not isinstance(
        values.dtype, pd.ArrowDtype
    ):
        return values
    if isinstance(values.dtype, pd.ArrowDtype) and pd.api.types.is_datetime64_any_dtype(
        values.dtype
    ):
        return pd.to_datetime(values.astype(object))
    try:
        with warnings.catch_warnings():
            # pandas 2.x warns about mixed offsets and returns objects;
            # pandas 3 raises. Both fall through to the UTC parse below.
            warnings.simplefilter("ignore", FutureWarning)
            parsed = pd.to_datetime(values, errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            return parsed
    except (TypeError, ValueError):
        pass
    warnings.warn(
        f"time column {values.name!r} mixes UTC offsets; times are converted to UTC",
        DataQualityWarning,
        stacklevel=caller_stacklevel(),
    )
    return pd.to_datetime(values, errors="coerce", utc=True, format="mixed")


def signal_groups(
    dataframe: pd.DataFrame, uuid_column: str, time_column: str
) -> list[tuple[str, pd.DataFrame]]:
    """Rows per signal, keyed by the uuid as text, sorted by key.

    Nullable, pyarrow and string columns are converted to NumPy dtypes
    first (the input is not changed). Rows without a uuid are
    dropped and uuids that differ only in type (``1`` and ``"1"``) are one
    signal; both cases emit a :class:`~ts_shape.errors.DataQualityWarning`.

    Args:
        dataframe: Long-format frame.
        uuid_column: Signal identifier column.
        time_column: Timestamp column.

    Returns:
        ``(uuid, rows)`` pairs.

    Raises:
        ColumnNotFoundError: When ``uuid_column`` or ``time_column`` is missing.
    """
    return list(_iter_signal_groups(dataframe, uuid_column, time_column))


def _iter_signal_groups(
    dataframe: pd.DataFrame, uuid_column: str, time_column: str
) -> Iterator[tuple[str, pd.DataFrame]]:
    """:func:`signal_groups`, one signal at a time.

    Only one signal's rows are copied at once, so profiling a large frame
    needs memory for one signal instead of all of them.

    Args:
        dataframe: Long-format frame.
        uuid_column: Signal identifier column.
        time_column: Timestamp column.

    Yields:
        ``(uuid, rows)`` pairs, sorted by uuid.

    Raises:
        ColumnNotFoundError: When ``uuid_column`` or ``time_column`` is missing.
    """
    require_columns(dataframe, (uuid_column, time_column), where="signal profile")
    # reset_index returns a new frame (a copy on pandas 2, copy-on-write on
    # pandas 3), so the column assignments below never reach the input.
    frame = dataframe.reset_index(drop=True)
    for column in frame.columns:
        is_signal_column = column == uuid_column or (
            isinstance(column, str) and column.startswith("value_")
        )
        if is_signal_column and _is_extension(frame[column]):
            # NumPy dtypes; text in a numeric column stays text, so the
            # profile can report it as non-numeric.
            frame[column] = _plain(frame[column])
    frame[time_column] = parse_times(frame[time_column])
    missing = frame[uuid_column].isna()
    if missing.any():
        warnings.warn(
            f"{int(missing.sum())} row(s) have no {uuid_column}; they are dropped",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
        frame = frame[~missing.to_numpy()]
    # Group by the uuid as text, working on the distinct values: one key per
    # distinct uuid, then each row's group from its factorize code.
    codes, uniques = pd.factorize(frame[uuid_column])
    unique_keys = pd.Series(uniques).astype(str).to_numpy()
    names, key_of_unique, per_key = np.unique(
        unique_keys, return_inverse=True, return_counts=True
    )
    clashes = sorted(str(k) for k in names[per_key > 1])
    if clashes:
        warnings.warn(
            f"{uuid_column} values {clashes[:5]} occur with different types "
            "(e.g. 1 and '1'); they are treated as one signal",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    row_key = key_of_unique[codes]
    order = np.argsort(row_key, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(row_key, minlength=names.size))]
    for k, name in enumerate(names):
        yield str(name), frame.take(order[bounds[k] : bounds[k + 1]])


def infer_spec(
    group: pd.DataFrame, uuid: str, time_column: str = "systime"
) -> SignalSpec:
    """Infer the spec of one signal from its rows.

    Args:
        group: The rows of one signal.
        uuid: The signal's uuid, written to the spec.
        time_column: Timestamp column.

    Returns:
        The inferred :class:`SignalSpec`; ``notes`` explain each field.

    Raises:
        ColumnNotFoundError: When ``time_column`` is missing.
    """
    require_columns(group, (time_column,), where="infer_spec")
    notes: dict[str, str] = {}
    g = group.sort_values(time_column)
    times = parse_times(g[time_column]).dropna()
    tz = str(times.dt.tz) if getattr(times.dt, "tz", None) is not None else None
    period, cv, regular = _interval_stats(times)

    value_column = most_populated_value_column(g) or "value_double"
    values = g[value_column].dropna() if value_column in g.columns else pd.Series()
    notes["value_column"] = "most populated value_* column"

    kind: Kind = "continuous"
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
    # Regular sampling: at least REGULAR_SHARE of the intervals lie within
    # REGULAR_TOLERANCE of the median. A few long logging gaps do not make a
    # regularly sampled signal irregular (the mean/std CV would).
    irregular = regular < REGULAR_SHARE
    timing = (
        f"{regular:.0%} of intervals within {REGULAR_TOLERANCE:.0%} of the "
        f"median (irregular if < {REGULAR_SHARE:.0%}), interval CV {cv:.2f}"
    )
    if kind in ("state", "counter"):
        change_encoded = len(values) >= 3 and repeats < 0.05 and irregular
        notes["encoding"] = f"{repeats:.0%} repeated values, {timing}"
    else:
        change_encoded = irregular
        notes["encoding"] = f"{timing}; irregular means deadband/change-encoded"
    encoding: Encoding = "change_encoded" if change_encoded else "sampled"

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
        kind=kind,
        encoding=encoding,
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
        :class:`SignalSpecs` keyed by uuid (as text); each spec's ``notes``
        explain the inferred fields.

    Raises:
        ColumnNotFoundError: When ``uuid_column`` or ``time_column`` is missing.
    """
    specs = {
        key: infer_spec(group, key, time_column)
        for key, group in _iter_signal_groups(dataframe, uuid_column, time_column)
    }
    result = SignalSpecs(specs)
    return result.override(overrides) if overrides else result
