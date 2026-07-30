"""Value-column resolution and coercion for messy production signal inputs.

Across a fleet of hundreds of machines the *same* logical quantity arrives in
whatever column and dtype the edge gateway happened to pick: a part counter may
land in ``value_integer`` on one line, ``value_double`` on the next, and
``value_string`` on an OPC-UA gateway that stringifies everything. A machine
state may be ``True``/``False``, ``1``/``0``, ``"RUN"``/``"STOP"`` or ``"on"``.
Serial numbers arrive padded, upper/lower-cased, and interleaved with
placeholder values such as ``""``, ``"n/a"`` or ``"#NV"``.

This module is the single place where those input variations are absorbed, so
the production normalizers above it can assume clean, typed series:

* :func:`resolve_value_column` -- pick the value column a signal actually uses.
* :func:`coerce_numeric` / :func:`coerce_bool` / :func:`coerce_label` -- convert
  a raw column to the dtype a normalizer needs, tolerating string encodings.
* :func:`signal_frame` -- filter to one uuid, sort, drop unusable rows and
  collapse duplicate timestamps into one reading.

All helpers are pure functions over pandas objects; nothing here reads the
network or mutates its inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_SENTINELS",
    "FALSE_TOKENS",
    "TRUE_TOKENS",
    "VALUE_COLUMNS",
    "coerce_bool",
    "coerce_label",
    "coerce_numeric",
    "resolve_value_column",
    "signal_frame",
    "union_timeline",
]

#: The value columns of the standard ts-shape timeseries schema, in the order
#: :func:`resolve_value_column` prefers them when several are equally populated.
VALUE_COLUMNS: tuple[str, ...] = (
    "value_double",
    "value_integer",
    "value_bool",
    "value_string",
)

#: Placeholder strings that industrial systems write when *no* identifier is
#: assigned. Treated as missing by :func:`coerce_label`. Note that ``"0"`` is
#: deliberately absent -- it is a legitimate part number on some controllers --
#: enable it per dimension via ``zero_is_sentinel``.
DEFAULT_SENTINELS: tuple[str, ...] = (
    "",
    "-",
    "--",
    "---",
    ".",
    "?",
    "??",
    "n/a",
    "n.a.",
    "na",
    "none",
    "null",
    "nan",
    "nil",
    "#nv",
    "#n/a",
    "unknown",
    "undefined",
    "no part",
    "nopart",
    "leer",
    "empty",
)

#: String tokens read as boolean true by :func:`coerce_bool`.
TRUE_TOKENS: frozenset[str] = frozenset(
    {
        "1",
        "1.0",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "ok",
        "good",
        "run",
        "running",
        "active",
        "auto",
        "high",
        "set",
        "ein",
        "ja",
        "wahr",
    }
)

#: String tokens read as boolean false by :func:`coerce_bool`.
FALSE_TOKENS: frozenset[str] = frozenset(
    {
        "0",
        "0.0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "nok",
        "bad",
        "stop",
        "stopped",
        "inactive",
        "idle",
        "manual",
        "low",
        "reset",
        "aus",
        "nein",
        "falsch",
    }
)


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------
def resolve_value_column(
    dataframe: pd.DataFrame,
    *,
    uuid: str | None = None,
    uuid_column: str = "uuid",
    preferred: str | None = None,
    candidates: Sequence[str] = VALUE_COLUMNS,
) -> str:
    """Return the value column that carries data for ``uuid``.

    Real gateways populate whichever column matches the PLC datatype, and the
    same logical role (counter, state, identifier) is not consistently typed
    across a fleet. Rather than forcing every caller to pass ``value_column=``
    per machine, resolve it from the data: the candidate column with the most
    non-null values for this signal wins, ties broken by ``candidates`` order.

    Args:
        dataframe: Frame in the standard ts-shape schema.
        uuid: Restrict the check to this signal; ``None`` inspects all rows.
        uuid_column: Column holding signal identifiers.
        preferred: If given and that column exists and has at least one
            non-null value for the signal, it is returned unconditionally.
            This keeps explicit caller intent authoritative.
        candidates: Columns to consider, most-preferred first.

    Returns:
        The name of the chosen value column.

    Raises:
        ValueError: If no candidate column exists, or none holds a non-null
            value for the signal. The message lists what was inspected so the
            caller can see whether the signal is simply empty.
    """
    present = [c for c in candidates if c in dataframe.columns]
    if not present:
        raise ValueError(
            f"None of the value columns {tuple(candidates)} exist in the frame. "
            f"Got columns: {list(dataframe.columns)}"
        )

    scope = dataframe
    if uuid is not None and uuid_column in dataframe.columns:
        scope = dataframe[dataframe[uuid_column] == uuid]

    if preferred is not None:
        if preferred not in dataframe.columns:
            raise ValueError(
                f"Requested value_column {preferred!r} does not exist. "
                f"Available: {list(dataframe.columns)}"
            )
        # An explicit request wins whenever it carries data; when the caller's
        # column is entirely empty for this signal we fall through to
        # auto-detection rather than silently returning nothing.
        if not scope.empty and scope[preferred].notna().any():
            return preferred

    counts = {c: int(scope[c].notna().sum()) for c in present}
    best = max(present, key=lambda c: (counts[c], -present.index(c)))
    if counts[best] == 0:
        raise ValueError(
            f"Signal {uuid!r} has no non-null values in any of {tuple(present)}. "
            "Check the uuid and the loaded time range."
        )
    return best


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------
def _decimal_comma_to_point(series: pd.Series) -> pd.Series:
    """Normalize ``"1,5"`` to ``"1.5"`` without breaking ``"1,234.5"``.

    Only a string carrying exactly one comma and no dot is treated as a
    German-style decimal; anything else keeps its commas so ``to_numeric``
    can reject it rather than silently mangling a thousands separator.
    """
    text = series.astype("string").str.strip()
    single_comma = (text.str.count(",") == 1) & (~text.str.contains(".", regex=False))
    return text.mask(single_comma, text.str.replace(",", ".", regex=False))


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce any value column to ``float64``, mapping unparseable values to NaN.

    Handles the three shapes seen in the wild: already-numeric columns (passed
    through), boolean columns (``True`` -> ``1.0``), and stringified numbers,
    including German decimal commas and surrounding whitespace.

    Args:
        series: Raw value column.

    Returns:
        A float series with the same index.
    """
    if series.empty:
        return series.astype("float64")
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    converted = pd.to_numeric(_decimal_comma_to_point(series), errors="coerce")
    return converted.astype("float64")


def coerce_bool(series: pd.Series, *, default: bool = False) -> pd.Series:
    """Coerce any value column to a plain ``bool`` series.

    Numeric input is truthy when non-zero. String input is matched against
    :data:`TRUE_TOKENS` / :data:`FALSE_TOKENS` case-insensitively, so
    ``"RUN"``, ``"on"``, ``"OK"`` and ``"1"`` all read as true. Values that
    match neither token set -- and genuine nulls -- fall back to ``default``.

    Args:
        series: Raw value column.
        default: Value used for nulls and unrecognised tokens.

    Returns:
        A boolean series with the same index.
    """
    if series.empty:
        return series.astype("bool")
    if pd.api.types.is_bool_dtype(series):
        return series.astype("bool")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.ne(0).where(numeric.notna(), default).astype("bool")

    text = series.astype("string").str.strip().str.lower()
    out = pd.Series(default, index=series.index, dtype="bool")
    out = out.mask(text.isin(TRUE_TOKENS), True)
    out = out.mask(text.isin(FALSE_TOKENS), False)
    # A stringified number that is not a known token still carries truthiness.
    unmatched = ~text.isin(TRUE_TOKENS) & ~text.isin(FALSE_TOKENS) & text.notna()
    if unmatched.any():
        numeric = pd.to_numeric(text.where(unmatched), errors="coerce")
        out = out.mask(numeric.notna(), numeric.ne(0).fillna(default))
    return out.astype("bool")


def coerce_label(
    series: pd.Series,
    *,
    sentinels: Sequence[str] = DEFAULT_SENTINELS,
    strip: bool = True,
    upper: bool = False,
    zero_is_sentinel: bool = False,
    strip_leading_zeros: bool = False,
) -> pd.Series:
    """Coerce any value column to a clean identifier (label) series.

    Identifiers -- order, batch, part, serial, customer part numbers -- reach a
    reporting layer as strings regardless of how the PLC typed them. This
    normalizes them so the same physical order does not split into several
    reporting rows because one controller emitted ``" A-100 "`` and another
    ``"a-100"``.

    Numeric input is rendered without a trailing ``.0`` so an order number held
    in ``value_double`` matches the same number held in ``value_string``.

    Args:
        series: Raw value column.
        sentinels: Case-insensitive placeholder values mapped to ``pd.NA``.
        strip: Strip surrounding whitespace.
        upper: Upper-case the result (recommended for machine-written IDs).
        zero_is_sentinel: Also treat ``"0"`` / ``0`` as "no identifier".
        strip_leading_zeros: Drop leading zeros, so ``"000123"`` and ``"123"``
            are the same order. Left off by default because some part-number
            schemes are significant in their padding.

    Returns:
        A nullable ``string`` series with the same index; missing/placeholder
        entries are ``pd.NA``.
    """
    if series.empty:
        return series.astype("string")

    if pd.api.types.is_bool_dtype(series):
        text = series.astype("string")
    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        # Render integral floats as integers: 123.0 -> "123".
        integral = numeric.notna() & np.isclose(numeric % 1, 0)
        text = numeric.astype("string")
        if integral.any():
            as_int = numeric.where(integral).astype("Float64").astype("Int64")
            text = text.mask(integral, as_int.astype("string"))
    else:
        text = series.astype("string")

    if strip:
        text = text.str.strip()
    if strip_leading_zeros:
        stripped = text.str.replace(r"^0+(?=.)", "", regex=True)
        text = stripped.where(stripped.ne(""), text)
    if upper:
        text = text.str.upper()

    blocked = {str(s).strip().lower() for s in sentinels}
    if zero_is_sentinel:
        blocked |= {"0", "0.0"}
    lowered = text.str.lower()
    return text.mask(lowered.isin(blocked) | text.isna(), pd.NA).astype("string")


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def signal_frame(
    dataframe: pd.DataFrame,
    uuid: str,
    *,
    time_column: str = "systime",
    value_column: str | None = None,
    uuid_column: str = "uuid",
    duplicate_policy: str = "last",
) -> pd.DataFrame:
    """Extract one signal as a tidy two-column frame.

    Every production normalizer starts from this shape, so the awkward parts of
    real inputs are dealt with once: the value column is resolved when not
    given, rows without a usable timestamp are dropped, ordering is made
    deterministic, and duplicate timestamps -- common when two gateways report
    the same PLC tag, or on a coarse timestamp resolution -- are collapsed to a
    single reading.

    Args:
        dataframe: Frame in the standard ts-shape schema.
        uuid: Signal to extract.
        time_column: Timestamp column.
        value_column: Value column to read; auto-resolved when ``None``.
        uuid_column: Column holding signal identifiers.
        duplicate_policy: How to collapse rows sharing a timestamp --
            ``"last"`` (keep the newest reading, the safe default for
            counters), ``"first"``, ``"max"``, ``"min"``, or ``"keep"`` to
            leave duplicates in place.

    Returns:
        A frame with columns ``[time_column, "value"]``, sorted by time and
        re-indexed from zero. Empty (with those columns) when the signal is
        absent, so callers can branch on ``.empty`` instead of catching.

    Raises:
        ValueError: If ``time_column`` is missing, ``duplicate_policy`` is
            unknown, or an explicitly requested ``value_column`` does not exist.
    """
    if time_column not in dataframe.columns:
        raise ValueError(
            f"Time column {time_column!r} not found. "
            f"Available columns: {list(dataframe.columns)}"
        )
    if duplicate_policy not in {"last", "first", "max", "min", "keep"}:
        raise ValueError(
            f"Unknown duplicate_policy {duplicate_policy!r}; expected one of "
            "'last', 'first', 'max', 'min', 'keep'."
        )

    empty = pd.DataFrame({time_column: pd.Series(dtype="datetime64[ns]"), "value": []})
    if dataframe.empty:
        return empty

    scope = dataframe
    if uuid_column in dataframe.columns:
        scope = dataframe[dataframe[uuid_column] == uuid]
    if scope.empty:
        return empty

    try:
        column = resolve_value_column(
            scope, uuid=uuid, uuid_column=uuid_column, preferred=value_column
        )
    except ValueError:
        # An all-null signal is a data-availability fact, not a programming
        # error: hand back an empty frame and let the caller report coverage.
        if value_column is not None and value_column not in dataframe.columns:
            raise
        return empty

    out = scope[[time_column, column]].rename(columns={column: "value"}).copy()
    out[time_column] = pd.to_datetime(out[time_column], errors="coerce")
    out = out.dropna(subset=[time_column])
    if out.empty:
        return empty

    # mergesort keeps equal timestamps in arrival order, which is what makes
    # "first"/"last" meaningful rather than arbitrary.
    out = out.sort_values(time_column, kind="mergesort")
    if duplicate_policy != "keep":
        dupes = out.duplicated(subset=[time_column], keep=False)
        if dupes.any():
            if duplicate_policy in {"first", "last"}:
                out = out.drop_duplicates(subset=[time_column], keep=duplicate_policy)
            else:
                agg = "max" if duplicate_policy == "max" else "min"
                out = (
                    out.groupby(time_column, as_index=False, sort=True)["value"]
                    .agg(agg)
                    .rename(columns={"value": "value"})
                )
    return out.reset_index(drop=True)


def union_timeline(
    frames: dict[str, pd.DataFrame],
    *,
    time_column: str = "systime",
) -> pd.DataFrame:
    """Combine several single-signal frames onto one forward-filled timeline.

    Used wherever a logical quantity is spread across several tags -- one-hot
    machine states, a stacklight's three lamps, separate OK/NOK counters. Each
    input contributes a column named after its key; values are held constant
    between that signal's own updates (step semantics), so any row of the
    result is the best-known snapshot of all signals at that instant.

    Args:
        frames: Mapping of output column name to a frame from
            :func:`signal_frame`.
        time_column: Timestamp column shared by the inputs.

    Returns:
        A frame with ``time_column`` plus one column per key, sorted by time.
        Leading rows hold ``NaN`` for signals that had not reported yet.
    """
    usable = {name: f for name, f in frames.items() if not f.empty}
    if not usable:
        cols: dict[str, pd.Series] = {
            time_column: pd.Series(dtype="datetime64[ns]"),
        }
        for name in frames:
            cols[name] = pd.Series(dtype="object")
        return pd.DataFrame(cols)

    parts = [
        f.rename(columns={"value": name})[[time_column, name]]
        for name, f in usable.items()
    ]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=time_column, how="outer")
    merged = merged.sort_values(time_column, kind="mergesort")
    # Collapse rows that share a timestamp before filling, otherwise a
    # simultaneous update of two lamps yields two half-populated rows.
    merged = merged.groupby(time_column, as_index=False, sort=True).last()
    value_cols = [c for c in merged.columns if c != time_column]
    merged[value_cols] = merged[value_cols].ffill()
    for name in frames:
        if name not in merged.columns:
            merged[name] = np.nan
    return merged.reset_index(drop=True)
