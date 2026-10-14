"""Shape-specific evaluators that turn a compiled trigger into a detector
DataFrame in the same format the shape-driven adapter already consumes.

Point and summary rules pass matching rows through; interval rules
coalesce contiguous True runs (per ``group_by``) into single rows with
``start``/``end``/duration; static rules emit a single summary row.

Whatever the shape, the returned frame is fed verbatim to
:func:`~ts_shape.eventlog.to_event_log`, which then exercises the same
adapter, EID generation, severity bucketing, object auto-extraction and
schema validation as built-in detectors. No parallel pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pandas as pd

from ...utils.coerce import to_numeric_checked
from .expression import compile_expression
from .spec import RuleSpec

_POINT_TIME_CANDIDATES = (
    "systime",
    "timestamp",
    "time",
    "event_time",
    "ts",
    "datetime",
)


def evaluate(spec: RuleSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Run ``spec`` against ``df`` and return a detector frame.

    The output column conventions match what the shape adapter expects:

    * ``shape="point" | "summary"``: original rows with a ``systime``-like
      timestamp column preserved (no renaming).
    * ``shape="interval"``: rows with ``start``, ``end``, ``source_uuid``
      (if present), the ``group_by`` columns, the value column (mean over the
      run), the severity column (max) and ``sample_count``. A run is
      half-open: it ends at the next sample after its last True row (at the
      last True row when no sample follows), like
      :func:`ts_shape.signals.primitives.intervals`.
    * ``shape="static"``: a single row carrying summary fields.

    Args:
        spec: The rule.
        df: Input frame the trigger is evaluated on.

    Returns:
        The detector frame; an empty (typed, for interval rules) frame when
        nothing triggers.

    Raises:
        ColumnNotFoundError: When the trigger names a column ``df`` lacks.
        ValueError: When the trigger is not boolean, or an interval rule finds
            no time column.
    """
    if df is None:
        return pd.DataFrame()
    if spec.shape == "interval" and len(df) == 0:
        return _empty_interval_frame(df, spec, _pick_time_col(df))
    if len(df) == 0:
        return df.iloc[0:0].copy()

    mask_fn = compile_expression(spec.trigger.expression)
    mask = mask_fn(df)

    if spec.shape == "point":
        return _evaluate_point(df, mask)
    if spec.shape == "interval":
        return _evaluate_interval(df, mask, spec)
    if spec.shape == "summary":
        return _evaluate_summary(df, mask, spec)
    if spec.shape == "static":
        return _evaluate_static(df, mask, spec)
    raise ValueError(f"unsupported shape {spec.shape!r}")  # pragma: no cover


def _interval_columns(df: pd.DataFrame, spec: RuleSpec) -> list[str]:
    cols = ["start", "end", *spec.trigger.group_by]
    for field in (spec.value_field, spec.severity_field):
        if field and field in df.columns and field not in cols:
            cols.append(field)
    if "source_uuid" in df.columns and "source_uuid" not in cols:
        cols.append("source_uuid")
    cols.append("sample_count")
    return cols


def _empty_interval_frame(
    df: pd.DataFrame, spec: RuleSpec, time_col: str | None
) -> pd.DataFrame:
    """Zero-row interval frame with the columns a non-empty result has."""
    time_dtype: Any = (
        df[time_col].dtype
        if time_col is not None
        and pd.api.types.is_datetime64_any_dtype(df[time_col].dtype)
        else "datetime64[ns]"
    )
    out: dict[str, pd.Series] = {}
    for col in _interval_columns(df, spec):
        if col in ("start", "end"):
            out[col] = pd.Series(dtype=time_dtype)
        elif col == "sample_count":
            out[col] = pd.Series(dtype="int64")
        elif col in (spec.value_field, spec.severity_field):
            out[col] = pd.Series(dtype="float64")
        else:
            out[col] = pd.Series(dtype=df[col].dtype if col in df.columns else object)
    return pd.DataFrame(out)


def _evaluate_point(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return df.loc[mask].reset_index(drop=True).copy()


def _pick_time_col(df: pd.DataFrame) -> str | None:
    for c in _POINT_TIME_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            # pandas is untyped here (no stubs), so column labels are Any;
            # rule frames use string column names.
            return cast(str, c)
    return None


def _evaluate_interval(
    df: pd.DataFrame,
    mask: pd.Series,
    spec: RuleSpec,
) -> pd.DataFrame:
    time_col = _pick_time_col(df)
    if time_col is None:
        raise ValueError(
            "interval lambda rule requires a datetime column in the input "
            "(systime / timestamp / time / event_time / ts / datetime)"
        )
    work = df.copy()
    work["_mask"] = mask.astype(bool).to_numpy()
    # Coerce the carried numeric columns once, so a non-numeric value is
    # reported once rather than per run.
    for field in (spec.value_field, spec.severity_field):
        if field and field in work.columns:
            work[field] = to_numeric_checked(work[field], what=f"column {field!r}")

    rows: list[dict[str, object]] = []
    group_by = list(spec.trigger.group_by)
    iterator: Iterator[tuple[tuple[Any, ...], pd.DataFrame]]
    if group_by:
        grouped = work.groupby(group_by, sort=False, dropna=False)
        iterator = (
            (tuple(k) if not isinstance(k, tuple) else k, g) for k, g in grouped
        )
    else:
        iterator = iter([((), work)])

    for key, sub in iterator:
        sub = sub.sort_values(time_col, kind="stable").reset_index(drop=True)
        times = sub[time_col]
        run_id = (sub["_mask"] != sub["_mask"].shift()).cumsum()
        for _, run in sub.groupby(run_id, sort=False):
            if not bool(run["_mask"].iloc[0]):
                continue
            start_ts = pd.Timestamp(run[time_col].iloc[0])
            # Half-open: the run lasts until the next sample of its group.
            after = int(run.index[-1]) + 1
            end_ts = pd.Timestamp(
                times.iloc[after] if after < len(sub) else run[time_col].iloc[-1]
            )
            duration_s = (end_ts - start_ts).total_seconds()
            if (
                spec.trigger.min_duration_s is not None
                and duration_s < spec.trigger.min_duration_s
            ):
                continue
            row: dict[str, object] = {
                "start": start_ts,
                "end": end_ts,
            }
            for col_name, col_val in zip(group_by, key):
                row[col_name] = col_val
            # Carry the value column through as the mean of the run, if set.
            if spec.value_field and spec.value_field in run.columns:
                row[spec.value_field] = run[spec.value_field].mean()
            # Carry severity_score as max over the run.
            if spec.severity_field and spec.severity_field in run.columns:
                row[spec.severity_field] = run[spec.severity_field].max()
            # Pass through source_uuid (asset auto-extract) when grouped on it.
            if "source_uuid" not in row and "source_uuid" in run.columns:
                row["source_uuid"] = run["source_uuid"].iloc[0]
            row["sample_count"] = int(len(run))
            rows.append(row)

    if not rows:
        return _empty_interval_frame(df, spec, time_col)
    return pd.DataFrame(rows).reindex(columns=_interval_columns(df, spec))


def _evaluate_summary(
    df: pd.DataFrame,
    mask: pd.Series,
    spec: RuleSpec,  # noqa: ARG001 — kept for signature symmetry
) -> pd.DataFrame:
    return df.loc[mask].reset_index(drop=True).copy()


def _evaluate_static(
    df: pd.DataFrame,
    mask: pd.Series,
    spec: RuleSpec,  # noqa: ARG001
) -> pd.DataFrame:
    hits = int(mask.sum())
    return pd.DataFrame([{"sample_count": hits}])
