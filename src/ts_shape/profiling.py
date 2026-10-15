"""One-call data-quality and semantics report for long-format time series.

Start every analysis here::

    import ts_shape

    report = ts_shape.profile(df)
    print(report.summary())     # what each signal is, and what is wrong with it
    report.issues()             # actionable list, most severe first
    report.specs                # SignalSpecs to pass on to detectors

The report infers each signal's semantics (see :mod:`ts_shape.signals`) and
checks the things that silently corrupt industrial analytics: logging gaps,
counter resets, duplicate or out-of-order timestamps, unparseable times,
missing or non-numeric values, signals stuck at one value, and uuids that spread their
values over several ``value_*`` columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ts_shape.signals.infer import infer_spec, signal_groups
from ts_shape.signals.spec import SignalSpec, SignalSpecs
from ts_shape.utils.counters import count_resets

__all__ = ["profile", "ProfileReport"]

_VALUE_COLUMNS = ("value_double", "value_integer", "value_bool", "value_string")
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

#: A continuous signal constant for this many nominal periods is "stuck".
STUCK_PERIODS = 20

PROFILE_COLUMNS = [
    "uuid",
    "kind",
    "encoding",
    "value_column",
    "rows",
    "first",
    "last",
    "nominal_period",
    "max_gap",
    "gap_count",
    "gap_seconds",
    "longest_gap_seconds",
    "resets",
    "duplicates",
    "out_of_order",
    "nat_timestamps",
    "missing_values",
    "value_columns_used",
    "longest_constant_seconds",
    "min",
    "max",
    "tz",
]
ISSUE_COLUMNS = ["uuid", "severity", "issue", "detail"]


@dataclass(frozen=True)
class ProfileReport:
    """Result of :func:`profile`.

    Attributes:
        specs: Inferred (and overridden) :class:`~ts_shape.signals.SignalSpecs`.
        table: One row per signal with the measured characteristics.
        issue_table: One row per finding (``uuid``, ``severity``, ``issue``,
            ``detail``), most severe first.
    """

    specs: SignalSpecs
    table: pd.DataFrame
    issue_table: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        """The per-signal table."""
        return self.table.copy()

    def issues(self, severity: str | None = None) -> pd.DataFrame:
        """Findings, optionally filtered to one severity.

        Args:
            severity: ``"error"``, ``"warning"`` or ``"info"`` to keep only that
                severity. None returns every finding.

        Returns:
            DataFrame with columns ``uuid``, ``severity``, ``issue`` and ``detail``,
                most severe first.
        """
        out = self.issue_table
        if severity is not None:
            out = out[out["severity"] == severity]
        return out.reset_index(drop=True)

    def summary(self) -> str:
        """Human-readable overview."""
        n = len(self.table)
        lines = [f"ts-shape profile: {n} signal(s)"]
        if n:
            kinds = self.table["kind"].value_counts().to_dict()
            lines.append(
                "  kinds: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
            )
        counts = self.issue_table["severity"].value_counts().to_dict()
        lines.append(
            "  issues: "
            + (
                ", ".join(f"{counts.get(s, 0)} {s}" for s in _SEVERITY_ORDER)
                if counts
                else "none"
            )
        )
        for _, row in self.issue_table.head(20).iterrows():
            lines.append(
                f"  [{row['severity']}] {row['uuid']}: {row['issue']} -- {row['detail']}"
            )
        if len(self.issue_table) > 20:
            lines.append(f"  ... {len(self.issue_table) - 20} more (see .issues())")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _longest_constant_seconds(values: pd.Series, times: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    run = (values != values.shift()).cumsum().to_numpy()
    # Span of each run of equal values: its last time minus its first
    # (cythonized group first/last; ``times`` has no NaT here).
    grouped = times.reset_index(drop=True).groupby(run, sort=False)
    spans = grouped.last() - grouped.first()
    return float(spans.max().total_seconds()) if len(spans) else 0.0


def _profile_signal(
    raw: pd.DataFrame, spec: SignalSpec, time_column: str
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    issues: list[tuple[str, str, str]] = []
    times_raw = raw[time_column]
    if not pd.api.types.is_datetime64_any_dtype(times_raw):
        # (to_datetime of a datetime column is a no-op, but scans it first)
        times_raw = pd.to_datetime(times_raw, errors="coerce")
    nat = int(times_raw.isna().sum())
    ordered = times_raw.dropna()
    out_of_order = int((ordered.diff() < pd.Timedelta(0)).sum())

    g = raw.assign(**{time_column: times_raw}).dropna(subset=[time_column])
    g = g.sort_values(time_column, kind="stable")
    times = g[time_column]
    duplicates = int(times.duplicated().sum())
    used = [c for c in _VALUE_COLUMNS if c in g.columns and g[c].notna().any()]
    values = g[spec.value_column] if spec.value_column in g.columns else pd.Series()
    missing = int(values.isna().sum())
    present = values.dropna()
    present_times = times.loc[present.index]

    diffs = times.diff().dropna()
    gap_count = 0
    gap_seconds = 0.0
    if spec.max_gap is not None and not diffs.empty:
        gaps = diffs[diffs > spec.max_gap]
        gap_count = len(gaps)
        gap_seconds = float(gaps.dt.total_seconds().sum())
    longest_gap = float(diffs.max().total_seconds()) if not diffs.empty else 0.0

    resets = 0
    non_numeric = 0
    vmin = vmax = None
    if spec.value_column in ("value_double", "value_integer") and not present.empty:
        numeric = pd.to_numeric(present, errors="coerce")
        # ``present`` has no nulls, so every NaN here is a value that is not
        # numeric; it is reported as an issue below rather than dropped.
        non_numeric = int(numeric.isna().sum())
        vmin, vmax = float(numeric.min()), float(numeric.max())
        if spec.kind == "counter":
            resets = count_resets(numeric)
    longest_constant = _longest_constant_seconds(present, present_times)

    row = {
        "uuid": spec.uuid,
        "kind": spec.kind,
        "encoding": spec.encoding,
        "value_column": spec.value_column,
        "rows": len(raw),
        "first": times.iloc[0] if len(times) else pd.NaT,
        "last": times.iloc[-1] if len(times) else pd.NaT,
        "nominal_period": spec.nominal_period,
        "max_gap": spec.max_gap,
        "gap_count": gap_count,
        "gap_seconds": gap_seconds,
        "longest_gap_seconds": longest_gap,
        "resets": resets,
        "duplicates": duplicates,
        "out_of_order": out_of_order,
        "nat_timestamps": nat,
        "missing_values": missing,
        "value_columns_used": ",".join(used),
        "longest_constant_seconds": longest_constant,
        "min": vmin,
        "max": vmax,
        "tz": spec.tz,
    }

    if nat:
        issues.append(
            ("error", "unparseable timestamps", f"{nat} row(s) with NaT systime")
        )
    if len(present) == 0:
        issues.append(("error", "no values", f"{spec.value_column} is empty"))
    if non_numeric:
        issues.append(
            (
                "error",
                "non-numeric values",
                f"{non_numeric} value(s) in {spec.value_column} are not numeric",
            )
        )
    if duplicates:
        issues.append(
            (
                "warning",
                "duplicate timestamps",
                f"{duplicates} repeated systime value(s)",
            )
        )
    if out_of_order:
        issues.append(
            (
                "info",
                "rows out of time order",
                f"{out_of_order} backward step(s); ts-shape sorts, but check the source",
            )
        )
    if len(used) > 1:
        issues.append(
            ("warning", "mixed value columns", f"values spread over {', '.join(used)}")
        )
    if missing:
        share = missing / max(len(values), 1)
        sev = "warning" if share > 0.05 else "info"
        issues.append(
            (
                sev,
                "missing values",
                f"{missing} NaN in {spec.value_column} ({share:.1%})",
            )
        )
    if gap_count:
        issues.append(
            (
                "warning",
                "logging gaps",
                f"{gap_count} gap(s) > {spec.max_gap}, {gap_seconds / 3600:.2f} h total, "
                f"longest {longest_gap / 3600:.2f} h",
            )
        )
    if resets:
        issues.append(
            (
                "info",
                "counter resets",
                f"{resets} reset(s): use reset-aware quantities (handle_resets / "
                "ts_shape.signals.primitives.increments)",
            )
        )
    if spec.encoding == "change_encoded":
        issues.append(
            (
                "info",
                "change-encoded",
                "one row per change: durations must close at the next change "
                "(interval_end='next_change'); set max_gap to the outage threshold",
            )
        )
    if (
        spec.kind == "continuous"
        and spec.nominal_period is not None
        and longest_constant >= STUCK_PERIODS * spec.nominal_period.total_seconds()
        and longest_constant > 0
    ):
        issues.append(
            (
                "warning",
                "possibly stuck",
                f"constant for {longest_constant / 3600:.2f} h "
                f"(>= {STUCK_PERIODS} sampling periods)",
            )
        )
    if spec.kind == "counter" and spec.value_column == "value_double":
        issues.append(
            ("info", "float counter", "cumulative float register (e.g. kWh meter)")
        )
    return row, issues


def profile(
    dataframe: pd.DataFrame,
    specs: SignalSpecs | Mapping[str, Mapping[str, Any]] | None = None,
    *,
    uuid_column: str = "uuid",
    time_column: str = "systime",
) -> ProfileReport:
    """Profile every signal in a long-format frame.

    Nullable, pyarrow and string dtypes are converted to NumPy dtypes first;
    a time column that mixes UTC offsets is converted to UTC with a
    :class:`~ts_shape.errors.DataQualityWarning`. The input is not changed.

    Args:
        dataframe: Long-format data (``systime``, ``uuid``, ``value_*``).
        specs: Known semantics. Either complete :class:`SignalSpecs`, or
            ``{uuid: {field: value}}`` overrides applied on top of inference.
        uuid_column: Column identifying the signal.
        time_column: Timestamp column.

    Returns:
        :class:`ProfileReport` with ``specs``, a per-signal ``table`` and the
        ``issues()`` list.

    Raises:
        ColumnNotFoundError: When ``uuid_column`` or ``time_column`` is
            missing (also for ``pd.DataFrame()``).
        ValueError: When ``specs`` overrides name unknown fields or invalid
            values.
    """
    if not isinstance(specs, SignalSpecs):
        overrides = {str(k): dict(v) for k, v in dict(specs or {}).items()}
        for key, fields in overrides.items():
            # Same validation as SignalSpec.from_dict: unknown fields raise.
            SignalSpec.from_dict({**fields, "uuid": key})
        specs = None
    else:
        overrides = {}

    inferred: dict[str, SignalSpec] = {}
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for key, raw in signal_groups(dataframe, uuid_column, time_column):
        if specs is not None and key in specs:
            spec = specs[key]
        else:
            parsed = raw.dropna(subset=[time_column])
            spec = infer_spec(parsed, key, time_column)
            if key in overrides:
                spec = spec.with_(**overrides[key])
        inferred[key] = spec
        row, found = _profile_signal(raw, spec, time_column)
        rows.append(row)
        issues.extend(
            {"uuid": key, "severity": s, "issue": i, "detail": d} for s, i, d in found
        )

    table = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    issue_table = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    if not issue_table.empty:
        issue_table = issue_table.sort_values(
            ["severity", "uuid"],
            key=lambda c: c.map(_SEVERITY_ORDER) if c.name == "severity" else c,
        ).reset_index(drop=True)
    return ProfileReport(
        specs=SignalSpecs(inferred), table=table, issue_table=issue_table
    )
