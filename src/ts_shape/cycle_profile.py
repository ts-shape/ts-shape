"""Cycle time profiling of one PLC from the two inputs ts-shape loads.

The inputs are exactly what the ts-shape loaders return, without any
mapping or machine context:

1. ``timeseries``: the raw frame of
   :class:`~ts_shape.loader.timeseries.azure_blob_loader.AzureBlobParquetLoader`
   (``uuid``, ``systime`` as Unix nanoseconds or timestamps, ``value_*``,
   ``is_delta`` and whatever else the parquet files carry);
2. ``metadata`` (optional): the datapoint table of a ts-shape metadata loader
   (:class:`~ts_shape.loader.metadata.metadata_json_loader.MetadataJsonLoader`
   ``.to_df()``, indexed by ``uuid``, or any frame with a ``uuid`` column).
   It supplies **labels only** (the OPC UA node path, else the display name,
   else the label); no signal role is read from it.

:func:`profile_cycles` runs :class:`~ts_shape.CycleProfileEvents` on them and
returns every table plus a summary of plain numbers. From the command line::

    python -m ts_shape.cycle_profile --container <name> --prefix <path/> \\
        --start 2026-11-17T00:00Z --end 2026-11-17T23:59:59Z \\
        --metadata-json datapoints.json --out profile/

The Azure credentials come from ``--sas-url`` or the environment
(``AZURE_STORAGE_CONNECTION_STRING``), as for the loader itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ts_shape.events.production.cycle_profiling import CycleProfileEvents
from ts_shape.signals.infer import infer_specs

if TYPE_CHECKING:
    from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

__all__ = [
    "LABEL_FIELDS",
    "CycleProfileResult",
    "load_azure_timeseries",
    "prepare_signals",
    "profile_cycles",
    "signal_labels",
]

#: Metadata columns tried for a signal's label, first filled one wins. The
#: ``[string]`` spellings are those of a flattened datapoint export.
LABEL_FIELDS: tuple[str, ...] = (
    "config.nodePath",
    "nodePath",
    "config.nodePath[string]",
    "config.displayName",
    "displayName",
    "config.displayName[string]",
    "label",
)
#: Signals dropped by default: personal data must not reach a report.
DEFAULT_EXCLUDE: tuple[str, ...] = ("Worker",)
_VALUE_COLUMNS = ("value_integer", "value_double", "value_string", "value_bool")


def signal_labels(metadata: pd.DataFrame | None) -> pd.Series:
    """``uuid -> label`` from a metadata table (labels only, no roles).

    Args:
        metadata: A metadata loader's table, indexed by ``uuid`` or with a
            ``uuid`` column; ``None`` gives no labels.

    Returns:
        Labels indexed by uuid (text); uuids without a label are left out.

    Example:
        >>> import pandas as pd
        >>> from ts_shape.cycle_profile import signal_labels
        >>> meta = pd.DataFrame({"uuid": ["a1", "b2"], "label": ["x", None],
        ...                      "config.nodePath": ["/Press/Counter", None]})
        >>> signal_labels(meta).to_dict()
        {'a1': '/Press/Counter'}
    """
    if metadata is None or len(metadata) == 0:
        return pd.Series(dtype=object)
    table = metadata.reset_index() if "uuid" not in metadata.columns else metadata
    if "uuid" not in table.columns:
        raise ValueError("metadata needs a 'uuid' column or index")
    uuids = table["uuid"].astype(str).to_numpy()
    labels = pd.Series([None] * len(table), index=uuids, dtype=object)
    for name in LABEL_FIELDS:
        if name in table.columns:
            values = table[name].to_numpy(dtype=object)
            usable = pd.Series(
                [v if isinstance(v, str) and v.strip() else None for v in values],
                index=uuids,
                dtype=object,
            )
            labels = labels.where(labels.notna(), usable)
    labels = labels.dropna()
    return labels[~labels.index.duplicated(keep="first")]


def prepare_signals(
    timeseries: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    *,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
) -> pd.DataFrame:
    """Turn a loader's raw frame into the long frame the profile reads.

    ``systime`` as integer Unix nanoseconds becomes a UTC timestamp,
    ``value_bool`` a nullable boolean, the ``uuid`` the signal's label from
    ``metadata`` (unlabeled uuids stay as they are), and columns the profile
    does not read (``sequence_number``, ``plctime``, ``value_bytes``) are
    dropped. Signals whose label matches a pattern in ``exclude`` are removed.

    Args:
        timeseries: Raw frame of a ts-shape timeseries loader.
        metadata: Optional metadata table (see :func:`signal_labels`).
        exclude: Regular expressions of signals to drop (default:
            ``Worker``, personal data). Empty keeps every signal.

    Returns:
        ``systime``, ``uuid``, ``is_delta`` (when present) and the
        ``value_*`` columns, sorted by time and signal.

    Example:
        >>> import pandas as pd
        >>> from ts_shape.cycle_profile import prepare_signals
        >>> raw = pd.DataFrame({"uuid": ["a1", "a1"],
        ...     "systime": [1794873604342112633, 1794873616342894346],
        ...     "value_double": [1.0, 2.0], "value_bool": [None, None],
        ...     "plctime": [0, 0]})
        >>> meta = pd.DataFrame({"uuid": ["a1"], "label": ["/Press/Temp"]})
        >>> out = prepare_signals(raw, meta)
        >>> out["uuid"].unique().tolist(), str(out["systime"].dt.tz)
        (['/Press/Temp'], 'UTC')
    """
    if "uuid" not in timeseries.columns or "systime" not in timeseries.columns:
        raise ValueError("timeseries needs 'uuid' and 'systime' columns")
    keep = ["systime", "uuid", "is_delta", *_VALUE_COLUMNS]
    frame = timeseries[[c for c in keep if c in timeseries.columns]].copy()
    for column in _VALUE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series([None] * len(frame), dtype=object)
    times = frame["systime"]
    if pd.api.types.is_integer_dtype(times):
        frame["systime"] = pd.to_datetime(times, unit="ns", utc=True)
    else:
        frame["systime"] = pd.to_datetime(times, utc=True)
    frame["value_bool"] = frame["value_bool"].astype("boolean")
    frame["value_integer"] = pd.to_numeric(frame["value_integer"], errors="coerce")
    frame["value_double"] = pd.to_numeric(frame["value_double"], errors="coerce")
    frame["uuid"] = frame["uuid"].astype(str)
    labels = signal_labels(metadata)
    if not labels.empty:
        frame["uuid"] = frame["uuid"].map(labels).fillna(frame["uuid"])
    if exclude:
        pattern = re.compile("|".join(f"(?:{p})" for p in exclude))
        frame = frame[~frame["uuid"].map(lambda u: bool(pattern.search(u)))]
    frame = frame.sort_values(["systime", "uuid"], kind="stable")
    return frame.reset_index(drop=True)


def signal_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """Kind, encoding and sample period of every signal (kickoff check LL-1)."""
    columns = [
        "signal",
        "kind",
        "encoding",
        "value_column",
        "nominal_period_seconds",
        "row_count",
        "distinct_value_count",
        "is_constant",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    counts = frame.groupby("uuid").size()
    for uuid, spec in infer_specs(frame).items():
        part = frame[frame["uuid"] == uuid]
        values = (
            part[spec.value_column].dropna()
            if spec.value_column in part
            else part.iloc[0:0]
        )
        distinct = int(values.nunique())
        period = spec.nominal_period
        rows.append(
            {
                "signal": uuid,
                "kind": spec.kind,
                "encoding": spec.encoding,
                "value_column": spec.value_column,
                "nominal_period_seconds": (
                    period.total_seconds() if period is not None else np.nan
                ),
                "row_count": int(counts.get(uuid, 0)),
                "distinct_value_count": distinct,
                "is_constant": distinct <= 1,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns).sort_values("signal").reset_index(drop=True)
    )


@dataclass
class CycleProfileResult:
    """Every table of one profile run and a summary of plain numbers.

    Attributes:
        tables: ``signals``, ``anchors``, ``evaluation_window``,
            ``standard_view``, ``cycle_time_statistics``,
            ``contribution_register``, ``segment_durations``,
            ``influencers``, ``cycle_records``, ``unusual_cycles``, ``pauses``,
            ``cycle_events`` (every event with its cycle and its offset from
            the cycle start; the input of event heatmaps).
        summary: Plain numbers (counts, seconds, percent), JSON-able.
    """

    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> list[Path]:
        """Write ``summary.json`` and one CSV per table into ``directory``.

        Args:
            directory: Target folder; created when missing.

        Returns:
            The written files.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written = [out / "summary.json"]
        written[0].write_text(
            json.dumps(self.summary, indent=1, allow_nan=False), encoding="utf-8"
        )
        for name, table in self.tables.items():
            path = out / f"{name}.csv"
            table.to_csv(path, index=False)
            written.append(path)
        return written

    @classmethod
    def load(cls, directory: str | Path) -> CycleProfileResult:
        """Read a result written by :meth:`save` (for dashboards).

        Args:
            directory: Folder written by :meth:`save`.

        Returns:
            The result with its summary and tables.
        """
        path = Path(directory)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        tables = {p.stem: pd.read_csv(p) for p in sorted(path.glob("*.csv"))}
        return cls(tables=tables, summary=summary)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(number) else number


def _pauses(records: pd.DataFrame) -> pd.DataFrame:
    pauses = records[records["cycle_class"] == "pause"]
    start = pd.to_datetime(pauses["start_ns"], unit="ns", utc=True)
    end = pd.to_datetime(pauses["end_ns"], unit="ns", utc=True)
    return pd.DataFrame(
        {
            "start": start.to_numpy(),
            "end": end.to_numpy(),
            "duration_seconds": pauses["duration_seconds"].to_numpy(dtype=float),
            "variant": pauses["variant"].to_numpy(),
        }
    )


def cycle_events(events: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Every event placed in its cycle, with its offset from the cycle start.

    Args:
        events: ``CycleProfileEvents.signal_events()``.
        records: ``CycleProfileEvents.cycle_records()``.

    Returns:
        ``cycle_uuid``, ``cycle_class``, ``variant``, ``event``,
        ``source_uuid``, ``offset_seconds`` (from the cycle start) and
        ``phase_ratio`` (offset / cycle duration), in time order. Events
        outside every complete cycle are left out.

    Example:
        >>> import ts_shape
        >>> from ts_shape.cycle_profile import profile_cycles
        >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=60, seed=1)
        >>> ev = profile_cycles(frame).tables["cycle_events"]
        >>> bool(ev["phase_ratio"].between(0, 1, inclusive="left").all())
        True
    """
    columns = [
        "cycle_uuid",
        "cycle_class",
        "variant",
        "event",
        "source_uuid",
        "offset_seconds",
        "phase_ratio",
    ]
    if events.empty or records.empty:
        return pd.DataFrame(columns=columns)
    ns = pd.DatetimeIndex(events["systime"]).as_unit("ns").asi8
    starts = records["start_ns"].to_numpy(dtype=np.int64)
    ends = records["end_ns"].to_numpy(dtype=np.int64)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    pos = np.searchsorted(starts, ns, side="right") - 1
    inside = pos >= 0
    inside[inside] &= ns[inside] < ends[pos[inside]]
    rec = records.iloc[order].reset_index(drop=True)
    idx = pos[inside]
    offset = (ns[inside] - starts[idx]) / 1e9
    duration = (ends[idx] - starts[idx]) / 1e9
    out = pd.DataFrame(
        {
            "cycle_uuid": rec["cycle_uuid"].to_numpy()[idx],
            "cycle_class": rec["cycle_class"].to_numpy()[idx],
            "variant": rec["variant"].to_numpy()[idx],
            "event": events["event"].to_numpy()[inside],
            "source_uuid": events["source_uuid"].to_numpy()[inside],
            "offset_seconds": offset,
            "phase_ratio": offset / duration,
        }
    )
    return out[columns].reset_index(drop=True)


def _summary(frame: pd.DataFrame, t: dict[str, pd.DataFrame]) -> dict[str, Any]:
    window = t["evaluation_window"]
    records = t["cycle_records"]
    stats = t["cycle_time_statistics"]
    register = t["contribution_register"]
    all_row = window[window["variant"] == "all"]
    w = all_row.iloc[0] if not all_row.empty else pd.Series(dtype=object)
    non_pause = records[records["cycle_class"] != "pause"]
    variants = []
    for row in stats.itertuples():
        reg = register[register["variant"] == row.variant]
        top_potential = reg.loc[reg["potential_seconds"].idxmax()] if len(reg) else None
        top_variance = (
            reg.loc[reg["variance_contribution_factor"].idxmax()] if len(reg) else None
        )
        gain = (
            (row.median_seconds / row.target_seconds - 1.0) * 100.0
            if row.target_seconds and row.target_seconds > 0
            else np.nan
        )
        variants.append(
            {
                "variant": row.variant,
                "cycle_count": int(row.cycle_count),
                "normal_count": int(row.normal_count),
                "long_count": int(row.long_count),
                "short_count": int(row.short_count),
                "pause_count": int(row.pause_count),
                "median_seconds": _num(row.median_seconds),
                "target_seconds": _num(row.target_seconds),
                "p90_seconds": _num(row.p90_seconds),
                "lower_limit_seconds": _num(row.lower_limit_seconds),
                "upper_limit_seconds": _num(row.upper_limit_seconds),
                "throughput_gain_pct": _num(gain),
                "cycles_per_hour_median": _num(3600.0 / row.median_seconds),
                "cycles_per_hour_target": _num(3600.0 / row.target_seconds),
                "top_potential_segment": (
                    None if top_potential is None else top_potential["segment"]
                ),
                "top_potential_seconds": (
                    None
                    if top_potential is None
                    else _num(top_potential["potential_seconds"])
                ),
                "top_variance_segment": (
                    None if top_variance is None else top_variance["segment"]
                ),
                "top_variance_share_pct": (
                    None
                    if top_variance is None
                    else _num(top_variance["variance_contribution_factor"] * 100.0)
                ),
            }
        )
    times = frame["systime"]
    return {
        "signal_count": int(frame["uuid"].nunique()),
        "row_count": len(frame),
        "data_start": None if frame.empty else times.min().isoformat(),
        "data_end": None if frame.empty else times.max().isoformat(),
        "anchor_event": w.get("anchor_event"),
        "anchor_uuid": w.get("anchor_uuid"),
        "variant_uuid": w.get("variant_uuid"),
        "time_resolution_seconds": _num(w.get("time_resolution_seconds")),
        "evaluable": bool(w.get("evaluable", False)),
        "reason": w.get("reason"),
        "cycle_count": len(records),
        "non_pause_cycle_count": len(non_pause),
        "long_count": int((records["cycle_class"] == "long").sum()),
        "short_count": int((records["cycle_class"] == "short").sum()),
        "pause_count": int((records["cycle_class"] == "pause").sum()),
        "pause_seconds": _num(
            records.loc[records["cycle_class"] == "pause", "duration_seconds"].sum()
        ),
        "longest_coherent_cycle_count": int(w.get("longest_coherent_cycle_count", 0)),
        "sequence_deviation_count": (
            int(records["sequence_deviation"].sum())
            if "sequence_deviation" in records
            else 0
        ),
        "variants": variants,
    }


def profile_cycles(
    timeseries: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    *,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    top_n: int = 10,
    **profile_options: Any,
) -> CycleProfileResult:
    """Profile the cycle time of one PLC from a loader's two outputs.

    Args:
        timeseries: Raw frame of a ts-shape timeseries loader (one PLC).
        metadata: Optional metadata table; labels only.
        exclude: Signals to drop (see :func:`prepare_signals`).
        top_n: Influencers kept per comparison.
        **profile_options: Passed to :class:`~ts_shape.CycleProfileEvents`
            (``min_cycles``, ``n_sigma``, ``time_resolution``, ...).

    Returns:
        The :class:`CycleProfileResult`.

    Example:
        >>> import ts_shape
        >>> from ts_shape.cycle_profile import profile_cycles
        >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
        >>> result = profile_cycles(frame)
        >>> result.summary["anchor_event"], result.summary["cycle_count"]
        ('plc:part_counter+1', 119)
    """
    frame = prepare_signals(timeseries, metadata, exclude=exclude)
    profile = CycleProfileEvents(frame, **profile_options)
    tables = {
        "signals": signal_inventory(frame),
        "anchors": profile.cycle_anchor_candidates(),
        "evaluation_window": profile.evaluation_window(),
        "standard_view": profile.standard_view(),
        "cycle_time_statistics": profile.cycle_time_statistics(),
        "contribution_register": profile.contribution_register(),
        "segment_durations": profile.segment_durations(),
        "influencers": profile.detect_cycle_influencers(top_n=top_n),
        "cycle_records": profile.cycle_records(),
        "unusual_cycles": profile.detect_unusual_cycles(),
    }
    tables["pauses"] = _pauses(tables["cycle_records"])
    tables["cycle_events"] = cycle_events(
        profile.signal_events(), tables["cycle_records"]
    )
    return CycleProfileResult(tables=tables, summary=_summary(frame, tables))


def load_azure_timeseries(
    loader: AzureBlobParquetLoader,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    signal_uuids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read one PLC's raw frame with the Azure loader as it is.

    Args:
        loader: A configured :class:`AzureBlobParquetLoader`.
        start: Start of the range.
        end: End of the range (inclusive).
        signal_uuids: The PLC's datapoint uuids (e.g. the metadata index);
            ``None`` reads every file in the hour folders.
    """
    if signal_uuids:
        return loader.load_files_by_time_range_and_uuids(start, end, list(signal_uuids))
    return loader.load_by_time_range(start, end)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: load from Azure, profile, write the result."""
    parser = argparse.ArgumentParser(
        prog="python -m ts_shape.cycle_profile",
        description="Cycle time profiling of one PLC from Azure Blob parquet files.",
    )
    parser.add_argument("--container", help="container name (not with --sas-url)")
    parser.add_argument("--sas-url", default=os.environ.get("AZURE_STORAGE_SAS_URL"))
    parser.add_argument(
        "--prefix", default="", help="path prefix before YYYY/MM/DD/HH/"
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--metadata-json", help="datapoint metadata (MetadataJsonLoader)"
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="read every file of the hour folders instead of the metadata's uuids",
    )
    parser.add_argument("--out", default="cycle_profile")
    args = parser.parse_args(argv)

    from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
    from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

    if args.sas_url:
        loader = AzureBlobParquetLoader(sas_url=args.sas_url, prefix=args.prefix)
    else:
        loader = AzureBlobParquetLoader(
            container_name=args.container,
            connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
            prefix=args.prefix,
        )
    metadata = (
        MetadataJsonLoader.from_file(args.metadata_json, strict=False).to_df()
        if args.metadata_json
        else None
    )
    uuids = None if (metadata is None or args.all_files) else list(metadata.index)
    raw = load_azure_timeseries(loader, args.start, args.end, uuids)
    result = profile_cycles(raw, metadata)
    result.save(args.out)
    print(
        json.dumps(
            {k: v for k, v in result.summary.items() if k != "variants"}, indent=1
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
