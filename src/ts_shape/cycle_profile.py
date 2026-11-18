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
from collections.abc import Mapping, Sequence
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
            the cycle start; the input of event heatmaps),
            ``recurring_events``, ``sequence_variants``, ``cycle_sequences``,
            ``process_graph``, ``parts_per_cycle``, ``parallel_lanes``,
            ``curve_bands``, ``curve_scores``, ``curve_flagged`` and
            ``cycle_time_check`` (see the functions of the same name).
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


#: An event recurs when it occurs in at least this share of non-pause cycles.
RECURRING_MIN_PRESENCE = 0.10
#: ... and at most this many times per cycle on average (else it is a clock).
RECURRING_MAX_PER_CYCLE = 2.0
#: A lane event repeats about once per cycle but is not locked to its phase.
LANE_MIN_RATE, LANE_MAX_RATE, LANE_MAX_LOCKING = 0.8, 1.25, 0.5
#: Sequence variants and the process graph use the events present in at
#: least half of the cycles (a short pulse that polling misses now and then
#: is lost data, not another route).
ROUTE_MIN_PRESENCE = 0.5
#: Points on the time-since-start grid of a cycle curve.
CURVE_GRID_POINTS = 64
#: Grid points a curve may shift before a difference counts as shape.
CURVE_SHIFT = 2
#: Value changes per cycle an analog signal needs to have a curve (a value
#: published once per cycle has no shape, however often it is polled).
CURVE_MIN_CHANGES = 2
#: A shape anomaly lies on average more than this many robust sigmas outside
#: the P10..P90 band over the whole window ...
CURVE_MIN_DEVIATION = 2.0
#: ... or is this many times farther from its nearest other cycle than usual.
CURVE_MIN_DISCORD = 3.0
#: Analog signals profiled as curves (the most variable ones).
CURVE_MAX_SIGNALS = 8
#: Unit factors tried when matching a reported cycle time (s, ms, min).
_TIME_UNITS = {"seconds": 1.0, "milliseconds": 1e-3, "minutes": 60.0}


def _non_pause(records: pd.DataFrame) -> pd.DataFrame:
    return records[records["cycle_class"] != "pause"]


def recurring_events(
    events: pd.DataFrame, records: pd.DataFrame, anchor_event: str | None = None
) -> pd.DataFrame:
    """The events that recur in the cycles, in their usual order.

    An event recurs when it occurs in at least 10 % of the non-pause cycles
    and at most twice per cycle on average (more is a clock, a heartbeat).

    Args:
        events: The ``cycle_events`` table.
        records: The ``cycle_records`` table.
        anchor_event: The cycle event, left out (it is the cycle boundary).

    Returns:
        ``event``, ``source_uuid``, ``presence_ratio``, ``events_per_cycle``
        and ``median_offset_seconds`` (first occurrence), ordered by the
        median offset.

    Example:
        >>> import ts_shape
        >>> from ts_shape.cycle_profile import profile_cycles, recurring_events
        >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=80, seed=1)
        >>> t = profile_cycles(frame).tables
        >>> recurring_events(t["cycle_events"], t["cycle_records"], "plc:part_counter+1")["event"].iloc[0]
        'plc:step=10'
    """
    columns = [
        "event",
        "source_uuid",
        "presence_ratio",
        "events_per_cycle",
        "median_offset_seconds",
    ]
    run = _non_pause(records)
    if events.empty or run.empty:
        return pd.DataFrame(columns=columns)
    body = events[events["cycle_uuid"].isin(run["cycle_uuid"])]
    if anchor_event is not None:
        body = body[body["event"] != anchor_event]
    n = len(run)
    rate = body.groupby("event").size() / n
    presence = body.groupby("event")["cycle_uuid"].nunique() / n
    keep = presence[
        (presence >= RECURRING_MIN_PRESENCE) & (rate <= RECURRING_MAX_PER_CYCLE)
    ].index
    first = (
        body[body["event"].isin(keep)]
        .groupby(["event", "cycle_uuid"])["offset_seconds"]
        .min()
    )
    out = pd.DataFrame(
        {
            "presence_ratio": presence.loc[keep],
            "events_per_cycle": rate.loc[keep],
            "median_offset_seconds": first.groupby("event").median(),
        }
    )
    out["source_uuid"] = body.drop_duplicates("event").set_index("event")["source_uuid"]
    out = out.reset_index().rename(columns={"index": "event"})
    return out.sort_values(["median_offset_seconds", "event"], kind="stable")[
        columns
    ].reset_index(drop=True)


def _ordered_first(
    events: pd.DataFrame, records: pd.DataFrame, rows: list[str], resolution: float
) -> pd.DataFrame:
    """The recurring events of every non-pause cycle as an ordered sequence.

    With a time resolution (polled signals) the order of events closer than
    one resolution is not observable:

    - events whose typical times (median first offset) lie within one
      resolution of each other form one step, labelled ``a + b``, at the
      earliest member's time in the cycle;
    - events typically within 1.5 resolutions of the cycle start or end are
      left out (they may fall on either side of the boundary).

    A real reordering (beyond the resolution) still changes the sequence.
    """
    run = _non_pause(records)
    sub = events[
        events["event"].isin(rows) & events["cycle_uuid"].isin(run["cycle_uuid"])
    ]
    first = sub.groupby(["cycle_uuid", "event"], as_index=False)["offset_seconds"].min()
    if first.empty:
        return first.assign(slot=pd.Series(dtype=float))
    typical = first.groupby("event")["offset_seconds"].median().sort_values()
    if resolution > 0:
        margin = 1.5 * resolution
        end = float(run["duration_seconds"].median())
        typical = typical[(typical >= margin) & (typical <= end - margin)]
    # Group events whose typical times lie within one resolution.
    group: dict[str, int] = {}
    start, gid = None, -1
    for event, t in typical.items():
        if start is None or t - start > resolution:
            gid, start = gid + 1, t
        group[str(event)] = gid
    members: dict[int, list[str]] = {}
    for event, g in group.items():
        members.setdefault(g, []).append(event)
    label = {g: " + ".join(sorted(m)) for g, m in members.items()}
    first = first[first["event"].isin(list(group))].copy()
    first["group"] = first["event"].map(group)
    steps = first.groupby(["cycle_uuid", "group"], as_index=False)[
        "offset_seconds"
    ].min()
    steps["event"] = steps["group"].map(label)
    steps["slot"] = steps["offset_seconds"]
    return steps.sort_values(["cycle_uuid", "slot", "event"], kind="stable")[
        ["cycle_uuid", "event", "offset_seconds", "slot"]
    ]


def sequence_variants(
    events: pd.DataFrame,
    records: pd.DataFrame,
    rows: list[str],
    *,
    resolution: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Group the non-pause cycles by the order of their recurring events.

    Args:
        events: The ``cycle_events`` table.
        records: The ``cycle_records`` table.
        rows: The recurring events (:func:`recurring_events`).
        resolution: Time resolution; events closer than this count as
            simultaneous.

    Returns:
        ``(variants, cycle_variant)``. ``variants``: ``sequence_rank``,
        ``cycle_count``, ``share_ratio``, ``median_seconds``,
        ``p90_seconds``, ``long_count``, ``short_count``, ``event_count``,
        ``signature`` (events joined by `` → ``), ``example_cycle_uuid``.
        ``cycle_variant``: the sequence rank of every non-pause cycle.
    """
    columns = [
        "sequence_rank",
        "cycle_count",
        "share_ratio",
        "median_seconds",
        "p90_seconds",
        "long_count",
        "short_count",
        "event_count",
        "signature",
        "example_cycle_uuid",
    ]
    run = _non_pause(records).set_index("cycle_uuid")
    if not rows or run.empty:
        return pd.DataFrame(columns=columns), pd.Series(dtype=float)
    first = _ordered_first(events, records, rows, resolution)
    signature = first.groupby("cycle_uuid", sort=False)["event"].agg(" → ".join)
    signature = signature.reindex(run.index).fillna("(no recurring event)")
    table = pd.DataFrame(
        {
            "signature": signature,
            "duration": run["duration_seconds"],
            "cls": run["cycle_class"],
        }
    )
    groups = table.groupby("signature", sort=False)
    out = pd.DataFrame(
        {
            "cycle_count": groups.size(),
            "median_seconds": groups["duration"].median(),
            "p90_seconds": groups["duration"].quantile(0.9),
            "long_count": groups["cls"].apply(lambda c: int((c == "long").sum())),
            "short_count": groups["cls"].apply(lambda c: int((c == "short").sum())),
            "example_cycle_uuid": groups.apply(
                lambda g: g.index[0], include_groups=False
            ),
        }
    ).reset_index()
    out["share_ratio"] = out["cycle_count"] / len(run)
    out["event_count"] = out["signature"].str.count(" → ") + 1
    out = out.sort_values(
        ["cycle_count", "signature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    out["sequence_rank"] = np.arange(1, len(out) + 1)
    rank = dict(zip(out["signature"], out["sequence_rank"]))
    return out[columns], table["signature"].map(rank)


def process_graph(
    events: pd.DataFrame,
    records: pd.DataFrame,
    rows: list[str],
    *,
    resolution: float = 0.0,
) -> pd.DataFrame:
    """Directly-follows graph of the recurring events within the cycles.

    Every non-pause cycle is a case: ``▶ cycle start`` → its recurring
    events in order → ``■ cycle end``. This is the process-mining view of
    the cycle (edges between steps with their frequency and time).

    Returns:
        ``source``, ``target``, ``cycle_count``, ``share_ratio`` (of the
        non-pause cycles), ``median_seconds`` and ``p90_seconds`` (time from
        source to target).
    """
    columns = [
        "source",
        "target",
        "cycle_count",
        "share_ratio",
        "median_seconds",
        "p90_seconds",
    ]
    run = _non_pause(records).set_index("cycle_uuid")
    if not rows or run.empty:
        return pd.DataFrame(columns=columns)
    first = _ordered_first(events, records, rows, resolution)
    starts = pd.DataFrame(
        {"cycle_uuid": run.index, "event": "▶ cycle start", "offset_seconds": 0.0}
    )
    ends = pd.DataFrame(
        {
            "cycle_uuid": run.index,
            "event": "■ cycle end",
            "offset_seconds": run["duration_seconds"].to_numpy(),
        }
    )
    seq = pd.concat(
        [starts, first[["cycle_uuid", "event", "offset_seconds"]], ends],
        ignore_index=True,
    )
    seq["order"] = seq.groupby("cycle_uuid").cumcount()
    seq.loc[seq["event"] == "■ cycle end", "order"] = 10**9
    seq = seq.sort_values(["cycle_uuid", "order"], kind="stable")
    seq["target"] = seq.groupby("cycle_uuid")["event"].shift(-1)
    seq["dt"] = (
        seq.groupby("cycle_uuid")["offset_seconds"].shift(-1) - seq["offset_seconds"]
    )
    edges = seq.dropna(subset=["target"]).rename(columns={"event": "source"})
    g = edges.groupby(["source", "target"])
    out = pd.DataFrame(
        {
            "cycle_count": g["cycle_uuid"].nunique(),
            "median_seconds": g["dt"].median(),
            "p90_seconds": g["dt"].quantile(0.9),
        }
    ).reset_index()
    out["share_ratio"] = out["cycle_count"] / len(run)
    return out.sort_values(
        ["cycle_count", "source", "target"], ascending=[False, True, True]
    )[columns].reset_index(drop=True)


def _lookup(frame: pd.DataFrame, signal: str, value_column: str) -> pd.DataFrame:
    """Time-sorted ``ns`` / ``v`` of one numeric signal (duplicates: last)."""
    part = frame.loc[frame["uuid"] == signal, ["systime", value_column]]
    out = pd.DataFrame(
        {
            "ns": pd.DatetimeIndex(part["systime"]).as_unit("ns").asi8,
            "v": pd.to_numeric(part[value_column], errors="coerce").to_numpy(
                dtype=float
            ),
        }
    ).dropna()
    return out.sort_values("ns", kind="stable").drop_duplicates("ns", keep="last")


def _value_at(lookup: pd.DataFrame, ns: np.ndarray, **merge: Any) -> np.ndarray:
    """Value of a signal at the times ``ns`` (any order)."""
    ns = np.asarray(ns, dtype=np.int64)
    order = np.argsort(ns, kind="stable")
    out = np.empty(len(ns))
    probe = pd.DataFrame({"ns": ns[order]})
    out[order] = pd.merge_asof(probe, lookup, on="ns", **merge)["v"].to_numpy(
        dtype=float
    )
    return out


def parts_per_cycle(
    frame: pd.DataFrame, records: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    """How much every counter advances per cycle (parts per cycle).

    A counter that advances by 4 in every cycle makes 4 parts per cycle (a
    washer basket, a multi-cavity mould). Counter resets are left out.

    Args:
        frame: The prepared long frame (:func:`prepare_signals`).
        records: The ``cycle_records`` table.
        inventory: The ``signals`` table (:func:`signal_inventory`).

    Returns:
        ``signal``, ``cycle_count``, ``median_parts_per_cycle``,
        ``mean_parts_per_cycle`` (robust to polling: an update just before
        or after a boundary moves a step to the neighbouring cycle),
        ``p10_parts_per_cycle``, ``p90_parts_per_cycle``.
    """
    columns = [
        "signal",
        "cycle_count",
        "median_parts_per_cycle",
        "mean_parts_per_cycle",
        "p10_parts_per_cycle",
        "p90_parts_per_cycle",
    ]
    run = _non_pause(records)
    rows = []
    for c in inventory[inventory["kind"] == "counter"].itertuples():
        lookup = _lookup(frame, c.signal, c.value_column)
        if lookup.empty or run.empty:
            continue
        delta = _value_at(lookup, run["end_ns"]) - _value_at(lookup, run["start_ns"])
        delta = delta[np.isfinite(delta) & (delta >= 0)]
        if len(delta) == 0:
            continue
        q10, q50, q90 = np.quantile(delta, [0.1, 0.5, 0.9])
        rows.append(
            {
                "signal": c.signal,
                "cycle_count": len(delta),
                "median_parts_per_cycle": float(q50),
                "mean_parts_per_cycle": float(np.mean(delta)),
                "p10_parts_per_cycle": float(q10),
                "p90_parts_per_cycle": float(q90),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def parallel_lanes(
    events: pd.DataFrame, records: pd.DataFrame, anchor_event: str | None = None
) -> pd.DataFrame:
    """Events that repeat at the cycle rate but are not locked to the cycle.

    A second press on the same controller (left/right) produces events about
    once per cycle at a phase that drifts through the cycle. The phase
    locking (mean resultant length of the event phases, 1 = always at the
    same point of the cycle, 0 = anywhere) separates them from the cycle's
    own steps.

    Args:
        events: The ``cycle_events`` table.
        records: The ``cycle_records`` table.
        anchor_event: The cycle event, left out.

    Returns:
        ``event``, ``source_uuid``, ``events_per_cycle``,
        ``phase_locking_ratio``, ``is_independent_lane``.
    """
    columns = [
        "event",
        "source_uuid",
        "events_per_cycle",
        "phase_locking_ratio",
        "is_independent_lane",
    ]
    run = _non_pause(records)
    body = events[events["cycle_uuid"].isin(run["cycle_uuid"])]
    if anchor_event is not None:
        body = body[body["event"] != anchor_event]
    if body.empty:
        return pd.DataFrame(columns=columns)
    rate = body.groupby("event").size() / max(len(run), 1)
    cand = rate[(rate >= LANE_MIN_RATE) & (rate <= LANE_MAX_RATE)].index
    sub = body[body["event"].isin(cand)]
    if sub.empty:
        return pd.DataFrame(columns=columns)
    angle = 2 * np.pi * sub["phase_ratio"].to_numpy(dtype=float)
    lock = (
        pd.DataFrame(
            {"event": sub["event"].to_numpy(), "c": np.cos(angle), "s": np.sin(angle)}
        )
        .groupby("event")[["c", "s"]]
        .mean()
    )
    out = pd.DataFrame(
        {
            "events_per_cycle": rate.loc[cand],
            "phase_locking_ratio": np.hypot(lock["c"], lock["s"]).reindex(cand),
        }
    )
    out["source_uuid"] = sub.drop_duplicates("event").set_index("event")["source_uuid"]
    out["is_independent_lane"] = out["phase_locking_ratio"] < LANE_MAX_LOCKING
    out = out.reset_index().rename(columns={"index": "event"})
    out = out.sort_values(["phase_locking_ratio", "event"], kind="stable")
    return out[columns].reset_index(drop=True)


def _shifted_nearest(curves: np.ndarray, shifts: range) -> np.ndarray:
    """RMS distance of every curve to its nearest other curve, shift-tolerant.

    For each shift the squared distances come from one matrix product
    (``|a|^2 + |b|^2 - 2 a.b``), so memory stays at cycles x cycles.
    """
    n, k = curves.shape
    best = np.full((n, n), np.inf)
    sq = (curves**2).sum(axis=1)
    for shift in shifts:
        moved = np.roll(curves, shift, axis=1)
        d2 = sq[:, None] + (moved**2).sum(axis=1)[None, :] - 2.0 * curves @ moved.T
        best = np.minimum(best, np.maximum(d2, 0.0) / k)
    np.fill_diagonal(best, np.inf)
    result: np.ndarray = np.sqrt(best.min(axis=1))
    return result


def _outlier(x: np.ndarray, n_sigma: float, floor: float) -> np.ndarray:
    """Beyond ``n_sigma`` robust sigmas, twice the median and ``floor``."""
    m = float(np.median(x))
    s = max(1.4826 * float(np.median(np.abs(x - m))), 1e-9)
    return ((x - m) / s > n_sigma) & (x > max(2.0 * m, floor))


def curve_profiles(
    frame: pd.DataFrame,
    records: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    n_sigma: float = 3.0,
    resolution: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shape of every analog signal within the cycle, and cycles that deviate.

    Each non-pause cycle at least as long as the window (cut-short cycles
    are already classified short) has its analog signal read on a fixed grid
    of :data:`CURVE_GRID_POINTS` points from the cycle start over the best
    demonstrated cycle time (P10), so every cycle is compared over the same
    span. Per signal:

    - ``deviation_score``: how far the curve leaves the P10..P90 band of all
      cycles, in robust sigmas, averaged over the grid; the band may shift by
      :data:`CURVE_SHIFT` grid points (at least one ``resolution``), so an
      edge that moves slightly is timing, not shape;
    - ``discord_score``: distance to the nearest other cycle's curve, with
      the same shift tolerance (matrix-profile discord: a shape no other
      cycle shares), scaled by the median nearest-neighbour distance;
    - ``is_shape_anomaly``: either score beyond ``n_sigma`` robust sigmas of
      its distribution over the cycles, above twice its median and above an
      absolute floor (deviation 2 sigmas on average over the window, discord
      3 times the usual nearest-neighbour distance).

    Args:
        frame: The prepared long frame.
        records: The ``cycle_records`` table.
        inventory: The ``signals`` table.
        n_sigma: Robust sigmas for a shape anomaly.
        resolution: Time resolution (s) of polled signals; the allowed shift
            of a curve is at least one resolution step.

    Returns:
        ``(bands, scores, flagged)``: ``bands`` per signal and grid point
        (``signal``, ``offset_seconds``, ``p10``, ``median``, ``p90``);
        ``scores`` per cycle and signal (``cycle_uuid``, ``signal``,
        ``deviation_score``, ``discord_score``, ``is_shape_anomaly``);
        ``flagged`` the curves of the anomalous cycles (``cycle_uuid``,
        ``signal``, ``offset_seconds``, ``value``).
    """
    b_cols = ["signal", "offset_seconds", "p10", "median", "p90"]
    s_cols = [
        "cycle_uuid",
        "signal",
        "deviation_score",
        "discord_score",
        "is_shape_anomaly",
    ]
    f_cols = ["cycle_uuid", "signal", "offset_seconds", "value"]

    def cat(parts: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=cols)
        return pd.concat(parts, ignore_index=True)[cols]

    run = _non_pause(records).reset_index(drop=True)
    # A curve needs readings within the cycle: at least CURVE_MIN_READINGS
    # per cycle (a value published once per cycle has no shape).
    analog = inventory[
        (inventory["kind"] == "continuous") & ~inventory["is_constant"].astype(bool)
    ]
    n_cycles = max(len(records), 1)

    def changes_per_cycle(signal: str, value_column: str) -> float:
        v = pd.to_numeric(
            frame.loc[frame["uuid"] == signal, value_column], errors="coerce"
        )
        return float((v.diff().fillna(0) != 0).sum()) / n_cycles

    analog = analog[
        [
            changes_per_cycle(a.signal, a.value_column) >= CURVE_MIN_CHANGES
            for a in analog.itertuples()
        ]
    ]
    if len(run) < 5 or analog.empty:
        return cat([], b_cols), cat([], s_cols), cat([], f_cols)
    span = float(np.quantile(run["duration_seconds"], 0.1))
    # Only cycles that cover the whole window: a cut-short cycle is already
    # "short"; its truncated curve is no shape fault.
    run = run[run["duration_seconds"] >= span].reset_index(drop=True)
    if len(run) < 5 or span <= 0:
        return cat([], b_cols), cat([], s_cols), cat([], f_cols)
    grid = np.linspace(0, span, CURVE_GRID_POINTS)
    step = span / (CURVE_GRID_POINTS - 1)
    shift = max(CURVE_SHIFT, int(np.ceil(resolution / step)) + 1 if step > 0 else 0)
    starts = run["start_ns"].to_numpy(dtype=np.int64)
    sample_ns = (starts[:, None] + (grid[None, :] * 1e9).astype(np.int64)).ravel()
    candidates = []
    for sig in analog.itertuples():
        lookup = _lookup(frame, sig.signal, sig.value_column)
        if len(lookup) < 3:
            continue
        curves = _value_at(lookup, sample_ns).reshape(len(run), CURVE_GRID_POINTS)
        if np.isnan(curves).all() or float(np.nanstd(curves)) == 0.0:
            continue
        median = np.nanmedian(curves, axis=0)
        candidates.append((float(np.nanstd(median)), str(sig.signal), curves))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    bands, scores, flagged = [], [], []
    shifts = range(-shift, shift + 1)
    for _, signal, curves in candidates[:CURVE_MAX_SIGNALS]:
        ok = ~np.isnan(curves).any(axis=1)
        if ok.sum() < 5:
            continue
        c = curves[ok]
        q10, q50, q90 = np.quantile(c, [0.1, 0.5, 0.9], axis=0)
        mad = 1.4826 * np.median(np.abs(c - q50), axis=0)
        floor = max(
            1e-9, 0.01 * float(np.median(np.abs(q50))), 0.25 * float(np.median(mad))
        )
        low = np.min([np.roll(q10, s) for s in shifts], axis=0)
        high = np.max([np.roll(q90, s) for s in shifts], axis=0)
        outside = np.maximum(c - high, 0) + np.maximum(low - c, 0)
        dev = np.mean(outside / np.maximum(mad, floor), axis=1)
        scale = max(float(np.median(q90 - q10)), floor)
        nn = _shifted_nearest(c / scale, shifts)
        disc = nn / max(float(np.median(nn)), 1e-9)
        anomaly = _outlier(dev, n_sigma, CURVE_MIN_DEVIATION) | _outlier(
            disc, n_sigma, CURVE_MIN_DISCORD
        )
        uuids = run.loc[ok, "cycle_uuid"].to_numpy()
        bands.append(
            pd.DataFrame(
                {
                    "signal": signal,
                    "offset_seconds": grid,
                    "p10": q10,
                    "median": q50,
                    "p90": q90,
                }
            )
        )
        scores.append(
            pd.DataFrame(
                {
                    "cycle_uuid": uuids,
                    "signal": signal,
                    "deviation_score": dev,
                    "discord_score": disc,
                    "is_shape_anomaly": anomaly,
                }
            )
        )
        for k in np.flatnonzero(anomaly):
            flagged.append(
                pd.DataFrame(
                    {
                        "cycle_uuid": uuids[k],
                        "signal": signal,
                        "offset_seconds": grid,
                        "value": c[k],
                    }
                )
            )
    return cat(bands, b_cols), cat(scores, s_cols), cat(flagged, f_cols)


def cycle_time_check(
    frame: pd.DataFrame, records: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    """Find signals that report the cycle time themselves, and compare.

    Many controllers publish the duration of the last cycle. A numeric
    signal whose value read right after each cycle end matches the derived
    cycle duration (in seconds, milliseconds or minutes) is such a signal;
    it is found from the data, not named. Its agreement is an independent
    check of the cycles the profile found.

    Returns:
        One row per numeric signal that matches within 10 % (median):
        ``signal``, ``unit``, ``cycle_count``,
        ``median_abs_difference_seconds``, ``p95_abs_difference_seconds``,
        ``correlation_ratio`` (Pearson r, clipped at 0), sorted best first.
    """
    columns = [
        "signal",
        "unit",
        "cycle_count",
        "median_abs_difference_seconds",
        "p95_abs_difference_seconds",
        "correlation_ratio",
    ]
    run = _non_pause(records)
    if len(run) < 5:
        return pd.DataFrame(columns=columns)
    numeric = inventory[
        inventory["value_column"].isin(["value_double", "value_integer"])
        & ~inventory["is_constant"]
    ]
    ends = run["end_ns"].to_numpy(dtype=np.int64)
    derived = run["duration_seconds"].to_numpy(dtype=float)
    tolerance = int(max(120.0, 0.25 * float(np.median(derived))) * 1e9)
    rows = []
    for sig in numeric.itertuples():
        lookup = _lookup(frame, sig.signal, sig.value_column)
        if len(lookup) < 5:
            continue
        reported = _value_at(lookup, ends, direction="forward", tolerance=tolerance)
        ok = np.isfinite(reported)
        if ok.sum() < 5:
            continue
        for unit, factor in _TIME_UNITS.items():
            seconds = reported[ok] * factor
            diff = np.abs(derived[ok] - seconds)
            if np.median(diff) <= 0.10 * np.median(derived[ok]):
                r = (
                    np.corrcoef(derived[ok], seconds)[0, 1]
                    if np.std(seconds) > 0
                    else 0.0
                )
                rows.append(
                    {
                        "signal": sig.signal,
                        "unit": unit,
                        "cycle_count": int(ok.sum()),
                        "median_abs_difference_seconds": float(np.median(diff)),
                        "p95_abs_difference_seconds": float(np.quantile(diff, 0.95)),
                        "correlation_ratio": (
                            float(max(r, 0.0)) if np.isfinite(r) else 0.0
                        ),
                    }
                )
                break
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(
        ["median_abs_difference_seconds", "signal"], kind="stable"
    ).reset_index(drop=True)


def case_log(
    events: pd.DataFrame, records: pd.DataFrame, rows: list[str] | None = None
) -> pd.DataFrame:
    """The cycles as a process-mining event log (one case per cycle).

    Columns follow the pm4py / XES conventions, so the frame goes straight
    into ``pm4py.discover_dfg`` or ``pm4py.discover_petri_net_inductive``:
    ``case:concept:name`` (cycle), ``concept:name`` (event),
    ``time:timestamp`` (UTC), plus ``case:cycle_class`` and
    ``case:variant``. ``rows`` limits the log to the recurring events.

    Example:
        >>> import ts_shape
        >>> from ts_shape.cycle_profile import case_log, profile_cycles
        >>> frame, _ = ts_shape.make_plc_cycles(n_cycles=60, seed=1)
        >>> t = profile_cycles(frame).tables
        >>> tuple(case_log(t["cycle_events"], t["cycle_records"]).columns[:3])
        ('case:concept:name', 'concept:name', 'time:timestamp')
    """
    columns = [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
        "case:cycle_class",
        "case:variant",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    ev = events if rows is None else events[events["event"].isin(rows)]
    start = (
        ev["cycle_uuid"]
        .map(records.set_index("cycle_uuid")["start_ns"])
        .astype("int64")
    )
    ts = pd.to_datetime(
        start + (ev["offset_seconds"] * 1e9).round().astype("int64"),
        unit="ns",
        utc=True,
    )
    out = pd.DataFrame(
        {
            "case:concept:name": ev["cycle_uuid"].to_numpy(),
            "concept:name": ev["event"].to_numpy(),
            "time:timestamp": ts.to_numpy(),
            "case:cycle_class": ev["cycle_class"].to_numpy(),
            "case:variant": ev["variant"].to_numpy(),
        }
    )
    return out.sort_values(
        ["time:timestamp", "concept:name"], kind="stable"
    ).reset_index(drop=True)


def _confidence(t: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """How far the profile can be trusted, from the data itself."""
    records = t["cycle_records"]
    run = records[records["cycle_class"] != "pause"]
    anchors = t["anchors"]
    chosen = anchors[anchors["is_anchor"].astype(bool)] if len(anchors) else anchors
    check = t.get("cycle_time_check", pd.DataFrame())
    variants = t.get("sequence_variants", pd.DataFrame())
    parts = t.get("parts_per_cycle", pd.DataFrame())
    lanes = t.get("parallel_lanes", pd.DataFrame())
    scores = t.get("curve_scores", pd.DataFrame())
    infl = t.get("influencers", pd.DataFrame())
    anchor_uuid = records["source_uuid"].iloc[0] if len(records) else None
    anchor_parts = parts[parts["signal"] == anchor_uuid] if len(parts) else parts
    return {
        "anchor_support_count": (
            int(chosen["support_count"].iloc[0]) if len(chosen) else 0
        ),
        "full_sequence_ratio": (
            _num(1 - run["sequence_deviation"].mean()) if len(run) else None
        ),
        "cycle_time_signal": None if check.empty else check["signal"].iloc[0],
        "cycle_time_signal_unit": None if check.empty else check["unit"].iloc[0],
        "cycle_time_median_abs_difference_seconds": (
            None
            if check.empty
            else _num(check["median_abs_difference_seconds"].iloc[0])
        ),
        "cycle_time_p95_abs_difference_seconds": (
            None if check.empty else _num(check["p95_abs_difference_seconds"].iloc[0])
        ),
        "sequence_variant_count": len(variants),
        "main_sequence_share_ratio": (
            None if variants.empty else _num(variants["share_ratio"].iloc[0])
        ),
        "parts_per_cycle": (
            None
            if anchor_parts.empty
            else _num(anchor_parts["median_parts_per_cycle"].iloc[0])
        ),
        "independent_lanes": (
            []
            if lanes.empty
            else lanes.loc[lanes["is_independent_lane"].astype(bool), "event"].tolist()
        ),
        "shape_anomaly_cycle_count": (
            0
            if scores.empty
            else int(
                scores.loc[
                    scores["is_shape_anomaly"].astype(bool), "cycle_uuid"
                ].nunique()
            )
        ),
        "significant_influencer_count": (
            0
            if infl.empty or "is_significant" not in infl
            else int(infl["is_significant"].astype(bool).sum())
        ),
    }


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
        "confidence": _confidence(t),
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
    records = tables["cycle_records"]
    tables["cycle_events"] = cycle_events(profile.signal_events(), records)
    window = tables["evaluation_window"]
    resolution = (
        float(window["time_resolution_seconds"].iloc[0]) if len(window) else 0.0
    )
    if not np.isfinite(resolution):
        resolution = 0.0
    anchor = records["anchor_event"].iloc[0] if len(records) else None
    lanes = parallel_lanes(tables["cycle_events"], records, anchor)
    tables["parallel_lanes"] = lanes
    # Events of an independent lane belong to another machine, not to the
    # cycle's own sequence.
    lane_events = set(lanes.loc[lanes["is_independent_lane"].astype(bool), "event"])
    recurring = recurring_events(tables["cycle_events"], records, anchor)
    recurring = recurring[~recurring["event"].isin(lane_events)]
    recurring = recurring.reset_index(drop=True)
    tables["recurring_events"] = recurring
    route = recurring.loc[recurring["presence_ratio"] >= ROUTE_MIN_PRESENCE, "event"]
    variants, per_cycle = sequence_variants(
        tables["cycle_events"], records, route.tolist(), resolution=resolution
    )
    tables["sequence_variants"] = variants
    tables["cycle_sequences"] = pd.DataFrame(
        {"cycle_uuid": per_cycle.index, "sequence_rank": per_cycle.to_numpy()}
    )
    tables["process_graph"] = process_graph(
        tables["cycle_events"], records, route.tolist(), resolution=resolution
    )
    tables["parts_per_cycle"] = parts_per_cycle(frame, records, tables["signals"])
    bands, scores, flagged = curve_profiles(
        frame,
        records,
        tables["signals"],
        resolution=resolution,
    )
    tables["curve_bands"], tables["curve_scores"], tables["curve_flagged"] = (
        bands,
        scores,
        flagged,
    )
    tables["cycle_time_check"] = cycle_time_check(frame, records, tables["signals"])
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


#: Benchmark scenarios of :func:`benchmark_profiles` (make_plc_cycles options).
BENCHMARK_SCENARIOS: dict[str, dict[str, Any]] = {
    "base": {},
    "parts_per_cycle": {"parts_per_cycle": 4},
    "reported_cycle_time": {"reported_cycle_time": True},
    "alternative_route": {"alt_sequence_probability": 0.15},
    "shape_faults": {"shape_fault_probability": 0.05},
    "parallel_lane": {"parallel_lane": True},
    "polled": {"poll_period": "2s", "reported_cycle_time": True},
}


def _prf(found: pd.Series, true: pd.Series) -> tuple[float, float]:
    tp = int((found & true).sum())
    precision = tp / int(found.sum()) if found.sum() else float("nan")
    recall = tp / int(true.sum()) if true.sum() else float("nan")
    return precision, recall


def benchmark_profiles(
    scenarios: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    seeds: Sequence[int] = (11,),
    n_cycles: int = 300,
) -> pd.DataFrame:
    """Score the profile against the ground truth of simulated machines.

    Every scenario (options of :func:`ts_shape.make_plc_cycles`) is run for
    every seed, profiled with :func:`profile_cycles` and compared with the
    generator's truth. Cycles are matched by start time (within 3 s when the
    signals are polled).

    Args:
        scenarios: ``{name: make_plc_cycles options}``; default
            :data:`BENCHMARK_SCENARIOS`.
        seeds: Random seeds.
        n_cycles: Cycles per run.

    Returns:
        One row per scenario and seed: ``anchor_ok``, ``cycle_match_ratio``,
        ``duration_mae_seconds``, precision / recall of ``long``, ``short``
        and ``pause``, ``route_ok`` (the alternative route is its own
        sequence variant), ``parts_ok``, ``lane_ok``,
        ``cycle_time_signal_ok``, ``shape_precision_ratio``,
        ``shape_recall_ratio``.

    Example:
        >>> from ts_shape.cycle_profile import benchmark_profiles
        >>> b = benchmark_profiles({"base": {}}, n_cycles=120)
        >>> bool(b["anchor_ok"].all()), float(b["cycle_match_ratio"].min())
        (True, 1.0)
    """
    from ts_shape.datasets import make_plc_cycles

    rows = []
    for name, options in (scenarios or BENCHMARK_SCENARIOS).items():
        for seed in seeds:
            frame, truth = make_plc_cycles(n_cycles=n_cycles, seed=seed, **options)
            result = profile_cycles(frame)
            t, conf = result.tables, result.summary["confidence"]
            rec = t["cycle_records"].sort_values("start_ns")
            truth = truth.assign(
                start_ns=pd.DatetimeIndex(truth["start"]).as_unit("ns").asi8
            ).sort_values("start_ns")
            tol = int(3e9) if options.get("poll_period") else 0
            m = pd.merge_asof(
                rec,
                truth[
                    ["start_ns", "duration_seconds", "kind", "sequence", "shape_fault"]
                ].rename(columns={"duration_seconds": "true_seconds"}),
                on="start_ns",
                direction="nearest",
                tolerance=tol,
            )
            matched = m["kind"].notna()
            row: dict[str, Any] = {
                "scenario": name,
                "seed": seed,
                "anchor_ok": str(result.summary["anchor_event"]).endswith(
                    "part_counter+1"
                ),
                "cycle_match_ratio": float(matched.mean()) if len(m) else 0.0,
                "duration_mae_seconds": float(
                    (
                        m.loc[matched, "duration_seconds"]
                        - m.loc[matched, "true_seconds"]
                    )
                    .abs()
                    .mean()
                ),
            }
            for cls in ("long", "short", "pause"):
                p, r = _prf(m["cycle_class"].eq(cls), m["kind"].eq(cls))
                row[f"{cls}_precision_ratio"], row[f"{cls}_recall_ratio"] = p, r
            seq = t["cycle_sequences"].merge(
                m[["cycle_uuid", "sequence"]], on="cycle_uuid"
            )
            swapped = seq[seq["sequence"] == "swapped"]["sequence_rank"]
            standard = seq[seq["sequence"] == "standard"]["sequence_rank"]
            row["route_ok"] = (
                bool(swapped.nunique() == 1 and not set(swapped) & set(standard))
                if len(swapped)
                else True
            )
            row["parts_ok"] = conf["parts_per_cycle"] == float(
                options.get("parts_per_cycle", 1)
            )
            lanes = conf["independent_lanes"]
            row["lane_ok"] = (
                bool(lanes) and all("lane2" in e for e in lanes)
                if options.get("parallel_lane")
                else not lanes
            )
            row["cycle_time_signal_ok"] = (
                str(conf["cycle_time_signal"]).endswith("last_cycle_time")
                if options.get("reported_cycle_time")
                else conf["cycle_time_signal"] is None
            )
            scores = t["curve_scores"]
            flagged = set(
                scores.loc[scores["is_shape_anomaly"].astype(bool), "cycle_uuid"]
            )
            faults = set(
                m.loc[m["shape_fault"].fillna(False).astype(bool), "cycle_uuid"]
            )
            row["shape_precision_ratio"] = (
                len(flagged & faults) / len(flagged)
                if flagged and faults
                else float("nan")
            )
            row["shape_recall_ratio"] = (
                len(flagged & faults) / len(faults) if faults else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)
