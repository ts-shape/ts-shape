"""Mapping-free stop profiling: why does the machine stop, what does it cost?

:class:`StopProfileEvents` takes the raw signals of one PLC -- no run-state
or alarm mapping -- and answers *which stop causes cost the most time?*

1. **Units.** With a cycle anchor (see
   :class:`~ts_shape.events.production.cycle_profiling.CycleProfileEvents`)
   every ``pause`` cycle is a *stop* and every ``long`` cycle a *micro-stop*;
   the lost time is the duration beyond the variant's median cycle and the
   stop is taken to start where the cycle should have ended. Without an
   anchor (a continuous line) the stops are the off-periods of a discovered
   run signal: a bool or state value during which a counter keeps counting.
   Stops that overlap a hole in which no signal logged anything are
   ``unknown`` time, not stops.
2. **Signatures.** The events in a window before each stop are compared with
   the same windows in normal running (:func:`~ts_shape.events.production.
   _unit_profile.enrichment`): an event that is much more frequent before
   stops (lift >= :data:`SIGNATURE_MIN_LIFT`, q < 0.05) is a stop signature.
   Each stop gets the earliest of them in its window (*first-out*); without
   one, a state that is enriched *during* stops (door open, manual mode);
   otherwise ``unexplained``.
3. **Register.** Per signature: count, lost time and its share, MTTR (median,
   P10, P90), MTBF and the potential (lost time beyond the signature's
   P10).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.production._unit_profile import (
    INCREMENT,
    REASON_NO_ANCHOR,
    REASON_NO_CANDIDATE,
    REASON_OK,
    _UnitProfiler,
    enrichment,
    evaluation_frame,
    first_label,
    free_windows,
    memo,
    overlaps_any,
    potential_register,
    silent_gaps,
    window_labels,
)
from ts_shape.events.production.cycle_profiling import (
    MAX_EVENTS_PER_CYCLE,
    CycleProfileEvents,
)
from ts_shape.utils.coerce import run_mask

__all__ = ["StopProfileEvents"]

#: A signature event is this many times more frequent before stops.
SIGNATURE_MIN_LIFT = 3.0
#: ... and seen before at least this many stops.
SIGNATURE_MIN_STOPS = 3
#: Shortest pre-window (seconds); otherwise one median cycle.
STOP_PRE_MIN_SECONDS = 30.0
#: An event seen before (or during) at least this share of all stops marks
#: the stop itself (the run signal falling), not a cause.
STOP_MARKER_SHARE = 0.9
#: The pre-window reaches this far past the estimated stop start.
STOP_POST_GRACE_SECONDS = 2.0
#: Baseline windows in normal running per stop.
BASELINE_WINDOWS_PER_STOP = 5
#: A held state is looked for in the first this many seconds of a stop (at
#: least the pre-window), against running windows of the same length.
HELD_WINDOW_SECONDS = 120.0
#: A gap with no row of any signal longer than this many median cycles (at
#: least :data:`HOLE_MIN_SECONDS`) is a data hole: unknown time.
HOLE_CYCLES = 10.0
HOLE_MIN_SECONDS = 7200.0
#: Without an anchor: a run value covers at least this share of the
#: counter's increments ...
RUN_STATE_MIN_COUNTER_SHARE = 0.95
#: ... and holds this share of the time (min, max).
RUN_STATE_TIME_SHARE = (0.3, 0.98)
#: Without an anchor, off-periods shorter than this are not stops (seconds).
MIN_STOP_SECONDS = 30.0
#: ``signal_kind`` of analog level crossings in the event frame.
ANALOG_KIND = "continuous"
#: Random seed of the baseline windows.
SEED = 0

_RECORD_COLUMNS = [
    "systime",
    "uuid",
    "source_uuid",
    "unit_uuid",
    "start_ns",
    "end_ns",
    "duration_seconds",
    "stop_class",
    "variant",
    "lost_seconds",
    "signature",
    "signature_kind",
    "preceding_events",
]
_SIGNATURE_COLUMNS = [
    "signature",
    "signature_kind",
    "stop_count",
    "lost_seconds",
    "lost_share_ratio",
    "mttr_median_seconds",
    "mttr_p10_seconds",
    "mttr_p90_seconds",
    "mtbf_seconds",
    "stops_per_hour",
    "potential_seconds",
    "lift_factor",
    "q_value",
    "potential_rank",
]
UNEXPLAINED = "unexplained"
DATA_HOLE = "data_hole"


def _drop_markers(evidence: pd.DataFrame) -> pd.DataFrame:
    """Mark events present at (almost) every stop as not significant.

    Such an event is the stop itself (the run signal falling, the counter
    stopping), not a cause that distinguishes stops.
    """
    if evidence.empty:
        return evidence
    out = evidence.copy()
    marker = out["unit_presence_ratio"] >= STOP_MARKER_SHARE
    out.loc[marker, "is_significant"] = False
    return out


class StopProfileEvents(_UnitProfiler):
    """Stops and their causes, learned from the raw signals of one PLC.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=300, seed=3, stop_causes=4, stop_probability=0.08)
        >>> stops = ts_shape.StopProfileEvents(frame)
        >>> bool(stops.evaluation_window()["evaluable"].iloc[0])
        True
    """

    PROFILE_KIND = "stops"
    PROFILE_TABLES = ("stop_records", "stop_signatures", "signature_evidence")
    PROFILE_ORDER = 10

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        pre_window: str | float | None = None,
        min_stops: int = 10,
        include_micro_stops: bool = True,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:stop_profile",
        **cycle_options: Any,
    ) -> None:
        """Initialize the stop profiler.

        Args:
            dataframe: Long-format signals of one PLC (``systime``, ``uuid``,
                ``value_*``).
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            pre_window: Length of the window before a stop in which its cause
                is looked for (offset string or seconds); default: one median
                cycle, at least :data:`STOP_PRE_MIN_SECONDS`.
            min_stops: Fewer stops are not evaluable.
            include_micro_stops: Treat ``long`` cycles as micro-stops (their
                time beyond the median is lost time).
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the stop records.
            **cycle_options: Options of the cycle profile built when
                ``cycle_profile`` is None (``n_sigma``, ``pause_factor`` ...).
        """
        super().__init__(
            dataframe,
            cycle_profile=cycle_profile,
            signal_uuids=signal_uuids,
            time_column=time_column,
            uuid_column=uuid_column,
            event_uuid=event_uuid,
            cycle_options=cycle_options,
        )
        self.pre_window = pre_window
        self.min_stops = int(min_stops)
        self.include_micro_stops = bool(include_micro_stops)

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------

    def _holes(self, median_cycle: float) -> list[tuple[int, int]]:
        ns = self._signals().all_ns
        limit = max(HOLE_MIN_SECONDS, HOLE_CYCLES * median_cycle)
        return silent_gaps(ns, min_gap_seconds=limit)

    def _stops_from_cycles(self) -> pd.DataFrame:
        cycles = self._units()
        limits = self._profile().variant_limits()
        classes = ["pause", "long"] if self.include_micro_stops else ["pause"]
        part = cycles[cycles["cycle_class"].isin(classes)].copy()
        median = part["variant"].map(
            lambda v: limits.get(str(v), {}).get("median", np.nan)
        )
        part["median_seconds"] = median.astype(float)
        start = part["start_ns"] + (part["median_seconds"] * 1e9).round().astype(
            "int64"
        )
        part["stop_start_ns"] = np.minimum(start, part["end_ns"] - 1)
        part["lost_seconds"] = (part["duration_seconds"] - part["median_seconds"]).clip(
            lower=0.0
        )
        part["stop_class"] = np.where(
            part["cycle_class"] == "pause", "stop", "micro_stop"
        )
        return part.rename(columns={"cycle_uuid": "unit_uuid"})

    def _run_signal(self) -> tuple[str, Any] | None:
        """``(uuid, running value)`` of a run signal, for a line without cycles."""
        specs = self._profile().signal_specs()
        events = self._events()
        counters = [u for u, s in specs.items() if s.kind == "counter"]
        if not counters:
            return None
        increments = events[events["event"].str.endswith(INCREMENT)]
        if increments.empty:
            return None
        busiest = increments["source_uuid"].value_counts().index[0]
        inc_ns = increments.loc[increments["source_uuid"] == busiest, "ns"].to_numpy()
        signals = self._signals()
        best: tuple[float, str, Any] | None = None
        for uuid, spec in specs.items():
            if spec.kind != "state" or uuid == busiest or uuid not in signals:
                continue
            values = signals.column(uuid, spec.value_column)
            ns = signals.ns(uuid)
            for value in pd.unique(values.dropna()):
                on = run_mask(values, run_states={value}, warn=False).fillna(False)
                on = on.to_numpy(dtype=bool)
                pos = np.searchsorted(ns, inc_ns, side="right") - 1
                covered = float(on[pos[pos >= 0]].mean()) if (pos >= 0).any() else 0.0
                dur = np.diff(np.append(ns, ns[-1]))
                share = float((dur * on).sum() / max(dur.sum(), 1))
                lo, hi = RUN_STATE_TIME_SHARE
                if covered >= RUN_STATE_MIN_COUNTER_SHARE and lo <= share <= hi:
                    score = covered - share  # covers the counting, little else
                    if best is None or score > best[0]:
                        best = (score, uuid, value)
        return None if best is None else (best[1], best[2])

    def _stops_from_run_signal(self, run: tuple[str, Any]) -> pd.DataFrame:
        uuid, value = run
        spec = self._profile().signal_specs()[uuid]
        signals = self._signals()
        ns = signals.ns(uuid)
        on = run_mask(
            signals.column(uuid, spec.value_column), run_states={value}, warn=False
        )
        on = on.fillna(True).to_numpy(dtype=bool)
        records = []
        i = 0
        while i < len(ns):
            if on[i]:
                i += 1
                continue
            j = i
            while j < len(ns) and not on[j]:
                j += 1
            end = ns[j] if j < len(ns) else ns[-1]
            seconds = (end - ns[i]) / 1e9
            if seconds >= MIN_STOP_SECONDS:
                records.append(
                    {
                        "unit_uuid": f"stop-{ns[i]}",
                        "start_ns": int(ns[i]),
                        "end_ns": int(end),
                        "stop_start_ns": int(ns[i]),
                        "duration_seconds": seconds,
                        "variant": "all",
                        "median_seconds": 0.0,
                        "lost_seconds": seconds,
                        "stop_class": "stop",
                    }
                )
            i = j
        return pd.DataFrame(records)

    @memo
    def _compute(self) -> dict[str, Any]:
        profile = self._profile()
        cycles = self._units()
        resolution = profile.resolution_seconds()
        source = "cycles"
        upstream: str | None = None
        if not cycles.empty:
            stops = self._stops_from_cycles()
            normal = cycles[cycles["cycle_class"] == "normal"]
            median_cycle = (
                float(normal["duration_seconds"].median()) if len(normal) else np.nan
            )
            # The stops are judged on their own duration: a cycle signal too
            # coarse for cycle times still marks minutes-long stops.
        else:
            run = self._run_signal()
            median_cycle = np.nan
            normal = cycles
            if run is None:
                stops = pd.DataFrame()
                source = "none"
            else:
                stops = self._stops_from_run_signal(run)
                source = f"run signal {run[0]}={run[1]!r}"
        result: dict[str, Any] = {
            "source": source,
            "resolution": resolution,
            "upstream": upstream,
            "median_cycle": median_cycle,
        }
        if stops.empty:
            result.update(stops=stops, evidence=pd.DataFrame(), held=pd.DataFrame())
            return result

        holes = self._holes(median_cycle if np.isfinite(median_cycle) else 0.0)
        in_hole = overlaps_any(stops["stop_start_ns"], stops["end_ns"], holes)
        stops = stops.reset_index(drop=True)
        stops["in_hole"] = in_hole

        pre = self._pre_window_seconds(median_cycle)
        pre_start = stops["stop_start_ns"].to_numpy() - int(pre * 1e9)
        pre_end = stops["stop_start_ns"].to_numpy() + int(STOP_POST_GRACE_SECONDS * 1e9)
        events = self._candidate_events(cycles)
        unit_sets = window_labels(events, pre_start, pre_end)
        # Baseline windows: as long as the pre-windows, outside every stop
        # (and data hole) -- normal running, not the start of another stop.
        busy = [
            *zip(stops["stop_start_ns"] - int(pre * 1e9), stops["end_ns"], strict=True),
            *holes,
        ]
        span = self._span()
        n_base = BASELINE_WINDOWS_PER_STOP * len(stops)
        pre_len = int((pre + STOP_POST_GRACE_SECONDS) * 1e9)
        base_start, base_end = free_windows(
            pre_len, n_base, span=span, busy=busy, seed=SEED
        )
        base_sets = window_labels(events, base_start, base_end)
        real = ~stops["in_hole"].to_numpy()
        evidence = enrichment(
            [s for s, keep in zip(unit_sets, real, strict=True) if keep],
            base_sets,
            min_units=SIGNATURE_MIN_STOPS,
            min_lift=SIGNATURE_MIN_LIFT,
        )
        evidence = _drop_markers(evidence)
        significant = (
            evidence.loc[evidence["is_significant"].astype(bool), "label"].tolist()
            if len(evidence)
            else []
        )
        first = first_label(events, pre_start, pre_end, significant)

        # States enriched *during* stops (door open, manual mode). Analog
        # crossings during a stop are its consequence (cooling), not a cause.
        discrete = events[events["signal_kind"] != ANALOG_KIND]
        # The same window length in stops and in running (a long stop would
        # otherwise collect any steady event): the first HELD_WINDOW of it.
        held_len = int(max(HELD_WINDOW_SECONDS, pre) * 1e9)
        held_end = np.minimum(stops["end_ns"], stops["stop_start_ns"] + held_len)
        during = window_labels(discrete, stops["stop_start_ns"], held_end)
        held_base = free_windows(held_len, n_base, span=span, busy=busy, seed=SEED + 1)
        held = enrichment(
            [s for s, keep in zip(during, real, strict=True) if keep],
            window_labels(discrete, *held_base),
            min_units=SIGNATURE_MIN_STOPS,
            min_lift=SIGNATURE_MIN_LIFT,
        )
        held = _drop_markers(held)
        held_labels = (
            held.loc[held["is_significant"].astype(bool), "label"].tolist()
            if len(held)
            else []
        )
        held_first = first_label(events, stops["stop_start_ns"], held_end, held_labels)

        signature, kind = [], []
        for f, h, hole in zip(first, held_first, stops["in_hole"], strict=True):
            if hole:
                signature.append(DATA_HOLE)
                kind.append("unknown")
            elif f is not None:
                signature.append(f)
                kind.append("first_out")
            elif h is not None:
                signature.append(h)
                kind.append("held_state")
            else:
                signature.append(UNEXPLAINED)
                kind.append(UNEXPLAINED)
        stops["signature"] = signature
        stops["signature_kind"] = kind
        stops["preceding_events"] = [
            ";".join(sorted(s & set(significant))) for s in unit_sets
        ]
        result.update(stops=stops, evidence=evidence, held=held)
        return result

    def _pre_window_seconds(self, median_cycle: float) -> float:
        if self.pre_window is not None:
            value = self.pre_window
            if isinstance(value, str):
                return float(pd.Timedelta(value).total_seconds())
            return float(value)
        base = median_cycle if np.isfinite(median_cycle) else 0.0
        return max(STOP_PRE_MIN_SECONDS, base)

    def _candidate_events(self, cycles: pd.DataFrame) -> pd.DataFrame:
        """Events rare enough to be a cause (not a heartbeat or the anchor)."""
        events = self._events()
        anchor = self._profile().cycle_anchor()
        if anchor is not None:
            events = events[events["event"] != anchor[0]]
        n = max(len(cycles), 1)
        counts = events["event"].value_counts()
        if len(cycles):
            keep = counts[counts <= MAX_EVENTS_PER_CYCLE * n].index
            events = events[events["event"].isin(set(keep))]
        return events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop_records(self) -> pd.DataFrame:
        """One record per stop (and micro-stop), with its signature.

        Returns:
            Point events at the stop end: ``systime``, ``uuid``,
            ``source_uuid``, ``unit_uuid``, ``start_ns`` (estimated stop
            start), ``end_ns``, ``duration_seconds`` (of the stop),
            ``stop_class`` (``stop`` / ``micro_stop``), ``variant``,
            ``lost_seconds``, ``signature`` (the first-out event, a held
            state, ``unexplained`` or ``data_hole``), ``signature_kind`` and
            ``preceding_events`` (every signature event in the pre-window).
        """
        stops = self._compute()["stops"]
        if stops.empty:
            return self._empty(_RECORD_COLUMNS)
        units = stops.assign(
            start_ns=stops["stop_start_ns"], unit_uuid=stops["unit_uuid"].astype(str)
        )
        return self._point_records(
            units,
            columns=_RECORD_COLUMNS,
            source=self._compute()["source"],
            stop_class=stops["stop_class"],
            variant=stops["variant"].astype(str),
            lost_seconds=stops["lost_seconds"].to_numpy(dtype=float),
            signature=stops["signature"],
            signature_kind=stops["signature_kind"],
            preceding_events=stops["preceding_events"],
        )

    def stop_signatures(self) -> pd.DataFrame:
        """The stop register: what each signature costs.

        Returns:
            One row per signature (``data_hole`` stops are left out):
            ``signature``, ``signature_kind``, ``stop_count``,
            ``lost_seconds``, ``lost_share_ratio`` (of all lost time),
            ``mttr_median_seconds`` / ``mttr_p10_seconds`` /
            ``mttr_p90_seconds`` (stop durations), ``mtbf_seconds`` (median
            time between two stops of the signature), ``stops_per_hour``,
            ``potential_seconds`` (lost time beyond the signature's P10
            stop), ``lift_factor`` and ``q_value`` (evidence) and
            ``potential_rank``, most potential first.
        """
        result = self._compute()
        stops = result["stops"]
        if stops.empty:
            return pd.DataFrame(columns=_SIGNATURE_COLUMNS)
        real = stops[stops["signature"] != DATA_HOLE].copy()
        if real.empty:
            return pd.DataFrame(columns=_SIGNATURE_COLUMNS)
        real["seconds"] = (real["end_ns"] - real["stop_start_ns"]) / 1e9
        reg = potential_register(real, keys=["signature"], value="seconds")
        total_lost = float(real["lost_seconds"].sum()) or np.nan
        span_ns = self._span_ns()
        # Each signature's evidence comes from the test that made it one.
        tests = {"first_out": result["evidence"], "held_state": result["held"]}
        rows = []
        for row in reg.itertuples(index=False):
            part = real[real["signature"] == row.signature]
            starts = np.sort(part["stop_start_ns"].to_numpy())
            mtbf = (
                float(np.median(np.diff(starts)) / 1e9) if len(starts) > 1 else np.nan
            )
            kind = str(part["signature_kind"].iloc[0])
            table = tests.get(kind, pd.DataFrame())
            ev = table[table["label"] == row.signature] if len(table) else table
            rows.append(
                {
                    "signature": row.signature,
                    "signature_kind": kind,
                    "stop_count": int(row.unit_count),
                    "lost_seconds": float(part["lost_seconds"].sum()),
                    "lost_share_ratio": float(part["lost_seconds"].sum()) / total_lost,
                    "mttr_median_seconds": row.median_seconds,
                    "mttr_p10_seconds": row.p10_seconds,
                    "mttr_p90_seconds": row.p90_seconds,
                    "mtbf_seconds": mtbf,
                    "stops_per_hour": (
                        row.unit_count / (span_ns / 3.6e12) if span_ns > 0 else np.nan
                    ),
                    "potential_seconds": row.total_potential_seconds,
                    "lift_factor": (
                        float(ev["lift_factor"].iloc[0]) if len(ev) else np.nan
                    ),
                    "q_value": float(ev["q_value"].iloc[0]) if len(ev) else np.nan,
                }
            )
        out = pd.DataFrame(rows).sort_values(
            ["potential_seconds", "lost_seconds"], ascending=False, kind="stable"
        )
        out["potential_rank"] = np.arange(1, len(out) + 1)
        return out[_SIGNATURE_COLUMNS].reset_index(drop=True)

    def signature_evidence(self) -> pd.DataFrame:
        """Why each event counts as a stop signature (or not).

        Returns:
            :func:`~ts_shape.events.production._unit_profile.enrichment` of
            the pre-stop windows (``window="before"``) and of the stop
            intervals (``window="during"``) against normal running:
            ``window``, ``label``, ``unit_count``, ``unit_presence_ratio``,
            ``baseline_presence_ratio``, ``lift_factor``, ``p_value``,
            ``q_value``, ``is_significant``.
        """
        result = self._compute()
        parts = []
        for name, table in (("before", result["evidence"]), ("during", result["held"])):
            if len(table):
                parts.append(table.assign(window=name))
        if not parts:
            return pd.DataFrame(
                columns=[
                    "window",
                    "label",
                    "unit_count",
                    "unit_presence_ratio",
                    "baseline_presence_ratio",
                    "lift_factor",
                    "p_value",
                    "q_value",
                    "is_significant",
                ]
            )
        out = pd.concat(parts, ignore_index=True)
        return out[["window", *[c for c in out.columns if c != "window"]]]

    def evaluation_window(self) -> pd.DataFrame:
        """Whether the stop profile can be trusted, and why not.

        Returns:
            One row: ``scope`` (``all``), ``unit_count`` (stops outside data
            holes), ``evaluable``, ``reason`` (``ok``, ``too_few_units``,
            ``below_resolution`` (judged on the stops' own durations, so a
            cycle signal too coarse for cycle times still marks minutes-long
            stops) or ``no_candidate_signal: run state`` without an anchor
            and a run signal),
            ``time_resolution_seconds``, ``source`` (``cycles`` or the run
            signal), ``explained_share_ratio`` (lost time with a signature)
            and ``unknown_seconds`` (stops inside data holes).
        """
        result = self._compute()
        stops = result["stops"]
        extra = ["source", "explained_share_ratio", "unknown_seconds"]
        if result["source"] == "none":
            reason = (
                f"{REASON_NO_CANDIDATE}: run state (no cycle anchor and no run signal)"
            )
            return evaluation_frame(
                [
                    {
                        "scope": "all",
                        "unit_count": 0,
                        "evaluable": False,
                        "reason": (
                            reason if not len(self.dataframe) == 0 else REASON_NO_ANCHOR
                        ),
                        "time_resolution_seconds": result["resolution"],
                        "source": "none",
                        "explained_share_ratio": np.nan,
                        "unknown_seconds": 0.0,
                    }
                ],
                extra,
            )
        real = stops[stops["signature"] != DATA_HOLE] if len(stops) else stops
        lost = float(real["lost_seconds"].sum()) if len(real) else 0.0
        explained = (
            float(real.loc[real["signature"] != UNEXPLAINED, "lost_seconds"].sum())
            if len(real)
            else 0.0
        )
        durations = (real["end_ns"] - real["stop_start_ns"]) / 1e9 if len(real) else []
        median = float(np.median(durations)) if len(durations) else np.nan
        unknown = (
            float(
                ((stops["end_ns"] - stops["stop_start_ns"]) / 1e9)[
                    stops["signature"] == DATA_HOLE
                ].sum()
            )
            if len(stops)
            else 0.0
        )
        return self._evaluation_row(
            len(real),
            self.min_stops,
            median_seconds=median,
            # Stops are judged on their own duration (a line has no cycles).
            upstream=result["upstream"] or REASON_OK,
            resolution=result["resolution"],
            source=result["source"],
            explained_share_ratio=explained / lost if lost > 0 else np.nan,
            unknown_seconds=unknown,
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``stop_count``, ``lost_hours``,
            ``explained_share_ratio``, ``top_signature`` and
            ``top_signature_lost_hours``.
        """
        window = self.evaluation_window().iloc[0]
        register = self.stop_signatures()
        explained = register[register["signature"] != UNEXPLAINED].sort_values(
            "lost_seconds", ascending=False, kind="stable"
        )
        top = explained.iloc[0] if len(explained) else None
        share = window["explained_share_ratio"]
        return {
            **self._summary_head("stop_count"),
            "lost_hours": (
                float(register["lost_seconds"].sum() / 3600.0) if len(register) else 0.0
            ),
            "explained_share_ratio": None if pd.isna(share) else float(share),
            "top_signature": None if top is None else str(top["signature"]),
            "top_signature_lost_hours": (
                None if top is None else float(top["lost_seconds"] / 3600.0)
            ),
        }

    def _span(self) -> tuple[int, int]:
        ns = self._signals().all_ns
        return (int(np.amin(ns)), int(np.amax(ns))) if len(ns) else (0, 0)

    def _span_ns(self) -> int:
        first, last = self._span()
        return last - first
