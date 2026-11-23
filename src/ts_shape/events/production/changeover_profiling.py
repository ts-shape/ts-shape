"""Mapping-free changeover (SMED) and start-up profiling.

:class:`ChangeoverProfileEvents` answers *which changeover step to shorten
first?* and :class:`StartupProfileEvents` *how long until the machine runs
normally again after a stop, and what holds it back?* -- both from the raw
signals of one PLC, on top of its cycle profile
(:class:`~ts_shape.events.production.cycle_profiling.CycleProfileEvents`).

Changeovers
    A changeover is a change of the product variant the cycle profile found
    (the article / recipe signal). It runs from the end of the last normal
    cycle of the old variant to the start of the first run of
    ``ramp_cycles`` normal cycles of the new one, and is split into the
    ``changeover`` itself (until the first new cycle ends) and the
    ``ramp_up``. Events much more frequent in changeovers than in running
    are step candidates; those present in most changeovers form the
    standard step sequence, ordered by their median offset. The time between
    consecutive steps is a segment, ranked by its potential (median - P10).
    A step that has also been done while the machine was still producing is
    tagged ``external_possible`` (SMED: it can move out of the stop).

Start-ups
    A start-up follows a stop (a ``pause`` cycle) or a changeover and lasts
    until ``ramp_cycles`` consecutive normal cycles. Each analog signal gets
    its running band (P10-P90 of its per-cycle mean in steady running); the
    time a start-up needs until a signal is back in its band, and which
    signal is last, shows what holds the restart back.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.production._unit_profile import (
    REASON_NO_CANDIDATE,
    _UnitProfiler,
    enrichment,
    evaluation_frame,
    evaluation_reason,
    potential_register,
    window_labels,
)
from ts_shape.events.production.cycle_profiling import CycleProfileEvents
from ts_shape.utils.frames import to_ns

__all__ = ["ChangeoverProfileEvents", "StartupProfileEvents"]

#: A step event is this many times more frequent in changeovers than in running.
CHANGEOVER_MIN_LIFT = 3.0
#: A text signal with more distinct values than this is a changing id
#: (fixture number, order number): its changes count as one step label.
ID_MAX_VALUES = 4
#: A ramp-up cycle is normal when its robust z is at most this.
RAMP_MAX_Z = 1.5
#: Smallest half-width of a running band, relative to the signal's median.
BAND_MIN_RELATIVE = 0.01
#: Cycles after a trigger in which a signal's settling is looked for.
MAX_STARTUP_CYCLES = 30
#: How far before a changeover a step counts as done while producing
#: (median cycles).
EXTERNAL_LOOKBACK_CYCLES = 10.0

_CHANGEOVER_RECORDS = [
    "systime",
    "uuid",
    "source_uuid",
    "unit_uuid",
    "start_ns",
    "end_ns",
    "duration_seconds",
    "from_variant",
    "to_variant",
    "changeover_seconds",
    "ramp_up_seconds",
    "ramp_up_cycle_count",
    "steps_seen",
]
_STARTUP_RECORDS = [
    "systime",
    "uuid",
    "source_uuid",
    "unit_uuid",
    "start_ns",
    "end_ns",
    "duration_seconds",
    "trigger",
    "variant",
    "startup_cycle_count",
    "lost_seconds",
    "last_signal",
]


def _step_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Events with id-like text changes folded into one label per signal."""
    text = events["signal_kind"] == "text"
    if not text.any():
        return events
    counts = events[text].groupby("source_uuid")["event"].nunique()
    ids = set(counts[counts > ID_MAX_VALUES].index)
    if not ids:
        return events
    out = events.copy()
    changing = out["source_uuid"].isin(ids) & text
    out.loc[changing, "event"] = out.loc[changing, "source_uuid"].astype(str) + "Δ"
    return out


def _normal_runs(classes: np.ndarray, z: np.ndarray, start: int, length: int) -> int:
    """Index of the first cycle from ``start`` that begins ``length`` normal cycles."""
    run = 0
    for i in range(start, len(classes)):
        ok = classes[i] == "normal" and np.isfinite(z[i]) and abs(z[i]) <= RAMP_MAX_Z
        run = run + 1 if ok else 0
        if run == length:
            return i - length + 1
    return -1


class ChangeoverProfileEvents(_UnitProfiler):
    """Changeovers between product variants and their standard steps.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=600, seed=2, article_block=40, changeover_steps=True)
        >>> co = ts_shape.ChangeoverProfileEvents(frame)
        >>> int(co.evaluation_window()["unit_count"].iloc[0]) >= 10
        True
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        presence_ratio: float = 0.6,
        ramp_cycles: int = 3,
        min_changeovers: int = 5,
        external_share: float = 0.2,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:changeover_profile",
        **cycle_options: Any,
    ) -> None:
        """Initialize the changeover profiler.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            presence_ratio: A step is standard when it occurs in at least this
                share of the changeovers.
            ramp_cycles: The changeover ends with this many consecutive
                normal cycles of the new variant.
            min_changeovers: Fewer changeovers are not evaluable.
            external_share: A step done while still producing in at least
                this share of the changeovers is ``external_possible``.
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the changeover records.
            **cycle_options: Options of the cycle profile built when
                ``cycle_profile`` is None.
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
        self.presence_ratio = float(presence_ratio)
        self.ramp_cycles = int(ramp_cycles)
        self.min_changeovers = int(min_changeovers)
        self.external_share = float(external_share)

    def _changeovers(self) -> pd.DataFrame:
        if "changeovers" in self._cache:
            return self._cache["changeovers"]
        cycles = self._units().reset_index(drop=True)
        rows = []
        if len(cycles) > 1 and self._profile().variant_signal() is not None:
            variant = cycles["variant"].astype(str).to_numpy()
            classes = cycles["cycle_class"].to_numpy(dtype=object)
            z = cycles["z"].to_numpy(dtype=float)
            for i in np.flatnonzero(variant[1:] != variant[:-1]) + 1:
                last_a = i - 1
                while last_a >= 0 and classes[last_a] != "normal":
                    last_a -= 1
                first_b = _normal_runs(classes, z, int(i), self.ramp_cycles)
                if last_a < 0 or first_b < 0:
                    continue
                start = int(cycles.at[last_a, "end_ns"])
                change_end = int(cycles.at[i, "end_ns"])
                end = int(cycles.at[first_b, "start_ns"])
                end = max(end, change_end)
                rows.append(
                    {
                        "unit_uuid": f"changeover-{cycles.at[i, 'cycle_uuid']}",
                        "start_ns": start,
                        "change_end_ns": change_end,
                        "end_ns": end,
                        "from_variant": variant[i - 1],
                        "to_variant": variant[i],
                        "ramp_up_cycle_count": int(max(first_b - i - 1, 0)),
                    }
                )
        out = pd.DataFrame(
            rows,
            columns=[
                "unit_uuid",
                "start_ns",
                "change_end_ns",
                "end_ns",
                "from_variant",
                "to_variant",
                "ramp_up_cycle_count",
            ],
        )
        self._cache["changeovers"] = out
        return out

    def _steps(self) -> dict[str, Any]:
        if "steps" in self._cache:
            cached: dict[str, Any] = self._cache["steps"]
            return cached
        units = self._changeovers()
        cycles = self._units()
        events = _step_labels(self._events())
        # Analog crossings in a changeover are its consequence (the mould
        # cools), not a step someone does.
        events = events[events["signal_kind"] != "continuous"]
        anchor = self._profile().cycle_anchor()
        if anchor is not None:
            events = events[events["event"] != anchor[0]]
        result: dict[str, Any] = {"standard": [], "evidence": pd.DataFrame()}
        if units.empty:
            self._cache["steps"] = result
            return result
        normal = cycles[cycles["cycle_class"] == "normal"]
        unit_sets = window_labels(events, units["start_ns"], units["change_end_ns"])
        base_sets = window_labels(events, normal["start_ns"], normal["end_ns"])
        evidence = enrichment(
            unit_sets,
            base_sets,
            min_units=max(2, int(np.ceil(self.presence_ratio * len(units)))),
            min_lift=CHANGEOVER_MIN_LIFT,
        )
        steps = (
            evidence.loc[evidence["is_significant"].astype(bool), "label"].tolist()
            if len(evidence)
            else []
        )
        # Offsets of each step's first occurrence in each changeover.
        ev = events[events["event"].isin(set(steps))].sort_values("ns")
        offsets: dict[str, list[float]] = {s: [] for s in steps}
        first_times: list[dict[str, int]] = []
        ns_all = ev["ns"].to_numpy(dtype=np.int64)
        names = ev["event"].to_numpy(dtype=object)
        for start, end in zip(units["start_ns"], units["change_end_ns"], strict=True):
            lo, hi = np.searchsorted(ns_all, [start, end])
            seen: dict[str, int] = {}
            for k in range(lo, hi):
                seen.setdefault(str(names[k]), int(ns_all[k]))
            first_times.append(seen)
            for label, at in seen.items():
                offsets[label].append((at - start) / 1e9)
        order = sorted(
            (s for s in steps if offsets[s]), key=lambda s: float(np.median(offsets[s]))
        )
        # External: the step event also occurs while still producing.
        median_cycle = (
            float(normal["duration_seconds"].median()) if len(normal) else 0.0
        )
        lookback = int(EXTERNAL_LOOKBACK_CYCLES * median_cycle * 1e9)
        before_sets = window_labels(
            events, units["start_ns"] - lookback, units["start_ns"]
        )
        external = {
            s: (
                float(np.mean([s in b for b in before_sets]))
                if len(before_sets)
                else 0.0
            )
            for s in order
        }
        result = {
            "standard": order,
            "evidence": evidence,
            "first_times": first_times,
            "external": external,
            "offsets": offsets,
        }
        self._cache["steps"] = result
        return result

    def changeover_records(self) -> pd.DataFrame:
        """One record per changeover.

        Returns:
            Point events at the changeover end: ``systime``, ``uuid``,
            ``source_uuid`` (the variant signal), ``unit_uuid``,
            ``start_ns``, ``end_ns``, ``duration_seconds``, ``from_variant``,
            ``to_variant``, ``changeover_seconds`` (until the first new cycle
            ends), ``ramp_up_seconds``, ``ramp_up_cycle_count`` and
            ``steps_seen`` (standard steps that occurred).
        """
        units = self._changeovers()
        if units.empty:
            return pd.DataFrame(columns=_CHANGEOVER_RECORDS)
        steps = self._steps()
        seen = [
            sum(1 for s in steps["standard"] if s in times)
            for times in steps.get("first_times", [{}] * len(units))
        ]
        out = pd.DataFrame(
            {
                "systime": self._times(units["end_ns"].to_numpy()),
                "uuid": self.event_uuid,
                "source_uuid": self._profile().variant_signal(),
                "unit_uuid": units["unit_uuid"].to_numpy(),
                "start_ns": units["start_ns"].to_numpy(dtype=np.int64),
                "end_ns": units["end_ns"].to_numpy(dtype=np.int64),
                "duration_seconds": (units["end_ns"] - units["start_ns"]).to_numpy()
                / 1e9,
                "from_variant": units["from_variant"].to_numpy(),
                "to_variant": units["to_variant"].to_numpy(),
                "changeover_seconds": (
                    units["change_end_ns"] - units["start_ns"]
                ).to_numpy()
                / 1e9,
                "ramp_up_seconds": (units["end_ns"] - units["change_end_ns"]).to_numpy()
                / 1e9,
                "ramp_up_cycle_count": units["ramp_up_cycle_count"].to_numpy(),
                "steps_seen": seen,
            }
        )
        return out[_CHANGEOVER_RECORDS]

    def changeover_view(self) -> pd.DataFrame:
        """The standard step sequence of a changeover.

        Returns:
            One row per standard step in order: ``position``, ``step`` (event
            label; ``<signal>Δ`` for a changing id such as a fixture number),
            ``presence_ratio``, ``median_offset_seconds`` (from the
            changeover start), ``lift_factor`` (vs running),
            ``external_demonstrated_ratio`` and ``step_hint``
            (``external_possible`` / ``internal``).
        """
        steps = self._steps()
        columns = [
            "position",
            "step",
            "presence_ratio",
            "median_offset_seconds",
            "lift_factor",
            "external_demonstrated_ratio",
            "step_hint",
        ]
        if not steps["standard"]:
            return pd.DataFrame(columns=columns)
        evidence = steps["evidence"].set_index("label")
        n = max(len(self._changeovers()), 1)
        rows = []
        for i, step in enumerate(steps["standard"], start=1):
            ext = steps["external"].get(step, 0.0)
            rows.append(
                {
                    "position": i,
                    "step": step,
                    "presence_ratio": len(steps["offsets"][step]) / n,
                    "median_offset_seconds": float(np.median(steps["offsets"][step])),
                    "lift_factor": float(evidence.at[step, "lift_factor"]),
                    "external_demonstrated_ratio": ext,
                    "step_hint": (
                        "external_possible"
                        if ext >= self.external_share
                        else "internal"
                    ),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def changeover_register(self) -> pd.DataFrame:
        """What each changeover segment costs: the SMED ranking.

        The segments run from the changeover start to the first standard
        step, between consecutive steps, and from the last step to the end of
        the changeover phase; ``ramp_up`` is the time after it until normal
        running. A segment whose closing step did not occur is left out for
        that changeover.

        Returns:
            One row per segment: ``segment``, ``unit_count``,
            ``median_seconds``, ``p10_seconds``, ``p90_seconds``,
            ``spread_seconds``, ``mean_seconds``, ``total_seconds``,
            ``potential_seconds`` (median - P10), ``total_potential_seconds``,
            ``potential_rank`` and ``step_hint`` of the closing step.
        """
        units = self._changeovers()
        steps = self._steps()
        if units.empty:
            return potential_register(
                pd.DataFrame(columns=["segment", "seconds"]), keys=["segment"]
            ).assign(step_hint=pd.Series(dtype=object))
        names = ["start", *steps["standard"], "end"]
        rows = []
        for (start, change_end, end), times in zip(
            units[["start_ns", "change_end_ns", "end_ns"]].itertuples(
                index=False, name=None
            ),
            steps.get("first_times", [{}] * len(units)),
            strict=True,
        ):
            marks = {"start": start, "end": change_end, **times}
            for a, b in zip(names[:-1], names[1:], strict=True):
                if a in marks and b in marks and marks[b] >= marks[a]:
                    rows.append(
                        {
                            "segment": f"{a} → {b}",
                            "seconds": (marks[b] - marks[a]) / 1e9,
                        }
                    )
            rows.append({"segment": "ramp_up", "seconds": (end - change_end) / 1e9})
        register = potential_register(pd.DataFrame(rows), keys=["segment"])
        hints = dict(
            zip(
                self.changeover_view()["step"],
                self.changeover_view()["step_hint"],
                strict=True,
            )
        )
        register["step_hint"] = [
            hints.get(seg.split(" → ")[-1], "") for seg in register["segment"]
        ]
        return register

    def changeover_matrix(self) -> pd.DataFrame:
        """Changeover times per from -> to variant.

        Returns:
            ``from_variant``, ``to_variant``, ``changeover_count``,
            ``median_seconds``, ``p10_seconds`` (whole changeover incl.
            ramp-up).
        """
        records = self.changeover_records()
        columns = [
            "from_variant",
            "to_variant",
            "changeover_count",
            "median_seconds",
            "p10_seconds",
        ]
        if records.empty:
            return pd.DataFrame(columns=columns)
        grouped = records.groupby(["from_variant", "to_variant"])["duration_seconds"]
        out = grouped.agg(
            changeover_count="count",
            median_seconds="median",
            p10_seconds=lambda v: float(np.percentile(v, 10)),
        ).reset_index()
        return out[columns]

    def evaluation_window(self) -> pd.DataFrame:
        """Whether the changeover profile can be trusted, and why not.

        Returns:
            One row: ``scope`` (``all``), ``unit_count`` (changeovers),
            ``evaluable``, ``reason`` (``ok``, ``too_few_units``,
            ``no_candidate_signal: variant`` when the cycle profile found no
            variant signal, ``upstream_not_evaluable`` without cycles),
            ``time_resolution_seconds``, ``variant_uuid`` and
            ``standard_step_count``.
        """
        profile = self._profile()
        resolution = profile.resolution_seconds()
        units = self._changeovers()
        extra = ["variant_uuid", "standard_step_count"]
        variant = profile.variant_signal()
        if self._units().empty:
            evaluable, reason = (
                False,
                f"upstream_not_evaluable: {profile.evaluation_reason()}",
            )
        elif variant is None:
            evaluable, reason = False, f"{REASON_NO_CANDIDATE}: variant"
        else:
            durations = (units["end_ns"] - units["start_ns"]) / 1e9
            evaluable, reason = evaluation_reason(
                len(units),
                self.min_changeovers,
                resolution=resolution,
                median_seconds=float(np.median(durations)) if len(units) else np.nan,
            )
        return evaluation_frame(
            [
                {
                    "scope": "all",
                    "unit_count": int(len(units)),
                    "evaluable": evaluable,
                    "reason": reason,
                    "time_resolution_seconds": resolution,
                    "variant_uuid": variant,
                    "standard_step_count": len(self._steps()["standard"]),
                }
            ],
            extra,
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``changeover_count``,
            ``median_changeover_seconds``, ``lost_hours`` (sum of changeover
            durations), ``top_segment`` and its ``top_segment_potential_seconds``.
        """
        window = self.evaluation_window().iloc[0]
        records = self.changeover_records()
        register = self.changeover_register()
        top = register.iloc[0] if len(register) else None
        return {
            "evaluable": bool(window["evaluable"]),
            "reason": str(window["reason"]),
            "changeover_count": int(window["unit_count"]),
            "median_changeover_seconds": (
                float(records["duration_seconds"].median()) if len(records) else None
            ),
            "lost_hours": float(records["duration_seconds"].sum() / 3600.0),
            "top_segment": None if top is None else str(top["segment"]),
            "top_segment_potential_seconds": (
                None if top is None else float(top["total_potential_seconds"])
            ),
        }


class StartupProfileEvents(_UnitProfiler):
    """Restarts after stops and changeovers, and what holds them back.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=400, seed=3, stop_causes=3, stop_probability=0.06,
        ...     rampup_cycles=4)
        >>> startups = ts_shape.StartupProfileEvents(frame)
        >>> bool(startups.evaluation_window()["evaluable"].iloc[0])
        True
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        ramp_cycles: int = 3,
        band_quantiles: tuple[float, float] = (0.1, 0.9),
        min_startups: int = 5,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:startup_profile",
        **cycle_options: Any,
    ) -> None:
        """Initialize the start-up profiler.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            ramp_cycles: A start-up ends with this many consecutive normal
                cycles (robust z at most :data:`RAMP_MAX_Z`).
            band_quantiles: Quantiles of a signal's per-cycle mean in steady
                running that bound its running band.
            min_startups: Fewer start-ups are not evaluable.
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the start-up records.
            **cycle_options: Options of the cycle profile built when
                ``cycle_profile`` is None.
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
        self.ramp_cycles = int(ramp_cycles)
        self.band_quantiles = (float(band_quantiles[0]), float(band_quantiles[1]))
        self.min_startups = int(min_startups)

    def _startups(self) -> pd.DataFrame:
        if "startups" in self._cache:
            return self._cache["startups"]
        cycles = self._units().reset_index(drop=True)
        rows = []
        if len(cycles):
            classes = cycles["cycle_class"].to_numpy(dtype=object)
            z = cycles["z"].to_numpy(dtype=float)
            variant = cycles["variant"].astype(str).to_numpy()
            limits = self._profile().variant_limits()
            for i in range(len(cycles) - 1):
                changeover = i > 0 and variant[i] != variant[i - 1]
                if classes[i] != "pause" and not changeover:
                    continue
                first = _normal_runs(classes, z, i + 1, self.ramp_cycles)
                if first < 0:
                    continue
                inside = cycles.iloc[i + 1 : first]
                median = limits.get(variant[i + 1], {}).get("median", np.nan)
                lost = float((inside["duration_seconds"] - median).clip(lower=0).sum())
                rows.append(
                    {
                        "unit_uuid": f"startup-{cycles.at[i, 'cycle_uuid']}",
                        "trigger_index": i,
                        "start_ns": int(cycles.at[i, "end_ns"]),
                        "end_ns": int(cycles.at[first, "start_ns"]),
                        "trigger": "changeover" if changeover else "pause",
                        "variant": variant[i + 1],
                        "startup_cycle_count": int(first - i - 1),
                        "lost_seconds": lost,
                    }
                )
        out = pd.DataFrame(
            rows,
            columns=[
                "unit_uuid",
                "trigger_index",
                "start_ns",
                "end_ns",
                "trigger",
                "variant",
                "startup_cycle_count",
                "lost_seconds",
            ],
        )
        self._cache["startups"] = out
        return out

    def _cycle_means(self) -> pd.DataFrame:
        """Per-cycle mean of every continuous signal (cycles x signals)."""
        if "means" in self._cache:
            return self._cache["means"]
        cycles = self._units().reset_index(drop=True)
        specs = self._profile().signal_specs()
        analog = [u for u, s in specs.items() if s.kind == "continuous"]
        frame = self.dataframe
        means = pd.DataFrame(index=cycles.index)
        if len(cycles) and analog:
            starts = cycles["start_ns"].to_numpy(dtype=np.int64)
            ends = cycles["end_ns"].to_numpy(dtype=np.int64)
            for uuid in analog:
                rows = frame[frame[self.uuid_column].astype(str) == uuid]
                ns = to_ns(rows[self.time_column])
                values = pd.to_numeric(rows[specs[uuid].value_column], errors="coerce")
                pos = np.searchsorted(starts, ns, side="right") - 1
                ok = (pos >= 0) & (ns < ends[np.clip(pos, 0, None)])
                series = pd.Series(values.to_numpy()[ok]).groupby(pos[ok]).mean()
                means[uuid] = series.reindex(cycles.index)
        self._cache["means"] = means
        return means

    def _settling(self) -> pd.DataFrame:
        if "settling" in self._cache:
            return self._cache["settling"]
        startups = self._startups()
        means = self._cycle_means()
        cycles = self._units().reset_index(drop=True)
        rows = []
        if len(startups) and means.shape[1]:
            excluded = np.zeros(len(cycles), dtype=bool)
            for i in startups["trigger_index"]:
                excluded[i : i + MAX_STARTUP_CYCLES + 1] = True
            steady = (cycles["cycle_class"] == "normal").to_numpy() & ~excluded
            lo_q, hi_q = self.band_quantiles
            for uuid in means.columns:
                ref = means.loc[steady, uuid].dropna()
                if len(ref) < 10:
                    continue
                lo, hi = ref.quantile(lo_q), ref.quantile(hi_q)
                # A very steady signal still needs a band it can reach.
                margin = max(
                    0.1 * (hi - lo), BAND_MIN_RELATIVE * abs(float(ref.median()))
                )
                lo, hi = lo - margin, hi + margin
                inside = means[uuid].between(lo, hi).to_numpy()
                for unit in startups.itertuples(index=False):
                    first = unit.trigger_index + 1
                    last = min(first + MAX_STARTUP_CYCLES, len(cycles))
                    settle = None
                    for k in range(first, last - self.ramp_cycles + 1):
                        if inside[k : k + self.ramp_cycles].all():
                            settle = k
                            break
                    seconds = (
                        (int(cycles.at[settle, "start_ns"]) - unit.start_ns) / 1e9
                        if settle is not None
                        else np.nan
                    )
                    rows.append(
                        {
                            "unit_uuid": unit.unit_uuid,
                            "signal": uuid,
                            "settle_seconds": seconds,
                            "band_low": float(lo),
                            "band_high": float(hi),
                        }
                    )
        out = pd.DataFrame(
            rows,
            columns=["unit_uuid", "signal", "settle_seconds", "band_low", "band_high"],
        )
        self._cache["settling"] = out
        return out

    def startup_records(self) -> pd.DataFrame:
        """One record per start-up.

        Returns:
            Point events at the start-up end: ``systime``, ``uuid``,
            ``source_uuid``, ``unit_uuid``, ``start_ns``, ``end_ns``,
            ``duration_seconds``, ``trigger`` (``pause`` / ``changeover``),
            ``variant``, ``startup_cycle_count``, ``lost_seconds`` (cycle
            time beyond the variant median) and ``last_signal`` (the analog
            signal that was last back in its running band).
        """
        startups = self._startups()
        if startups.empty:
            return pd.DataFrame(columns=_STARTUP_RECORDS)
        settling = self._settling()
        last = {}
        if len(settling):
            valid = settling.dropna(subset=["settle_seconds"])
            for unit, part in valid.groupby("unit_uuid"):
                last[unit] = str(part.loc[part["settle_seconds"].idxmax(), "signal"])
        anchor = self._profile().cycle_anchor()
        out = pd.DataFrame(
            {
                "systime": self._times(startups["end_ns"].to_numpy()),
                "uuid": self.event_uuid,
                "source_uuid": None if anchor is None else anchor[1],
                "unit_uuid": startups["unit_uuid"].to_numpy(),
                "start_ns": startups["start_ns"].to_numpy(dtype=np.int64),
                "end_ns": startups["end_ns"].to_numpy(dtype=np.int64),
                "duration_seconds": (
                    startups["end_ns"] - startups["start_ns"]
                ).to_numpy()
                / 1e9,
                "trigger": startups["trigger"].to_numpy(),
                "variant": startups["variant"].to_numpy(),
                "startup_cycle_count": startups["startup_cycle_count"].to_numpy(),
                "lost_seconds": startups["lost_seconds"].to_numpy(dtype=float),
                "last_signal": [last.get(u) for u in startups["unit_uuid"]],
            }
        )
        return out[_STARTUP_RECORDS]

    def startup_signals(self) -> pd.DataFrame:
        """Which signals hold the restart back.

        Returns:
            One row per analog signal: ``signal``, ``band_low`` /
            ``band_high`` (running band), ``startup_count`` (start-ups in
            which it settled), ``median_settle_seconds``,
            ``p90_settle_seconds``, ``last_to_band_ratio`` (share of
            start-ups in which it was the last signal back in its band) and
            ``never_settled_count``; slowest first.
        """
        settling = self._settling()
        columns = [
            "signal",
            "band_low",
            "band_high",
            "startup_count",
            "median_settle_seconds",
            "p90_settle_seconds",
            "last_to_band_ratio",
            "never_settled_count",
        ]
        if settling.empty:
            return pd.DataFrame(columns=columns)
        records = self.startup_records()
        n = max(len(records), 1)
        rows = []
        for signal, part in settling.groupby("signal"):
            valid = part["settle_seconds"].dropna()
            rows.append(
                {
                    "signal": signal,
                    "band_low": float(part["band_low"].iloc[0]),
                    "band_high": float(part["band_high"].iloc[0]),
                    "startup_count": int(len(valid)),
                    "median_settle_seconds": (
                        float(valid.median()) if len(valid) else np.nan
                    ),
                    "p90_settle_seconds": (
                        float(valid.quantile(0.9)) if len(valid) else np.nan
                    ),
                    "last_to_band_ratio": float(
                        (records["last_signal"] == signal).sum()
                    )
                    / n,
                    "never_settled_count": int(part["settle_seconds"].isna().sum()),
                }
            )
        out = pd.DataFrame(rows, columns=columns)
        return out.sort_values(
            "median_settle_seconds", ascending=False, kind="stable", na_position="last"
        ).reset_index(drop=True)

    def startup_register(self) -> pd.DataFrame:
        """Lost time of start-ups per trigger, ranked by potential.

        Returns:
            :func:`~ts_shape.events.production._unit_profile.potential_register`
            of the start-up durations per ``trigger``, plus ``lost_seconds``
            (cycle time beyond the median, summed).
        """
        startups = self._startups()
        table = startups.assign(
            seconds=(startups["end_ns"] - startups["start_ns"]) / 1e9
        )
        register = potential_register(table, keys=["trigger"])
        lost = startups.groupby("trigger")["lost_seconds"].sum()
        register["lost_seconds"] = register["trigger"].map(lost).astype(float)
        return register

    def evaluation_window(self) -> pd.DataFrame:
        """Whether the start-up profile can be trusted, and why not.

        Returns:
            One row: ``scope``, ``unit_count`` (start-ups), ``evaluable``,
            ``reason``, ``time_resolution_seconds`` and
            ``analog_signal_count``.
        """
        profile = self._profile()
        resolution = profile.resolution_seconds()
        startups = self._startups()
        if self._units().empty:
            evaluable, reason = (
                False,
                f"upstream_not_evaluable: {profile.evaluation_reason()}",
            )
        else:
            durations = (startups["end_ns"] - startups["start_ns"]) / 1e9
            evaluable, reason = evaluation_reason(
                len(startups),
                self.min_startups,
                resolution=resolution,
                median_seconds=float(np.median(durations)) if len(startups) else np.nan,
            )
        return evaluation_frame(
            [
                {
                    "scope": "all",
                    "unit_count": int(len(startups)),
                    "evaluable": evaluable,
                    "reason": reason,
                    "time_resolution_seconds": resolution,
                    "analog_signal_count": int(self._cycle_means().shape[1]),
                }
            ],
            ["analog_signal_count"],
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``startup_count``,
            ``median_startup_cycles``, ``lost_hours`` and ``slowest_signal``.
        """
        window = self.evaluation_window().iloc[0]
        records = self.startup_records()
        signals = self.startup_signals()
        return {
            "evaluable": bool(window["evaluable"]),
            "reason": str(window["reason"]),
            "startup_count": int(window["unit_count"]),
            "median_startup_cycles": (
                float(records["startup_cycle_count"].median()) if len(records) else None
            ),
            "lost_hours": float(records["lost_seconds"].sum() / 3600.0),
            "slowest_signal": (
                None if signals.empty else str(signals["signal"].iloc[0])
            ),
        }
