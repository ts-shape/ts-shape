"""Derive per-cycle durations from whatever a machine actually publishes.

Cycle time is the metric most often computed from the wrong signal. Machines
expose it in at least six different ways, and only the first is unambiguous:

* the measured cycle time as a value -- in seconds, milliseconds, or minutes,
  and the tag name rarely says which,
* nothing at all, so it must come from the gap between counter increments,
* a single trigger whose edges bracket the cycle,
* separate start and end triggers,
* a part-present flag whose true-duration is the in-process time,
* only a run state and a part count, giving an average over each production run.

:class:`CycleTimeNormalizer` implements all six behind one interface and, just
as importantly, refuses the cycles that are artefacts rather than measurements:

* The **break problem.** Edge- and counter-derived cycles measure the gap
  between parts, so the gap across a lunch break, a shift change or a
  three-hour breakdown arrives as one enormous "cycle". Left in, a handful of
  them dominate every mean. ``max_seconds`` and ``gate_states`` remove them, and
  :meth:`rejected` shows what was removed and why -- never a silent drop.
* The **multi-cavity problem.** A cycle producing eight parts is not an
  eight-times-slower machine. ``cavities`` (or the counter increment itself)
  divides the measured cycle into a comparable per-part time, while the raw
  figure stays available as ``cycle_time_raw_seconds``.
* The **unit problem.** ``TimeUnit.AUTO`` infers seconds vs milliseconds from
  magnitude, which is what makes one fleet-wide report possible when half the
  controllers publish each.

Example::

    from ts_shape.events.production.signals import (
        CycleMode, CycleSpec, CycleTimeNormalizer, MachineState, TimeUnit,
    )

    spec = CycleSpec(
        mode=CycleMode.EXPLICIT,
        uuid="m1.cycle_time",
        unit=TimeUnit.MILLISECONDS,
        cavities=4,
        min_seconds=1.0,
        max_seconds=300.0,
        gate_states=(MachineState.DOWN, MachineState.PLANNED_STOP),
    )
    cycles = CycleTimeNormalizer(df, spec, machine_id="M1", status=status_normalizer)
    cycles.cycles()                       # one row per cycle
    cycles.cycle_statistics()             # count / mean / median / p95
    cycles.deviations(target=12.0)        # cycles off nameplate
    cycles.rejected()                     # what was filtered, and why
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
    finalize_summary_df,
)
from ts_shape.events.production.signals._intervals import collapse_runs
from ts_shape.events.production.signals.counters import ProductionCounterNormalizer
from ts_shape.events.production.signals.spec import (
    CounterKind,
    CounterMode,
    CounterSpec,
    CycleMode,
    CycleSpec,
    MachineState,
    TimeUnit,
)
from ts_shape.events.production.signals.values import (
    coerce_bool,
    coerce_numeric,
    signal_frame,
)
from ts_shape.utils.base import Base

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ts_shape.events.production.signals.status import MachineStatusNormalizer

logger = logging.getLogger(__name__)

__all__ = ["CycleTimeNormalizer"]

#: Plausible band, in seconds, used by :attr:`TimeUnit.AUTO` when the spec gives
#: no explicit bounds. Half a second to an hour covers press lines through heat
#: treatment while still separating seconds from milliseconds.
DEFAULT_PLAUSIBLE_BAND: tuple[float, float] = (0.5, 3600.0)

_CYCLE_EXTRAS: tuple[str, ...] = (
    "machine_id",
    "cycle_time_seconds",
    "cycle_time_raw_seconds",
    "parts_per_cycle",
    "mode",
)

_REJECT_EXTRAS: tuple[str, ...] = (
    "machine_id",
    "cycle_time_seconds",
    "parts_per_cycle",
    "mode",
    "reason",
)


class CycleTimeNormalizer(Base):
    """Turn any cycle-time source into canonical per-cycle records.

    Args:
        dataframe: Timeseries in the standard ts-shape schema.
        spec: How to derive cycle time for this machine.
        time_column: Timestamp column.
        machine_id: Asset identifier broadcast onto every output row.
        event_uuid: ``uuid`` value stamped on emitted events.
        uuid_column: Column holding signal identifiers.
        counter_events: Canonical count events from
            :meth:`~ts_shape.events.production.signals.counters.ProductionCounterNormalizer.count_events`,
            needed by ``COUNTER_DELTA`` and ``STATE_INTERVAL``. When omitted, a
            counter is read straight from ``spec.uuid`` as a cumulative
            totalizer -- pass the events instead whenever the machine's counter
            is anything else, so the cycle count matches the production count.
        status: A configured
            :class:`~ts_shape.events.production.signals.status.MachineStatusNormalizer`.
            Needed for ``STATE_INTERVAL`` mode and for ``gate_states``.
        ideal_cycle_seconds: Nameplate per-part cycle time, used as the default
            target by :meth:`deviations` and :meth:`cycle_statistics`.

    Raises:
        ValueError: If the mode needs an input that was not supplied -- for
            example ``STATE_INTERVAL`` without a ``status`` normalizer.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        spec: CycleSpec,
        *,
        time_column: str = "systime",
        machine_id: str | None = None,
        event_uuid: str = "prod:cycle",
        uuid_column: str = "uuid",
        counter_events: pd.DataFrame | None = None,
        status: MachineStatusNormalizer | None = None,
        ideal_cycle_seconds: float | None = None,
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.spec = spec
        self.time_column = time_column
        self.machine_id = machine_id
        self.event_uuid = event_uuid
        self.uuid_column = uuid_column
        self.counter_events = counter_events
        self.status = status
        self.ideal_cycle_seconds = ideal_cycle_seconds
        if spec.mode is CycleMode.STATE_INTERVAL:
            if status is None:
                raise ValueError(
                    "CycleMode.STATE_INTERVAL derives cycle time from producing "
                    "intervals, so a MachineStatusNormalizer must be passed as "
                    "status=."
                )
            if counter_events is None and not spec.uuid:
                raise ValueError(
                    "CycleMode.STATE_INTERVAL divides producing time by the parts "
                    "produced, so it needs a part count: set CycleSpec.uuid to the "
                    "counter signal, or pass counter_events= from "
                    "ProductionCounterNormalizer.count_events()."
                )
        if spec.gate_states and status is None:
            raise ValueError(
                f"CycleSpec.gate_states={[s.value for s in spec.gate_states]} needs a "
                "MachineStatusNormalizer to know when those states applied; pass "
                "status=."
            )
        self._resolved_unit: TimeUnit | None = None
        self._candidates: pd.DataFrame | None = None
        self._rejected: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _signal(self, uuid: str) -> pd.DataFrame:
        return signal_frame(
            self.dataframe,
            uuid,
            time_column=self.time_column,
            value_column=self.spec.value_column,
            uuid_column=self.uuid_column,
        )

    def _counts(self) -> pd.DataFrame:
        """Canonical count events, supplied or read from ``spec.uuid``."""
        if self.counter_events is not None:
            events = self.counter_events
            if events.empty:
                return events
            keep = events["quantity"] > 0
            return events[keep].reset_index(drop=True)
        if not self.spec.uuid:
            return pd.DataFrame(columns=["systime", "quantity"])
        normalizer = ProductionCounterNormalizer(
            self.dataframe,
            counters=[
                CounterSpec(
                    uuid=self.spec.uuid,
                    kind=CounterKind.TOTAL,
                    mode=CounterMode.CUMULATIVE,
                    value_column=self.spec.value_column,
                )
            ],
            time_column=self.time_column,
            machine_id=self.machine_id,
            uuid_column=self.uuid_column,
        )
        events = normalizer.count_events()
        if events.empty:
            return events
        return events[events["quantity"] > 0].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Unit resolution
    # ------------------------------------------------------------------
    def _band(self) -> tuple[float, float]:
        low, high = DEFAULT_PLAUSIBLE_BAND
        if self.spec.min_seconds is not None:
            low = self.spec.min_seconds
        if self.spec.max_seconds is not None:
            high = self.spec.max_seconds
        return low, high

    def resolved_unit(self) -> TimeUnit:
        """Return the unit used for an explicit cycle-time signal.

        With :attr:`~ts_shape.events.production.signals.spec.TimeUnit.AUTO` the
        unit is inferred from the median magnitude: the candidate that lands the
        median inside the plausible band wins, preferring seconds, then
        milliseconds. A signal whose median is 12000 is milliseconds; one whose
        median is 12 is seconds. Ties are impossible because the bands differ by
        three orders of magnitude.

        Falls back to seconds -- and logs -- when nothing fits, so an odd signal
        degrades to an explicit assumption rather than to empty output.
        """
        if self.spec.unit is not TimeUnit.AUTO:
            return self.spec.unit
        if self._resolved_unit is not None:
            return self._resolved_unit

        frame = self._signal(self.spec.uuid or "")
        values = (
            coerce_numeric(frame["value"]).dropna()
            if not frame.empty
            else pd.Series(dtype="float64")
        )
        positive = values[values > 0]
        if positive.empty:
            self._resolved_unit = TimeUnit.SECONDS
            return self._resolved_unit

        # The plausibility band is stated per part, and so is the reported cycle
        # on a multi-cavity machine only after dividing by the cavities.
        median = float(positive.median()) / float(self.spec.cavities)
        low, high = self._band()
        for candidate in (
            TimeUnit.SECONDS,
            TimeUnit.MILLISECONDS,
            TimeUnit.MINUTES,
            TimeUnit.MICROSECONDS,
            TimeUnit.HOURS,
        ):
            if low <= median * candidate.seconds_factor <= high:
                self._resolved_unit = candidate
                logger.info(
                    "Inferred unit %s for cycle-time signal %s (median raw value %g).",
                    candidate.value,
                    self.spec.uuid,
                    median,
                )
                return candidate
        logger.warning(
            "Could not infer a unit for cycle-time signal %s: median raw value %g "
            "fits no unit within the plausible band %g-%g s. Assuming seconds; set "
            "CycleSpec.unit explicitly.",
            self.spec.uuid,
            median,
            low,
            high,
        )
        self._resolved_unit = TimeUnit.SECONDS
        return self._resolved_unit

    # ------------------------------------------------------------------
    # Candidate derivation, one method per mode
    # ------------------------------------------------------------------
    def _empty_candidates(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "start": pd.Series(dtype="datetime64[ns]"),
                "end": pd.Series(dtype="datetime64[ns]"),
                "cycle_time_raw_seconds": pd.Series(dtype="float64"),
                "parts_per_cycle": pd.Series(dtype="float64"),
                "source_uuid": pd.Series(dtype="object"),
            }
        )

    def _from_explicit(self) -> pd.DataFrame:
        """A published cycle time; the cycle ends when the value is reported."""
        frame = self._signal(self.spec.uuid or "")
        if frame.empty:
            return self._empty_candidates()
        seconds = coerce_numeric(frame["value"]) * self.resolved_unit().seconds_factor
        end = pd.to_datetime(frame[self.time_column])
        return pd.DataFrame(
            {
                "start": end - pd.to_timedelta(seconds.fillna(0.0), unit="s"),
                "end": end,
                "cycle_time_raw_seconds": seconds.to_numpy(),
                "parts_per_cycle": float(self.spec.cavities),
                "source_uuid": self.spec.uuid,
            }
        )

    def _from_counter_delta(self) -> pd.DataFrame:
        """Time between counter increments, divided by the parts in each."""
        counts = self._counts()
        if len(counts) < 2:
            return self._empty_candidates()
        times = pd.to_datetime(counts["systime"]).reset_index(drop=True)
        quantity = counts["quantity"].astype("float64").reset_index(drop=True)
        raw = (times - times.shift()).dt.total_seconds()
        return pd.DataFrame(
            {
                "start": times.shift().iloc[1:].to_numpy(),
                "end": times.iloc[1:].to_numpy(),
                "cycle_time_raw_seconds": raw.iloc[1:].to_numpy(),
                # The parts booked at the end of the gap are the parts the gap
                # produced, so a burst of 8 is one cycle of 8, not 8 cycles.
                "parts_per_cycle": quantity.iloc[1:].to_numpy(),
                "source_uuid": (
                    counts["source_uuid"].iloc[1:].to_numpy()
                    if "source_uuid" in counts.columns
                    else self.spec.uuid
                ),
            }
        )

    def _trigger_mask(self, values: pd.Series) -> pd.Series:
        """Boolean mask of the readings that count as trigger events.

        ``edge="any"`` takes every true reading rather than detecting a
        transition -- the correct reading for a historian that logs one row per
        pulse and never the intervening false, where edge detection would see a
        single unending pulse and find no cycles at all.
        """
        flags = coerce_bool(values)
        if self.spec.edge == "any":
            return flags
        previous = flags.shift(fill_value=False)
        rising = flags & ~previous
        falling = ~flags & previous
        if self.spec.edge == "rising":
            return rising
        if self.spec.edge == "falling":
            return falling
        return rising | falling

    def _from_trigger_edge(self) -> pd.DataFrame:
        """Time between consecutive edges of a single trigger."""
        frame = self._signal(self.spec.uuid or "")
        if frame.empty:
            return self._empty_candidates()
        mask = self._trigger_mask(frame["value"])
        times = pd.to_datetime(frame.loc[mask, self.time_column]).reset_index(drop=True)
        if len(times) < 2:
            return self._empty_candidates()
        raw = (times - times.shift()).dt.total_seconds()
        return pd.DataFrame(
            {
                "start": times.shift().iloc[1:].to_numpy(),
                "end": times.iloc[1:].to_numpy(),
                "cycle_time_raw_seconds": raw.iloc[1:].to_numpy(),
                "parts_per_cycle": float(self.spec.cavities),
                "source_uuid": self.spec.uuid,
            }
        )

    def _from_trigger_pair(self) -> pd.DataFrame:
        """Start signal to the next end signal after it."""
        starts = self._signal(self.spec.uuid or "")
        ends = self._signal(self.spec.end_uuid or "")
        if starts.empty or ends.empty:
            return self._empty_candidates()

        start_edges = pd.to_datetime(
            starts.loc[self._trigger_mask(starts["value"]), self.time_column]
        ).reset_index(drop=True)
        end_edges = pd.to_datetime(
            ends.loc[self._trigger_mask(ends["value"]), self.time_column]
        ).reset_index(drop=True)
        if start_edges.empty or end_edges.empty:
            return self._empty_candidates()

        matched = pd.merge_asof(
            pd.DataFrame({"start": start_edges.astype("datetime64[ns]")}),
            pd.DataFrame({"end": end_edges.astype("datetime64[ns]")}).assign(
                _key=end_edges.astype("datetime64[ns]")
            ),
            left_on="start",
            right_on="_key",
            direction="forward",
        ).drop(columns=["_key"])
        matched = matched.dropna(subset=["end"])
        # Several starts can point at the same end when a start pulse repeats;
        # the last one before that end is the cycle that actually ran.
        matched = matched.drop_duplicates(subset=["end"], keep="last")
        if matched.empty:
            return self._empty_candidates()
        return pd.DataFrame(
            {
                "start": matched["start"].to_numpy(),
                "end": matched["end"].to_numpy(),
                "cycle_time_raw_seconds": (matched["end"] - matched["start"])
                .dt.total_seconds()
                .to_numpy(),
                "parts_per_cycle": float(self.spec.cavities),
                "source_uuid": f"{self.spec.uuid},{self.spec.end_uuid}",
            }
        )

    def _from_part_present(self) -> pd.DataFrame:
        """The time a part-present flag stays true."""
        frame = self._signal(self.spec.uuid or "")
        if frame.empty:
            return self._empty_candidates()
        states = pd.DataFrame(
            {
                self.time_column: pd.to_datetime(frame[self.time_column]).to_numpy(),
                "present": coerce_bool(frame["value"]).to_numpy(),
            }
        )
        runs = collapse_runs(
            states, value_column="present", time_column=self.time_column
        )
        present = runs[runs["present"].astype(bool)]
        if present.empty:
            return self._empty_candidates()
        return pd.DataFrame(
            {
                "start": present["start"].to_numpy(),
                "end": present["end"].to_numpy(),
                "cycle_time_raw_seconds": present["duration_seconds"].to_numpy(),
                "parts_per_cycle": float(self.spec.cavities),
                "source_uuid": self.spec.uuid,
            }
        )

    def _from_state_interval(self) -> pd.DataFrame:
        """Producing time per run divided by the parts that run produced.

        The fallback for machines with only a run signal and a counter. Each
        record is the *average* cycle over one production run, so treat the
        spread as run-to-run variation rather than cycle-to-cycle.
        """
        assert self.status is not None  # guaranteed in __init__
        intervals = self.status.status_intervals()
        counts = self._counts()
        if intervals.empty or counts.empty:
            return self._empty_candidates()
        producing = intervals[intervals["state"] == MachineState.PRODUCING.value]
        if producing.empty:
            return self._empty_candidates()

        times = pd.to_datetime(counts["systime"]).astype("datetime64[ns]").to_numpy()
        quantity = counts["quantity"].astype("float64").to_numpy()
        cumulative = np.concatenate([[0.0], np.cumsum(quantity)])
        starts = pd.to_datetime(producing["start"]).astype("datetime64[ns]").to_numpy()
        ends = pd.to_datetime(producing["end"]).astype("datetime64[ns]").to_numpy()
        lo = np.searchsorted(times, starts, side="left")
        hi = np.searchsorted(times, ends, side="right")
        parts = cumulative[hi] - cumulative[lo]

        out = pd.DataFrame(
            {
                "start": starts,
                "end": ends,
                "cycle_time_raw_seconds": producing["duration_seconds"].to_numpy(),
                "parts_per_cycle": parts,
                "source_uuid": producing["source_uuid"].to_numpy(),
            }
        )
        # A producing run that booked no parts says nothing about cycle time.
        return out[out["parts_per_cycle"] > 0].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _gated_overlap(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Boolean mask of cycles overlapping any gated machine state.

        Exact overlap test, not an endpoint check: a cycle whose start and end
        both sit in producing time can still contain a two-minute fault, and
        that cycle is not a measurement of the machine's speed.
        """
        assert self.status is not None
        intervals = self.status.status_intervals()
        if intervals.empty:
            return np.zeros(len(starts), dtype=bool)
        wanted = {s.value for s in self.spec.gate_states}
        gated = intervals[intervals["state"].isin(wanted)]
        if gated.empty:
            return np.zeros(len(starts), dtype=bool)

        gated = gated.sort_values("start")
        gate_starts = pd.to_datetime(gated["start"]).astype("datetime64[ns]").to_numpy()
        gate_ends = pd.to_datetime(gated["end"]).astype("datetime64[ns]").to_numpy()
        # Running maximum of gated end times lets one searchsorted answer
        # "does any gated interval that starts before this cycle ends reach
        # past its start?".
        reach = np.maximum.accumulate(gate_ends)
        idx = np.searchsorted(gate_starts, ends, side="left")
        overlap = np.zeros(len(starts), dtype=bool)
        found = idx > 0
        overlap[found] = reach[idx[found] - 1] > starts[found]
        return overlap

    def _build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Derive, filter, and split candidates into kept and rejected (cached)."""
        if self._candidates is not None and self._rejected is not None:
            return self._candidates, self._rejected

        builders = {
            CycleMode.EXPLICIT: self._from_explicit,
            CycleMode.COUNTER_DELTA: self._from_counter_delta,
            CycleMode.TRIGGER_EDGE: self._from_trigger_edge,
            CycleMode.TRIGGER_PAIR: self._from_trigger_pair,
            CycleMode.PART_PRESENT: self._from_part_present,
            CycleMode.STATE_INTERVAL: self._from_state_interval,
        }
        candidates = builders[self.spec.mode]().reset_index(drop=True)
        if candidates.empty:
            self._candidates = candidates
            self._rejected = candidates.assign(reason=pd.Series(dtype="object"))
            return self._candidates, self._rejected

        candidates["start"] = pd.to_datetime(candidates["start"])
        candidates["end"] = pd.to_datetime(candidates["end"])
        candidates["parts_per_cycle"] = candidates["parts_per_cycle"].astype("float64")
        candidates["cycle_time_seconds"] = (
            candidates["cycle_time_raw_seconds"] / candidates["parts_per_cycle"]
        )
        candidates["mode"] = self.spec.mode.value
        candidates["machine_id"] = self.machine_id

        reason = pd.Series(pd.NA, index=candidates.index, dtype="object")
        per_part = candidates["cycle_time_seconds"]
        invalid = per_part.isna() | (per_part <= 0)
        reason = reason.mask(invalid & reason.isna(), "non_positive")
        if self.spec.min_seconds is not None:
            reason = reason.mask(
                (per_part < self.spec.min_seconds) & reason.isna(), "too_short"
            )
        if self.spec.max_seconds is not None:
            reason = reason.mask(
                (per_part > self.spec.max_seconds) & reason.isna(), "too_long"
            )
        if self.spec.gate_states:
            overlap = self._gated_overlap(
                candidates["start"].astype("datetime64[ns]").to_numpy(),
                candidates["end"].astype("datetime64[ns]").to_numpy(),
            )
            reason = reason.mask(
                pd.Series(overlap, index=candidates.index) & reason.isna(),
                "gated_state",
            )

        candidates["reason"] = reason
        kept = candidates[reason.isna()].drop(columns=["reason"]).reset_index(drop=True)
        rejected = candidates[reason.notna()].reset_index(drop=True)
        self._candidates = kept
        self._rejected = rejected
        return kept, rejected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cycles(self) -> pd.DataFrame:
        """Return one canonical record per accepted cycle.

        Returns:
            Interval-shaped frame with ``start``, ``end``, ``duration_seconds``,
            ``uuid``, ``source_uuid``, ``machine_id``, ``cycle_time_seconds``
            (per part -- the comparable figure), ``cycle_time_raw_seconds`` (as
            measured, before dividing by cavities), ``parts_per_cycle`` and
            ``mode``.
        """
        kept, _ = self._build()
        if kept.empty:
            return empty_event_df("interval", _CYCLE_EXTRAS)
        out = finalize_interval_df(kept, uuid=self.event_uuid, source_uuid=None)
        return out.reset_index(drop=True)

    def rejected(self) -> pd.DataFrame:
        """Return the candidate cycles that were filtered out, with the reason.

        Never look at a cycle-time average without looking here. A machine whose
        rejected count rivals its accepted count is not slow -- it is stopping,
        and the mean of what is left describes only the good runs.

        Returns:
            Interval-shaped frame as :meth:`cycles`, plus ``reason``:
            ``non_positive`` (zero or negative -- a clock or ordering problem),
            ``too_short`` (below ``min_seconds``, typically sensor bounce),
            ``too_long`` (above ``max_seconds`` -- a break or stoppage caught
            between two parts), ``gated_state`` (overlapped an excluded machine
            state).
        """
        _, rejected = self._build()
        if rejected.empty:
            return empty_event_df("interval", _REJECT_EXTRAS)
        out = finalize_interval_df(
            rejected, uuid=f"{self.event_uuid}:rejected", source_uuid=None
        )
        return out.reset_index(drop=True)

    def cycle_statistics(
        self,
        *,
        group_by: Sequence[str] = (),
        target: float | None = None,
        events: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Summarise accepted cycle times, optionally per group.

        Args:
            group_by: Columns to group by -- typically context columns attached
                by
                :class:`~ts_shape.events.production.signals.context.ProductionContextResolver`,
                giving cycle time per part number, order or tool.
            target: Per-part target cycle time. Defaults to
                ``ideal_cycle_seconds``.
            events: Pre-built cycle records carrying the ``group_by`` columns;
                defaults to :meth:`cycles`.

        Returns:
            Summary-shaped frame with ``start``, ``end``, ``duration_seconds``,
            the ``group_by`` columns, ``cycles``, ``parts``, ``mean_seconds``,
            ``median_seconds``, ``std_seconds``, ``min_seconds``,
            ``max_seconds``, ``p05_seconds``, ``p95_seconds``,
            ``total_seconds`` and ``machine_id``. When a target is known it also
            carries ``target_seconds``, ``mean_deviation_pct`` and
            ``within_target_pct``.

        Raises:
            ValueError: If a ``group_by`` column is not present on the records.
        """
        source = self.cycles() if events is None else events
        keys = list(group_by)
        extras = (
            *keys,
            "cycles",
            "parts",
            "mean_seconds",
            "median_seconds",
            "std_seconds",
            "min_seconds",
            "max_seconds",
            "p05_seconds",
            "p95_seconds",
            "total_seconds",
            "machine_id",
        )
        if source.empty:
            return empty_event_df("summary", extras)
        missing = [k for k in keys if k not in source.columns]
        if missing:
            raise ValueError(
                f"group_by column(s) {missing} are not present on the cycle records. "
                "Attach them first, e.g. "
                "ProductionContextResolver.attach(normalizer.cycles())."
            )

        aggregated = self._aggregate(source, keys)
        aggregated["machine_id"] = self.machine_id
        effective_target = target if target is not None else self.ideal_cycle_seconds
        if effective_target is not None and effective_target > 0:
            aggregated["target_seconds"] = effective_target
            aggregated["mean_deviation_pct"] = (
                (aggregated["mean_seconds"] - effective_target)
                / effective_target
                * 100.0
            )
            within = (
                source.assign(_ok=source["cycle_time_seconds"] <= effective_target)
                .groupby(keys, dropna=False, observed=True)["_ok"]
                .mean()
                * 100.0
                if keys
                else pd.Series(
                    [(source["cycle_time_seconds"] <= effective_target).mean() * 100.0]
                )
            )
            aggregated["within_target_pct"] = (
                within.reset_index(drop=True) if keys else within.to_numpy()
            )
        out = finalize_summary_df(aggregated, uuid=self.event_uuid, source_uuid=None)
        return out.reset_index(drop=True)

    def by_window(self, freq: str = "1h") -> pd.DataFrame:
        """Summarise cycle times per fixed window.

        A cycle is attributed to the window containing its **end**, matching how
        the parts it produced are counted, so cycle counts reconcile with
        production counts.

        Args:
            freq: Pandas offset alias for the window.

        Returns:
            Summary-shaped frame as :meth:`cycle_statistics`, one row per window.
        """
        source = self.cycles()
        if source.empty:
            return empty_event_df(
                "summary",
                (
                    "cycles",
                    "parts",
                    "mean_seconds",
                    "median_seconds",
                    "std_seconds",
                    "min_seconds",
                    "max_seconds",
                    "p05_seconds",
                    "p95_seconds",
                    "total_seconds",
                    "machine_id",
                ),
            )
        work = source.copy()
        work["_window"] = pd.to_datetime(work["end"]).dt.floor(freq)
        aggregated = self._aggregate(work, ["_window"])
        aggregated["start"] = aggregated.pop("_window")
        aggregated["end"] = aggregated["start"] + pd.tseries.frequencies.to_offset(freq)
        aggregated["machine_id"] = self.machine_id
        out = finalize_summary_df(aggregated, uuid=self.event_uuid, source_uuid=None)
        return out.reset_index(drop=True)

    def deviations(
        self,
        *,
        target: float | None = None,
        tolerance_pct: float = 10.0,
    ) -> pd.DataFrame:
        """Return cycles that deviate from target by more than ``tolerance_pct``.

        Args:
            target: Per-part target cycle time. Defaults to
                ``ideal_cycle_seconds``.
            tolerance_pct: Allowed deviation in percent, either direction.

        Returns:
            Point-shaped frame (timestamped at the cycle end) with
            ``machine_id``, ``cycle_time_seconds``, ``target_seconds``,
            ``deviation_seconds``, ``deviation_pct``, ``direction``
            (``slow`` / ``fast``) and ``severity``. A ``fast`` cycle matters
            too: it usually means a part was skipped or a sensor missed one.

        Raises:
            ValueError: If no target is available.
        """
        effective = target if target is not None else self.ideal_cycle_seconds
        if effective is None or effective <= 0:
            raise ValueError(
                "deviations() needs a positive target cycle time. Pass target= or "
                "set MachineSpec.ideal_cycle_seconds / "
                "CycleTimeNormalizer(ideal_cycle_seconds=...)."
            )
        extras = (
            "machine_id",
            "cycle_time_seconds",
            "target_seconds",
            "deviation_seconds",
            "deviation_pct",
            "direction",
            "severity",
        )
        source = self.cycles()
        if source.empty:
            return empty_event_df("point", extras)

        deviation = source["cycle_time_seconds"] - effective
        deviation_pct = deviation / effective * 100.0
        flagged = deviation_pct.abs() > tolerance_pct
        if not flagged.any():
            return empty_event_df("point", extras)

        out = pd.DataFrame(
            {
                "systime": source.loc[flagged, "end"].to_numpy(),
                "source_uuid": source.loc[flagged, "source_uuid"].to_numpy(),
                "machine_id": self.machine_id,
                "cycle_time_seconds": source.loc[
                    flagged, "cycle_time_seconds"
                ].to_numpy(),
                "target_seconds": effective,
                "deviation_seconds": deviation[flagged].to_numpy(),
                "deviation_pct": deviation_pct[flagged].to_numpy(),
            }
        )
        out["direction"] = np.where(out["deviation_seconds"] > 0, "slow", "fast")
        out["severity"] = np.where(
            out["deviation_pct"].abs() > 2 * tolerance_pct, "warning", "info"
        )
        out = finalize_point_df(
            out, uuid=f"{self.event_uuid}:deviation", source_uuid=None
        )
        return out.reindex(
            columns=["systime", "uuid", "source_uuid", *extras]
        ).reset_index(drop=True)

    def diagnostics(self) -> pd.DataFrame:
        """Return one health row for this machine's cycle-time derivation.

        Returns:
            Static-shaped single-row frame with ``machine_id``, ``mode``,
            ``unit`` (as resolved), ``cavities``, ``candidates``, ``accepted``,
            ``rejected``, a count per rejection reason, ``mean_seconds`` and
            ``median_seconds``.
        """
        kept, rejected = self._build()
        reasons = (
            rejected["reason"].value_counts().to_dict() if not rejected.empty else {}
        )
        row: dict[str, Any] = {
            "machine_id": self.machine_id,
            "mode": self.spec.mode.value,
            "unit": (
                self.resolved_unit().value
                if self.spec.mode is CycleMode.EXPLICIT
                else TimeUnit.SECONDS.value
            ),
            "cavities": self.spec.cavities,
            "candidates": int(len(kept) + len(rejected)),
            "accepted": int(len(kept)),
            "rejected": int(len(rejected)),
            "rejected_non_positive": int(reasons.get("non_positive", 0)),
            "rejected_too_short": int(reasons.get("too_short", 0)),
            "rejected_too_long": int(reasons.get("too_long", 0)),
            "rejected_gated_state": int(reasons.get("gated_state", 0)),
            "mean_seconds": (
                float(kept["cycle_time_seconds"].mean()) if not kept.empty else np.nan
            ),
            "median_seconds": (
                float(kept["cycle_time_seconds"].median()) if not kept.empty else np.nan
            ),
        }
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _aggregate(self, source: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        """Compute the standard cycle-time statistics, grouped or overall."""

        def _stats(group: pd.DataFrame) -> dict[str, Any]:
            values = group["cycle_time_seconds"]
            return {
                "cycles": int(len(values)),
                "parts": float(group["parts_per_cycle"].sum()),
                "mean_seconds": float(values.mean()),
                "median_seconds": float(values.median()),
                "std_seconds": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min_seconds": float(values.min()),
                "max_seconds": float(values.max()),
                "p05_seconds": float(values.quantile(0.05)),
                "p95_seconds": float(values.quantile(0.95)),
                "total_seconds": float(group["cycle_time_raw_seconds"].sum()),
                "start": group["start"].min(),
                "end": group["end"].max(),
            }

        if not keys:
            return pd.DataFrame([_stats(source)])
        rows = [
            {
                **dict(
                    zip(keys, key if isinstance(key, tuple) else (key,), strict=True)
                ),
                **_stats(group),
            }
            for key, group in source.groupby(
                keys, dropna=False, observed=True, sort=True
            )
        ]
        return pd.DataFrame(rows)
