"""Fleet-scale production reporting over hundreds of machines.

One machine's production report is a scripting exercise. Several hundred is a
different problem, and the hard parts are not the metrics:

* **Every machine encodes its signals differently**, so the report must be
  written against canonical quantities, never against tags. That is what the
  normalizers in this subpackage provide, and what this module composes.
* **The configuration is the deliverable.** Hundreds of assets cannot be mapped
  in code; they are described as data in a
  :class:`~ts_shape.events.production.signals.spec.FleetSpec` and reviewed like
  data.
* **Machines are unequally instrumented.** One has OK and NOK counters, a
  PackML state and a serial number; the next has a single pulse and nothing
  else. A fleet report must produce a *stable schema* regardless, marking what
  is unknown as unknown rather than zero -- otherwise the poorly instrumented
  machines quietly look like the best performers.
* **Mis-mapped tags are the normal state of the world.** At this scale a report
  is only as good as its coverage diagnostics, so
  :meth:`~FleetProductionReport.signal_coverage` and
  :meth:`~FleetProductionReport.anomalies` are first-class outputs rather than
  debugging aids.

Example::

    from ts_shape.events.production.signals import FleetProductionReport, FleetSpec

    # One row per signal, maintained by the people who know the machines.
    fleet = FleetSpec.from_records(tag_table.to_dict("records"))

    report = FleetProductionReport(fleet, df)

    report.production_by_window("1h")                     # machine x hour
    report.production_by_context(["order_number"])        # machine x order
    report.oee("1D")                                      # A x P x Q per day
    report.machine_summary()                              # one row per machine
    report.fleet_summary(group_by=["plant", "line"])      # roll-up
    report.signal_coverage()                              # what is mis-mapped
    report.anomalies()                                    # resets, unmapped codes

Memory: with hundreds of machines the timeseries rarely fits in one frame. Pass
``data_provider=`` instead of ``dataframe=`` and each machine's data is loaded,
reduced and released one at a time::

    report = FleetProductionReport(
        fleet,
        data_provider=lambda spec: loader.load(uuids=spec.signal_uuids),
    )
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.production.signals.context import ProductionContextResolver
from ts_shape.events.production.signals.counters import (
    QUANTITY_COLUMNS,
    ProductionCounterNormalizer,
)
from ts_shape.events.production.signals.cycles import CycleTimeNormalizer
from ts_shape.events.production.signals.spec import (
    CycleMode,
    FleetSpec,
    MachineSpec,
    MachineState,
)
from ts_shape.events.production.signals.status import MachineStatusNormalizer
from ts_shape.errors import DataQualityWarning

logger = logging.getLogger(__name__)

__all__ = ["FleetProductionReport", "MachineNormalizers"]


class MachineNormalizers:
    """The configured normalizers for one machine, built lazily.

    Returned by :meth:`FleetProductionReport.normalizers` so a machine that
    looks wrong in the fleet report can be drilled into with the same objects
    the report itself used -- no re-configuration, no risk of analysing
    something different from what was reported.

    Attributes:
        spec: The machine's configuration.
        counters: Counter normalizer, or ``None`` if no counters are configured.
        status: Status normalizer, or ``None``.
        cycles: Cycle-time normalizer, or ``None``.
        context: Context resolver, or ``None``.
    """

    __slots__ = ("context", "counters", "cycles", "spec", "status")

    def __init__(
        self,
        spec: MachineSpec,
        *,
        counters: ProductionCounterNormalizer | None,
        status: MachineStatusNormalizer | None,
        cycles: CycleTimeNormalizer | None,
        context: ProductionContextResolver | None,
    ) -> None:
        self.spec = spec
        self.counters = counters
        self.status = status
        self.cycles = cycles
        self.context = context

    def __repr__(self) -> str:
        configured = [
            name
            for name in ("counters", "status", "cycles", "context")
            if getattr(self, name) is not None
        ]
        return f"MachineNormalizers({self.spec.machine_id!r}, configured={configured})"


class FleetProductionReport:
    """Compose canonical production reporting across a fleet of machines.

    Args:
        fleet: The fleet configuration, or a bare sequence of
            :class:`~ts_shape.events.production.signals.spec.MachineSpec`.
        dataframe: One frame holding every machine's timeseries. Partitioned by
            signal uuid exactly once, not re-scanned per machine.
        data_provider: Called with each :class:`MachineSpec` to fetch that
            machine's data on demand. Use it instead of ``dataframe`` when the
            fleet's data does not fit in memory. Results are not cached, so each
            machine's frame is released after use.
        time_column: Timestamp column.
        uuid_column: Column holding signal identifiers.
        until: Closing timestamp for open state intervals -- normally the end of
            the reporting window. Without it, the last interval of each machine
            closes at that machine's last reading.
        assume_all_ok: Applied to machines that count a total but measure no
            quality at all. Off by default; see
            :meth:`~ts_shape.events.production.signals.counters.ProductionCounterNormalizer.counts_by_window`.

    Raises:
        ValueError: If neither ``dataframe`` nor ``data_provider`` is given, if
            both are, or if the fleet is empty.
    """

    def __init__(
        self,
        fleet: FleetSpec | Sequence[MachineSpec],
        dataframe: pd.DataFrame | None = None,
        *,
        data_provider: Callable[[MachineSpec], pd.DataFrame] | None = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        until: pd.Timestamp | str | None = None,
        assume_all_ok: bool = False,
    ) -> None:
        self.fleet = fleet if isinstance(fleet, FleetSpec) else FleetSpec(tuple(fleet))
        if not self.fleet.machines:
            raise ValueError(
                "FleetProductionReport needs at least one MachineSpec to report on."
            )
        if dataframe is None and data_provider is None:
            raise ValueError(
                "Provide either dataframe= (one frame for the whole fleet) or "
                "data_provider= (a callable fetching each machine's data)."
            )
        if dataframe is not None and data_provider is not None:
            raise ValueError(
                "Provide dataframe= or data_provider=, not both -- otherwise it is "
                "ambiguous which one a machine's numbers came from."
            )
        self.dataframe = dataframe
        self.data_provider = data_provider
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.until = pd.Timestamp(until) if until is not None else None
        self.assume_all_ok = assume_all_ok
        # Validate the mapping eagerly: a uuid on two machines double-counts
        # production, and finding that out from a report is expensive.
        self._uuid_index = self.fleet.uuid_index()
        self._partitions: dict[str, pd.DataFrame] | None = None
        self._normalizers: dict[str, MachineNormalizers] = {}

    def __repr__(self) -> str:
        source = "data_provider" if self.data_provider is not None else "dataframe"
        return (
            f"FleetProductionReport(machines={len(self.fleet)}, source={source}, "
            f"attributes={list(self.fleet.attribute_keys)})"
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def _partition(self) -> dict[str, pd.DataFrame]:
        """Split the fleet frame into one frame per machine, once."""
        if self._partitions is not None:
            return self._partitions
        assert self.dataframe is not None
        frame = self.dataframe
        if frame.empty or self.uuid_column not in frame.columns:
            self._partitions = {}
            return self._partitions
        # A single vectorised map beats one boolean scan per machine, which is
        # the difference between seconds and minutes at fleet scale.
        owner = frame[self.uuid_column].map(self._uuid_index)
        known = frame[owner.notna()]
        self._partitions = {
            str(machine_id): group.drop(columns=["_machine_id"], errors="ignore")
            for machine_id, group in known.assign(
                _machine_id=owner[owner.notna()]
            ).groupby("_machine_id", sort=False)
        }
        # Count the signals actually present but unclaimed, not the difference
        # between the two totals -- a configured-but-absent tag would otherwise
        # cancel out an unconfigured one and the log would understate both.
        unmapped = set(frame[self.uuid_column].dropna().unique()) - set(
            self._uuid_index
        )
        if unmapped:
            logger.info(
                "%d signal(s) in the input are not configured on any machine and are "
                "ignored (e.g. %s). FleetSpec.signal_uuids lists what the report "
                "reads.",
                len(unmapped),
                ", ".join(sorted(str(u) for u in unmapped)[:5]),
            )
        return self._partitions

    def _frame_for(self, spec: MachineSpec) -> pd.DataFrame:
        if self.data_provider is not None:
            return self.data_provider(spec)
        return self._partition().get(
            spec.machine_id,
            pd.DataFrame(columns=[self.time_column, self.uuid_column]),
        )

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------
    def normalizers(self, machine_id: str) -> MachineNormalizers:
        """Return the configured normalizers for one machine.

        Cached when the report owns a fleet frame, so drilling into a machine
        after a fleet run costs nothing. With a ``data_provider`` the data is
        re-fetched, since the point of that mode is not to hold it.

        Args:
            machine_id: The machine to build normalizers for.

        Returns:
            A :class:`MachineNormalizers` bundle.

        Raises:
            KeyError: If the machine is not in the fleet.
        """
        if machine_id in self._normalizers:
            return self._normalizers[machine_id]
        spec = self.fleet.machine(machine_id)
        frame = self._frame_for(spec)
        with warnings.catch_warnings():
            # A per-machine slice necessarily has empty value columns for the
            # datatypes that machine does not use, so this warning fires for
            # nearly every machine and says nothing actionable. Across hundreds
            # of machines it would bury the warnings that do matter -- notably
            # the duplicate-timestamp one, which is left enabled.
            warnings.filterwarnings(
                "ignore",
                message=r".*contain only NaN values.*",
                category=DataQualityWarning,
            )
            return self._build_normalizers(spec, frame)

    def _build_normalizers(
        self, spec: MachineSpec, frame: pd.DataFrame
    ) -> MachineNormalizers:
        """Construct the normalizer bundle for one machine's data."""
        counters = (
            ProductionCounterNormalizer(
                frame,
                counters=spec.counters,
                time_column=self.time_column,
                machine_id=spec.machine_id,
                uuid_column=self.uuid_column,
            )
            if spec.counters
            else None
        )
        status = (
            MachineStatusNormalizer(
                frame,
                spec.status,
                time_column=self.time_column,
                machine_id=spec.machine_id,
                uuid_column=self.uuid_column,
                until=self.until,
            )
            if spec.status is not None
            else None
        )
        context = (
            ProductionContextResolver(
                frame,
                specs=spec.context,
                time_column=self.time_column,
                machine_id=spec.machine_id,
                uuid_column=self.uuid_column,
            )
            if spec.context
            else None
        )
        cycles = None
        if spec.cycle is not None:
            needs_status = spec.cycle.mode is CycleMode.STATE_INTERVAL or bool(
                spec.cycle.gate_states
            )
            if needs_status and status is None:
                logger.warning(
                    "Machine %s configures cycle mode %s (or gate_states) but has no "
                    "status signal; cycle time is skipped for it.",
                    spec.machine_id,
                    spec.cycle.mode.value,
                )
            else:
                cycles = CycleTimeNormalizer(
                    frame,
                    spec.cycle,
                    time_column=self.time_column,
                    machine_id=spec.machine_id,
                    uuid_column=self.uuid_column,
                    counter_events=(
                        counters.count_events() if counters is not None else None
                    ),
                    status=status,
                    ideal_cycle_seconds=spec.ideal_cycle_seconds,
                )
        bundle = MachineNormalizers(
            spec, counters=counters, status=status, cycles=cycles, context=context
        )
        if self.data_provider is None:
            self._normalizers[spec.machine_id] = bundle
        return bundle

    def _selected(self, machines: Iterable[str] | None) -> tuple[MachineSpec, ...]:
        if machines is None:
            return self.fleet.machines
        wanted = list(machines)
        return tuple(self.fleet.machine(m) for m in wanted)

    def _attributes(self, spec: MachineSpec) -> dict[str, Any]:
        """Attribute values for one machine, keyed over the fleet-wide union.

        Every machine gets every key so the concatenated report has one stable
        schema and can be grouped by any attribute without checking which
        machines happen to define it.
        """
        return {key: spec.attributes.get(key) for key in self.fleet.attribute_keys}

    def _stamp(self, frame: pd.DataFrame, spec: MachineSpec) -> pd.DataFrame:
        out = frame.copy()
        out["machine_id"] = spec.machine_id
        for key, value in self._attributes(spec).items():
            out[key] = value
        return out

    def _count_events(
        self, bundle: MachineNormalizers, dimensions: Sequence[str]
    ) -> pd.DataFrame:
        """Count events with the requested context dimensions guaranteed present."""
        assert bundle.counters is not None
        events = bundle.counters.count_events(include_zero=True)
        wanted = list(dimensions)
        if not wanted:
            return events
        resolver = bundle.context
        available = (
            [d for d in wanted if d in resolver.dimensions]
            if resolver is not None
            else []
        )
        if resolver is not None and available:
            events = resolver.attach(events, dimensions=available)
        # A machine that does not report a dimension still needs the column, or
        # the fleet concat would produce a ragged frame.
        for dimension in wanted:
            if dimension not in events.columns:
                events[dimension] = pd.NA
        return events

    @staticmethod
    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        usable = [p for p in parts if not p.empty]
        if not usable:
            return pd.concat(parts) if parts else pd.DataFrame()
        return pd.concat(usable, ignore_index=True)

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------
    def production_by_window(
        self,
        freq: str = "1h",
        *,
        dimensions: Sequence[str] = (),
        machines: Iterable[str] | None = None,
        drop_empty: bool = True,
    ) -> pd.DataFrame:
        """Return canonical production quantities per machine and window.

        The tidy fact table a fleet dashboard is built on: one row per machine,
        window and (optionally) context combination, with the same columns for
        every machine regardless of how it counts.

        Args:
            freq: Window size as a pandas offset alias.
            dimensions: Context dimensions to break down by. A machine that does
                not report one still appears, with that column missing.
            machines: Restrict to these machine ids; defaults to the whole fleet.
            drop_empty: Drop windows with no production.

        Returns:
            Summary-shaped frame with ``start``, ``end``, ``duration_seconds``,
            ``machine_id``, the fleet's attribute columns, the requested
            ``dimensions``, ``qty_ok``, ``qty_nok``, ``qty_total``,
            ``qty_rework``, ``qty_scrap``, ``good_rate_pct``, ``nok_rate_pct``
            and ``nok_ppm``.
        """
        parts: list[pd.DataFrame] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            if bundle.counters is None:
                continue
            if dimensions:
                events = self._count_events(bundle, dimensions)
                frame = bundle.counters.counts_by_context(
                    list(dimensions),
                    freq=freq,
                    events=events,
                    assume_all_ok=self.assume_all_ok,
                )
            else:
                frame = bundle.counters.counts_by_window(
                    freq,
                    assume_all_ok=self.assume_all_ok,
                    drop_empty=drop_empty,
                )
            if frame.empty:
                continue
            parts.append(self._stamp(frame, spec))
        return self._concat(parts)

    def production_by_context(
        self,
        dimensions: Sequence[str],
        *,
        freq: str | None = None,
        machines: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return production quantities per machine and context combination.

        This is the answer to "how many good parts of customer part 4711 did we
        make last week, and on which machines" -- across the fleet, in one frame.

        Args:
            dimensions: Context dimensions to group by, e.g.
                ``["order_number", "part_number"]``.
            freq: Optional window inside each context group.
            machines: Restrict to these machine ids.

        Returns:
            Summary-shaped frame as :meth:`production_by_window`.

        Raises:
            ValueError: If ``dimensions`` is empty.
        """
        if not dimensions:
            raise ValueError(
                "production_by_context requires at least one dimension; use "
                "production_by_window for a plain time series."
            )
        parts: list[pd.DataFrame] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            if bundle.counters is None:
                continue
            events = self._count_events(bundle, dimensions)
            frame = bundle.counters.counts_by_context(
                list(dimensions),
                freq=freq,
                events=events,
                assume_all_ok=self.assume_all_ok,
            )
            if frame.empty:
                continue
            parts.append(self._stamp(frame, spec))
        return self._concat(parts)

    # ------------------------------------------------------------------
    # State and cycle time
    # ------------------------------------------------------------------
    def status_by_window(
        self,
        freq: str | None = "1h",
        *,
        machines: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return per-state seconds and availability per machine and window.

        Args:
            freq: Window size; ``None`` for one row per machine.
            machines: Restrict to these machine ids.

        Returns:
            Summary-shaped frame with the ``<state>_seconds`` columns,
            ``observed_seconds``, ``planned_busy_seconds``, ``availability_pct``,
            ``utilization_pct``, ``machine_id`` and the attribute columns.
        """
        parts: list[pd.DataFrame] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            if bundle.status is None:
                continue
            frame = bundle.status.state_summary(freq)
            if frame.empty:
                continue
            parts.append(self._stamp(frame, spec))
        return self._concat(parts)

    def status_intervals(
        self, *, machines: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Return canonical state intervals for the fleet, one frame.

        Args:
            machines: Restrict to these machine ids.

        Returns:
            Interval-shaped frame as
            :meth:`~ts_shape.events.production.signals.status.MachineStatusNormalizer.status_intervals`,
            with ``machine_id`` and the attribute columns.
        """
        parts: list[pd.DataFrame] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            if bundle.status is None:
                continue
            frame = bundle.status.status_intervals()
            if frame.empty:
                continue
            parts.append(self._stamp(frame, spec))
        return self._concat(parts)

    def cycle_by_window(
        self,
        freq: str = "1h",
        *,
        machines: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return cycle-time statistics per machine and window.

        Args:
            freq: Window size as a pandas offset alias.
            machines: Restrict to these machine ids.

        Returns:
            Summary-shaped frame with ``cycles``, ``parts``, ``mean_seconds``,
            ``median_seconds``, ``p05_seconds``, ``p95_seconds``,
            ``machine_id`` and the attribute columns.
        """
        parts: list[pd.DataFrame] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            if bundle.cycles is None:
                continue
            frame = bundle.cycles.by_window(freq)
            if frame.empty:
                continue
            parts.append(self._stamp(frame, spec))
        return self._concat(parts)

    def cycle_by_context(
        self,
        dimensions: Sequence[str],
        *,
        machines: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return cycle-time statistics per machine and context combination.

        Cycle time per part number is the figure that tells you whether a part is
        genuinely slower or a machine simply ran it on a bad day.

        Args:
            dimensions: Context dimensions to group by.
            machines: Restrict to these machine ids.

        Returns:
            Summary-shaped frame as :meth:`cycle_by_window`, grouped by
            ``dimensions`` instead of time.
        """
        parts: list[pd.DataFrame] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            if bundle.cycles is None:
                continue
            records = bundle.cycles.cycles()
            if records.empty:
                continue
            resolver = bundle.context
            available = (
                [d for d in dimensions if d in resolver.dimensions]
                if resolver is not None
                else []
            )
            if resolver is not None and available:
                # Bind context at the cycle's *end*, the instant the part was
                # finished and counted. Binding at the start would attribute a
                # cycle that began before the first context reading -- or across
                # a changeover -- to the wrong order.
                records = resolver.attach(
                    records, dimensions=available, time_column="end"
                )
            for dimension in dimensions:
                if dimension not in records.columns:
                    records[dimension] = pd.NA
            frame = bundle.cycles.cycle_statistics(
                group_by=tuple(dimensions), events=records
            )
            if frame.empty:
                continue
            parts.append(self._stamp(frame, spec))
        return self._concat(parts)

    # ------------------------------------------------------------------
    # OEE
    # ------------------------------------------------------------------
    def oee(
        self,
        freq: str = "1D",
        *,
        machines: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return the OEE factors per machine and window.

        Computed the ISO 22400 way from canonical inputs:

        * ``availability_pct`` -- producing time / planned busy time, from the
          status intervals. Planned stops and off time are excluded from the
          denominator; ``unknown`` (no data) time is excluded from both, so an
          outage lowers confidence rather than inventing availability.
        * ``performance_pct`` -- ``ideal_cycle_seconds x parts / producing
          seconds``, where *parts* is the total produced (falling back to
          good + bad when no total is measured). Needs
          ``MachineSpec.ideal_cycle_seconds``; ``NaN`` without it, never
          assumed. A figure above 100% is meaningful information: the machine
          beat its nameplate cycle, so the nameplate is wrong.
        * ``quality_pct`` -- good parts / total parts. ``NaN`` when the machine
          measures no quality.
        * ``oee_pct`` -- the product of the three, ``NaN`` if any factor is
          unknown.

        A ``NaN`` factor is the point, not a gap: an OEE number assembled from
        guessed inputs is worse than no number, because it gets compared against
        machines whose inputs were real.

        Args:
            freq: Window size as a pandas offset alias.
            machines: Restrict to these machine ids.

        Returns:
            Summary-shaped frame with ``start``, ``end``, ``duration_seconds``,
            ``machine_id``, the attribute columns, ``qty_ok``, ``qty_nok``,
            ``qty_total``, ``producing_seconds``, ``planned_busy_seconds``,
            ``ideal_cycle_seconds``, ``availability_pct``, ``performance_pct``,
            ``quality_pct`` and ``oee_pct``.
        """
        selected = self._selected(machines)
        production = self.production_by_window(
            freq, machines=[s.machine_id for s in selected], drop_empty=False
        )
        status = self.status_by_window(freq, machines=[s.machine_id for s in selected])

        keys = ["machine_id", "start"]
        production_columns = keys + ["end", "duration_seconds", *QUANTITY_COLUMNS]
        status_columns = keys + [
            f"{MachineState.PRODUCING.value}_seconds",
            "planned_busy_seconds",
            "availability_pct",
        ]
        left = (
            production[[c for c in production_columns if c in production.columns]]
            if not production.empty
            else pd.DataFrame(columns=production_columns)
        )
        right = (
            status[[c for c in status_columns if c in status.columns]]
            if not status.empty
            else pd.DataFrame(columns=status_columns)
        )
        merged = left.merge(right, on=keys, how="outer")
        if merged.empty:
            return merged

        merged = merged.rename(
            columns={f"{MachineState.PRODUCING.value}_seconds": "producing_seconds"}
        )
        ideal = {s.machine_id: s.ideal_cycle_seconds for s in selected}
        merged["ideal_cycle_seconds"] = merged["machine_id"].map(ideal)

        # Performance needs a count of everything the machine made, good or not.
        produced = merged["qty_total"]
        if "qty_ok" in merged.columns and "qty_nok" in merged.columns:
            produced = produced.fillna(
                merged["qty_ok"].fillna(0.0) + merged["qty_nok"].fillna(0.0)
            )
        for column in ("producing_seconds", "planned_busy_seconds", "availability_pct"):
            if column not in merged.columns:
                # No machine in this selection reports a status signal, so there
                # is no run time to rate the output against.
                merged[column] = np.nan
        producing = merged["producing_seconds"]
        with np.errstate(invalid="ignore", divide="ignore"):
            merged["performance_pct"] = np.where(
                merged["ideal_cycle_seconds"].notna()
                & producing.notna()
                & (producing > 0),
                merged["ideal_cycle_seconds"] * produced / producing * 100.0,
                np.nan,
            )
            merged["quality_pct"] = np.where(
                merged["qty_total"].notna() & (merged["qty_total"] > 0),
                merged["qty_ok"] / merged["qty_total"] * 100.0,
                np.nan,
            )
        merged["oee_pct"] = (
            merged["availability_pct"]
            * merged["performance_pct"]
            * merged["quality_pct"]
            / 10000.0
        )
        # Restore the attributes the merge dropped, and the window end for rows
        # that only exist on the status side.
        attributes = {s.machine_id: self._attributes(s) for s in selected}
        for key in self.fleet.attribute_keys:
            merged[key] = merged["machine_id"].map(
                {mid: attrs.get(key) for mid, attrs in attributes.items()}
            )
        offset = pd.tseries.frequencies.to_offset(freq)
        merged["start"] = pd.to_datetime(merged["start"])
        merged["end"] = pd.to_datetime(merged.get("end")).fillna(
            merged["start"] + offset
        )
        merged["duration_seconds"] = (
            merged["end"] - merged["start"]
        ).dt.total_seconds()

        ordered = [
            "start",
            "end",
            "duration_seconds",
            "machine_id",
            *self.fleet.attribute_keys,
            *QUANTITY_COLUMNS,
            "producing_seconds",
            "planned_busy_seconds",
            "ideal_cycle_seconds",
            "availability_pct",
            "performance_pct",
            "quality_pct",
            "oee_pct",
        ]
        present = [c for c in ordered if c in merged.columns]
        return (
            merged[present].sort_values(["machine_id", "start"]).reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Roll-ups
    # ------------------------------------------------------------------
    def machine_summary(self, *, machines: Iterable[str] | None = None) -> pd.DataFrame:
        """Return one row per machine for the whole loaded range.

        The fleet league table, and the first thing to look at: it carries both
        the performance figures and the instrumentation facts next to each other,
        so a machine at the bottom can be read as "slow" or as "badly mapped"
        without opening a second report.

        Args:
            machines: Restrict to these machine ids.

        Returns:
            Static-shaped frame with ``machine_id``, the attribute columns,
            ``qty_ok``, ``qty_nok``, ``qty_total``, ``good_rate_pct``,
            ``nok_ppm``, ``producing_seconds``, ``planned_busy_seconds``,
            ``availability_pct``, ``cycles``, ``mean_cycle_seconds``,
            ``ideal_cycle_seconds``, ``counter_anomalies``,
            ``unmapped_state_values``, ``context_dimensions``, ``first_seen``
            and ``last_seen``.
        """
        rows: list[dict[str, Any]] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            row: dict[str, Any] = {
                "machine_id": spec.machine_id,
                **self._attributes(spec),
            }
            for column in (*QUANTITY_COLUMNS, "good_rate_pct", "nok_ppm"):
                row[column] = np.nan
            row["counter_anomalies"] = 0
            if bundle.counters is not None:
                totals = bundle.counters.totals(assume_all_ok=self.assume_all_ok)
                if not totals.empty:
                    for column in (*QUANTITY_COLUMNS, "good_rate_pct", "nok_ppm"):
                        if column in totals.columns:
                            row[column] = float(totals[column].iloc[0])
                row["counter_anomalies"] = int(len(bundle.counters.anomalies()))

            row.update(
                {
                    "producing_seconds": np.nan,
                    "planned_busy_seconds": np.nan,
                    "availability_pct": np.nan,
                    "unmapped_state_values": 0,
                }
            )
            if bundle.status is not None:
                summary = bundle.status.state_summary()
                if not summary.empty:
                    row["producing_seconds"] = float(
                        summary[f"{MachineState.PRODUCING.value}_seconds"].iloc[0]
                    )
                    row["planned_busy_seconds"] = float(
                        summary["planned_busy_seconds"].iloc[0]
                    )
                    row["availability_pct"] = float(summary["availability_pct"].iloc[0])
                row["unmapped_state_values"] = int(len(bundle.status.unmapped_values()))

            row.update({"cycles": 0, "mean_cycle_seconds": np.nan})
            if bundle.cycles is not None:
                diagnostics = bundle.cycles.diagnostics().iloc[0]
                row["cycles"] = int(diagnostics["accepted"])
                row["mean_cycle_seconds"] = float(diagnostics["mean_seconds"])
            row["ideal_cycle_seconds"] = spec.ideal_cycle_seconds
            row["context_dimensions"] = ",".join(spec.context_names)

            frame = self._frame_for(spec)
            times = (
                pd.to_datetime(frame[self.time_column], errors="coerce")
                if not frame.empty and self.time_column in frame.columns
                else pd.Series(dtype="datetime64[ns]")
            )
            row["first_seen"] = times.min() if len(times) else pd.NaT
            row["last_seen"] = times.max() if len(times) else pd.NaT
            rows.append(row)
        return pd.DataFrame(rows)

    def fleet_summary(
        self,
        *,
        group_by: Sequence[str] = (),
        machines: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Roll the machine summary up by asset attributes.

        Rates are recomputed from the summed quantities, never averaged: the mean
        of per-machine quality percentages is not the quality of the group, and
        the difference grows with how unequal the machines' output is.

        The quality rate is computed only over the machines that measure both
        good parts *and* a total, and ``quality_machines`` reports how many that
        was. Summing across machines with different measured kinds -- one that
        counts only good parts, one that counts only a total -- yields a ratio of
        unrelated quantities, which can exceed 100%.

        Args:
            group_by: Attribute columns to group by -- e.g. ``["plant"]`` or
                ``["plant", "line"]``. Empty gives one row for the whole fleet.
            machines: Restrict to these machine ids.

        Returns:
            Static-shaped frame with the ``group_by`` columns, ``machines``,
            ``qty_ok``, ``qty_nok``, ``qty_total``, ``good_rate_pct``,
            ``nok_ppm``, ``quality_machines``, ``producing_seconds``,
            ``planned_busy_seconds``, ``availability_pct`` and ``cycles``.

        Raises:
            ValueError: If a ``group_by`` column is not a configured attribute.
        """
        summary = self.machine_summary(machines=machines)
        keys = list(group_by)
        if summary.empty:
            return pd.DataFrame(
                columns=[*keys, "machines", *QUANTITY_COLUMNS, "good_rate_pct"]
            )
        missing = [k for k in keys if k not in summary.columns]
        if missing:
            raise ValueError(
                f"group_by column(s) {missing} are not machine attributes. "
                f"Available: {list(self.fleet.attribute_keys)}"
            )

        sums = [
            *QUANTITY_COLUMNS,
            "producing_seconds",
            "planned_busy_seconds",
            "cycles",
        ]
        # min_count=1 keeps an all-unknown column unknown. A plain sum would
        # report 0 good parts for a group whose machines measure no quality, and
        # a 0% quality rate reads as a catastrophe rather than as "not measured".
        if keys:
            grouped = summary.groupby(keys, dropna=False, observed=True)
            aggregated = grouped[sums].sum(min_count=1).reset_index()
            aggregated["machines"] = grouped["machine_id"].nunique().to_numpy()
        else:
            aggregated = pd.DataFrame([{c: summary[c].sum(min_count=1) for c in sums}])
            aggregated["machines"] = summary["machine_id"].nunique()

        # A quality rate must be computed over machines that measure *both*
        # numerator and denominator. Dividing a group's summed good parts by its
        # summed total mixes machines that measure only good parts with machines
        # that measure only a total, and can exceed 100%. Restrict the ratio to
        # the machines where the comparison is defined, and say how many those
        # were, so a rate covering half the group cannot be mistaken for all of it.
        comparable = summary[summary["qty_ok"].notna() & summary["qty_total"].notna()]
        basis_columns = ["qty_ok", "qty_nok", "qty_total"]
        if keys:
            if comparable.empty:
                basis = pd.DataFrame(
                    columns=[*keys, *basis_columns, "quality_machines"]
                )
            else:
                basis_group = comparable.groupby(keys, dropna=False, observed=True)
                basis = basis_group[basis_columns].sum(min_count=1).reset_index()
                basis["quality_machines"] = (
                    basis_group["machine_id"].nunique().to_numpy()
                )
            aggregated = aggregated.merge(
                basis, on=keys, how="left", suffixes=("", "_basis")
            )
        else:
            for column in basis_columns:
                aggregated[f"{column}_basis"] = (
                    comparable[column].sum(min_count=1)
                    if not comparable.empty
                    else np.nan
                )
            aggregated["quality_machines"] = comparable["machine_id"].nunique()
        for column in basis_columns:
            if f"{column}_basis" not in aggregated.columns:
                aggregated[f"{column}_basis"] = np.nan
        aggregated["quality_machines"] = (
            aggregated["quality_machines"].fillna(0).astype("int64")
        )

        basis_total = aggregated["qty_total_basis"]
        with np.errstate(invalid="ignore", divide="ignore"):
            aggregated["good_rate_pct"] = np.where(
                basis_total > 0,
                aggregated["qty_ok_basis"] / basis_total * 100.0,
                np.nan,
            )
            aggregated["nok_ppm"] = np.where(
                basis_total > 0,
                aggregated["qty_nok_basis"] / basis_total * 1e6,
                np.nan,
            )
            aggregated["availability_pct"] = np.where(
                aggregated["planned_busy_seconds"] > 0,
                aggregated["producing_seconds"]
                / aggregated["planned_busy_seconds"]
                * 100.0,
                np.nan,
            )
        ordered = [
            *keys,
            "machines",
            *QUANTITY_COLUMNS,
            "good_rate_pct",
            "nok_ppm",
            "quality_machines",
            "producing_seconds",
            "planned_busy_seconds",
            "availability_pct",
            "cycles",
        ]
        return aggregated[[c for c in ordered if c in aggregated.columns]]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def signal_coverage(self, *, machines: Iterable[str] | None = None) -> pd.DataFrame:
        """Return one row per configured signal saying whether data arrived.

        The commissioning report for a fleet rollout and the daily health check
        afterwards. ``readings == 0`` is the single most common defect at this
        scale -- a renamed tag, a gateway that was never started, a machine
        moved to another cell -- and it is invisible in a production report,
        where the machine simply shows no output.

        Args:
            machines: Restrict to these machine ids.

        Returns:
            Static-shaped frame with ``machine_id``, the attribute columns,
            ``role`` (``counter_ok`` / ``status`` / ``cycle`` / ``context:
            order_number`` ...), ``uuid``, ``present``, ``readings``,
            ``value_column``, ``first_seen`` and ``last_seen``.
        """
        rows: list[dict[str, Any]] = []
        for spec in self._selected(machines):
            frame = self._frame_for(spec)
            has_uuids = not frame.empty and self.uuid_column in frame.columns
            counts = (
                frame[self.uuid_column].value_counts()
                if has_uuids
                else pd.Series(dtype="int64")
            )
            times = (
                frame.assign(
                    _t=pd.to_datetime(frame[self.time_column], errors="coerce")
                )
                .groupby(self.uuid_column, observed=True)["_t"]
                .agg(["min", "max"])
                if has_uuids and self.time_column in frame.columns
                else pd.DataFrame(columns=["min", "max"])
            )

            roles: list[tuple[str, str, str | None]] = []
            for counter in spec.counters:
                roles.append(
                    (
                        f"counter_{counter.kind.value}",
                        counter.uuid,
                        counter.value_column,
                    )
                )
                if counter.trigger_uuid:
                    roles.append(
                        (
                            f"counter_{counter.kind.value}:trigger",
                            counter.trigger_uuid,
                            counter.value_column,
                        )
                    )
            if spec.status is not None:
                for uuid in spec.status.signal_uuids:
                    roles.append(("status", uuid, spec.status.value_column))
            if spec.cycle is not None:
                for uuid in spec.cycle.signal_uuids:
                    roles.append(("cycle", uuid, spec.cycle.value_column))
            for dimension in spec.context:
                roles.append(
                    (
                        f"context:{dimension.name}",
                        dimension.uuid,
                        dimension.value_column,
                    )
                )

            for role, uuid, value_column in roles:
                readings = int(counts.get(uuid, 0))
                rows.append(
                    {
                        "machine_id": spec.machine_id,
                        **self._attributes(spec),
                        "role": role,
                        "uuid": uuid,
                        "present": readings > 0,
                        "readings": readings,
                        "value_column": value_column,
                        "first_seen": (
                            times.loc[uuid, "min"] if uuid in times.index else pd.NaT
                        ),
                        "last_seen": (
                            times.loc[uuid, "max"] if uuid in times.index else pd.NaT
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def anomalies(self, *, machines: Iterable[str] | None = None) -> pd.DataFrame:
        """Return every data-quality finding across the fleet, one frame.

        Counter resets, register wrap-arounds, implausible jumps and status codes
        no mapping covers -- the list to work through before questioning a
        machine's performance.

        Args:
            machines: Restrict to these machine ids.

        Returns:
            Static-shaped frame with ``machine_id``, the attribute columns,
            ``source_uuid``, ``category`` (``counter`` / ``status``),
            ``anomaly``, ``severity``, ``occurrences``, ``first_seen`` and
            ``last_seen``, worst first.
        """
        parts: list[pd.DataFrame] = []
        for spec in self._selected(machines):
            bundle = self.normalizers(spec.machine_id)
            if bundle.counters is not None:
                found = bundle.counters.anomalies()
                if not found.empty:
                    grouped = (
                        found.groupby(
                            ["source_uuid", "anomaly", "severity"], observed=True
                        )
                        .agg(
                            occurrences=("systime", "size"),
                            first_seen=("systime", "min"),
                            last_seen=("systime", "max"),
                        )
                        .reset_index()
                    )
                    grouped["category"] = "counter"
                    parts.append(self._stamp(grouped, spec))
            if bundle.status is not None:
                unmapped = bundle.status.unmapped_values()
                if not unmapped.empty:
                    grouped = unmapped.rename(columns={"raw_value": "anomaly"})[
                        [
                            "source_uuid",
                            "anomaly",
                            "occurrences",
                            "first_seen",
                            "last_seen",
                        ]
                    ].copy()
                    grouped["anomaly"] = "unmapped_state:" + grouped["anomaly"].astype(
                        "string"
                    )
                    grouped["severity"] = "warning"
                    grouped["category"] = "status"
                    parts.append(self._stamp(grouped, spec))
        combined = self._concat(parts)
        if combined.empty:
            return pd.DataFrame(
                columns=[
                    "machine_id",
                    *self.fleet.attribute_keys,
                    "source_uuid",
                    "category",
                    "anomaly",
                    "severity",
                    "occurrences",
                    "first_seen",
                    "last_seen",
                ]
            )
        rank = {"error": 0, "warning": 1, "info": 2}
        combined["_rank"] = combined["severity"].map(rank).fillna(3)
        ordered = [
            "machine_id",
            *self.fleet.attribute_keys,
            "source_uuid",
            "category",
            "anomaly",
            "severity",
            "occurrences",
            "first_seen",
            "last_seen",
        ]
        return (
            combined.sort_values(["_rank", "occurrences"], ascending=[True, False])
            .drop(columns=["_rank"])[ordered]
            .reset_index(drop=True)
        )
