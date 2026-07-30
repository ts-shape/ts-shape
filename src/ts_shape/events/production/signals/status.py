"""Normalize any machine-status encoding into canonical state intervals.

"Is the machine running" is answered differently by every controller. Across a
fleet you will find, all reporting into the same dashboard:

* a single ``running`` boolean,
* an integer state code from a bespoke PLC state machine (``0`` = off,
  ``3`` = auto, ``7`` = fault, and nobody wrote down the rest),
* a string status (``"AUTO"``, ``"MANUAL"``, ``"STOERUNG"``),
* four separate boolean lamps that are sometimes on at the same time,
* a stacklight colour,
* a PackML state number.

:class:`MachineStatusNormalizer` maps all of them onto the nine
:class:`~ts_shape.events.production.signals.spec.MachineState` values and emits
intervals, so availability is computed once and compared across machines that
share nothing but the concept of running.

Three things it does that a naive implementation does not:

* **Correct interval ends.** A state holds until the *next* update, not until
  the last sample that repeated it. Ending intervals at the last repeated sample
  silently discards most of the duration on change-triggered signals.
* **Data gaps are ``UNKNOWN``, not "still running".** With ``stale_after`` set,
  a machine that stops reporting stops accumulating state time. This is the
  difference between an availability figure and a fiction: the most common cause
  of a suspiciously good machine is a dead gateway.
* **Unmapped codes are surfaced, not guessed.** Anything absent from
  ``state_map`` becomes ``UNKNOWN`` and is listed by :meth:`unmapped_values`,
  which is how you discover what a controller really emits.

Example::

    from ts_shape.events.production.signals import (
        MachineState, MachineStatusNormalizer, StatusMode, StatusSpec,
    )

    spec = StatusSpec(
        mode=StatusMode.CODE_MAP,
        uuid="m1.state",
        state_map={
            "0": MachineState.OFF, "1": MachineState.IDLE,
            "3": MachineState.PRODUCING, "5": MachineState.SETUP,
            "7": MachineState.DOWN,
        },
        min_duration="5s",      # debounce a flickering bit
        stale_after="10min",    # no data -> UNKNOWN
    )
    status = MachineStatusNormalizer(df, spec, machine_id="M1")
    status.status_intervals()
    status.state_summary("1h")     # per-state seconds + availability_pct
    status.unmapped_values()       # codes nobody mapped yet
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
    finalize_summary_df,
)
from ts_shape.events.production.signals._intervals import (
    collapse_runs,
    debounce_intervals,
    split_intervals,
)
from ts_shape.events.production.signals.spec import (
    MachineState,
    StatusMode,
    StatusSpec,
)
from ts_shape.events.production.signals.values import (
    coerce_bool,
    coerce_label,
    signal_frame,
    union_timeline,
)
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

__all__ = ["MachineStatusNormalizer", "STATE_SECONDS_COLUMNS"]

#: Per-state duration columns of :meth:`MachineStatusNormalizer.state_summary`.
#: Every state always gets a column, whether or not the machine ever reached it,
#: so summaries from different machines concatenate into one fleet frame.
STATE_SECONDS_COLUMNS: tuple[str, ...] = tuple(
    f"{state.value}_seconds" for state in MachineState
)

_INTERVAL_EXTRAS: tuple[str, ...] = (
    "machine_id",
    "state",
    "raw_value",
    "availability_class",
    "is_productive",
)


class MachineStatusNormalizer(Base):
    """Map a machine's status signal(s) onto canonical state intervals.

    Args:
        dataframe: Timeseries in the standard ts-shape schema.
        spec: How this machine encodes its status.
        time_column: Timestamp column.
        machine_id: Asset identifier broadcast onto every output row.
        event_uuid: ``uuid`` value stamped on emitted events.
        uuid_column: Column holding signal identifiers.
        until: Closing timestamp for the final, still-open interval. Defaults
            to the last timestamp anywhere in ``dataframe`` -- other signals
            still arriving is evidence the machine was observed that long.
            Pass the end of the reporting window for exact edges.

    Raises:
        ValueError: If the spec is invalid for the data (see
            :class:`~ts_shape.events.production.signals.spec.StatusSpec`).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        spec: StatusSpec,
        *,
        time_column: str = "systime",
        machine_id: str | None = None,
        event_uuid: str = "prod:machine_status",
        uuid_column: str = "uuid",
        until: pd.Timestamp | str | None = None,
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.spec = spec
        self.time_column = time_column
        self.machine_id = machine_id
        self.event_uuid = event_uuid
        self.uuid_column = uuid_column
        self.until = pd.Timestamp(until) if until is not None else None
        self._states: pd.DataFrame | None = None
        self._intervals: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Horizon
    # ------------------------------------------------------------------
    def _horizon(self) -> pd.Timestamp | None:
        if self.until is not None:
            return self.until
        if self.dataframe.empty or self.time_column not in self.dataframe.columns:
            return None
        last = pd.to_datetime(self.dataframe[self.time_column], errors="coerce").max()
        return None if pd.isna(last) else last

    # ------------------------------------------------------------------
    # Raw -> canonical state samples
    # ------------------------------------------------------------------
    def _map_values(self, raw: pd.Series) -> pd.Series:
        """Map raw state values through ``state_map``, keyed case-insensitively."""
        keys = coerce_label(raw, sentinels=()).str.strip().str.lower()
        mapping = {k: v.value for k, v in self.spec.state_map.items()}
        mapped = keys.map(mapping)
        return mapped.fillna(self.spec.default_state.value).astype("object")

    def _boolean_states(self, frame: pd.DataFrame) -> pd.DataFrame:
        flags = coerce_bool(frame["value"])
        states = np.where(
            flags, self.spec.true_state.value, self.spec.false_state.value
        )
        return pd.DataFrame(
            {
                self.time_column: frame[self.time_column].to_numpy(),
                "state": states,
                "raw_value": coerce_label(frame["value"], sentinels=()).to_numpy(),
            }
        )

    def _mapped_states(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = coerce_label(frame["value"], sentinels=())
        return pd.DataFrame(
            {
                self.time_column: frame[self.time_column].to_numpy(),
                "state": self._map_values(frame["value"]).to_numpy(),
                "raw_value": raw.to_numpy(),
            }
        )

    def _one_hot_states(self) -> pd.DataFrame:
        """Resolve several boolean flags into one state per instant.

        Machines routinely assert ``running`` and ``fault`` simultaneously --
        one bit is the drive, the other the alarm word. ``priority`` decides,
        defaulting to fault-before-production, because the alternative is
        counting faulted time as production.
        """
        uuids = list(self.spec.state_map.keys())
        frames = {
            uuid: signal_frame(
                self.dataframe,
                uuid,
                time_column=self.time_column,
                value_column=self.spec.value_column,
                uuid_column=self.uuid_column,
            )
            for uuid in uuids
        }
        timeline = union_timeline(frames, time_column=self.time_column)
        if timeline.empty:
            return self._empty_states()

        ranking = {state: rank for rank, state in enumerate(self.spec.priority)}
        # Highest-priority asserted flag wins; ties are impossible because the
        # priority order is total over the states present.
        order = sorted(
            uuids,
            key=lambda u: ranking.get(self.spec.state_map[u], len(ranking)),
        )
        chosen = pd.Series(
            self.spec.default_state.value, index=timeline.index, dtype="object"
        )
        raw = pd.Series(pd.NA, index=timeline.index, dtype="object")
        any_known = pd.Series(False, index=timeline.index)
        for uuid in reversed(order):
            asserted = coerce_bool(timeline[uuid])
            state = self.spec.state_map[uuid].value
            chosen = chosen.mask(asserted, state)
            raw = raw.mask(asserted, uuid)
            any_known |= asserted
        # No flag asserted at all is itself information: with a false_state
        # configured it means "not producing", which is more useful than
        # UNKNOWN for a machine that simply drops all its bits when idle.
        chosen = chosen.mask(~any_known, self.spec.false_state.value)
        raw = raw.mask(~any_known, pd.NA)
        return pd.DataFrame(
            {
                self.time_column: timeline[self.time_column].to_numpy(),
                "state": chosen.to_numpy(),
                "raw_value": raw.to_numpy(),
            }
        )

    def _empty_states(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.time_column: pd.Series(dtype="datetime64[ns]"),
                "state": pd.Series(dtype="object"),
                "raw_value": pd.Series(dtype="object"),
            }
        )

    def _state_samples(self) -> pd.DataFrame:
        """Canonical ``(time, state, raw_value)`` samples, cached."""
        if self._states is not None:
            return self._states

        if self.spec.mode is StatusMode.ONE_HOT:
            states = self._one_hot_states()
        else:
            assert self.spec.uuid is not None  # guaranteed by StatusSpec
            frame = signal_frame(
                self.dataframe,
                self.spec.uuid,
                time_column=self.time_column,
                value_column=self.spec.value_column,
                uuid_column=self.uuid_column,
            )
            if frame.empty:
                logger.info(
                    "Status signal %s has no readings in the loaded range.",
                    self.spec.uuid,
                )
                states = self._empty_states()
            elif self.spec.mode is StatusMode.BOOLEAN:
                states = self._boolean_states(frame)
            else:
                states = self._mapped_states(frame)

        if not states.empty and self.spec.planned_stop_values:
            planned = {v.strip().lower() for v in self.spec.planned_stop_values}
            is_planned = (
                states["raw_value"]
                .astype("string")
                .str.strip()
                .str.lower()
                .isin(planned)
            )
            # ``mask`` rather than ``.loc[...] = ...``: the state column can be
            # backed by a read-only numpy array from ``np.where``.
            states["state"] = states["state"].mask(
                is_planned, MachineState.PLANNED_STOP.value
            )

        states = self._insert_stale_gaps(states)
        self._states = states
        return states

    def _insert_stale_gaps(self, states: pd.DataFrame) -> pd.DataFrame:
        """Insert ``UNKNOWN`` samples where the signal went quiet.

        Without this, one lost connection makes the last known state run to the
        end of the report. A machine that "produced" through a two-day outage is
        the classic symptom.
        """
        if states.empty or self.spec.stale_after is None:
            return states
        limit = pd.to_timedelta(self.spec.stale_after)
        times = pd.to_datetime(states[self.time_column])
        gaps = times.diff().shift(-1)
        stale = gaps > limit
        horizon = self._horizon()
        if horizon is not None:
            # The trailing sample also goes stale if nothing arrives before the
            # horizon.
            tail_gap = horizon - times.iloc[-1]
            if tail_gap > limit:
                stale.iloc[-1] = True

        if not stale.any():
            return states
        inserted = pd.DataFrame(
            {
                self.time_column: times[stale] + limit,
                "state": MachineState.UNKNOWN.value,
                "raw_value": pd.NA,
            }
        )
        combined = pd.concat([states, inserted], ignore_index=True)
        return (
            combined.sort_values(self.time_column, kind="mergesort")
            .drop_duplicates(subset=[self.time_column], keep="first")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def status_intervals(self) -> pd.DataFrame:
        """Return canonical state intervals.

        Returns:
            Interval-shaped frame with ``start``, ``end``, ``duration_seconds``,
            ``uuid``, ``source_uuid``, ``machine_id``, ``state`` (a
            :class:`~ts_shape.events.production.signals.spec.MachineState`
            value), ``raw_value`` (what the machine actually said, kept for
            drill-down), ``availability_class`` (``productive`` / ``loss`` /
            ``planned`` / ``unknown``) and ``is_productive``.

            Intervals are contiguous and non-overlapping: each ends where the
            next begins, so durations sum to the observed span.
        """
        if self._intervals is not None:
            return self._intervals

        states = self._state_samples()
        if states.empty:
            out = empty_event_df("interval", _INTERVAL_EXTRAS)
            self._intervals = out
            return out

        intervals = collapse_runs(
            states,
            value_column="state",
            time_column=self.time_column,
            until=self._horizon(),
            carry_columns=("raw_value",),
        )
        if self.spec.min_duration is not None:
            intervals = debounce_intervals(
                intervals, min_duration=self.spec.min_duration, value_column="state"
            )
        intervals = intervals[intervals["duration_seconds"] > 0].reset_index(drop=True)
        if intervals.empty:
            out = empty_event_df("interval", _INTERVAL_EXTRAS)
            self._intervals = out
            return out

        intervals["machine_id"] = self.machine_id
        intervals["availability_class"] = intervals["state"].map(
            {s.value: s.availability_class for s in MachineState}
        )
        intervals["is_productive"] = intervals["state"].eq(MachineState.PRODUCING.value)
        source = (
            self.spec.uuid
            if self.spec.mode is not StatusMode.ONE_HOT
            else ",".join(self.spec.state_map.keys())
        )
        out = finalize_interval_df(intervals, uuid=self.event_uuid, source_uuid=source)
        self._intervals = out.reset_index(drop=True)
        return self._intervals

    def downtime_intervals(self, *, min_duration: str | None = None) -> pd.DataFrame:
        """Return only the intervals that consume planned busy time.

        Args:
            min_duration: Discard shorter intervals -- use it to separate
                micro-stops from reportable downtime.

        Returns:
            Interval-shaped frame, a subset of :meth:`status_intervals`.
        """
        intervals = self.status_intervals()
        if intervals.empty:
            return intervals
        out = intervals[intervals["availability_class"] == "loss"]
        if min_duration is not None:
            threshold = pd.to_timedelta(min_duration).total_seconds()
            out = out[out["duration_seconds"] >= threshold]
        return out.reset_index(drop=True)

    def transitions(self) -> pd.DataFrame:
        """Return point events at every canonical state change.

        Returns:
            Point-shaped frame with ``systime``, ``uuid``, ``source_uuid``,
            ``machine_id``, ``from_state``, ``to_state``, ``transition``
            (``"producing->down"``) and ``previous_duration_seconds`` -- how
            long the machine had been in the state it just left.
        """
        extras = (
            "machine_id",
            "from_state",
            "to_state",
            "transition",
            "previous_duration_seconds",
        )
        intervals = self.status_intervals()
        if len(intervals) < 2:
            return empty_event_df("point", extras)
        out = pd.DataFrame(
            {
                self.time_column: intervals["start"].iloc[1:].to_numpy(),
                "source_uuid": intervals["source_uuid"].iloc[1:].to_numpy(),
                "machine_id": self.machine_id,
                "from_state": intervals["state"].iloc[:-1].to_numpy(),
                "to_state": intervals["state"].iloc[1:].to_numpy(),
                "previous_duration_seconds": intervals["duration_seconds"]
                .iloc[:-1]
                .to_numpy(),
            }
        )
        out["transition"] = out["from_state"] + "->" + out["to_state"]
        out = finalize_point_df(
            out,
            uuid=f"{self.event_uuid}:transition",
            source_uuid=None,
            time_col=self.time_column,
        )
        return out.reindex(
            columns=["systime", "uuid", "source_uuid", *extras]
        ).reset_index(drop=True)

    def state_summary(
        self,
        freq: str | None = None,
        *,
        drop_empty: bool = True,
    ) -> pd.DataFrame:
        """Aggregate state durations into per-state seconds and availability.

        Intervals crossing a window boundary are split, so the seconds in each
        window are exact and hourly figures sum to the daily one.

        Args:
            freq: Window size as a pandas offset alias; ``None`` summarises the
                whole loaded range as one row.
            drop_empty: Drop windows with no observed time.

        Returns:
            Summary-shaped frame with ``start``, ``end``, ``duration_seconds``,
            one ``<state>_seconds`` column per canonical state,
            ``observed_seconds`` (everything except ``unknown``),
            ``planned_busy_seconds`` (productive + loss), ``availability_pct``
            (productive / planned busy) and ``utilization_pct`` (productive /
            observed), plus ``machine_id``.

            ``availability_pct`` is ``NaN`` when there was no planned busy time
            -- a machine that was off all window has no availability, which is
            not the same as 0%.
        """
        extras = (
            *STATE_SECONDS_COLUMNS,
            "observed_seconds",
            "planned_busy_seconds",
            "availability_pct",
            "utilization_pct",
            "machine_id",
        )
        intervals = self.status_intervals()
        if intervals.empty:
            return empty_event_df("summary", extras)

        if freq is None:
            pieces = intervals.assign(window_start=intervals["start"].min())
        else:
            pieces = split_intervals(intervals, freq)

        wide = (
            pieces.pivot_table(
                index="window_start",
                columns="state",
                values="duration_seconds",
                aggfunc="sum",
                observed=True,
            )
            .reindex(columns=[s.value for s in MachineState])
            .fillna(0.0)
        )
        wide.columns = [f"{c}_seconds" for c in wide.columns]
        wide = wide.reset_index()
        wide.columns.name = None

        productive = wide[f"{MachineState.PRODUCING.value}_seconds"]
        loss = wide[
            [
                f"{s.value}_seconds"
                for s in MachineState
                if s.availability_class == "loss"
            ]
        ].sum(axis=1)
        unknown = wide[f"{MachineState.UNKNOWN.value}_seconds"]
        total = wide[list(STATE_SECONDS_COLUMNS)].sum(axis=1)

        wide["observed_seconds"] = total - unknown
        wide["planned_busy_seconds"] = productive + loss
        wide["availability_pct"] = np.where(
            wide["planned_busy_seconds"] > 0,
            productive / wide["planned_busy_seconds"] * 100.0,
            np.nan,
        )
        wide["utilization_pct"] = np.where(
            wide["observed_seconds"] > 0,
            productive / wide["observed_seconds"] * 100.0,
            np.nan,
        )
        wide["machine_id"] = self.machine_id

        wide["start"] = wide.pop("window_start")
        if freq is None:
            wide["end"] = intervals["end"].max()
        else:
            wide["end"] = wide["start"] + pd.tseries.frequencies.to_offset(freq)
        if drop_empty:
            wide = wide[total.to_numpy() > 0]
        out = finalize_summary_df(wide, uuid=self.event_uuid, source_uuid=None)
        return out.reset_index(drop=True)

    def state_at(self, timestamps: Sequence[pd.Timestamp] | pd.Series) -> pd.Series:
        """Return the canonical state in force at each timestamp.

        The as-of lookup other modules use to gate a metric by machine state --
        for example to discard a "cycle" that is really a lunch break.

        Args:
            timestamps: Instants to look up.

        Returns:
            An object series of state values, aligned to the input order.
            ``unknown`` before the first interval starts.
        """
        wanted = pd.to_datetime(pd.Series(list(timestamps), dtype="datetime64[ns]"))
        if wanted.empty:
            return pd.Series(dtype="object")
        intervals = self.status_intervals()
        if intervals.empty:
            return pd.Series(
                MachineState.UNKNOWN.value, index=wanted.index, dtype="object"
            )
        order = np.argsort(wanted.to_numpy(), kind="mergesort")
        # merge_asof refuses to join datetime64[ns] against datetime64[us], and
        # which resolution a frame carries depends on how it was built.
        probe = pd.DataFrame(
            {"_t": pd.to_datetime(wanted.to_numpy()[order]).astype("datetime64[ns]")}
        )
        lookup = intervals[["start", "state"]].rename(columns={"start": "_t"}).copy()
        lookup["_t"] = pd.to_datetime(lookup["_t"]).astype("datetime64[ns]")
        merged = pd.merge_asof(probe, lookup, on="_t", direction="backward")
        result = pd.Series(
            MachineState.UNKNOWN.value, index=range(len(wanted)), dtype="object"
        )
        result.iloc[order] = (
            merged["state"].fillna(MachineState.UNKNOWN.value).to_numpy()
        )
        result.index = wanted.index
        return result

    def unmapped_values(self) -> pd.DataFrame:
        """Return raw status values that no mapping covers.

        The commissioning tool for a new line and the drift detector for an old
        one: a controller that starts emitting state ``12`` after a software
        update shows up here rather than quietly becoming ``UNKNOWN`` time.

        Returns:
            Static-shaped frame with ``machine_id``, ``source_uuid``,
            ``raw_value``, ``occurrences``, ``first_seen`` and ``last_seen``,
            most frequent first. Empty for encodings that cannot have unmapped
            values (boolean and one-hot).
        """
        columns = [
            "machine_id",
            "source_uuid",
            "raw_value",
            "occurrences",
            "first_seen",
            "last_seen",
        ]
        if self.spec.mode in (StatusMode.BOOLEAN, StatusMode.ONE_HOT):
            return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

        assert self.spec.uuid is not None
        frame = signal_frame(
            self.dataframe,
            self.spec.uuid,
            time_column=self.time_column,
            value_column=self.spec.value_column,
            uuid_column=self.uuid_column,
        )
        if frame.empty:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

        raw = coerce_label(frame["value"], sentinels=())
        keys = raw.str.strip().str.lower()
        known = set(self.spec.state_map.keys())
        unknown = frame[~keys.isin(known)].assign(raw_value=raw[~keys.isin(known)])
        if unknown.empty:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

        grouped = (
            unknown.groupby("raw_value", dropna=False, observed=True)
            .agg(
                occurrences=(self.time_column, "size"),
                first_seen=(self.time_column, "min"),
                last_seen=(self.time_column, "max"),
            )
            .reset_index()
            .sort_values("occurrences", ascending=False)
        )
        grouped["machine_id"] = self.machine_id
        grouped["source_uuid"] = self.spec.uuid
        return grouped[columns].reset_index(drop=True)

    def diagnostics(self) -> pd.DataFrame:
        """Return one health row for this machine's status signal.

        Returns:
            Static-shaped single-row frame with ``machine_id``,
            ``source_uuid``, ``mode``, ``readings``, ``intervals``,
            ``states_seen``, ``unmapped_values``, ``unknown_seconds``,
            ``observed_seconds``, ``first_seen`` and ``last_seen``.
        """
        intervals = self.status_intervals()
        states = self._state_samples()
        unmapped = self.unmapped_values()
        unknown_seconds = (
            float(
                intervals.loc[
                    intervals["state"] == MachineState.UNKNOWN.value,
                    "duration_seconds",
                ].sum()
            )
            if not intervals.empty
            else 0.0
        )
        observed = (
            float(intervals["duration_seconds"].sum()) if not intervals.empty else 0.0
        )
        row: dict[str, Any] = {
            "machine_id": self.machine_id,
            "source_uuid": (
                self.spec.uuid
                if self.spec.mode is not StatusMode.ONE_HOT
                else ",".join(self.spec.state_map.keys())
            ),
            "mode": self.spec.mode.value,
            "readings": int(len(states)),
            "intervals": int(len(intervals)),
            "states_seen": (
                ",".join(sorted(intervals["state"].unique()))
                if not intervals.empty
                else ""
            ),
            "unmapped_values": int(len(unmapped)),
            "unknown_seconds": unknown_seconds,
            "observed_seconds": observed - unknown_seconds,
            "first_seen": (
                states[self.time_column].iloc[0] if not states.empty else pd.NaT
            ),
            "last_seen": (
                states[self.time_column].iloc[-1] if not states.empty else pd.NaT
            ),
        }
        return pd.DataFrame([row])
