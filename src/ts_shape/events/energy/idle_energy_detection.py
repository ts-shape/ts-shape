import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._windows import with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.shifts import assign_shift
from ts_shape.utils.time_weighted import time_weighted_resample

logger = logging.getLogger(__name__)

_DEFAULT_SHIFTS: dict[str, tuple[str, str]] = {
    "shift_1": ("06:00", "14:00"),
    "shift_2": ("14:00", "22:00"),
    "shift_3": ("22:00", "06:00"),
}


class IdleEnergyDetectionEvents(Base):
    """Energy: Idle Energy Detection

    Cross-reference an energy meter signal with a boolean machine-state signal
    to detect and quantify energy consumed during idle (non-production) periods.

    Supports two data models via constructor parameters:

    - Standard (ts-shape default)::

        IdleEnergyDetectionEvents(df)
        # expects: systime | uuid | value_double | value_bool

    - Raw CSV::

        IdleEnergyDetectionEvents(df, time_column="time", uuid_column="id")

    Methods:
    - idle_energy_by_window: Idle vs running energy per time window.
    - idle_energy_by_shift: Idle waste aggregated per shift.
    - idle_energy_trend: Rolling trend of idle energy waste.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "energy:idle",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        """Initialize the idle energy detector.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            event_uuid: UUID stamped on emitted events.
            time_column: Timestamp column.
            uuid_column: Column identifying the signal (e.g. ``"id"`` for a raw CSV
                frame).
        """
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    @renamed_kwargs(
        state_uuid="run_state_uuid",
        energy_column="value_column_energy",
        state_column="value_column_state",
    )
    def idle_energy_by_window(
        self,
        meter_uuid: str,
        run_state_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_state: str = "value_bool",
        window: str = "1h",
        idle_threshold: float = 0.1,
        time_weighted: bool = True,
    ) -> pd.DataFrame:
        """Aggregate energy consumed during idle periods per time window.

        The machine-running share is the share of each window the boolean state
        signal was True. It is reported as ``machine_running_ratio`` (0-1) and
        as ``machine_running_pct`` (0-100). Windows whose ratio is below
        ``idle_threshold`` count as fully idle; otherwise the idle fraction is
        ``1 - machine_running_ratio``.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            run_state_uuid: Identifier of the boolean machine-state signal (True=running).
            value_column_energy: Column containing energy readings.
            value_column_state: Column containing boolean state values.
            window: Resample window (e.g. '1h', '15min').
            idle_threshold: Running ratio (0-1) below which the window is
                classified as fully idle.
            time_weighted: ``True`` (default) weights the machine state by how
                long it was held (until the next state change) and keeps every
                energy window. ``False`` averages state *samples* and keeps only
                windows that contain a state row; that is only right for a
                densely, uniformly sampled state signal.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       is_delta, total_energy, idle_energy, running_energy,
                       machine_running_pct (0-100), idle_fraction,
                       machine_running_ratio (0-1)
        """
        energy, state = self._filter_two(meter_uuid, run_state_uuid)
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "is_delta",
            "total_energy",
            "idle_energy",
            "running_energy",
            "machine_running_pct",
            "idle_fraction",
            "machine_running_ratio",
        ]
        if energy.empty or state.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        energy = energy.set_index(self.time_column)
        state = state.set_index(self.time_column)

        energy_agg = (
            energy[value_column_energy].resample(window).sum().to_frame("total_energy")
        )
        if time_weighted:
            state_agg = (
                time_weighted_resample(
                    state[value_column_state].astype("boolean").astype(float),
                    window,
                    data_end=energy.index.max(),
                )
                .reindex(energy_agg.index)
                .to_frame("machine_running_pct")
            )
            out = energy_agg.join(state_agg, how="left").reset_index()
        else:
            state_agg = (
                bool_or_false(state[value_column_state])
                .astype(float)
                .resample(window)
                .mean()
                .to_frame("machine_running_pct")
            )
            out = energy_agg.join(state_agg, how="inner").reset_index()
        out = out.rename(columns={self.time_column: "start"})

        out["machine_running_ratio"] = out["machine_running_pct"].fillna(0.0)
        out["machine_running_pct"] = out["machine_running_ratio"] * 100.0
        out["idle_fraction"] = self._idle_fraction(
            out["machine_running_ratio"], idle_threshold
        )
        out["idle_energy"] = out["total_energy"] * out["idle_fraction"]
        out["running_energy"] = out["total_energy"] - out["idle_energy"]
        out["uuid"] = self.event_uuid
        out["source_uuid"] = meter_uuid
        out["is_delta"] = True

        return with_window_bounds(out[empty_cols], window)

    @renamed_kwargs(
        state_uuid="run_state_uuid",
        energy_column="value_column_energy",
        state_column="value_column_state",
    )
    def idle_energy_by_shift(
        self,
        meter_uuid: str,
        run_state_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_state: str = "value_bool",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
        idle_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Aggregate idle energy waste per shift across all dates.

        The idle fraction of a shift is ``1 - running ratio`` (the mean of
        the state samples in that shift). Pass ``idle_threshold`` to apply
        the same rule as :meth:`idle_energy_by_window`: a shift whose running
        ratio is below the threshold counts as fully idle.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            run_state_uuid: Identifier of the boolean machine-state signal.
            value_column_energy: Column containing energy readings.
            value_column_state: Column containing boolean state values.
            shift_definitions: Dict mapping shift name → (start_time, end_time).
                Default: three-shift operation (06-14, 14-22, 22-06).
            idle_threshold: Running ratio (0-1) below which a shift counts as
                fully idle. Default None: no threshold, the idle fraction is
                always ``1 - running ratio``; 0.1 matches
                :meth:`idle_energy_by_window`.

        Returns:
            DataFrame: shift, total_energy, idle_energy, waste_fraction
        """
        shifts = shift_definitions or _DEFAULT_SHIFTS
        empty_cols = ["shift", "total_energy", "idle_energy", "waste_fraction"]

        energy, state = self._filter_two(meter_uuid, run_state_uuid)
        if energy.empty or state.empty:
            return pd.DataFrame(columns=empty_cols)

        energy["_shift"] = assign_shift(energy[self.time_column], shifts)
        state["_shift"] = assign_shift(state[self.time_column], shifts)

        energy_by_shift = (
            energy.groupby("_shift")[value_column_energy].sum().rename("total_energy")
        )
        running_by_shift = (
            bool_or_false(state[value_column_state])
            .astype(float)
            .groupby(state["_shift"])
            .mean()
            .rename("machine_running_pct")
        )

        out = pd.concat([energy_by_shift, running_by_shift], axis=1).reset_index()
        out = out.rename(columns={"_shift": "shift"})
        out["machine_running_pct"] = out["machine_running_pct"].fillna(0.0)
        out["idle_fraction"] = self._idle_fraction(
            out["machine_running_pct"], idle_threshold
        )
        out["idle_energy"] = out["total_energy"] * out["idle_fraction"]
        out["waste_fraction"] = np.where(
            out["total_energy"] > 0,
            out["idle_energy"] / out["total_energy"],
            0.0,
        )
        return out[empty_cols]

    @renamed_kwargs(
        state_uuid="run_state_uuid",
        energy_column="value_column_energy",
        state_column="value_column_state",
    )
    def idle_energy_trend(
        self,
        meter_uuid: str,
        run_state_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_state: str = "value_bool",
        window: str = "1D",
        trend_window: int = 7,
        idle_threshold: float = 0.1,
    ) -> pd.DataFrame:
        """Rolling trend of idle energy waste over time.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            run_state_uuid: Identifier of the boolean machine-state signal.
            value_column_energy: Column containing energy readings.
            value_column_state: Column containing boolean state values.
            window: Aggregation window (default daily).
            trend_window: Number of windows for rolling average.
            idle_threshold: Running ratio (0-1) below which a window counts
                as fully idle (see :meth:`idle_energy_by_window`).

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       idle_energy, rolling_avg_idle_energy, trend_direction
        """
        by_window = self.idle_energy_by_window(
            meter_uuid,
            run_state_uuid,
            value_column_energy=value_column_energy,
            value_column_state=value_column_state,
            window=window,
            idle_threshold=idle_threshold,
        )
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "idle_energy",
            "rolling_avg_idle_energy",
            "trend_direction",
        ]
        if by_window.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        by_window["rolling_avg_idle_energy"] = (
            by_window["idle_energy"].rolling(window=trend_window, min_periods=1).mean()
        )
        slope = by_window["rolling_avg_idle_energy"].diff()
        by_window["trend_direction"] = pd.cut(
            slope,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=["improving", "stable", "worsening"],
        )
        # idle_energy_by_window already carries the true window bounds.
        return by_window[["start", "end", "duration_seconds", *empty_cols[1:]]]

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _idle_fraction(running_ratio: pd.Series, idle_threshold: float | None) -> Any:
        """Idle share from a 0-1 running ratio; fully idle below the threshold."""
        base = (1.0 - running_ratio).clip(lower=0.0)
        if idle_threshold is None:
            return base
        return np.where(running_ratio < idle_threshold, 1.0, base)

    def _filter_two(
        self, uuid_a: str, uuid_b: str
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """Filter dataframe for two signal UUIDs and prepare timestamps."""
        a = (
            self.dataframe[self.dataframe[self._uuid_column] == uuid_a]
            .copy()
            .sort_values(self.time_column)
        )
        b = (
            self.dataframe[self.dataframe[self._uuid_column] == uuid_b]
            .copy()
            .sort_values(self.time_column)
        )
        for df in (a, b):
            if not df.empty:
                df[self.time_column] = pd.to_datetime(df[self.time_column])
        return a, b
