import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.events._windows import with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs
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
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    @renamed_kwargs(
        energy_column="value_column_energy", state_column="value_column_state"
    )
    def idle_energy_by_window(
        self,
        meter_uuid: str,
        state_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_state: str = "value_bool",
        window: str = "1h",
        idle_threshold: float = 0.1,
        time_weighted: bool = False,
    ) -> pd.DataFrame:
        """Aggregate energy consumed during idle periods per time window.

        The machine-running share is the mean of the boolean state signal within
        each window, a ratio between 0 and 1. It is reported as
        ``machine_running_ratio`` and, for backward compatibility, as
        ``machine_running_pct`` (also a 0-1 ratio despite its name). Windows
        whose ratio is below ``idle_threshold`` count as fully idle; otherwise
        the idle fraction is ``1 - machine_running_ratio``.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            state_uuid: Identifier of the boolean machine-state signal (True=running).
            value_column_energy: Column containing energy readings.
            value_column_state: Column containing boolean state values.
            window: Resample window (e.g. '1h', '15min').
            idle_threshold: Running ratio (0-1) below which the window is
                classified as fully idle.
            time_weighted: Weight the machine state by how long it was held
                (until the next state change) and keep every energy window. Default
                ``False`` averages state *samples* and inner-joins, so a change-encoded
                state signal reads as a sample count and windows without a state row
                drop their energy.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       is_delta, total_energy, idle_energy, running_energy,
                       machine_running_pct (0-1 ratio, kept for backward
                       compatibility), idle_fraction, machine_running_ratio
        """
        energy, state = self._filter_two(meter_uuid, state_uuid)
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
                state[value_column_state]
                .fillna(False)
                .astype(float)
                .resample(window)
                .mean()
                .to_frame("machine_running_pct")
            )
            out = energy_agg.join(state_agg, how="inner").reset_index()
        out = out.rename(columns={self.time_column: "start"})

        out["machine_running_pct"] = out["machine_running_pct"].fillna(0.0)
        out["machine_running_ratio"] = out["machine_running_pct"]
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
        energy_column="value_column_energy", state_column="value_column_state"
    )
    def idle_energy_by_shift(
        self,
        meter_uuid: str,
        state_uuid: str,
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
            state_uuid: Identifier of the boolean machine-state signal.
            value_column_energy: Column containing energy readings.
            value_column_state: Column containing boolean state values.
            shift_definitions: Dict mapping shift name → (start_time, end_time).
                Default: three-shift operation (06-14, 14-22, 22-06).
            idle_threshold: Running ratio (0-1) below which a shift counts as
                fully idle. Default None (no threshold, the previous
                behavior); 0.1 matches :meth:`idle_energy_by_window`.

        Returns:
            DataFrame: shift, total_energy, idle_energy, waste_fraction
        """
        shifts = shift_definitions or _DEFAULT_SHIFTS
        empty_cols = ["shift", "total_energy", "idle_energy", "waste_fraction"]

        energy, state = self._filter_two(meter_uuid, state_uuid)
        if energy.empty or state.empty:
            return pd.DataFrame(columns=empty_cols)

        def _assign_shift(ts: pd.Timestamp) -> str:
            t = ts.time()
            for name, (start, end) in shifts.items():
                st = pd.to_datetime(start).time()
                et = pd.to_datetime(end).time()
                if st < et:
                    if st <= t < et:
                        return name
                else:
                    if t >= st or t < et:
                        return name
            return "unknown"

        energy["_shift"] = energy[self.time_column].apply(_assign_shift)
        state["_shift"] = state[self.time_column].apply(_assign_shift)

        energy_by_shift = (
            energy.groupby("_shift")[value_column_energy].sum().rename("total_energy")
        )
        running_by_shift = (
            state[value_column_state]
            .fillna(False)
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
        energy_column="value_column_energy", state_column="value_column_state"
    )
    def idle_energy_trend(
        self,
        meter_uuid: str,
        state_uuid: str,
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
            state_uuid: Identifier of the boolean machine-state signal.
            value_column_energy: Column containing energy readings.
            value_column_state: Column containing boolean state values.
            window: Aggregation window (default daily).
            trend_window: Number of windows for rolling average.
            idle_threshold: machine_running_pct below this → idle.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       idle_energy, rolling_avg_idle_energy, trend_direction
        """
        by_window = self.idle_energy_by_window(
            meter_uuid,
            state_uuid,
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
        return with_window_bounds(by_window[empty_cols], window)

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
