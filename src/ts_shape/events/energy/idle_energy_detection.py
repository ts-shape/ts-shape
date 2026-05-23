import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, Optional, Tuple

from ts_shape.utils.base import Base

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
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    def idle_energy_by_window(
        self,
        meter_uuid: str,
        state_uuid: str,
        *,
        energy_column: str = "value_double",
        state_column: str = "value_bool",
        window: str = "1h",
        idle_threshold: float = 0.1,
    ) -> pd.DataFrame:
        """Aggregate energy consumed during idle periods per time window.

        Machine-running percentage is the mean of the boolean state signal within
        each window. Windows below ``idle_threshold`` are classified as idle.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            state_uuid: Identifier of the boolean machine-state signal (True=running).
            energy_column: Column containing energy readings.
            state_column: Column containing boolean state values.
            window: Resample window (e.g. '1h', '15min').
            idle_threshold: machine_running_pct below this classifies the window as idle.

        Returns:
            DataFrame: start, uuid, source_uuid, is_delta,
                       total_energy, idle_energy, running_energy,
                       machine_running_pct, idle_fraction
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
        ]
        if energy.empty or state.empty:
            return pd.DataFrame(columns=empty_cols)

        energy = energy.set_index(self.time_column)
        state = state.set_index(self.time_column)

        energy_agg = (
            energy[energy_column].resample(window).sum().to_frame("total_energy")
        )
        state_agg = (
            state[state_column]
            .fillna(False)
            .astype(float)
            .resample(window)
            .mean()
            .to_frame("machine_running_pct")
        )

        out = energy_agg.join(state_agg, how="inner").reset_index()
        out = out.rename(columns={self.time_column: "start"})

        out["machine_running_pct"] = out["machine_running_pct"].fillna(0.0)
        out["idle_fraction"] = np.where(
            out["machine_running_pct"] < idle_threshold,
            1.0,
            1.0 - out["machine_running_pct"],
        )
        out["idle_energy"] = out["total_energy"] * out["idle_fraction"]
        out["running_energy"] = out["total_energy"] - out["idle_energy"]
        out["uuid"] = self.event_uuid
        out["source_uuid"] = meter_uuid
        out["is_delta"] = True

        return out[empty_cols]

    def idle_energy_by_shift(
        self,
        meter_uuid: str,
        state_uuid: str,
        *,
        energy_column: str = "value_double",
        state_column: str = "value_bool",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Aggregate idle energy waste per shift across all dates.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            state_uuid: Identifier of the boolean machine-state signal.
            energy_column: Column containing energy readings.
            state_column: Column containing boolean state values.
            shift_definitions: Dict mapping shift name → (start_time, end_time).
                Default: three-shift operation (06-14, 14-22, 22-06).

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
            energy.groupby("_shift")[energy_column].sum().rename("total_energy")
        )
        running_by_shift = (
            state[state_column]
            .fillna(False)
            .astype(float)
            .groupby(state["_shift"])
            .mean()
            .rename("machine_running_pct")
        )

        out = pd.concat([energy_by_shift, running_by_shift], axis=1).reset_index()
        out = out.rename(columns={"_shift": "shift"})
        out["machine_running_pct"] = out["machine_running_pct"].fillna(0.0)
        out["idle_fraction"] = (1.0 - out["machine_running_pct"]).clip(lower=0.0)
        out["idle_energy"] = out["total_energy"] * out["idle_fraction"]
        out["waste_fraction"] = np.where(
            out["total_energy"] > 0,
            out["idle_energy"] / out["total_energy"],
            0.0,
        )
        return out[empty_cols]

    def idle_energy_trend(
        self,
        meter_uuid: str,
        state_uuid: str,
        *,
        energy_column: str = "value_double",
        state_column: str = "value_bool",
        window: str = "1D",
        trend_window: int = 7,
        idle_threshold: float = 0.1,
    ) -> pd.DataFrame:
        """Rolling trend of idle energy waste over time.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            state_uuid: Identifier of the boolean machine-state signal.
            energy_column: Column containing energy readings.
            state_column: Column containing boolean state values.
            window: Aggregation window (default daily).
            trend_window: Number of windows for rolling average.
            idle_threshold: machine_running_pct below this → idle.

        Returns:
            DataFrame: start, uuid, source_uuid, idle_energy,
                       rolling_avg_idle_energy, trend_direction
        """
        by_window = self.idle_energy_by_window(
            meter_uuid,
            state_uuid,
            energy_column=energy_column,
            state_column=state_column,
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
            return pd.DataFrame(columns=empty_cols)

        by_window["rolling_avg_idle_energy"] = (
            by_window["idle_energy"].rolling(window=trend_window, min_periods=1).mean()
        )
        slope = by_window["rolling_avg_idle_energy"].diff()
        by_window["trend_direction"] = pd.cut(
            slope,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=["improving", "stable", "worsening"],
        )
        return by_window[empty_cols]

    # ── Internal helpers ────────────────────────────────────────────────────

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
