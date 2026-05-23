import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class EnergyEfficiencyEvents(Base):
    """Energy: Efficiency Tracking

    Track energy efficiency metrics against production and machine state.

    Supports two data models via constructor parameters:

    - Standard (ts-shape default)::

        EnergyEfficiencyEvents(df)
        # expects: systime | uuid | value_double / value_integer / value_bool

    - Raw CSV (time + id + value)::

        EnergyEfficiencyEvents(df, time_column="time", uuid_column="id")
        # pass value_column="value" to each method

    Methods:
    - efficiency_trend: Rolling efficiency metric over time.
    - idle_energy_waste: Detect energy consumption during idle periods.
    - specific_energy_consumption: Energy per unit output trend.
    - efficiency_comparison: Compare efficiency across shifts or periods.
    - normalize: Static helper to convert raw CSV format to standard schema.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "energy:efficiency",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        *,
        series_id: str,
        time_column: str = "time",
        value_column: str = "value",
        id_column: str | None = None,
    ) -> pd.DataFrame:
        """Convert a raw energy DataFrame to the standard ts-shape schema.

        Args:
            df: Raw DataFrame.
            series_id: UUID to assign when no id_column is provided.
            time_column: Name of the timestamp column in df.
            value_column: Name of the value column in df.
            id_column: Optional column name whose values become the uuid.

        Returns:
            DataFrame with columns: systime, uuid, value_double, is_delta
        """
        out = pd.DataFrame()
        out["systime"] = pd.to_datetime(df[time_column])
        if id_column and id_column in df.columns:
            out["uuid"] = df[id_column].astype(str)
        else:
            out["uuid"] = series_id
        out["value_double"] = pd.to_numeric(df[value_column], errors="coerce")
        out["is_delta"] = True
        return out.sort_values("systime").reset_index(drop=True)

    def efficiency_trend(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        energy_column: str = "value_double",
        counter_column: str = "value_integer",
        window: str = "1h",
        trend_window: int = 24,
    ) -> pd.DataFrame:
        """Rolling energy efficiency trend (units produced per kWh).

        Args:
            meter_uuid: UUID of the energy meter signal.
            counter_uuid: UUID of the production counter signal.
            energy_column: Column with energy readings.
            counter_column: Column with counter readings.
            window: Time window for aggregation.
            trend_window: Number of windows for rolling average.

        Returns:
            DataFrame: start, uuid, source_uuid, is_delta,
                       energy, units, efficiency, rolling_avg_efficiency,
                       trend_direction
        """
        # Aggregate energy
        energy_data = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        counter_data = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if energy_data.empty or counter_data.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "energy",
                    "units",
                    "efficiency",
                    "rolling_avg_efficiency",
                    "trend_direction",
                ]
            )

        energy_data[self.time_column] = pd.to_datetime(energy_data[self.time_column])
        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        energy_data = energy_data.set_index(self.time_column)
        counter_data = counter_data.set_index(self.time_column)

        energy_agg = (
            energy_data[energy_column].resample(window).sum().to_frame("energy")
        )
        counter_agg = (
            counter_data[counter_column]
            .resample(window)
            .agg(lambda x: x.max() - x.min() if len(x) > 1 else 0)
            .clip(lower=0)
            .to_frame("units")
        )

        merged = energy_agg.join(counter_agg, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "start"})

        if merged.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "energy",
                    "units",
                    "efficiency",
                    "rolling_avg_efficiency",
                    "trend_direction",
                ]
            )

        merged["efficiency"] = np.where(
            merged["energy"] > 0,
            merged["units"] / merged["energy"],
            0.0,
        )

        merged["rolling_avg_efficiency"] = (
            merged["efficiency"].rolling(window=trend_window, min_periods=1).mean()
        )

        slope = merged["rolling_avg_efficiency"].diff()
        merged["trend_direction"] = pd.cut(
            slope,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=["decreasing", "stable", "increasing"],
        )

        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True

        return merged[
            [
                "start",
                "uuid",
                "source_uuid",
                "is_delta",
                "energy",
                "units",
                "efficiency",
                "rolling_avg_efficiency",
                "trend_direction",
            ]
        ]

    def idle_energy_waste(
        self,
        meter_uuid: str,
        state_uuid: str,
        *,
        energy_column: str = "value_double",
        state_column: str = "value_bool",
        window: str = "15min",
        idle_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Detect energy consumed during idle periods (waste).

        Compares energy consumption with machine run/idle state to find
        windows where the machine is idle but still consuming energy.

        Args:
            meter_uuid: UUID of the energy meter signal.
            state_uuid: UUID of the boolean machine state signal (True=run).
            energy_column: Column with energy readings.
            state_column: Column with boolean state.
            window: Time window for analysis.
            idle_threshold: Energy above this during idle is waste.

        Returns:
            DataFrame: start, uuid, source_uuid, is_delta,
                       energy_consumed, machine_running_pct, is_idle_waste,
                       waste_energy
        """
        energy = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        state = (
            self.dataframe[self.dataframe[self._uuid_column] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if energy.empty or state.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "energy_consumed",
                    "machine_running_pct",
                    "is_idle_waste",
                    "waste_energy",
                ]
            )

        energy[self.time_column] = pd.to_datetime(energy[self.time_column])
        state[self.time_column] = pd.to_datetime(state[self.time_column])

        energy = energy.set_index(self.time_column)
        state = state.set_index(self.time_column)

        energy_agg = (
            energy[energy_column].resample(window).sum().to_frame("energy_consumed")
        )
        state_agg = (
            state[state_column]
            .fillna(False)
            .astype(float)
            .resample(window)
            .mean()
            .to_frame("machine_running_pct")
        )

        merged = energy_agg.join(state_agg, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "start"})

        # Idle waste: machine mostly idle but still consuming energy
        merged["is_idle_waste"] = (merged["machine_running_pct"] < 0.1) & (
            merged["energy_consumed"] > idle_threshold
        )
        merged["waste_energy"] = np.where(
            merged["is_idle_waste"], merged["energy_consumed"], 0.0
        )

        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True

        return merged[
            [
                "start",
                "uuid",
                "source_uuid",
                "is_delta",
                "energy_consumed",
                "machine_running_pct",
                "is_idle_waste",
                "waste_energy",
            ]
        ]

    def specific_energy_consumption(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        energy_column: str = "value_double",
        counter_column: str = "value_integer",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Daily/periodic specific energy consumption (SEC = energy / output).

        Lower SEC indicates better efficiency.

        Args:
            meter_uuid: UUID of the energy meter signal.
            counter_uuid: UUID of the production counter.
            energy_column: Column with energy readings.
            counter_column: Column with counter readings.
            window: Time window (default daily).

        Returns:
            DataFrame: start, uuid, source_uuid, is_delta,
                       total_energy, total_output, sec, sec_trend
        """
        energy = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        counter = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if energy.empty or counter.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "total_energy",
                    "total_output",
                    "sec",
                    "sec_trend",
                ]
            )

        energy[self.time_column] = pd.to_datetime(energy[self.time_column])
        counter[self.time_column] = pd.to_datetime(counter[self.time_column])

        energy = energy.set_index(self.time_column)
        counter = counter.set_index(self.time_column)

        energy_agg = (
            energy[energy_column].resample(window).sum().to_frame("total_energy")
        )
        counter_agg = (
            counter[counter_column]
            .resample(window)
            .agg(lambda x: x.max() - x.min() if len(x) > 1 else 0)
            .clip(lower=0)
            .to_frame("total_output")
        )

        merged = energy_agg.join(counter_agg, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "start"})

        merged["sec"] = np.where(
            merged["total_output"] > 0,
            merged["total_energy"] / merged["total_output"],
            np.nan,
        )

        # Trend: compare each period's SEC to rolling average
        rolling_sec = merged["sec"].rolling(window=7, min_periods=1).mean()
        merged["sec_trend"] = np.where(
            merged["sec"] > rolling_sec * 1.1,
            "worsening",
            np.where(merged["sec"] < rolling_sec * 0.9, "improving", "stable"),
        )

        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True

        return merged[
            [
                "start",
                "uuid",
                "source_uuid",
                "is_delta",
                "total_energy",
                "total_output",
                "sec",
                "sec_trend",
            ]
        ]

    def efficiency_comparison(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        energy_column: str = "value_double",
        counter_column: str = "value_integer",
        shift_definitions: dict[str, tuple] | None = None,
    ) -> pd.DataFrame:
        """Compare energy efficiency across shifts.

        Args:
            meter_uuid: UUID of the energy meter signal.
            counter_uuid: UUID of the production counter.
            energy_column: Column with energy readings.
            counter_column: Column with counter readings.
            shift_definitions: Dict mapping shift name to (start_time, end_time)
                             strings. Default: 3-shift operation.

        Returns:
            DataFrame: shift, avg_energy, avg_output, avg_efficiency,
                       total_energy, total_output
        """
        if shift_definitions is None:
            shift_definitions = {
                "shift_1": ("06:00", "14:00"),
                "shift_2": ("14:00", "22:00"),
                "shift_3": ("22:00", "06:00"),
            }

        energy = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        counter = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if energy.empty or counter.empty:
            return pd.DataFrame(
                columns=[
                    "shift",
                    "avg_energy",
                    "avg_output",
                    "avg_efficiency",
                    "total_energy",
                    "total_output",
                ]
            )

        energy[self.time_column] = pd.to_datetime(energy[self.time_column])
        counter[self.time_column] = pd.to_datetime(counter[self.time_column])

        # Assign shifts
        def _assign_shift(ts: pd.Timestamp) -> str:
            t = ts.time()
            for name, (start, end) in shift_definitions.items():
                st = pd.to_datetime(start).time()
                et = pd.to_datetime(end).time()
                if st < et:
                    if st <= t < et:
                        return name
                else:
                    if t >= st or t < et:
                        return name
            return "unknown"

        energy["shift"] = energy[self.time_column].apply(_assign_shift)
        counter["shift"] = counter[self.time_column].apply(_assign_shift)

        energy_by_shift = (
            energy.groupby("shift")[energy_column]
            .agg(total_energy="sum", avg_energy="mean")
            .reset_index()
        )

        counter_by_shift = (
            counter.groupby("shift")[counter_column]
            .agg(total_output="sum", avg_output="mean")
            .reset_index()
        )

        merged = energy_by_shift.merge(
            counter_by_shift, on="shift", how="outer"
        ).fillna(0)
        merged["avg_efficiency"] = np.where(
            merged["total_energy"] > 0,
            merged["total_output"] / merged["total_energy"],
            0.0,
        )

        return merged[
            [
                "shift",
                "avg_energy",
                "avg_output",
                "avg_efficiency",
                "total_energy",
                "total_output",
            ]
        ]
