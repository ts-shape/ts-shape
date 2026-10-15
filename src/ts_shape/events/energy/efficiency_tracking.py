import logging

import numpy as np
import pandas as pd

from ts_shape.events._windows import with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false, to_numeric_checked
from ts_shape.utils.counters import counter_increments, counter_per_window
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.shifts import assign_shift
from ts_shape.utils.time_weighted import time_weighted_resample

logger = logging.getLogger(__name__)


def _rolling_ratio(
    numerator: pd.Series, denominator: pd.Series, window: int
) -> pd.Series:
    """Ratio of rolling sums, ``sum(numerator) / sum(denominator)``.

    The aggregate intensity of a span of windows (ISO 50006: EnPI of a
    period = its total energy / its total output), not the mean of the
    per-window ratios, which over-weights low-output windows. NaN where the
    rolling denominator is not positive.
    """
    num = numerator.astype(float).rolling(window=window, min_periods=1).sum()
    den = denominator.astype(float).rolling(window=window, min_periods=1).sum()
    return (num / den.where(den > 0)).astype(float)


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
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    @staticmethod
    @renamed_kwargs(df="dataframe", id_column="uuid_column")
    def normalize(
        dataframe: pd.DataFrame,
        *,
        series_id: str,
        time_column: str = "time",
        value_column: str = "value",
        uuid_column: str | None = None,
    ) -> pd.DataFrame:
        """Convert a raw energy DataFrame to the standard ts-shape schema.

        Args:
            dataframe: Raw DataFrame.
            series_id: UUID to assign when no uuid_column is provided.
            time_column: Name of the timestamp column in dataframe.
            value_column: Name of the value column in dataframe.
            uuid_column: Optional column name whose values become the uuid.

        Returns:
            DataFrame with columns: systime, uuid, value_double, is_delta
        """
        out = pd.DataFrame()
        out["systime"] = pd.to_datetime(dataframe[time_column])
        if uuid_column and uuid_column in dataframe.columns:
            out["uuid"] = dataframe[uuid_column].astype(str)
        else:
            out["uuid"] = series_id
        out["value_double"] = to_numeric_checked(
            dataframe[value_column], what=f"column {value_column!r}"
        )
        out["is_delta"] = True
        return out.sort_values("systime").reset_index(drop=True)

    @renamed_kwargs(
        counter_column="value_column_counter", energy_column="value_column_energy"
    )
    def efficiency_trend(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_counter: str = "value_integer",
        window: str = "1h",
        handle_resets: bool = True,
        trend_window: int = 24,
    ) -> pd.DataFrame:
        """Rolling energy efficiency trend (units produced per kWh).

        Args:
            meter_uuid: UUID of the energy meter signal.
            counter_uuid: UUID of the production counter signal.
            value_column_energy: Column with energy readings.
            value_column_counter: Column with counter readings.
            window: Time window for aggregation.
            trend_window: Number of trailing windows (itself included) in
                ``rolling_avg_efficiency``: their summed units over their
                summed energy.
            handle_resets: ``True`` (default) treats a decreasing production
                counter as a reset and sums reset-aware increments per window
                (see :func:`ts_shape.utils.counters.counter_per_window`), so a
                window with a reset still reports what was produced.
                ``False`` uses the per-window ``max - min``.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       is_delta, energy, units, efficiency, rolling_avg_efficiency,
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
            return with_window_bounds(
                pd.DataFrame(
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
                ),
                window,
            )

        energy_data[self.time_column] = pd.to_datetime(energy_data[self.time_column])
        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        energy_data = energy_data.set_index(self.time_column)
        counter_data = counter_data.set_index(self.time_column)

        energy_agg = (
            energy_data[value_column_energy].resample(window).sum().to_frame("energy")
        )
        counter_agg = counter_per_window(
            counter_data[value_column_counter], window, handle_resets=handle_resets
        ).to_frame("units")

        merged = energy_agg.join(counter_agg, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "start"})

        if merged.empty:
            return with_window_bounds(
                pd.DataFrame(
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
                ),
                window,
            )

        merged["efficiency"] = np.where(
            merged["energy"] > 0,
            merged["units"] / merged["energy"],
            0.0,
        )

        # Efficiency of the trailing span as one period (summed units over
        # summed energy), not the mean of per-window ratios.
        merged["rolling_avg_efficiency"] = _rolling_ratio(
            merged["units"], merged["energy"], trend_window
        ).fillna(0.0)

        slope = merged["rolling_avg_efficiency"].diff()
        merged["trend_direction"] = pd.cut(
            slope,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=["decreasing", "stable", "increasing"],
        )

        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True

        return with_window_bounds(
            merged[
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
            ],
            window,
        )

    @renamed_kwargs(
        state_uuid="run_state_uuid",
        energy_column="value_column_energy",
        state_column="value_column_state",
    )
    def idle_energy_waste(
        self,
        meter_uuid: str,
        run_state_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_state: str = "value_bool",
        window: str = "15min",
        idle_threshold: float = 0.0,
        time_weighted: bool = True,
    ) -> pd.DataFrame:
        """Detect energy consumed during idle periods (waste).

        Compares energy consumption with machine run/idle state to find
        windows where the machine is idle but still consuming energy.

        Args:
            meter_uuid: UUID of the energy meter signal.
            run_state_uuid: UUID of the boolean machine state signal (True=run).
            value_column_energy: Column with energy readings.
            value_column_state: Column with boolean state.
            window: Time window for analysis.
            idle_threshold: Energy above this during idle is waste.
            time_weighted: ``True`` (default) weights the machine state by how
                long it was held (until the next state change) and keeps every
                energy window. ``False`` averages state *samples* and keeps only
                windows that contain a state row; that is only right for a
                densely, uniformly sampled state signal.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       is_delta, energy_consumed, machine_running_pct (0-100),
                       machine_running_ratio (0-1), is_idle_waste, waste_energy.
                       A window is idle waste when the machine ran less than
                       10 % of it and consumed more than ``idle_threshold``.
        """
        energy = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        state = (
            self.dataframe[self.dataframe[self._uuid_column] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if energy.empty or state.empty:
            return with_window_bounds(
                pd.DataFrame(
                    columns=[
                        "start",
                        "uuid",
                        "source_uuid",
                        "is_delta",
                        "energy_consumed",
                        "machine_running_pct",
                        "machine_running_ratio",
                        "is_idle_waste",
                        "waste_energy",
                    ]
                ),
                window,
            )

        energy[self.time_column] = pd.to_datetime(energy[self.time_column])
        state[self.time_column] = pd.to_datetime(state[self.time_column])

        energy = energy.set_index(self.time_column)
        state = state.set_index(self.time_column)

        energy_agg = (
            energy[value_column_energy]
            .resample(window)
            .sum()
            .to_frame("energy_consumed")
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
            merged = energy_agg.join(state_agg, how="left").reset_index()
        else:
            state_agg = (
                bool_or_false(state[value_column_state])
                .astype(float)
                .resample(window)
                .mean()
                .to_frame("machine_running_pct")
            )
            merged = energy_agg.join(state_agg, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "start"})

        merged["machine_running_ratio"] = merged.pop("machine_running_pct")
        merged["machine_running_pct"] = merged["machine_running_ratio"] * 100.0
        # Idle waste: machine mostly idle (running < 10 %) but still consuming
        merged["is_idle_waste"] = (merged["machine_running_ratio"] < 0.1) & (
            merged["energy_consumed"] > idle_threshold
        )
        merged["waste_energy"] = np.where(
            merged["is_idle_waste"], merged["energy_consumed"], 0.0
        )

        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True

        return with_window_bounds(
            merged[
                [
                    "start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "energy_consumed",
                    "machine_running_pct",
                    "machine_running_ratio",
                    "is_idle_waste",
                    "waste_energy",
                ]
            ],
            window,
        )

    @renamed_kwargs(
        counter_column="value_column_counter", energy_column="value_column_energy"
    )
    def specific_energy_consumption(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_counter: str = "value_integer",
        window: str = "1D",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Daily/periodic specific energy consumption (SEC = energy / output).

        Lower SEC indicates better efficiency.

        Args:
            meter_uuid: UUID of the energy meter signal.
            counter_uuid: UUID of the production counter.
            value_column_energy: Column with energy readings.
            value_column_counter: Column with counter readings.
            window: Time window (default daily).
            handle_resets: ``True`` (default) treats a decreasing production
                counter as a reset and sums reset-aware increments per window
                (see :func:`ts_shape.utils.counters.counter_per_window`), so a
                window with a reset still reports what was produced.
                ``False`` uses the per-window ``max - min``.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       is_delta, total_energy, total_output, sec, sec_trend
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
            return with_window_bounds(
                pd.DataFrame(
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
                ),
                window,
            )

        energy[self.time_column] = pd.to_datetime(energy[self.time_column])
        counter[self.time_column] = pd.to_datetime(counter[self.time_column])

        energy = energy.set_index(self.time_column)
        counter = counter.set_index(self.time_column)

        energy_agg = (
            energy[value_column_energy].resample(window).sum().to_frame("total_energy")
        )
        counter_agg = counter_per_window(
            counter[value_column_counter], window, handle_resets=handle_resets
        ).to_frame("total_output")

        merged = energy_agg.join(counter_agg, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "start"})

        merged["sec"] = np.where(
            merged["total_output"] > 0,
            merged["total_energy"] / merged["total_output"],
            np.nan,
        )

        # Trend: compare each period's SEC to the SEC of the trailing 7
        # periods taken as one period (summed energy / summed output, ISO
        # 50006), not the mean of the per-period ratios.
        rolling_sec = _rolling_ratio(merged["total_energy"], merged["total_output"], 7)
        merged["sec_trend"] = np.where(
            merged["sec"] > rolling_sec * 1.1,
            "worsening",
            np.where(merged["sec"] < rolling_sec * 0.9, "improving", "stable"),
        )

        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True

        return with_window_bounds(
            merged[
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
            ],
            window,
        )

    @renamed_kwargs(
        counter_column="value_column_counter", energy_column="value_column_energy"
    )
    def efficiency_comparison(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_counter: str = "value_integer",
        shift_definitions: dict[str, tuple] | None = None,
    ) -> pd.DataFrame:
        """Compare energy efficiency across shifts.

        Args:
            meter_uuid: UUID of the energy meter signal.
            counter_uuid: UUID of the production counter.
            value_column_energy: Column with energy readings.
            value_column_counter: Column with counter readings.
            shift_definitions: Dict mapping shift name to (start_time, end_time)
                             strings. Default: 3-shift operation.

        Returns:
            DataFrame: shift, avg_energy, avg_output, avg_efficiency,
                       total_energy, total_output. ``total_output`` is the
                       counter increase within the shift (reset-aware) and
                       ``avg_output`` the mean increase per counter reading.
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

        energy["shift"] = assign_shift(energy[self.time_column], shift_definitions)
        counter["shift"] = assign_shift(counter[self.time_column], shift_definitions)

        energy_by_shift = (
            energy.groupby("shift")[value_column_energy]
            .agg(total_energy="sum", avg_energy="mean")
            .reset_index()
        )

        # The counter is cumulative: output is the sum of reset-aware
        # per-reading increments, not the sum of raw readings (which made
        # 100, 101, 102 count as 303 parts instead of 2).
        counter["_increment"] = counter_increments(counter[value_column_counter])
        counter_by_shift = (
            counter.groupby("shift")["_increment"]
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
