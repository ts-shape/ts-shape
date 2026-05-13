import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class EnergyPerformanceIndicatorEvents(Base):
    """Energy: Performance Indicator (EnPI) per ISO 50001

    Calculate energy consumed per unit of production output (EnPI = kWh/unit).
    Tracks EnPI against a rolling baseline and identifies improvement/degradation
    trends. Supports comparison across multiple meters / production areas.

    Supports two data models via constructor parameters:

    - Standard (ts-shape default)::

        EnergyPerformanceIndicatorEvents(df)
        # expects: systime | uuid | value_double | value_integer

    - Raw CSV::

        EnergyPerformanceIndicatorEvents(df, time_column="time", uuid_column="id")

    Methods:
    - enpi_by_window: EnPI (energy / units) per time window.
    - enpi_vs_baseline: EnPI vs rolling baseline with anomaly flags.
    - enpi_by_hierarchy: EnPI across multiple meters for area comparison.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "energy:enpi",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    def enpi_by_window(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        energy_column: str = "value_double",
        counter_column: str = "value_integer",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Calculate energy per unit produced (EnPI) for each time window.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            counter_uuid: Identifier of the production counter signal.
            energy_column: Column containing energy readings.
            counter_column: Column containing counter readings.
            window: Aggregation window (e.g. '1D', '1h', '1W').

        Returns:
            DataFrame: window_start, uuid, source_uuid, is_delta,
                       energy_kwh, units_produced, enpi
        """
        energy, counter = self._aggregate_pair(
            meter_uuid,
            counter_uuid,
            energy_column=energy_column,
            counter_column=counter_column,
            window=window,
        )
        empty_cols = [
            "window_start",
            "uuid",
            "source_uuid",
            "is_delta",
            "energy_kwh",
            "units_produced",
            "enpi",
        ]
        if energy.empty:
            return pd.DataFrame(columns=empty_cols)

        merged = energy.join(counter, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "window_start"})
        merged["enpi"] = np.where(
            merged["units_produced"] > 0,
            merged["energy_kwh"] / merged["units_produced"],
            np.nan,
        )
        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True
        return merged[empty_cols]

    def enpi_vs_baseline(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        energy_column: str = "value_double",
        counter_column: str = "value_integer",
        window: str = "1D",
        baseline_window: int = 30,
        deviation_threshold: float = 0.1,
    ) -> pd.DataFrame:
        """Compare current EnPI against a rolling baseline.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            counter_uuid: Identifier of the production counter signal.
            energy_column: Column containing energy readings.
            counter_column: Column containing counter readings.
            window: Aggregation window.
            baseline_window: Number of windows for rolling baseline.
            deviation_threshold: Fractional deviation to flag as anomaly (0.1 = 10%).

        Returns:
            DataFrame: window_start, uuid, source_uuid, enpi, baseline_enpi,
                       deviation_pct, is_anomaly, trend
        """
        base = self.enpi_by_window(
            meter_uuid,
            counter_uuid,
            energy_column=energy_column,
            counter_column=counter_column,
            window=window,
        )
        empty_cols = [
            "window_start",
            "uuid",
            "source_uuid",
            "enpi",
            "baseline_enpi",
            "deviation_pct",
            "is_anomaly",
            "trend",
        ]
        if base.empty or len(base) < 2:
            return pd.DataFrame(columns=empty_cols)

        base["baseline_enpi"] = (
            base["enpi"].rolling(window=baseline_window, min_periods=1).mean()
        )
        base["deviation_pct"] = np.where(
            base["baseline_enpi"] > 0,
            (base["enpi"] - base["baseline_enpi"]) / base["baseline_enpi"],
            0.0,
        )
        base["is_anomaly"] = base["deviation_pct"].abs() > deviation_threshold

        # Trend: sign of rolling average slope over last few periods
        rolling = base["enpi"].rolling(window=min(7, len(base)), min_periods=1).mean()
        slope = rolling.diff()
        base["trend"] = np.where(
            slope < -0.01,
            "improving",
            np.where(slope > 0.01, "degrading", "stable"),
        )
        return base[empty_cols]

    def enpi_by_hierarchy(
        self,
        meter_uuids: List[str],
        counter_uuid: str,
        *,
        energy_column: str = "value_double",
        counter_column: str = "value_integer",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Calculate EnPI per meter for cross-area comparison.

        Useful for comparing energy intensity across production lines, buildings,
        or hierarchy levels. Combine with series metadata to map meter_uuid to
        label_lvl / hierarchy columns.

        Args:
            meter_uuids: List of energy meter identifiers.
            counter_uuid: Shared production counter identifier.
            energy_column: Column containing energy readings.
            counter_column: Column containing counter readings.
            window: Aggregation window.

        Returns:
            DataFrame: window_start, meter_uuid, energy_kwh, units_produced, enpi
        """
        frames = []
        for meter_uuid in meter_uuids:
            df = self.enpi_by_window(
                meter_uuid,
                counter_uuid,
                energy_column=energy_column,
                counter_column=counter_column,
                window=window,
            )
            if not df.empty:
                df = df.rename(columns={"source_uuid": "meter_uuid"})
                frames.append(
                    df[
                        [
                            "window_start",
                            "meter_uuid",
                            "energy_kwh",
                            "units_produced",
                            "enpi",
                        ]
                    ]
                )

        if not frames:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "meter_uuid",
                    "energy_kwh",
                    "units_produced",
                    "enpi",
                ]
            )

        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["window_start", "meter_uuid"])
            .reset_index(drop=True)
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _aggregate_pair(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        energy_column: str,
        counter_column: str,
        window: str,
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """Resample energy and counter to window, return as two indexed DataFrames."""
        energy_raw = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        counter_raw = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if energy_raw.empty or counter_raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        energy_raw[self.time_column] = pd.to_datetime(energy_raw[self.time_column])
        counter_raw[self.time_column] = pd.to_datetime(counter_raw[self.time_column])

        energy_idx = energy_raw.set_index(self.time_column)
        counter_idx = counter_raw.set_index(self.time_column)

        energy_agg = (
            energy_idx[energy_column].resample(window).sum().to_frame("energy_kwh")
        )
        counter_agg = (
            counter_idx[counter_column]
            .resample(window)
            .agg(lambda x: max(x.max() - x.min(), 0) if len(x) > 1 else 0)
            .to_frame("units_produced")
        )

        return energy_agg, counter_agg
