import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class EnergyConsumptionEvents(Base):
    """Energy: Consumption Analysis

    Analyze energy consumption patterns from meter/sensor signals.

    Supports two data models via constructor parameters:

    - Standard (ts-shape default)::

        EnergyConsumptionEvents(df)
        # expects: systime | uuid | value_double

    - Raw CSV (time + id + value)::

        EnergyConsumptionEvents(df, time_column="time", uuid_column="id")
        # expects: time | id | value
        # pass value_column="value" to each method

    Methods:
    - consumption_by_window: Aggregate energy per time window from a meter UUID.
    - peak_demand_detection: Flag windows where consumption exceeds a threshold.
    - consumption_baseline_deviation: Compare actual vs rolling baseline.
    - energy_per_unit: Energy per production unit when paired with a counter.
    - normalize: Static helper to convert raw CSV format to standard schema.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "energy:consumption",
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
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert a raw energy DataFrame to the standard ts-shape schema.

        Handles two input formats:

        - Two-column CSV: (time, value) — ``series_id`` is assigned as uuid.
        - Three-column with explicit id: (time, id_column, value) — values from
          ``id_column`` are used as uuid; ``series_id`` is ignored.

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

    def consumption_by_window(
        self,
        meter_uuid: str,
        *,
        value_column: str = "value_double",
        window: str = "1h",
        agg: str = "sum",
    ) -> pd.DataFrame:
        """Aggregate energy consumption per time window.

        Args:
            meter_uuid: UUID of the energy meter signal.
            value_column: Column containing energy readings.
            window: Resample window (e.g. '1h', '15min', '1D').
            agg: Aggregation method ('sum', 'mean', 'max').

        Returns:
            DataFrame: window_start, uuid, source_uuid, is_delta, consumption
        """
        s = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if s.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "consumption",
                ]
            )

        s[self.time_column] = pd.to_datetime(s[self.time_column])
        s = s.set_index(self.time_column)

        if agg == "sum":
            resampled = s[value_column].resample(window).sum()
        elif agg == "mean":
            resampled = s[value_column].resample(window).mean()
        elif agg == "max":
            resampled = s[value_column].resample(window).max()
        else:
            resampled = s[value_column].resample(window).sum()

        out = resampled.to_frame("consumption").reset_index()
        out = out.rename(columns={self.time_column: "window_start"})
        out["uuid"] = self.event_uuid
        out["source_uuid"] = meter_uuid
        out["is_delta"] = True
        return out[["window_start", "uuid", "source_uuid", "is_delta", "consumption"]]

    def peak_demand_detection(
        self,
        meter_uuid: str,
        *,
        value_column: str = "value_double",
        window: str = "15min",
        threshold: Optional[float] = None,
        percentile: float = 0.95,
    ) -> pd.DataFrame:
        """Detect peak demand periods exceeding a threshold.

        If threshold is None, uses the given percentile of windowed consumption.

        Args:
            meter_uuid: UUID of the energy meter signal.
            value_column: Column containing energy readings.
            window: Resample window for demand calculation.
            threshold: Absolute demand threshold. If None, auto-calculated.
            percentile: Percentile to use for auto-threshold (default 95th).

        Returns:
            DataFrame: window_start, uuid, source_uuid, is_delta, demand,
                       threshold, is_peak
        """
        consumption = self.consumption_by_window(
            meter_uuid, value_column=value_column, window=window
        )
        if consumption.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "demand",
                    "threshold",
                    "is_peak",
                ]
            )

        if threshold is None:
            threshold = consumption["consumption"].quantile(percentile)

        consumption["demand"] = consumption["consumption"]
        consumption["threshold"] = threshold
        consumption["is_peak"] = consumption["demand"] > threshold

        return consumption[
            [
                "window_start",
                "uuid",
                "source_uuid",
                "is_delta",
                "demand",
                "threshold",
                "is_peak",
            ]
        ]

    def consumption_baseline_deviation(
        self,
        meter_uuid: str,
        *,
        value_column: str = "value_double",
        window: str = "1h",
        baseline_periods: int = 24,
        deviation_threshold: float = 0.2,
    ) -> pd.DataFrame:
        """Compare actual consumption vs rolling baseline.

        Args:
            meter_uuid: UUID of the energy meter signal.
            value_column: Column containing energy readings.
            window: Resample window for consumption.
            baseline_periods: Number of windows for rolling baseline.
            deviation_threshold: Fractional deviation to flag (0.2 = 20%).

        Returns:
            DataFrame: window_start, uuid, source_uuid, is_delta,
                       consumption, baseline, deviation_pct, is_anomaly
        """
        consumption = self.consumption_by_window(
            meter_uuid, value_column=value_column, window=window
        )
        if consumption.empty or len(consumption) < baseline_periods:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "consumption",
                    "baseline",
                    "deviation_pct",
                    "is_anomaly",
                ]
            )

        consumption["baseline"] = (
            consumption["consumption"]
            .rolling(window=baseline_periods, min_periods=1)
            .mean()
        )
        consumption["deviation_pct"] = np.where(
            consumption["baseline"] > 0,
            (consumption["consumption"] - consumption["baseline"])
            / consumption["baseline"],
            0.0,
        )
        consumption["is_anomaly"] = (
            consumption["deviation_pct"].abs() > deviation_threshold
        )

        return consumption[
            [
                "window_start",
                "uuid",
                "source_uuid",
                "is_delta",
                "consumption",
                "baseline",
                "deviation_pct",
                "is_anomaly",
            ]
        ]

    def energy_per_unit(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        energy_column: str = "value_double",
        counter_column: str = "value_integer",
        window: str = "1h",
    ) -> pd.DataFrame:
        """Calculate energy consumption per production unit.

        Args:
            meter_uuid: UUID of the energy meter signal.
            counter_uuid: UUID of the production counter signal.
            energy_column: Column with energy readings.
            counter_column: Column with counter readings.
            window: Time window for aggregation.

        Returns:
            DataFrame: window_start, uuid, source_uuid, is_delta,
                       energy, units_produced, energy_per_unit
        """
        energy = self.consumption_by_window(
            meter_uuid, value_column=energy_column, window=window
        )

        # Get production counts
        counter = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if energy.empty or counter.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "energy",
                    "units_produced",
                    "energy_per_unit",
                ]
            )

        counter[self.time_column] = pd.to_datetime(counter[self.time_column])
        counter = counter.set_index(self.time_column)
        counts = (
            counter[counter_column]
            .resample(window)
            .agg(lambda x: x.max() - x.min() if len(x) > 1 else 0)
        )
        counts = counts.clip(lower=0).to_frame("units_produced").reset_index()
        counts = counts.rename(columns={self.time_column: "window_start"})

        merged = energy.merge(counts, on="window_start", how="inner")
        merged["energy"] = merged["consumption"]
        merged["energy_per_unit"] = np.where(
            merged["units_produced"] > 0,
            merged["energy"] / merged["units_produced"],
            np.nan,
        )

        return merged[
            [
                "window_start",
                "uuid",
                "source_uuid",
                "is_delta",
                "energy",
                "units_produced",
                "energy_per_unit",
            ]
        ]
