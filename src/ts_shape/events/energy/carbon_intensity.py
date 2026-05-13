import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class CarbonIntensityEvents(Base):
    """Energy: Carbon Intensity Tracking (Scope 1 & 2)

    Converts energy and fuel consumption signals to CO2-equivalent emissions
    using configurable emission factors. Supports Scope 1 (direct fuel) and
    Scope 2 (electricity) calculations, plus carbon intensity per unit produced.
    Designed for CSRD and ISO 14064 reporting.

    Supports two data models via constructor parameters:

    - Standard (ts-shape default)::

        CarbonIntensityEvents(df, emission_factors={"meter:elec": 0.233})
        # expects: systime | uuid | value_double

    - Raw CSV::

        CarbonIntensityEvents(df, emission_factors={"sensor_01": 0.233},
                              time_column="time", uuid_column="id")

    ``emission_factors`` is a dict mapping signal identifier → kgCO2e per unit
    of the energy/fuel reading::

        emission_factors = {
            "meter:electricity": 0.233,   # kgCO2e/kWh  (UK grid average 2026)
            "meter:gas":         2.034,   # kgCO2e/m³   (natural gas)
        }

    Each factor's scope is inferred from ``scope_map``. Any uuid not in
    ``scope_map`` defaults to Scope 2.

    Methods:
    - emissions_by_window: CO2e per source per time window.
    - total_emissions_by_window: Aggregated Scope 1 + 2 per window.
    - carbon_intensity_per_unit: kgCO2e per unit produced.
    - emission_factor_audit: Return configured factors for audit trail.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        emission_factors: Dict[str, float],
        *,
        scope_map: Optional[Dict[str, int]] = None,
        event_uuid: str = "energy:carbon",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        """
        Args:
            dataframe: Input signal DataFrame.
            emission_factors: Mapping of signal identifier → kgCO2e per unit.
            scope_map: Optional mapping of signal identifier → scope (1 or 2).
                       Defaults to Scope 2 for any uuid not in the map.
            event_uuid: UUID assigned to output events.
            time_column: Name of the timestamp column.
            uuid_column: Name of the signal identifier column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.emission_factors = emission_factors
        self.scope_map = scope_map or {}
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    def emissions_by_window(
        self,
        *,
        scope: int = 0,
        value_column: str = "value_double",
        window: str = "1D",
    ) -> pd.DataFrame:
        """CO2e emissions per configured source per time window.

        Args:
            scope: Filter by scope: 1 = fuel only, 2 = electricity only,
                   0 = all sources (default).
            value_column: Column containing consumption readings.
            window: Aggregation window (e.g. '1D', '1h').

        Returns:
            DataFrame: window_start, uuid, source_uuid, scope,
                       consumption, emission_factor, kgco2e
        """
        frames = []
        for src_uuid, factor in self.emission_factors.items():
            src_scope = self.scope_map.get(src_uuid, 2)
            if scope != 0 and src_scope != scope:
                continue

            raw = (
                self.dataframe[self.dataframe[self._uuid_column] == src_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            if raw.empty:
                logger.debug("No data found for emission source '%s'", src_uuid)
                continue

            raw[self.time_column] = pd.to_datetime(raw[self.time_column])
            raw = raw.set_index(self.time_column)
            agg = (
                raw[value_column]
                .resample(window)
                .sum()
                .to_frame("consumption")
                .reset_index()
            )
            agg = agg.rename(columns={self.time_column: "window_start"})
            agg["uuid"] = self.event_uuid
            agg["source_uuid"] = src_uuid
            agg["scope"] = src_scope
            agg["emission_factor"] = factor
            agg["kgco2e"] = agg["consumption"] * factor
            frames.append(
                agg[
                    [
                        "window_start",
                        "uuid",
                        "source_uuid",
                        "scope",
                        "consumption",
                        "emission_factor",
                        "kgco2e",
                    ]
                ]
            )

        if not frames:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "uuid",
                    "source_uuid",
                    "scope",
                    "consumption",
                    "emission_factor",
                    "kgco2e",
                ]
            )

        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["window_start", "source_uuid"])
            .reset_index(drop=True)
        )

    def total_emissions_by_window(
        self,
        *,
        value_column: str = "value_double",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Aggregate Scope 1 + Scope 2 emissions across all configured meters.

        Args:
            value_column: Column containing consumption readings.
            window: Aggregation window.

        Returns:
            DataFrame: window_start, uuid, scope1_kgco2e, scope2_kgco2e,
                       total_kgco2e
        """
        all_emissions = self.emissions_by_window(
            scope=0, value_column=value_column, window=window
        )
        empty_cols = [
            "window_start",
            "uuid",
            "scope1_kgco2e",
            "scope2_kgco2e",
            "total_kgco2e",
        ]
        if all_emissions.empty:
            return pd.DataFrame(columns=empty_cols)

        scope1 = (
            all_emissions[all_emissions["scope"] == 1]
            .groupby("window_start")["kgco2e"]
            .sum()
            .rename("scope1_kgco2e")
        )
        scope2 = (
            all_emissions[all_emissions["scope"] == 2]
            .groupby("window_start")["kgco2e"]
            .sum()
            .rename("scope2_kgco2e")
        )

        out = pd.concat([scope1, scope2], axis=1).fillna(0.0).reset_index()
        out["total_kgco2e"] = out["scope1_kgco2e"] + out["scope2_kgco2e"]
        out["uuid"] = self.event_uuid
        return out[empty_cols].sort_values("window_start").reset_index(drop=True)

    def carbon_intensity_per_unit(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_double",
        counter_column: str = "value_integer",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Carbon intensity per unit produced (kgCO2e / unit).

        Args:
            counter_uuid: Identifier of the production counter signal.
            value_column: Column containing energy/fuel readings.
            counter_column: Column containing counter readings.
            window: Aggregation window.

        Returns:
            DataFrame: window_start, uuid, total_kgco2e, units_produced,
                       carbon_intensity, trend
        """
        totals = self.total_emissions_by_window(
            value_column=value_column, window=window
        )
        empty_cols = [
            "window_start",
            "uuid",
            "total_kgco2e",
            "units_produced",
            "carbon_intensity",
            "trend",
        ]
        if totals.empty:
            return pd.DataFrame(columns=empty_cols)

        counter_raw = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if counter_raw.empty:
            return pd.DataFrame(columns=empty_cols)

        counter_raw[self.time_column] = pd.to_datetime(counter_raw[self.time_column])
        counter_raw = counter_raw.set_index(self.time_column)
        counter_agg = (
            counter_raw[counter_column]
            .resample(window)
            .agg(lambda x: max(x.max() - x.min(), 0) if len(x) > 1 else 0)
            .to_frame("units_produced")
            .reset_index()
            .rename(columns={self.time_column: "window_start"})
        )

        merged = totals.merge(counter_agg, on="window_start", how="inner")
        merged["carbon_intensity"] = np.where(
            merged["units_produced"] > 0,
            merged["total_kgco2e"] / merged["units_produced"],
            np.nan,
        )

        rolling = (
            merged["carbon_intensity"]
            .rolling(window=min(7, len(merged)), min_periods=1)
            .mean()
        )
        slope = rolling.diff()
        merged["trend"] = np.where(
            slope < -0.001,
            "improving",
            np.where(slope > 0.001, "worsening", "stable"),
        )
        return merged[empty_cols]

    def emission_factor_audit(self) -> pd.DataFrame:
        """Return the configured emission factors for audit and reporting.

        Returns:
            DataFrame: source_uuid, scope, emission_factor_kgco2e_per_unit
        """
        rows = [
            {
                "source_uuid": uid,
                "scope": self.scope_map.get(uid, 2),
                "emission_factor_kgco2e_per_unit": factor,
            }
            for uid, factor in self.emission_factors.items()
        ]
        return pd.DataFrame(
            rows, columns=["source_uuid", "scope", "emission_factor_kgco2e_per_unit"]
        )
