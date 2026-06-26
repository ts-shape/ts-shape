"""Energy Events

Detectors for energy-related patterns: consumption analysis, efficiency
tracking, idle waste detection, EnPI, and carbon intensity on manufacturing
and industrial IoT time series data.

All classes accept both the standard ts-shape schema (systime | uuid | value_*)
and the raw CSV model (time | id | value) via time_column and uuid_column
constructor parameters.

Classes:
- EnergyConsumptionEvents: Analyze energy consumption patterns.
  - consumption_by_window: Aggregate energy consumption per time window.
  - peak_demand_detection: Detect peak demand periods exceeding thresholds.
  - consumption_baseline_deviation: Compare actual vs baseline consumption.
  - energy_per_unit: Calculate energy consumption per production unit.
  - normalize: Convert raw CSV DataFrame to standard schema.

- EnergyEfficiencyEvents: Track energy efficiency metrics.
  - efficiency_trend: Rolling efficiency metric over time.
  - idle_energy_waste: Detect energy consumption during idle periods.
  - specific_energy_consumption: Energy per unit output over time.
  - efficiency_comparison: Compare efficiency across shifts or periods.
  - normalize: Convert raw CSV DataFrame to standard schema.

- IdleEnergyDetectionEvents: Detect and quantify idle energy waste.
  - idle_energy_by_window: Idle vs running energy per time window.
  - idle_energy_by_shift: Idle waste aggregated per shift.
  - idle_energy_trend: Rolling trend of idle energy waste.

- EnergyPerformanceIndicatorEvents: ISO 50001 EnPI (energy per unit produced).
  - enpi_by_window: EnPI for each time window.
  - enpi_vs_baseline: EnPI vs rolling baseline with anomaly flags.
  - enpi_by_hierarchy: EnPI across multiple meters for area comparison.

- CarbonIntensityEvents: Scope 1 & 2 CO2e emissions tracking.
  - emissions_by_window: CO2e per source per time window.
  - total_emissions_by_window: Aggregated Scope 1 + 2 per window.
  - carbon_intensity_per_unit: kgCO2e per unit produced.
  - emission_factor_audit: Return configured factors for audit trail.
"""

from .carbon_intensity import CarbonIntensityEvents
from .consumption_analysis import EnergyConsumptionEvents
from .efficiency_tracking import EnergyEfficiencyEvents
from .energy_performance_indicator import EnergyPerformanceIndicatorEvents
from .idle_energy_detection import IdleEnergyDetectionEvents

__all__ = [
    "EnergyConsumptionEvents",
    "EnergyEfficiencyEvents",
    "IdleEnergyDetectionEvents",
    "EnergyPerformanceIndicatorEvents",
    "CarbonIntensityEvents",
]
