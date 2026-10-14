"""Calculator

Numeric column calculations for engineered features.

Classes:
- IntegerCalc: Operations for integer-like numeric columns.
  - scale_column: Multiply by a factor.
  - offset_column: Add a constant.
  - divide_column: Divide by a constant.
  - subtract_column: Subtract a constant.
  - calculate_with_fixed_factors: Multiply then add.
  - mod_column: Modulo operation.
  - power_column: Raise to a power.

- PowerIntegrator: Integrate an instantaneous power signal (kW) into energy (kWh).
  - energy_by_window: Time-weighted kWh and mean kW per window.
  - to_energy_deltas: Emit kWh deltas in the standard ts-shape schema.
  - segment_energy: Per-sample energy contributions (audit trail).
"""
