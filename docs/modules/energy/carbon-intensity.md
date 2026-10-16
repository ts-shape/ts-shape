# CarbonIntensityEvents

CO2-equivalent emissions from energy and fuel meters with configurable emission factors: emissions per source and window, Scope 1 (fuel) and Scope 2 (electricity) totals, carbon intensity per produced unit, and the factors used, for an audit trail (CSRD, ISO 14064).

`ts_shape.events.energy.carbon_intensity` · [Energy guide](../../guides/energy.md) · [API reference](../../reference/ts_shape/events/energy/carbon_intensity.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.energy.carbon_intensity import CarbonIntensityEvents

# one week of hourly readings: kWh of electricity and m3 of gas per hour, a part counter
t = pd.date_range("2024-01-01", periods=7 * 24, freq="1h")
rng = np.random.default_rng(1)
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "meter:electricity", "value_double": rng.normal(120, 10, len(t))}),
    pd.DataFrame({"systime": t, "uuid": "meter:gas", "value_double": rng.normal(15, 2, len(t))}),
    pd.DataFrame({"systime": t, "uuid": "counter:parts", "value_integer": np.arange(len(t)) * 60}),
], ignore_index=True)

carbon = CarbonIntensityEvents(
    df,
    emission_factors={"meter:electricity": 0.233, "meter:gas": 2.034},  # kgCO2e per kWh, per m3
    scope_map={"meter:gas": 1, "meter:electricity": 2},
)
per_source = carbon.emissions_by_window(window="1D")
totals = carbon.total_emissions_by_window(window="1D")
print(totals[["start", "scope1_kgco2e", "scope2_kgco2e", "total_kgco2e"]].head(3))
intensity = carbon.carbon_intensity_per_unit("counter:parts", window="1D")
print(intensity[["start", "units_produced", "carbon_intensity", "trend"]].head(3))
print(carbon.emission_factor_audit())
```

## Methods

| Method | Returns |
|---|---|
| `emissions_by_window(scope=0, value_column='value_double', window='1D')` | per source and window: `source_uuid`, `scope`, `consumption`, `emission_factor`, `kgco2e` |
| `total_emissions_by_window(value_column='value_double', window='1D')` | per window: `scope1_kgco2e`, `scope2_kgco2e`, `total_kgco2e` |
| `carbon_intensity_per_unit(counter_uuid, value_column='value_double', value_column_counter='value_integer', window='1D', handle_resets=True)` | per window: `total_kgco2e`, `units_produced`, `carbon_intensity` (kgCO2e per unit), `trend` |
| `emission_factor_audit()` | one row per configured source: `source_uuid`, `scope`, `emission_factor_kgco2e_per_unit` |

## Notes

- Meter values are consumption per reading (deltas), summed per window. A cumulative meter reading must be differenced first; an instantaneous kW signal must be integrated with [PowerIntegrator](power-integration.md).
- `emission_factors` maps each meter uuid to kgCO2e per unit of its reading. A uuid missing from `scope_map` counts as Scope 2. `scope=1` or `scope=2` restricts `emissions_by_window` to one scope.
- The part counter is reset-aware (`handle_resets=True`). The first window counts only the increments after the first reading.
- `trend` compares a rolling mean over up to 7 windows with the previous one: `improving` when it falls by more than 0.001 kgCO2e per unit, `worsening` when it rises by more, `stable` otherwise.
- For raw CSV data, pass `time_column=` and `uuid_column=` to the constructor.
