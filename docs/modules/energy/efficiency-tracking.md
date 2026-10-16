# EnergyEfficiencyEvents

Energy efficiency against production and machine state: output per kWh over time, energy used while the machine is idle, specific energy consumption (kWh per unit), and a comparison across shifts.

`ts_shape.events.energy.efficiency_tracking` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/energy/efficiency_tracking.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents

t = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
running = (t.hour >= 6) & (t.hour < 22)
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "meter:main", "value_double": np.where(running, 12.0, 3.0)}),  # kWh deltas
    pd.DataFrame({"systime": t, "uuid": "counter:parts", "value_integer": np.cumsum(running * 30)}),
    pd.DataFrame({"systime": t, "uuid": "machine:running", "value_bool": running}),
], ignore_index=True)

ee = EnergyEfficiencyEvents(df)
waste = ee.idle_energy_waste("meter:main", "machine:running", window="1h")
print(f"Idle energy: {waste['waste_energy'].sum():.1f} kWh")
sec = ee.specific_energy_consumption("meter:main", "counter:parts", window="1D")
```

## Methods

| Method | Returns |
|---|---|
| `efficiency_trend(meter_uuid, counter_uuid, window='1h', trend_window=24)` | per window: `energy`, `units`, `efficiency` (units per kWh), `rolling_avg_efficiency`, `trend_direction` |
| `idle_energy_waste(meter_uuid, run_state_uuid, window='15min', idle_threshold=0.0)` | per window: `energy_consumed`, `machine_running_pct`, `machine_running_ratio`, `is_idle_waste`, `waste_energy` |
| `specific_energy_consumption(meter_uuid, counter_uuid, window='1D')` | per window: `total_energy`, `total_output`, `sec` (kWh per unit), `sec_trend` |
| `efficiency_comparison(meter_uuid, counter_uuid, shift_definitions=None)` | per shift: `avg_energy`, `avg_output`, `avg_efficiency`, `total_energy`, `total_output` |

## Notes

- The meter must carry kWh deltas (see [PowerIntegrator](power-integration.md) for kW signals). The state signal is boolean, True = running; counters are reset-aware.
- `idle_energy_waste` books a window's energy as waste when the machine ran less than 10 % of it and it consumed more than `idle_threshold`. The run state is held until the next change (`time_weighted=True`); `time_weighted=False` averages the samples.
- `rolling_avg_efficiency` is the trailing `trend_window` windows taken as one period (Σunits / Σenergy), and `sec_trend` compares each window's SEC with the trailing 7 windows' Σenergy / Σoutput (ISO 50006), not with the mean of the per-window ratios, which over-weights low-output windows.
- `efficiency_comparison` defaults to three 8-hour shifts from 06:00.
