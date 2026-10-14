# EnergyConsumptionEvents

Energy consumption per window, peak-demand windows, deviation from a rolling baseline, and energy per produced unit, from a meter signal of kWh deltas.

`ts_shape.events.energy.consumption_analysis` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/energy/consumption_analysis.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents

df = pd.concat([
    ts_shape.make_timeseries(["meter:main"], n_points=2880, freq="15min", baseline=12.0, noise=2.0),  # kWh per 15 min
    pd.DataFrame({"systime": pd.date_range("2025-01-01", periods=2880, freq="15min"),
                  "uuid": "counter:parts", "value_integer": np.arange(2880) * 25}),
], ignore_index=True)

energy = EnergyConsumptionEvents(df)
per_shift = energy.consumption_by_window("meter:main", window="8h", agg="sum")
peaks = energy.peak_demand_detection("meter:main", window="15min", percentile=0.95)
print(energy.energy_per_unit("meter:main", "counter:parts", window="1D").head())
```

## Methods

| Method | Returns (per window) |
|---|---|
| `consumption_by_window(meter_uuid, window='1h', agg='sum')` | `consumption` (`agg`: `sum`, `mean`, `max`) |
| `peak_demand_detection(meter_uuid, window='15min', threshold=None, percentile=0.95)` | `demand`, `threshold`, `is_peak` |
| `consumption_baseline_deviation(meter_uuid, window='1h', baseline_periods=24, deviation_threshold=0.2)` | `consumption`, `baseline` (rolling over `baseline_periods` windows), `deviation_pct`, `is_anomaly` |
| `energy_per_unit(meter_uuid, counter_uuid, window='1h')` | `energy`, `units_produced`, `energy_per_unit` |
| `normalize(df, series_id, ...)` (static) | a raw `(time, value)` or `(time, id, value)` frame converted to the standard schema |

## Notes

- Values must be energy deltas (kWh since the previous reading): the methods sum `value_double` per window. For an instantaneous kW signal, integrate it first with [PowerIntegrator](power-integration.md); summing kW samples measures the logging rate.
- Without `threshold`, the peak threshold is the `percentile` of the windowed consumption. `deviation_threshold` is a fraction (0.2 = 20 %).
- The part counter is reset-aware (`handle_resets=True`).
- For raw CSV data, pass `time_column=` and `uuid_column=` to the constructor and `value_column=` to each method.
