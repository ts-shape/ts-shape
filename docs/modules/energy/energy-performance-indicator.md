# EnergyPerformanceIndicatorEvents

The energy performance indicator (EnPI, ISO 50001) of a line: energy per produced unit per window, compared with a rolling or a regression baseline (ISO 50006), and compared across meters or areas.

`ts_shape.events.energy.energy_performance_indicator` · [Energy guide](../../guides/energy.md) · [API reference](../../reference/ts_shape/events/energy/energy_performance_indicator.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.energy.energy_performance_indicator import EnergyPerformanceIndicatorEvents

# 60 days: 400 kWh/day base load plus 2.0 kWh per part; from Feb 15 each part takes 2.4 kWh
days = pd.date_range("2024-01-01", periods=60, freq="1D")
rng = np.random.default_rng(2)
parts = rng.integers(800, 1200, len(days))
kwh = 400 + np.where(days < "2024-02-15", 2.0, 2.4) * parts
df = pd.concat([
    pd.DataFrame({"systime": days, "uuid": "meter:line1", "value_double": kwh}),
    pd.DataFrame({"systime": days, "uuid": "meter:line2", "value_double": 0.9 * kwh}),
    pd.DataFrame({"systime": days, "uuid": "counter:parts", "value_integer": np.cumsum(parts)}),
], ignore_index=True)

enpi = EnergyPerformanceIndicatorEvents(df)
per_day = enpi.enpi_by_window("meter:line1", "counter:parts", window="1D")
check = enpi.enpi_vs_baseline(
    "meter:line1", "counter:parts",
    baseline="regression", baseline_period=("2024-01-02", "2024-02-01"),
    deviation_threshold=0.05,
)
print(check[["start", "enpi", "baseline_enpi", "deviation_pct", "is_anomaly"]].iloc[43:47])
areas = enpi.enpi_by_hierarchy(["meter:line1", "meter:line2"], "counter:parts", window="1D")
print(areas.groupby("meter_uuid")["enpi"].mean())
```

From February 15 the line uses 17 % more energy than the baseline model expects for the same output.

## Methods

| Method | Returns |
|---|---|
| `enpi_by_window(meter_uuid, counter_uuid, value_column_energy='value_double', value_column_counter='value_integer', window='1D', handle_resets=True)` | per window: `energy_kwh`, `units_produced`, `enpi` (kWh per unit) |
| `enpi_vs_baseline(meter_uuid, counter_uuid, ..., baseline_window=30, deviation_threshold=0.1, baseline='rolling', baseline_period=None)` | per window: `enpi`, `baseline_enpi`, `deviation_pct`, `is_anomaly`, `trend`; with `baseline='regression'` also `expected_energy_kwh`, `energy_savings_kwh` |
| `enpi_by_hierarchy(meter_uuids, counter_uuid, ...)` | per meter and window: `meter_uuid`, `energy_kwh`, `units_produced`, `enpi` |

## Notes

- Meter values are kWh per reading (deltas), summed per window; the production counter is cumulative and reset-aware.
- The first window has no counter increment (there is no earlier reading), so its `enpi` is NaN. The regression fit leaves out windows without output (zero or missing units, such as this first window or a stop day) and names them in a `DataQualityWarning`; their energy is base load only and would pull the fit.
- `baseline='rolling'` (default) divides the summed energy of the last `baseline_window` windows by their summed output. It follows the process, so it never shows a lasting improvement, and a fixed base load makes kWh/unit look worse whenever output is low.
- `baseline='regression'` fits `energy = a + b * units` on `baseline_period` (default: the first `baseline_window` windows) and compares each window's energy with the model's expectation for its output, so the comparison is normalized for volume. The cumulative sum of `energy_savings_kwh` is the CUSUM of savings.
- `deviation_threshold` is a fraction (0.1 = 10 %). `trend` is `improving`, `degrading` or `stable` by the slope of a 7-window rolling EnPI.
- For raw CSV data, pass `time_column=` and `uuid_column=` to the constructor.
