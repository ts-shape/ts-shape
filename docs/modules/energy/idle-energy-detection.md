# IdleEnergyDetectionEvents

Energy a machine draws while it is not producing: idle and running energy per window from a meter and a boolean run signal, idle waste per shift, and the trend of idle energy over days.

`ts_shape.events.energy.idle_energy_detection` · [Energy guide](../../guides/energy.md) · [API reference](../../reference/ts_shape/events/energy/idle_energy_detection.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.energy.idle_energy_detection import IdleEnergyDetectionEvents

# kWh per 15 min for three days; the press runs 06:00-22:00 and draws 5 kWh per 15 min when idle
t = pd.date_range("2024-01-01", periods=3 * 96, freq="15min")
running = (t.hour >= 6) & (t.hour < 22)
rng = np.random.default_rng(3)
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "meter:press",
                  "value_double": np.where(running, 20.0, 5.0) + rng.normal(0, 0.5, len(t))}),
    pd.DataFrame({"systime": t, "uuid": "press:running", "value_bool": running}),
], ignore_index=True)

idle = IdleEnergyDetectionEvents(df)
hourly = idle.idle_energy_by_window("meter:press", "press:running", window="1h")
print(hourly[["start", "total_energy", "idle_energy", "machine_running_pct"]].iloc[4:8])
shifts = idle.idle_energy_by_shift("meter:press", "press:running")
print(shifts)
trend = idle.idle_energy_trend("meter:press", "press:running", window="1D", trend_window=2)
```

The night shift (22:00-06:00, `shift_3`) is all idle: about 480 kWh over three nights.

## Methods

| Method | Returns |
|---|---|
| `idle_energy_by_window(meter_uuid, state_uuid, window='1h', idle_threshold=0.1, time_weighted=True)` | per window: `total_energy`, `idle_energy`, `running_energy`, `machine_running_pct` (0-100), `machine_running_ratio` (0-1), `idle_fraction` |
| `idle_energy_by_shift(meter_uuid, state_uuid, shift_definitions=None, idle_threshold=None)` | one row per shift over all days: `shift`, `total_energy`, `idle_energy`, `waste_fraction` |
| `idle_energy_trend(meter_uuid, state_uuid, window='1D', trend_window=7, idle_threshold=0.1)` | per window: `idle_energy`, `rolling_avg_idle_energy`, `trend_direction` |

## Notes

- Meter values are kWh per reading (deltas), summed per window. The run signal is `value_bool` (True = running).
- A window whose running share is below `idle_threshold` counts as fully idle; otherwise the idle share is `1 - machine_running_ratio`, and `idle_energy` is that share of the window's energy.
- The running share is time-weighted by default: each state holds until the next state change, so a change-only state signal is measured correctly. `time_weighted=False` averages the state samples, which is only right for a dense, evenly sampled signal.
- `idle_energy_by_shift` defaults to three shifts, 06-14, 14-22 and 22-06, named `shift_1` to `shift_3`; pass `shift_definitions={"early": ("06:00", "14:00"), ...}` for your own. It averages the state samples per shift and applies no threshold unless `idle_threshold` is given.
- For raw CSV data, pass `time_column=` and `uuid_column=` to the constructor.
