# PowerIntegrator

Integrates an instantaneous active-power signal (kW, e.g. `*.Total_Active_Power-kW`) into energy (kWh) per window, weighting each sample by how long it was valid. Use it when instantaneous power is all you have; if the meter also has a cumulative kWh register, the counter difference per window is billing-accurate and needs no integration.

`ts_shape.transform.calculator.power_integration` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/transform/calculator/power_integration.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.transform.calculator.power_integration import PowerIntegrator
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents

t = pd.date_range("2024-01-01 06:00", periods=240, freq="1min")
df = pd.DataFrame({"systime": t, "uuid": "press7:Total_Active_Power-kW",
                   "value_double": np.where(np.arange(240) % 60 < 45, 140.0, 20.0)})

result = PowerIntegrator.energy_by_window(df, window="1h", max_gap="15min")
print(result[["start", "kwh", "avg_kw", "coverage"]])
deltas = PowerIntegrator.to_energy_deltas(df, window="1h", max_gap="15min")
daily = EnergyConsumptionEvents(deltas).consumption_by_window("press7:Total_Active_Power-kW", window="1D")
```

## Methods

All are static and keyword-only after the frame.

| Method | Returns |
|---|---|
| `energy_by_window(df, window='1h', method='locf', max_gap=None, total=True)` | per window: `start`, `end`, `kwh`, `avg_kw`, `coverage`, `sample_count` (+ `uuid` with `total=False`) |
| `to_energy_deltas(df, window='1h', ...)` | the same energy in the standard schema: `systime`, `uuid`, `value_double` (kWh), `is_delta`; feeds the energy detectors |
| `segment_energy(df, window='1h', ...)` | per-sample contributions (audit trail): `window_start`, `start`, `end`, `kw_start`, `kw_end`, `kwh` |

## Notes

- Summing kW samples scales with the logging rate, and a plain mean is a sample-count average; on irregular historian data both are wrong. Over a full hour the kWh equal the time-weighted mean kW.
- `method="locf"` (default) holds each value until the next sample, correct for on-change PLC and meter logging. `method="linear"` is trapezoidal, correct for a smooth analog signal on a fixed cycle; on a 0 → 100 kW ramp over one hour `linear` gives exactly 50 kWh and `locf` slightly less.
- Always set `max_gap` on historian data: without it one 500 kW sample before a three-hour outage counts 1500 kWh. A segment longer than `max_gap` is dropped and counts towards neither energy nor coverage. Set the cap above the normal logging cadence; an on-change signal on a steady machine can be silent for a long time.
- `coverage` is the fraction of the window backed by data (with `total=True`, the least-covered signal's). Windows with no data are omitted, never reported as 0 kWh. `avg_kw` = `kwh` / covered hours.
- `kwh` is net energy: negative power (PV export, regenerative drives) is integrated as-is.
- `total=True` integrates each signal separately and sums them, which is the plant total (averaging power across machines would not be). `to_energy_deltas` keeps the signal UUIDs by default (`total=False`); with `total=True` the summed series is named `event_uuid` (default `energy:kwh`).
- `power_unit="W"` or `"MW"` for differently scaled signals; the result is always kWh. A `-kW` suffix is only a naming convention, so compare a full day with the meter's own counter.
