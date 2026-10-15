# Energy

How to get from a power signal to kWh, energy per part, idle waste and CO2e.
The classes live in `ts_shape.events.energy`; `PowerIntegrator` is in
`ts_shape.transform.calculator`.

All energy detectors expect **energy per reading** (kWh since the previous
reading) and sum it per window. A meter that logs instantaneous power (kW) is
integrated first; a cumulative meter reading is differenced first.

## Example data

A press over three days: active power in kW every minute, a boolean run
signal and a part counter. The press runs 06:00 to 22:00 at about 140 kW and
idles at 20 kW.

```python
import numpy as np
import pandas as pd

t = pd.date_range("2024-01-01", periods=3 * 1440, freq="1min")
running = (t.hour >= 6) & (t.hour < 22)
rng = np.random.default_rng(5)
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "press:power_kw",
                  "value_double": np.where(running, 140.0, 20.0) + rng.normal(0, 3, len(t))}),
    pd.DataFrame({"systime": t, "uuid": "press:running", "value_bool": running}),
    pd.DataFrame({"systime": t, "uuid": "press:parts", "value_integer": np.cumsum(running * 2)}),
], ignore_index=True)
```

## From power to energy

`PowerIntegrator.to_energy_deltas` integrates the kW signal to kWh per window,
in the long format the detectors read. `max_gap` treats a longer gap as no data
instead of holding the last value across it.

```python
from ts_shape.transform.calculator.power_integration import PowerIntegrator

kwh = PowerIntegrator.to_energy_deltas(df[df["uuid"] == "press:power_kw"], window="15min", max_gap="10min")
energy = pd.concat([kwh, df[df["uuid"] != "press:power_kw"]], ignore_index=True)
```

## Consumption, idle waste and energy per part

```python
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents
from ts_shape.events.energy.energy_performance_indicator import EnergyPerformanceIndicatorEvents
from ts_shape.events.energy.idle_energy_detection import IdleEnergyDetectionEvents

daily = EnergyConsumptionEvents(energy).consumption_by_window("press:power_kw", window="1D")
print(daily[["start", "consumption"]].round({"consumption": 0}))

idle = IdleEnergyDetectionEvents(energy).idle_energy_by_window("press:power_kw", "press:running", window="1h")
print(f"idle energy: {idle['idle_energy'].sum():.0f} kWh of {idle['total_energy'].sum():.0f} kWh")

enpi = EnergyPerformanceIndicatorEvents(energy).enpi_by_window("press:power_kw", "press:parts", window="1D")
print(enpi[["start", "energy_kwh", "units_produced", "enpi"]].round({"energy_kwh": 0, "enpi": 3}))
```

The EnPI includes the idle energy: 160 of the 2400 kWh a day are drawn while
the press waits. The production counter is reset-aware; its first window counts
only the parts after the first reading, so start the data before production
starts or compare whole periods.

## CO2e

```python
from ts_shape.events.energy.carbon_intensity import CarbonIntensityEvents

carbon = CarbonIntensityEvents(energy, emission_factors={"press:power_kw": 0.233})   # kgCO2e per kWh
per_part = carbon.carbon_intensity_per_unit("press:parts", window="1D")
print(per_part[["start", "total_kgco2e", "carbon_intensity"]].round({"total_kgco2e": 0, "carbon_intensity": 3}))
```

## Which class answers which question

| Question | Class | Page |
|---|---|---|
| kWh from a kW signal? | `PowerIntegrator` | [Power integration](../modules/energy/power-integration.md) |
| Consumption per window, peak demand, deviation from a baseline? | `EnergyConsumptionEvents` | [Consumption analysis](../modules/energy/consumption-analysis.md) |
| Specific energy per part, idle waste, efficiency trend? | `EnergyEfficiencyEvents` | [Efficiency tracking](../modules/energy/efficiency-tracking.md) |
| EnPI against a baseline (ISO 50001 / 50006), across lines? | `EnergyPerformanceIndicatorEvents` | [Energy performance indicator](../modules/energy/energy-performance-indicator.md) |
| Energy while the machine does not produce, per shift? | `IdleEnergyDetectionEvents` | [Idle energy](../modules/energy/idle-energy-detection.md) |
| Scope 1 and 2 emissions, CO2e per part? | `CarbonIntensityEvents` | [Carbon intensity](../modules/energy/carbon-intensity.md) |
