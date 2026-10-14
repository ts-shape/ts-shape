# PowerIntegrator

> Integrate an instantaneous power signal (kW) into energy (kWh).

**Module:** `ts_shape.transform.calculator.power_integration`
**Guide:** [Production Guide](../../guides/production.md)

---

## When to Use

Use when a meter or PLC gives you **instantaneous active power** — a signal like
`*.Total_Active_Power-kW` — and you need consumption, not demand.

Active power (kW) is the real, working part of the draw: the part that becomes
torque, heat and light, and the part you are billed for, as opposed to reactive
power (kvar), which magnetises motors and transformers without doing work.
"Total" means summed over all three phases (L1 + L2 + L3).

Power is a **rate**; energy is its integral:

```
E[kWh] = ∫ P[kW] dt
```

Over a full hour that collapses to a convenient identity: the kWh consumed in an
hour equals the *time-weighted* mean kW during that hour. The emphasis matters,
because historians log on change or on deadband, so samples are irregular and
both shortcuts are wrong:

| Shortcut | Why it fails |
|----------|--------------|
| `sum()` of the kW samples | Scales with the logging rate, not with consumption — double the log rate and the "energy" doubles. This is what happens when a kW signal is fed straight into `EnergyConsumptionEvents`, whose methods assume kWh deltas. |
| `mean()` of the kW samples | A *sample-count* average. A machine at 200 kW for 59 minutes that logs 40 samples around a 900 kW spike in the last minute reads far too high. |

`PowerIntegrator` weights every sample by how long it was valid, splits segments
at window boundaries, and caps logging outages.

!!! tip "Prefer a counter when you have one"
    If the meter also exposes a cumulative kWh register (`label ILIKE '%Energy%'`),
    use it — a counter difference per window is billing-accurate and needs no
    integration. Use this class when instantaneous power is all you have.

---

## Quick Example

```python
from ts_shape.transform.calculator.power_integration import PowerIntegrator

# df: systime | uuid | value_double   (value_double = kW)
PowerIntegrator.energy_by_window(df, freq="1h", max_gap="15min")
#                start                 end  duration_seconds    kwh  avg_kw  coverage  sample_count
# 2024-01-01 06:00:00 2024-01-01 07:00:00            3600.0  140.0   140.0       1.0            37
```

Per machine instead of a plant total:

```python
PowerIntegrator.energy_by_window(df, freq="1h", total=False)
# adds a `uuid` column, one row per signal per window
```

Hand off to the energy detectors, which expect kWh deltas:

```python
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents

deltas = PowerIntegrator.to_energy_deltas(df, freq="1h", max_gap="15min")
EnergyConsumptionEvents(deltas).consumption_by_window(meter_uuid, window="1h")
```

In a pipeline:

```python
from ts_shape.pipeline import Pipeline

(
    Pipeline(name="hourly-kwh")
    .transform(PowerIntegrator, "to_energy_deltas", freq="1h", max_gap="15min")
    .detect(
        EnergyConsumptionEvents,
        "consumption_by_window",
        name="kwh",
        meter_uuid=meter_uuid,
        window="1h",
        agg="sum",
    )
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `energy_by_window()` | kWh and time-weighted mean kW per window | DataFrame: start, end, duration_seconds, kwh, avg_kw, coverage, sample_count |
| `to_energy_deltas()` | The same energy in the standard ts-shape schema | DataFrame: systime, uuid, value_double (kWh), is_delta |
| `segment_energy()` | Per-sample energy contributions — the audit trail | DataFrame: uuid, window_start, start, end, duration_seconds, kw_start, kw_end, kwh |

### Output columns

- **`kwh`** — integrated energy in the window. Negative power (PV export,
  regenerative drives) is integrated as-is, so this is *net* energy; filter the
  input first if you want gross import.
- **`avg_kw`** — time-weighted mean power over the *covered* time, i.e. `kwh`
  divided by the hours actually logged. With full coverage and `freq="1h"` it
  equals `kwh`.
- **`coverage`** — fraction of the window backed by data. With `total=True` it
  is the least-covered signal's. Anything below `1.0` means the window is
  under-reported.

Windows with no data at all are **omitted**, never reported as `0.0` kWh, so
"we did not measure" never renders as "the machine used nothing".

---

## Tips & Hints

!!! tip "Choose the interpolation that matches the logger"
    `method="locf"` (default) holds each value until the next sample — correct
    for on-change PLC and meter logging, where a value is by definition valid
    until the next one is written. `method="linear"` is trapezoidal — correct
    for a smoothly varying analog signal on a fixed cycle. On a 0 → 100 kW ramp
    over one hour, `linear` gives exactly 50 kWh and `locf` slightly less; that
    is the interpolation bias, not an error.

!!! warning "Always set `max_gap` on historian data"
    Without it, one sample at 500 kW before a three-hour logger outage bills
    1500 kWh. A segment longer than `max_gap` is dropped entirely — the power
    over that span is unknown — so it counts towards neither energy nor
    coverage. Set the cap *above* the signal's normal logging cadence: an
    on-change signal on a steady machine can legitimately stay silent for a long
    time, and too tight a cap silently discards real consumption. Watch
    `coverage` to see when that is happening.

!!! warning "Energy adds across machines, power does not"
    A SQL helper that flattens every matching UUID across every selected PLC
    into one list is fine as *input* — `total=True` integrates each signal
    separately and then sums, which is the plant total. Averaging that same
    list instead gives the average power of a typical machine, which is not a
    quantity anyone wants.

!!! info "Confirm the unit before you trust the number"
    A `-kW` suffix is a naming convention, not a guarantee. Pass
    `power_unit="W"` or `power_unit="MW"` when the signal is scaled differently;
    the result is always kWh. Then sanity-check a full day against the meter's
    own counter — they should agree within a percent or two.

!!! info "Related modules"
    - [`EnergyConsumptionEvents`](consumption-analysis.md) - Windowed consumption, peak demand and baseline deviation on kWh deltas
    - [`EnergyEfficiencyEvents`](efficiency-tracking.md) - Efficiency trends and idle energy waste
    - [`UnitConverter`](../../reference/ts_shape/transform/calculator/unit_conversion.md) - General engineering unit conversion

---

## See Also

- [Production Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/transform/calculator/power_integration.md)
