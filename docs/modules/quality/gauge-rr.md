# GaugeRepeatabilityEvents

Gauge R&R (measurement system analysis) for an inline sensor from repeated measurements of reference parts: equipment variation (repeatability), appraiser variation (reproducibility), %GRR, ndc and bias against reference values.

`ts_shape.events.quality.gauge_repeatability` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/gauge_repeatability.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.quality.gauge_repeatability import GaugeRepeatabilityEvents

rng = np.random.default_rng(0)
parts, operators, trials = [f"P{i}" for i in range(10)], ["A", "B", "C"], 3
rows = [(p, o, 10 + i * 0.5 + 0.02 * j + rng.normal(0, 0.05)) for i, p in enumerate(parts)
        for j, o in enumerate(operators) for _ in range(trials)]
df = pd.DataFrame(rows, columns=["value_string", "operator", "value_double"])
df["systime"], df["uuid"] = pd.date_range("2024-01-01", periods=len(df), freq="1min"), "gauge_01"

grr = GaugeRepeatabilityEvents(df, "gauge_01", operator_column="operator")
print(grr.gauge_rr_summary(tolerance_range=10.0))
bias = grr.measurement_bias({"P0": 10.0, "P1": 10.5})
```

## Methods

| Method | Returns |
|---|---|
| `repeatability(n_trials=None)` | per part: `mean`, `range`, `repeatability_std`, `EV` |
| `reproducibility()` | per operator: `mean`, `reproducibility_std`, `AV` (needs `operator_column`) |
| `gauge_rr_summary(tolerance_range=None)` | one row: `EV`, `AV`, `GRR`, `PV`, `TV`, `pct_GRR`, `pct_PV`, `ndc` (+ `pct_GRR_tolerance`) |
| `measurement_bias(reference_values)` | per part: `measured_mean`, `reference`, `bias`, `bias_pct` |

## Notes

- The part ID is read from `part_column` (default `value_string`) and the measurement from `value_column` (`value_double`); `operator_column` names an optional operator or station column.
- With `operator_column`, ranges and standard deviations are taken within each operator × part cell and averaged per part (AIAG average-and-range method).
- `AV = sqrt((X_diff · K2)² − EV² / (n · r))` with `X_diff` the range of operator means, `n` parts and `r` trials; a negative radicand gives `AV = 0`.
- `n_trials=None` detects the number of trials from the data.
- `pct_GRR_tolerance` (needs `tolerance_range` = USL − LSL) is the share of the tolerance used by the gauge. Under AIAG guidance below 10 % is acceptable, 10-30 % may be acceptable, above 30 % is not.
