# MeasurementUncertaintyEvents

Measurement uncertainty per the GUM (ISO/IEC Guide 98-3): Type A from the scatter of repeated readings per window, Type B from sources you supply (calibration certificate, resolution, temperature), combined and expanded uncertainty, and the uncertainty budget.

`ts_shape.events.quality.measurement_uncertainty` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/measurement_uncertainty.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.measurement_uncertainty import MeasurementUncertaintyEvents

df = ts_shape.make_timeseries(["cmm1:bore_diameter"], n_points=2000, freq="1min", baseline=25.0, noise=0.003)
sources = [
    {"source": "calibration", "half_width": 0.004, "distribution": "normal"},
    {"source": "resolution", "half_width": 0.0005, "distribution": "rectangular"},
]

mu = MeasurementUncertaintyEvents(df, "cmm1:bore_diameter")
print(mu.combined_uncertainty(window="8h", type_b_sources=sources, coverage_factor=2.0))
budget = mu.uncertainty_budget(window="8h", type_b_sources=sources)
```

## Methods

| Method | Returns |
|---|---|
| `type_a_uncertainty(window='8h')` | per window: `window_start`, `mean`, `std`, `n_samples`, `type_a_uncertainty` |
| `type_b_uncertainty(sources)` (static) | per source: `source`, `distribution`, `half_width`, `divisor`, `type_b_uncertainty` |
| `combined_uncertainty(window='8h', type_b_sources=None, coverage_factor=2.0)` | per window: `type_a`, `type_b_combined`, `combined_uncertainty`, `expanded_uncertainty`, `coverage_factor` |
| `uncertainty_budget(window='8h', type_b_sources=None)` | per source: `source`, `uncertainty`, `pct_contribution` |

## Notes

- Type A: `u_A = std / sqrt(n)` per window; windows with fewer than 2 samples are skipped.
- Type B: `u_B = half_width / divisor`, with divisors `rectangular` √3 (default, also for unknown names), `triangular` √6, `normal` 2, `u-shaped` √2. Each source needs `source` and `half_width`; `distribution` is optional.
- Combined: `u_c = sqrt(u_A² + Σ u_Bi²)`; expanded: `U = k · u_c` with `k = coverage_factor` (2.0 ≈ 95.45 % coverage).
- Budget: Type A is averaged over all windows; `pct_contribution` is each source's share of the total variance (`u²`), so the rows sum to 100.
- `part_uuid` is deprecated and has no effect.
