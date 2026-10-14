# MeasurementUncertaintyEvents

> Measurement uncertainty per the GUM: Type A from repeated readings, Type B from external sources, combined and expanded uncertainty, and the uncertainty budget.

**Module:** `ts_shape.events.quality.measurement_uncertainty`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when you need to state how much a measurement can be trusted, for example to report `value ± U (k=2)` or to check that an instrument is good enough for a tolerance. It follows ISO/IEC Guide 98-3 (GUM):

- **Type A** is computed from the signal itself: the scatter of repeated readings per time window.
- **Type B** comes from sources you supply: calibration certificate, display resolution, temperature effects.
- The **budget** shows which source dominates, so you know whether to improve repeatability or buy a better reference.

---

## Quick Example

```python
from ts_shape.events.quality.measurement_uncertainty import MeasurementUncertaintyEvents

sources = [
    {"source": "calibration", "half_width": 0.004, "distribution": "normal"},
    {"source": "resolution", "half_width": 0.0005, "distribution": "rectangular"},
]

mu = MeasurementUncertaintyEvents(df, "cmm1:bore_diameter")

mu.type_a_uncertainty(window="8h")
MeasurementUncertaintyEvents.type_b_uncertainty(sources)
mu.combined_uncertainty(window="8h", type_b_sources=sources, coverage_factor=2.0)
mu.uncertainty_budget(window="8h", type_b_sources=sources)
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `type_a_uncertainty(window)` | summary | per window: `window_start`, `mean`, `std`, `n_samples`, `type_a_uncertainty` |
| `type_b_uncertainty(sources)` | static | per source: `source`, `distribution`, `half_width`, `divisor`, `type_b_uncertainty` |
| `combined_uncertainty(window, type_b_sources, coverage_factor)` | summary | per window: `window_start`, `type_a`, `type_b_combined`, `combined_uncertainty`, `expanded_uncertainty`, `coverage_factor` |
| `uncertainty_budget(window, type_b_sources)` | summary | per source: `source`, `uncertainty`, `pct_contribution` |

## Definitions

- **Type A:** `u_A = std / sqrt(n)` per window; windows with fewer than 2 samples are skipped.
- **Type B:** `u_B = half_width / divisor`. Divisors: `rectangular` √3 (default, also used for unknown names), `triangular` √6, `normal` 2, `u-shaped` √2. Each source dict needs `source` and `half_width`; `distribution` is optional.
- **Combined:** `u_c = sqrt(u_A² + Σ u_Bi²)`; **expanded:** `U = k · u_c`, with `k = coverage_factor` (2.0 ≈ 95.45 % coverage).
- **Budget:** Type A is averaged over all windows; `pct_contribution` is each source's share of the total variance (`u²`), so the rows sum to 100.
- `part_uuid` is deprecated: it has no effect, passing it emits a `DeprecationWarning`, and it will be removed in the next major release.
