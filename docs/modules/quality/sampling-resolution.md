# SamplingResolutionEvents

> Find the limits of a digital measurement chain: A/D quantization, effective vs. theoretical resolution, aliasing risk and sampling jitter.

**Module:** `ts_shape.events.quality.sampling_resolution`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when you suspect the data acquisition, not the process, shapes what you see:

- a signal that only takes a few discrete values (coarse A/D converter, a historian that rounds, a deadband);
- noise larger than the instrument's step size, so the resolution on paper is not the resolution you get;
- a process frequency (vibration, a pump stroke, a control oscillation) close to half the sampling rate, so the recorded waveform may be aliased;
- irregular PLC or OPC UA sampling intervals.

Run it per signal before frequency analysis, SPC on small variations, or a Gauge R&R.

---

## Quick Example

```python
from ts_shape.events.quality.sampling_resolution import SamplingResolutionEvents

sr = SamplingResolutionEvents(df, "pump3:discharge_pressure")

sr.detect_quantization(window="1h", threshold_ratio=0.3)
sr.effective_resolution(window="1h")

# Pump runs at 1450 rpm, about 24.2 Hz
sr.detect_aliasing(window="1h", expected_frequency=24.2)

sr.sampling_jitter(window="1h")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `detect_quantization(window, threshold_ratio)` | summary | per window: `window_start`, `n_unique_values`, `expected_unique`, `quantization_step`, `quantization_ratio`, `quantized` |
| `effective_resolution(window)` | summary | per window: `window_start`, `theoretical_resolution`, `effective_resolution`, `resolution_ratio`, `assessment` |
| `detect_aliasing(window, expected_frequency)` | summary | per window: `window_start`, `sampling_rate_hz`, `nyquist_hz`, `dominant_frequency`, `aliasing_risk` |
| `sampling_jitter(window)` | summary | per window: `window_start`, `mean_interval_ms`, `std_interval_ms`, `jitter_pct`, `max_gap_ms`, `regularity_score` |

## Notes

- **Quantization.** The step is estimated from the smallest gaps between distinct values; `expected_unique` = value range / step + 1. A window is `quantized` when `quantization_ratio` (distinct / expected) is below `threshold_ratio` and fewer than half the samples are distinct. Needs at least 10 samples per window.
- **Resolution.** `theoretical_resolution` is the smallest step between distinct values; `effective_resolution` is the standard deviation of the absolute sample-to-sample differences. `assessment` is `adequate` (ratio ≤ 2), `degraded` (≤ 5) or `insufficient`.
- **Aliasing.** The sampling rate is the mean interval in the window; the dominant frequency comes from an FFT (needs at least 16 samples). `aliasing_risk` compares `expected_frequency` (or the dominant frequency if not given) with the Nyquist frequency: `high` above 90 %, `medium` above 70 %, `low` above 50 %, else `none`. A frequency already aliased by the sampling cannot be detected from the data alone, so pass `expected_frequency` when you know it. Frequencies are in Hz.
- **Jitter.** `jitter_pct` is the coefficient of variation of the sampling intervals; `regularity_score` = 100 − 2 × `jitter_pct`, floored at 0. Duplicate timestamps are ignored.
