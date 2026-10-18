# SamplingResolutionEvents

Checks whether a signal's sampling is fine enough for the process: quantization (too few distinct values), effective vs theoretical resolution, aliasing risk against the Nyquist frequency, and sampling jitter.

`ts_shape.events.quality.sampling_resolution` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/sampling_resolution.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.sampling_resolution import SamplingResolutionEvents

df = ts_shape.make_timeseries(["pump3:discharge_pressure"], n_points=7200, freq="1s", baseline=6.0, noise=0.05)
df["value_double"] = df["value_double"].round(1)  # coarse 0.1 bar ADC steps

sr = SamplingResolutionEvents(df, "pump3:discharge_pressure")
result = sr.detect_quantization(window="1h", threshold_ratio=0.3)
print(result[["start", "n_unique_values", "quantization_step", "quantized"]])
resolution = sr.effective_resolution(window="1h")
aliasing = sr.detect_aliasing(window="1h", expected_frequency_hz=24.2)  # 1450 rpm pump
jitter = sr.sampling_jitter(window="1h")
```

## Methods

All arguments are keyword-only.

| Method | Returns (per window) |
|---|---|
| `detect_quantization(window='1h', threshold_ratio=0.3)` | `start`, `end`, `duration_seconds`, `n_unique_values`, `expected_unique`, `quantization_step`, `quantization_ratio`, `quantized` |
| `effective_resolution(window='1h')` | `start`, `end`, `duration_seconds`, `theoretical_resolution`, `effective_resolution`, `resolution_ratio`, `assessment` |
| `detect_aliasing(window='1h', expected_frequency_hz=None)` | `start`, `end`, `duration_seconds`, `sampling_rate_hz`, `nyquist_hz`, `dominant_frequency`, `aliasing_risk` |
| `sampling_jitter(window='1h')` | `start`, `end`, `duration_seconds`, `mean_interval_seconds`, `std_interval_seconds`, `jitter_pct`, `max_gap_seconds`, `regularity_score` |

## Notes

- Quantization: the step is estimated from the smallest gaps between distinct values; `expected_unique` = value range / step + 1. A window is `quantized` when `quantization_ratio` (distinct / expected) is below `threshold_ratio` and fewer than half the samples are distinct. Needs at least 10 samples per window.
- Resolution: `theoretical_resolution` is the smallest step between distinct values; `effective_resolution` is the standard deviation of the absolute sample-to-sample differences. `assessment` is `adequate` (ratio ≤ 2), `degraded` (≤ 5) or `insufficient`.
- Intervals are float seconds from the nanosecond timestamps, so sub-millisecond spacing (0.5 ms, 1.5 ms) keeps its precision.
- Aliasing: the sampling rate comes from the mean interval in the window; the dominant frequency from an FFT (at least 16 samples). `aliasing_risk` compares `expected_frequency_hz` (or the dominant frequency) with the Nyquist frequency: `high` above 90 %, `medium` above 70 %, `low` above 50 %, else `none`. A frequency already aliased by the sampling cannot be seen in the data, so pass `expected_frequency_hz` (Hz) when you know it.
- Jitter: `jitter_pct` is the coefficient of variation of the sampling intervals; `regularity_score` = 100 − 2 × `jitter_pct`, floored at 0. Duplicate timestamps are ignored.
