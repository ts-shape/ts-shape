# VibrationAnalysisEvents

Rolling RMS above a healthy baseline, growth of peak-to-peak amplitude, and bearing health indicators (RMS, peak, crest factor, kurtosis) per window, from a vibration signal.

`ts_shape.events.maintenance.vibration_analysis` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/maintenance/vibration_analysis.md)

## Example

```python
import ts_shape
from ts_shape.events.maintenance.vibration_analysis import VibrationAnalysisEvents

df = ts_shape.make_timeseries(["accel_x"], n_points=20000, freq="1s", baseline=0.0, noise=2.5, n_outliers=50)
df.loc[15000:, "value_double"] *= 2.0  # bearing wear

va = VibrationAnalysisEvents(df, "accel_x")
rms = va.detect_rms_exceedance(2.5, threshold_factor=1.5, window="10min")
growth = va.detect_amplitude_growth(window="1h", growth_threshold=0.1)
print(va.bearing_health_indicators(window="1h")[["start", "rms", "peak", "crest_factor", "kurtosis"]])
```

## Methods

| Method | Returns |
|---|---|
| `detect_rms_exceedance(baseline_rms, threshold_factor=1.5, window='1min')` | intervals with rolling RMS above `baseline_rms × threshold_factor`: `rms_value`, `baseline_rms`, `ratio`, `duration_seconds` |
| `detect_amplitude_growth(window='1h', growth_threshold=0.1)` | windows whose peak-to-peak amplitude grew more than `growth_threshold` (fraction) over the first window: `amplitude`, `baseline_amplitude`, `growth_pct` |
| `bearing_health_indicators(window='5min')` | per window: `rms`, `peak`, `crest_factor` (peak / RMS), `kurtosis` |

## Notes

- `baseline_rms` is the RMS of the healthy machine, e.g. from a known-good period or the severity zones of ISO 10816 / ISO 20816 for the machine class.
- Amplitude windows do not overlap; the first window is the baseline.
- The signal is read from `value_double`. These are time-domain indicators; no spectrum is computed.
