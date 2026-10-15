# SignalQualityEvents

Data-quality checks for one numeric signal: gaps against an expected sampling interval, sampling regularity, values outside a physical range, and completeness per window. For gap summaries, coverage and interpolation candidates see [DataGapAnalysisEvents](data-gap-analysis.md).

`ts_shape.events.quality.signal_quality` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/signal_quality.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.signal_quality import SignalQualityEvents

df = ts_shape.make_timeseries(["tank_level"], n_points=7200, freq="1s", baseline=50.0, noise=20.0)
df = df.drop(index=range(3000, 3100))  # 100 s outage

sq = SignalQualityEvents(df, signal_uuid="tank_level")
result = sq.detect_missing_data(expected_freq="1s", tolerance_factor=2.0)
print(result[["start", "gap_duration", "expected_samples_missing"]])
out_of_range = sq.detect_out_of_range(min_value=0.0, max_value=100.0)
completeness = sq.data_completeness(window="1h", expected_freq="1s")
```

## Methods

| Method | Returns |
|---|---|
| `detect_missing_data(expected_freq='1s', tolerance_factor=2.0)` | gaps longer than `tolerance_factor × expected_freq`: `start`, `end`, `gap_duration`, `expected_samples_missing` |
| `sampling_regularity(window='1h')` | per window: `mean_interval`, `std_interval`, `min_interval`, `max_interval`, `regularity_score` |
| `detect_out_of_range(min_value, max_value, interval_end='next_change')` | intervals outside the range: `start`, `end`, `duration`, `min_observed`, `max_observed`, `direction` |
| `data_completeness(window='1h', expected_freq='1s')` | per window: `expected_count`, `actual_count`, `completeness_pct` |

## Notes

- An out-of-range interval ends at the next sample back in range by default, so change-only signals report the real time out of range. `interval_end="last_sample"` ends it at the last flagged sample.
- For historian data that is compressed or batched, raise `tolerance_factor` (e.g. to 3) to avoid false gaps.
