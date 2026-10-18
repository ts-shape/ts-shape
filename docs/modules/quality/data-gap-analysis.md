# DataGapAnalysisEvents

Gaps in a signal's history (historian outages, network drops, a PLC that stopped publishing): gap intervals, one summary row, coverage per period, and gaps short enough to interpolate. For individual missing-data events and range checks see [SignalQualityEvents](signal-quality.md).

`ts_shape.events.quality.data_gap_analysis` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/data_gap_analysis.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents

df = ts_shape.make_timeseries(["boiler2:steam_temp"], n_points=3000, freq="1s")
df = df.drop(index=list(range(500, 520)) + list(range(1500, 1900)))  # two outages

gaps = DataGapAnalysisEvents(df, "boiler2:steam_temp")
result = gaps.gap_summary(min_gap="5s")
print(result[["total_gaps", "total_missing_seconds", "longest_gap_seconds", "gap_fraction"]])
coverage = gaps.coverage_by_period(window="10min")
fill = gaps.interpolation_candidates(max_gap="30s", min_gap="2s")
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `detect_gaps(min_gap=None)` | `start` (last sample before), `end` (first sample after), `gap_duration_seconds`, `samples_before_gap`, `samples_after_gap` |
| `gap_summary(min_gap=None)` | one row: `total_gaps`, `total_missing_seconds`, `longest_gap_seconds`, `shortest_gap_seconds`, `mean_gap_seconds`, `data_span_seconds`, `gap_fraction` |
| `coverage_by_period(window='1h', min_gap=None)` | per window: `sample_count`, `coverage_pct`, `gap_count`, `gap_seconds` |
| `interpolation_candidates(max_gap='10s', min_gap='0s')` | `start`, `end`, `gap_duration_seconds`, `value_before`, `value_after`, `value_jump`, `safe_to_interpolate` |

## Notes

- A gap is an interval between two consecutive samples strictly longer than `min_gap`: regular 10-minute sampling with `min_gap="10min"` has no gaps. Without `min_gap` (the default of `detect_gaps`, `gap_summary` and `coverage_by_period`) the threshold is twice the median positive sampling interval, so a regularly sampled signal has none. A zero-length step (duplicate timestamp) is never a gap, even with `min_gap="0s"`.
- `gap_fraction` = total gap time / time from first to last sample (`data_span_seconds`, reported with or without gaps); time outside that span is not counted. Without gaps, `shortest_gap_seconds` and `mean_gap_seconds` are NaN and the totals are 0.
- `coverage_by_period` clips each gap to the window. Windows follow the calendar, so `"1D"` on a DST day is 23 or 25 hours.
- `interpolation_candidates` reports every interval between `min_gap` and `max_gap`; with `min_gap="0s"` that is every sample interval, so set `min_gap` just above the normal sampling interval. A gap is `safe_to_interpolate` when the value jump across it is within 2 standard deviations of the whole signal.
