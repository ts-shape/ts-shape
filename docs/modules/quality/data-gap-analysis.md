# DataGapAnalysisEvents

> Where are the holes in my data? Gap intervals, an overall gap summary, coverage per period and gaps short enough to interpolate.

**Module:** `ts_shape.events.quality.data_gap_analysis`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use to judge whether a signal's history is complete enough to analyze, and where it is not: historian outages, network drops, a PLC that stopped publishing. It complements [SignalQualityEvents](signal-quality.md), which flags individual missing-data events, with pattern-level views: one summary row, coverage per hour or day, and which gaps can be filled by interpolation without inventing a step that never happened.

---

## Quick Example

```python
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents

gaps = DataGapAnalysisEvents(df, "boiler2:steam_temp")

gaps.detect_gaps(min_gap="1min")
gaps.gap_summary(min_gap="1min")
gaps.coverage_by_period(window="1D")

fill = gaps.interpolation_candidates(max_gap="30s", min_gap="5s")
print(fill[fill["safe_to_interpolate"]])
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `detect_gaps(min_gap)` | interval | `start` (last sample before), `end` (first sample after), `gap_duration_seconds`, `samples_before_gap`, `samples_after_gap` |
| `gap_summary(min_gap)` | summary | one row: `total_gaps`, `total_missing_seconds`, `longest_gap_seconds`, `shortest_gap_seconds`, `mean_gap_seconds`, `data_span_seconds`, `gap_fraction` |
| `coverage_by_period(window, min_gap)` | summary | per window: `start`, `sample_count`, `coverage_pct`, `gap_count`, `gap_seconds` |
| `interpolation_candidates(max_gap, min_gap)` | interval | `start`, `end`, `gap_duration_seconds`, `value_before`, `value_after`, `value_jump`, `safe_to_interpolate` |

## Notes

- A gap is the interval between two consecutive samples that is at least `min_gap` long.
- `gap_fraction` = total gap time / time from the first to the last sample. Time before the first or after the last sample is not counted.
- `coverage_by_period` clips each gap to the window. Without `min_gap` it uses twice the median positive sampling interval. Window lengths follow the calendar, so a `"1D"` window on a DST day is 23 or 25 hours.
- `interpolation_candidates` reports every interval between `min_gap` and `max_gap`; with the default `min_gap="0s"` that is every sample interval up to `max_gap`, so set `min_gap` to just above your normal sampling interval. A gap is `safe_to_interpolate` when the value jump across it is within 2 standard deviations of the whole signal.
