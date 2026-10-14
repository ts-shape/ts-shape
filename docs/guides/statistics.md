# Signal Analytics

Compute descriptive statistics, detect process cycles, and recognize patterns across plant signals.

---

## Numeric Statistics

Standard statistical measures for any numeric signal.

```python
from ts_shape.features.stats.numeric_stats import NumericStatistics

stats = NumericStatistics(df, "value_double")

print(f"Count: {stats.count()}")
print(f"Mean: {stats.mean():.2f}")
print(f"Std: {stats.std():.2f}")
print(f"Min: {stats.min():.2f}")
print(f"Max: {stats.max():.2f}")
print(f"Median: {stats.median():.2f}")

# Percentiles
print(f"P95: {stats.percentile(95):.2f}")
print(f"P99: {stats.percentile(99):.2f}")
```

---

## Time Coverage Statistics

Understand signal availability and temporal coverage.

```python
from ts_shape.features.stats.timestamp_stats import TimestampStatistics

time_stats = TimestampStatistics(df, "systime")

print(f"First: {time_stats.first()}")
print(f"Last: {time_stats.last()}")
print(f"Duration: {time_stats.duration()}")
print(f"Count: {time_stats.count()}")
```

---

## String Value Counts

Analyze categorical signals — signal names, machine states, product types.

```python
from ts_shape.features.stats.string_stats import StringStatistics

str_stats = StringStatistics(df, "uuid")

print(str_stats.value_counts())
#          uuid  count
# 0  temperature   1440
# 1     pressure   1440
# 2     humidity   1440
```

---

## Cycle Extraction

Detect and validate production cycles from boolean or numeric trigger signals.

```python
from ts_shape.features.cycles.cycles_extractor import CycleExtractor

extractor = CycleExtractor(
    dataframe=df,
    start_uuid="cycle_start_signal",
    end_uuid="cycle_end_signal",
    value_change_threshold=0.1
)

# Auto-detect best extraction method
suggestions = extractor.suggest_method()
print(f"Recommended: {suggestions['recommended_methods']}")
print(f"Reason: {suggestions['reasoning']}")

# Extract cycles using recommended method
if 'process_persistent_cycle' in suggestions['recommended_methods']:
    cycles = extractor.process_persistent_cycle()
elif 'process_step_sequence' in suggestions['recommended_methods']:
    cycles = extractor.process_step_sequence(start_step=1, end_step=10)
else:
    cycles = extractor.process_value_change_cycle()

# Validate cycle durations
validated = extractor.validate_cycles(cycles, min_duration='1s', max_duration='1h')

# Detect and resolve overlapping cycles
clean_cycles = extractor.detect_overlapping_cycles(validated, resolve='keep_longest')

# Extraction statistics
stats = extractor.get_extraction_stats()
print(f"Total: {stats['total_cycles']}, Complete: {stats['complete_cycles']}")
```

---

## Module Deep Dives

**Correlation:** [Signal Correlation](../modules/correlation/signal-correlation.md) | [Anomaly Correlation](../modules/correlation/anomaly-correlation.md)

---

## Next Steps

- [Quality Control & SPC](quality.md) — Outlier detection and process control
- [Production Monitoring](production.md) — Machine states and shop floor tracking
- [API Reference](../reference/index.md) — Full features API documentation
