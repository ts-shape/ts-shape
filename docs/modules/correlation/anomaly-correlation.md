# AnomalyCorrelationEvents

> Correlate anomaly events across multiple signals to find coincident patterns, cascading failures, and root cause candidates.

**Module:** `ts_shape.events.correlation.anomaly_correlation`
**Guide:** [Statistics Guide](../../guides/statistics.md)

---

## When to Use

Use to understand how anomalies on individual signals relate to each other. The class flags anomalies itself (absolute Z-score at or above `z_threshold`, default 3.0, per signal on `value_double`), so pass raw long-format signals, not pre-computed anomaly flags. Finds root cause candidates by identifying which signals' anomalies consistently precede others. Essential for complex systems where a fault in one subsystem cascades through multiple sensors.

---

## Quick Example

```python
from ts_shape.events.correlation.anomaly_correlation import AnomalyCorrelationEvents

# df: systime | uuid | value_double
correlator = AnomalyCorrelationEvents(df)
signals = ["pump:pressure", "pump:flow", "motor:current"]

# Find anomalies on at least 2 signals within a 10-minute window
coincident = correlator.coincident_anomalies(signals, coincidence_window="10min")
print(f"Found {len(coincident)} coincident anomaly groups")

# Detect cascades: pressure anomaly followed by a flow anomaly within 15 minutes
cascades = correlator.cascade_detection(
    "pump:pressure", "pump:flow", max_delay="15min"
)

# Rank signals by how often their anomalies precede others
ranking = correlator.root_cause_ranking(signals)
print("Top root cause candidates:")
print(ranking.head(5)[["signal_uuid", "leader_ratio", "rank"]])
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `coincident_anomalies(signal_uuids)` | Anomalies on at least `min_signals` (default 2) signals within `coincidence_window` (default `"5min"`) | `start`, `end`, `uuid`, `is_delta`, `anomaly_count`, `signal_uuids_involved` |
| `cascade_detection(leader_uuid, follower_uuid)` | Leader anomaly followed by a follower anomaly within `max_delay` (default `"10min"`) | One row per cascade: `leader_time`, `follower_time`, `uuid`, `is_delta`, `leader_uuid`, `follower_uuid`, `delay_seconds` |
| `root_cause_ranking(signal_uuids)` | Rank signals by how often their anomalies lead others | `signal_uuid`, `leader_count`, `follower_count`, `leader_ratio`, `rank` |

---

## Tips & Hints

!!! tip "Tune the cascade window carefully"
    A window that is too short misses real cascades; too long produces false associations. Start with the known propagation time of your process (e.g., fluid transit time between sensors) and adjust from there.

!!! info "Related modules"
    - [`SignalCorrelationEvents`](signal-correlation.md) - Correlate raw signal values instead of anomaly events
    - [`FailurePredictionEvents`](../maintenance/failure-prediction.md) - Predict failure timing once root cause is identified
    - [`DegradationDetectionEvents`](../maintenance/degradation-detection.md) - Detect the degradation patterns that generate anomalies

---

## See Also

- [Statistics Guide](../../guides/statistics.md)
- [API Reference](../../reference/ts_shape/events/correlation/anomaly_correlation.md)
