# AnomalyCorrelationEvents

Correlates anomalies (z-score outliers) across signals: anomalies that occur together, cascades where one signal's anomaly is followed by another's, and a ranking of which signals tend to lead.

`ts_shape.events.correlation.anomaly_correlation` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/correlation/anomaly_correlation.md)

## Example

```python
import ts_shape
from ts_shape.events.correlation.anomaly_correlation import AnomalyCorrelationEvents

signals = ["pump:pressure", "pump:flow", "motor:current"]
df = ts_shape.make_timeseries(signals, n_points=2880, freq="30s", n_outliers=6)

ac = AnomalyCorrelationEvents(df)
coincident = ac.coincident_anomalies(signals, coincidence_window="10min", min_signals=2)
cascades = ac.cascade_detection("pump:pressure", "pump:flow", max_delay="15min")
print(ac.root_cause_ranking(signals)[["signal_uuid", "leader_count", "leader_ratio", "rank"]])
```

## Methods

| Method | Returns |
|---|---|
| `coincident_anomalies(signal_uuids, n_sigma=3.0, coincidence_window='5min', min_signals=2)` | one row per group: `start`, `end`, `anomaly_count`, `signal_uuids_involved` |
| `cascade_detection(leader_uuid, follower_uuid, n_sigma=3.0, max_delay='10min')` | one row per cascade: `leader_time`, `follower_time`, `delay_seconds` |
| `root_cause_ranking(signal_uuids, n_sigma=3.0, max_delay='10min')` | per signal: `leader_count`, `follower_count`, `leader_ratio`, `rank` |

## Notes

- An anomaly is a sample whose z-score within its own signal exceeds `n_sigma`; values are read from `value_double`.
- `root_cause_ranking` counts, for every pair, how often one signal's anomaly precedes the other's within `max_delay`. Frequent leaders are root-cause candidates, not proof.
- Set `max_delay` from the known propagation time between the sensors (e.g. transit time); too long a delay pairs unrelated anomalies.
