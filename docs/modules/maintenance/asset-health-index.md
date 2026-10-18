# AssetHealthIndexEvents

One 0-100 health score per asset and window, fused from several condition signals (vibration, temperature, motor current), each tagged with the failure mode it watches: alerts when the score falls below warning or critical, the signals that cost the most health, and a projection of when the score reaches a critical level.

`ts_shape.events.maintenance.asset_health_index` · [Engineering guide](../../guides/engineering.md) · [Maintenance guide](../../guides/maintenance.md) · [API reference](../../reference/ts_shape/events/maintenance/asset_health_index.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.maintenance.asset_health_index import AssetHealthIndexEvents

# three days at 10 min: vibration rises steadily, temperature and current stay put
kw = dict(n_points=432, freq="10min", start="2024-03-01")
df = pd.concat([
    ts_shape.make_timeseries(["vib:press-01"], baseline=2.0, noise=0.05, drift=3.0, **kw),
    ts_shape.make_timeseries(["temp:press-01"], baseline=50.0, noise=0.2, **kw),
    ts_shape.make_timeseries(["curr:press-01"], baseline=12.0, noise=0.1, **kw),
], ignore_index=True)

health = AssetHealthIndexEvents(
    df,
    asset_id="press-01",
    signals={
        "vib:press-01": "bearing",
        "temp:press-01": {"failure_mode": "thermal", "weight": 2.0},
        "curr:press-01": {"failure_mode": "electrical"},
    },
)
index = health.health_index(window="6h", baseline_window="12h")
print(index[["start", "asset_health_score", "dominant_failure_mode"]].tail(3))
alerts = health.health_events(window="6h", baseline_window="12h")
ranking = health.signal_contributions(window="6h", baseline_window="12h")
print(ranking[["signal_uuid", "failure_mode", "mean_health", "contribution_pct"]])
forecast = health.maintenance_forecast(window="6h", baseline_window="12h", critical_threshold=40.0)
print(forecast[["current_health", "decline_per_day", "recommended_maintenance_days"]])
```

The bearing (vibration) signal drives the loss of health, although temperature carries twice its weight.

## Methods

| Method | Returns |
|---|---|
| `health_index(window='1h', baseline_window='24h')` | per window: `asset_id`, `asset_health_score`, `dominant_failure_mode`, `worst_signal_uuid`, `signals_available`, `signals_expected`, `confidence_level` |
| `health_events(window='1h', baseline_window='24h', warning_threshold=70.0, critical_threshold=50.0, only_crossings=True)` | point events when the score enters `warning` or `critical`: `asset_health_score`, `severity`, `threshold`, `dominant_failure_mode` |
| `signal_contributions(window='1h', baseline_window='24h')` | one row per signal: `failure_mode`, `weight`, `mean_health`, `min_health`, `weighted_deficit`, `contribution_pct`, `rank` |
| `maintenance_forecast(window='1h', baseline_window='24h', critical_threshold=50.0, min_confidence=0.5)` | one row: `current_health`, `decline_per_day`, `recommended_maintenance_days`, `forecast_time`, `confidence` |

## Notes

- `signals` maps each signal uuid to a failure mode, or to `{"failure_mode": ..., "weight": ...}`; a bare string has weight 1. Weights must be positive.
- Each signal is scored per window against the opening `baseline_window` of its own data: up to a third of the score for mean drift, for growing variance and for the trend slope. Start the data at a known-good period.
- The asset score is the weighted mean of the signals present in the window. A signal without data in a window is left out rather than scored 0; `confidence_level` is the share of the configured weight that was present. `dominant_failure_mode` is the failure mode of the signal with the largest weighted health deficit.
- `health_events` reports only the window that enters a band by default; `only_crossings=False` reports every window below a threshold. `direction` is `below` when the band gets worse (and for repeat rows) and `above` for a recovery from `critical` to `warning` (then `threshold` is the `critical_threshold` the score rose above); a recovery to `ok` emits no row. `critical_threshold` must not exceed `warning_threshold`.
- `maintenance_forecast` fits a straight line through the score. `confidence` is the fit's R²; below `min_confidence`, and for a flat or improving score, the horizon columns are NaN.
- For one signal, [DegradationDetectionEvents](degradation-detection.md) gives the same kind of health score per sample.
