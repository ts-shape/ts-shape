# ChangeoverEvents

> Detect product/recipe changes and compute changeover windows.

**Module:** `ts_shape.events.production.changeover`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use to track product/recipe changes on production lines. Detects when the product signal changes value and computes changeover duration. Essential for understanding SMED (Single-Minute Exchange of Die) improvements and tracking setup time as a component of planned downtime.

---

## Quick Example

```python
from ts_shape.events.production.changeover import ChangeoverEvents
import pandas as pd

# Long format: a product/recipe string signal
df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01 06:00", periods=360, freq="1min"),
    "uuid": "line1:product",
    "value_string": ["SKU-A"] * 120 + ["SKU-B"] * 100 + ["SKU-A"] * 140,
})

co = ChangeoverEvents(df)
changes = co.detect_changeover(product_uuid="line1:product", include_initial=False)
windows = co.changeover_window(
    product_uuid="line1:product",
    until="fixed_window",
    config={"duration": "10min"},
)
metrics = co.changeover_quality_metrics(product_uuid="line1:product")
print(changes)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_changeover(product_uuid, min_hold='0s', include_initial=True)` | Generate point events at each product/recipe change | DataFrame with `systime`, `uuid`, `source_uuid`, `is_delta`, `new_value` |
| `changeover_window(product_uuid, until='fixed_window', config=None)` | Compute changeover windows from each change point: a fixed duration (`config={'duration': '10min'}`) or until process metrics stabilize (`until='stable_band'`) | DataFrame with `start`, `end`, `uuid`, `source_uuid`, `is_delta`, `method`, `completed` |
| `changeover_quality_metrics(product_uuid)` | Compute changeover frequency and time-between-changeovers statistics per product | DataFrame with `product`, `changeover_count`, `avg_`/`min_`/`max_`/`std_time_between_seconds` |

---

## Tips & Hints

!!! tip "Combine with machine state"
    Pair changeover detection with `MachineStateEvents` to distinguish changeover downtime from unplanned downtime — changeovers typically show a planned idle pattern.

!!! info "Related modules"
    - [Machine State](machine-state.md) — run/idle detection to separate changeover idle from unplanned stops
    - [Batch Tracking](batch-tracking.md) — similar concept for batch-ID signals in process industries
    - [Downtime Tracking](downtime.md) — categorize changeover as planned downtime

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/changeover.md)
