# QualityEvents

> Track NOK (defective) parts and quality metrics from OK/NOK counter signals.

**Module:** `ts_shape.events.production.quality_tracking`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use for quality tracking on production lines. Computes scrap rates, first-pass yield, and defect Pareto by reason code per shift. Pairs naturally with the OEE calculator's quality component and with part-level tracking for per-SKU quality analysis.

---

## Quick Example

```python
from ts_shape.events.production.quality_tracking import QualityEvents
import pandas as pd
import numpy as np

np.random.seed(42)
t = pd.date_range("2024-01-01 06:00", periods=480, freq="1min")
ok_counts = np.cumsum(np.random.choice([0, 1], size=480, p=[0.05, 0.95]))
nok_counts = np.cumsum(np.random.choice([0, 1], size=480, p=[0.97, 0.03]))

# Long format: one row per signal reading, identified by uuid
df = pd.concat([
    pd.DataFrame({"uuid": "line1:ok", "systime": t, "value_integer": ok_counts}),
    pd.DataFrame({"uuid": "line1:nok", "systime": t, "value_integer": nok_counts}),
    pd.DataFrame({"uuid": "line1:part_id", "systime": t,
                  "value_string": ["PN-100"] * 240 + ["PN-200"] * 240}),
    pd.DataFrame({"uuid": "line1:defect_reason", "systime": t,
                  "value_string": np.random.choice(
                      ["SCRATCH", "DIMENSION", "CRACK", "COSMETIC"], size=480)}),
], ignore_index=True)

qt = QualityEvents(df)
by_shift = qt.nok_by_shift(ok_counter_uuid="line1:ok", nok_counter_uuid="line1:nok")
by_part = qt.quality_by_part(
    ok_counter_uuid="line1:ok", nok_counter_uuid="line1:nok", part_id_uuid="line1:part_id"
)
by_reason = qt.nok_by_reason(
    nok_counter_uuid="line1:nok", defect_reason_uuid="line1:defect_reason"
)
daily = qt.daily_quality_summary(ok_counter_uuid="line1:ok", nok_counter_uuid="line1:nok")
print(by_shift)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `nok_by_shift(ok_counter_uuid, nok_counter_uuid)` | Compute NOK count and NOK rate per shift | DataFrame with `date`, `shift`, `ok_parts`, `nok_parts`, `total_parts`, `nok_rate_pct`, `first_pass_yield_pct`, `quality_pct` |
| `quality_by_part(ok_counter_uuid, nok_counter_uuid, part_id_uuid)` | Break down quality metrics by part number | DataFrame with `part_number`, `ok_parts`, `nok_parts`, `total_parts`, `nok_rate_pct`, `first_pass_yield_pct`, `quality_pct` |
| `nok_by_reason(nok_counter_uuid, defect_reason_uuid)` | Pareto of defect reasons by count | DataFrame with `reason`, `nok_parts`, `pct_of_total` |
| `daily_quality_summary(ok_counter_uuid, nok_counter_uuid)` | Daily summary with total parts, rejects, and quality rates | DataFrame with `date`, `ok_parts`, `nok_parts`, `total_parts`, `nok_rate_pct`, `first_pass_yield_pct`, `quality_pct` |

All methods accept `handle_resets=True` to count production correctly across counter resets; the result then gains a `resets` column.

---

## Tips & Hints

!!! tip "Use monotonic counters, not deltas"
    Pass cumulative OK and NOK counters — the module computes deltas internally. Pass `handle_resets=True` if the counters can reset. If your PLC provides per-cycle OK/NOK flags instead, convert them to cumulative sums first.

!!! info "Related modules"
    - [OEE Calculator](oee-calculator.md) — uses quality ratio as the third OEE factor
    - [Part Production Tracking](part-tracking.md) — production volumes to contextualize quality rates
    - [Batch Tracking](batch-tracking.md) — quality metrics per batch for process industries
    - [Cycle Time Tracking](cycle-time.md) — correlate slow cycles with quality defects

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/quality_tracking.md)
