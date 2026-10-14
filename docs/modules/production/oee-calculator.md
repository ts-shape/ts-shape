# OEEEvents

> Calculate Overall Equipment Effectiveness from timeseries signals: Availability x Performance x Quality.

**Module:** `ts_shape.events.production.oee_calculator`
**Guide:** [OEE Analytics](../../guides/oee-analytics.md)

---

## When to Use

Use for daily OEE reporting. Combines availability (from machine state), performance (from part counters vs ideal cycle time), and quality (from good/reject counters) into the standard OEE metric. World-class OEE is typically 85% or above; this module helps identify which of the three factors is the biggest loss.

---

## Quick Example

```python
from ts_shape.events.production.oee_calculator import OEEEvents
import pandas as pd
import numpy as np

# Long format: run state (bool), part counter and reject counter (integer)
np.random.seed(42)
times = pd.date_range("2024-01-01 06:00", periods=480, freq="1min")
df = pd.concat([
    pd.DataFrame({"systime": times, "uuid": "line1:running",
                  "value_bool": [True] * 400 + [False] * 30 + [True] * 50}),
    pd.DataFrame({"systime": times, "uuid": "line1:part_counter",
                  "value_integer": np.arange(1, 481)}),
    pd.DataFrame({"systime": times, "uuid": "line1:reject_counter",
                  "value_integer": np.cumsum(np.random.choice([0, 1], size=480, p=[0.97, 0.03]))}),
], ignore_index=True)

oee = OEEEvents(df)
availability = oee.availability(run_state_uuid="line1:running", planned_time_hours=8)
performance = oee.performance(
    counter_uuid="line1:part_counter", ideal_cycle_time=55.0, run_state_uuid="line1:running"
)
quality = oee.quality(total_uuid="line1:part_counter", reject_uuid="line1:reject_counter")
daily = oee.oee(
    run_state_uuid="line1:running",
    counter_uuid="line1:part_counter",
    ideal_cycle_time=55.0,  # seconds per part
    total_uuid="line1:part_counter",
    reject_uuid="line1:reject_counter",
    planned_time_hours=8,
)
print(daily[["start", "availability_pct", "performance_pct", "quality_pct", "oee_pct"]])
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `availability(run_state_uuid, planned_time_hours=None, max_gap=None)` | Compute daily availability from a boolean run-state signal (planned time defaults to the data span per day) | DataFrame per day with `start`, `end`, `duration_seconds`, `run_seconds`, `planned_seconds`, `availability_pct`, `excluded_seconds` (time `max_gap` removed as no data) |
| `performance(counter_uuid, ideal_cycle_time, run_state_uuid=None, handle_resets=True)` | Compute daily performance: actual output vs theoretical maximum at the ideal cycle time (seconds per part) | DataFrame per day with `actual_parts`, `ideal_parts`, `run_seconds`, `performance_pct` |
| `quality(total_uuid, reject_uuid, handle_resets=True)` | Compute daily quality: good parts divided by total parts produced | DataFrame per day with `total_parts`, `reject_parts`, `good_parts`, `quality_pct` |
| `oee(run_state_uuid, counter_uuid, ideal_cycle_time, total_uuid=None, reject_uuid=None, planned_time_hours=None)` | Compute daily OEE as Availability x Performance x Quality (quality is 100% without `total_uuid`/`reject_uuid`) | DataFrame per day with `start`, `end`, `duration_seconds`, `availability_pct`, `performance_pct`, `quality_pct`, `oee_pct`, then the basis: `run_seconds`, `planned_seconds`, `excluded_seconds`, `actual_parts`, `ideal_parts`, `total_parts`, `reject_parts`, `good_parts` (NaN without quality counters), `resets` |

---

## Tips & Hints

!!! tip "Validate each factor independently"
    Always check availability, performance, and quality separately before looking at the combined OEE number. A single factor can mask improvement in the others.

!!! info "Related modules"
    - [Machine State](machine-state.md) — provides the run/idle signal for availability
    - [Line Throughput](line-throughput.md) — provides part counts for performance
    - [Quality Tracking](quality-tracking.md) — provides OK/NOK counts for quality
    - [Micro-Stop Detection](micro-stops.md) — uncovers hidden availability losses

---

## See Also

- [OEE Analytics Guide](../../guides/oee-analytics.md)
- [API Reference](../../reference/ts_shape/events/production/oee_calculator.md)
