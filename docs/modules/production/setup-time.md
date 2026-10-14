# SetupTimeEvents

> Track and analyze setup/changeover durations for SMED analysis.

**Module:** `ts_shape.events.production.setup_time_tracking`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use for SMED (Single-Minute Exchange of Dies) analysis. Tracks setup durations and analyzes them by product transition to identify which changeovers take longest. Essential for lean manufacturing initiatives aimed at reducing changeover time and increasing available production time.

---

## Quick Example

```python
from ts_shape.events.production.setup_time_tracking import SetupTimeEvents

tracker = SetupTimeEvents(production_df)

# List every setup event with duration
# (state signal in value_string; rows equal to setup_value count as setup)
setups = tracker.setup_durations(state_uuid="line1:state", setup_value="Setup")

# Setup statistics grouped by product transition
by_product = tracker.setup_by_product(
    state_uuid="line1:state",
    part_id_uuid="line1:part_id"
)

# Overall setup time statistics
stats = tracker.setup_statistics(state_uuid="line1:state")

# Weekly trend of setup durations
trend = tracker.setup_trend(state_uuid="line1:state", window="1W")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `setup_durations(state_uuid, setup_value='Setup')` | List every setup event with start, end, and duration | `DataFrame` with `start`, `end`, `duration_minutes`, `date`, `shift` |
| `setup_by_product(state_uuid, part_id_uuid, setup_value='Setup')` | Setup time statistics grouped by product transition pair | `DataFrame` with `from_product`, `to_product`, `avg_minutes`, `min_minutes`, `max_minutes`, `count` |
| `setup_statistics(state_uuid, setup_value='Setup')` | Overall setup time statistics (single row) | `DataFrame` with `total_setups`, `total_minutes`, `avg_minutes`, `median_minutes`, `std_minutes`, `pct_of_available_time` |
| `setup_trend(state_uuid, setup_value='Setup', window='1W')` | Setup durations per time period | `DataFrame` with `period`, `avg_setup_minutes`, `setup_count`, `total_setup_minutes` |

---

## Tips & Hints

!!! tip "Separate internal and external setup tasks"
    SMED distinguishes between internal setup (machine stopped) and external setup (done while running). If your state signal encodes both, filter for internal-only to get the true changeover impact.

!!! info "Related modules"
    - [Changeover](changeover.md) — changeover event detection from state transitions
    - [Performance Loss](performance-loss.md) — speed losses that may occur after setup
    - [OEE Calculator](oee-calculator.md) — setup time feeds into availability losses

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/setup_time_tracking.md)
