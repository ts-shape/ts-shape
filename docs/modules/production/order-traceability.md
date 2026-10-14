# ValueTraceabilityEvents

> Trace a shared identifier across multiple stations (serial number, order ID, batch code).

**Module:** `ts_shape.events.production.order_traceability`
**Guide:** [Traceability Guide](../../guides/traceability.md)

---

## When to Use

Use to track parts, orders, or batches as they move through production stations. Each station has its own UUID signal carrying the current identifier being processed; `station_uuids` maps each of those UUIDs to a readable station name. This module correlates those signals into a unified timeline showing where each item has been and how long it spent at each station.

---

## Quick Example

```python
from ts_shape.events.production.order_traceability import ValueTraceabilityEvents

tracer = ValueTraceabilityEvents(
    production_df,
    station_uuids={
        "line1:cutting:serial": "Cutting",
        "line1:welding:serial": "Welding",
        "line1:painting:serial": "Painting",
        "line1:assembly:serial": "Assembly",
    },
)

# Build full timeline per identifier per station
timeline = tracer.build_timeline()

# End-to-end lead time per identifier
lead_times = tracer.lead_time()

# Where is each item right now?
status = tracer.current_status()
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `build_timeline()` | Full timeline per identifier per station | `DataFrame` |
| `lead_time()` | End-to-end lead time per identifier | `DataFrame` |
| `current_status()` | Last known station for each identifier | `DataFrame` |
| `station_dwell_statistics()` | Dwell time statistics per station | `DataFrame` |

---

## Tips & Hints

!!! tip "Use consistent identifier formats"
    Ensure all stations emit the same identifier format (e.g., zero-padded serial numbers). Mismatched formats will cause the tracer to treat "Order-42" and "Order-042" as different items.

!!! info "Related modules"
    - [Routing Traceability](routing-traceability.md) — adds routing/state signal correlation
    - [Multi-Process Traceability](multi-process-traceability.md) — handles parallel paths and complex topologies
    - [Part Tracking](part-tracking.md) — single-station part tracking

---

## See Also

- [Traceability Guide](../../guides/traceability.md)
- [API Reference](../../reference/ts_shape/events/production/order_traceability.md)
