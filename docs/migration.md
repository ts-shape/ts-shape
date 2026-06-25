# Migration & Deprecations

A running list of renames and deprecated aliases, so existing code keeps working
while you migrate. Deprecated names still function but emit a
`DeprecationWarning`; they may be removed in a future major release. The
authoritative, dated record is the [Changelog](changelog.md) — this page is the
quick old → new lookup.

## Deprecated function aliases

The event-log exporters were renamed into a consistent `to_event_log_*` family.
The old names remain as deprecated aliases.

| Deprecated | Use instead | Notes |
|---|---|---|
| `to_flat_df(...)` | `to_event_log_xes(...)` | XES-flavoured flat export. Signature unchanged. |
| `to_ocel_tables(...)` | `to_event_log_ocel(...)` | OCEL 2.0 `(events, objects, relations)` tables. Signature unchanged. |

```python
# Before
from ts_shape.eventlog import to_flat_df, to_ocel_tables

# After
from ts_shape.eventlog import to_event_log_xes, to_event_log_ocel
```

## Class aliases

| Alias (back-compat) | Canonical name | Notes |
|---|---|---|
| `OrderTraceabilityEvents` | `ValueTraceabilityEvents` | Same class; the alias tracks single-ID traceability that is no longer order-specific. New code should import `ValueTraceabilityEvents`. |

## Standardized detector output (May 2026)

Every public DataFrame-returning detector method now emits one of three
canonical shapes — `point`, `interval`, or `summary` — defined in
`src/ts_shape/events/_output.py`. If you previously read shape-specific column
names, switch to the canonical ones:

| Old column(s) | Canonical column |
|---|---|
| `start_time`, `gap_start`, `window_start`, `period_start`, `disturbance_start` | `start` |
| `end_time`, `gap_end`, `window_end`, `period_end`, `disturbance_end` | `end` |
| `severity_score`, `severity_level` | `severity` |
| `OEECalculator`: `date` | `start` + `end` + `duration_seconds` (86400) |

See the [Changelog](changelog.md) for the full dated history of breaking changes.
