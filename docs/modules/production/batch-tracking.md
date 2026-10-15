# BatchTrackingEvents

Batches, their durations and yields from a batch-ID string signal. A batch is a run of constant batch-ID values; a new batch starts when the value changes.

`ts_shape.events.production.batch_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/batch_tracking.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.production.batch_tracking import BatchTrackingEvents

df = pd.concat([
    ts_shape.make_id_signal("line1:batch_id", ["B-001", "B-002", "B-001"], hold=60, freq="1min"),
    ts_shape.make_timeseries(["line1:part_counter"], n_points=180, freq="1min",
                             baseline=0, noise=0, drift=360, value_column="value_integer"),
], ignore_index=True)

bt = BatchTrackingEvents(df, batch_uuid="line1:batch_id")
print(bt.detect_batches()[["batch_id", "start", "end", "duration_seconds"]])
yields = bt.batch_yield(counter_uuid="line1:part_counter")
matrix = bt.batch_transition_matrix(layout="wide")
```

## Methods

| Method | Returns |
|---|---|
| `detect_batches()` | one row per batch: `batch_id`, `start` (first sample), `end` (last sample), `duration_seconds`, `sample_count` |
| `batch_duration_statistics()` | per batch ID: `count`, `min_`/`avg_`/`max_`/`total_duration_seconds` |
| `batch_yield(counter_uuid, handle_resets=True)` | per batch: `batch_id`, `start`, `end`, `duration_seconds`, `quantity` (+ `resets`) |
| `batch_transition_matrix(layout='long')` | `from_state`, `to_state`, `transition_count`; `layout='wide'` gives the from/to pivot table with a `total` column |

## Notes

- `batch_yield` with `handle_resets=True` books every reset-aware counter increment to the batch that was running when it was produced: a reading reports the production since the previous reading, so it belongs to the batch whose ID was last seen strictly before it. Production between the last sample of one batch and the first sample of the next counts for the earlier batch, and the batch quantities add up to the counter total.
- Readings before the first or after the last batch-ID sample, or while the batch ID is empty, belong to no batch.
- `handle_resets=False` keeps `max(0, last - first)` over each batch's `[start, end]`.
- With fewer than two batches the transition matrix is empty (long layout keeps its columns).
