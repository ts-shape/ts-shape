# ChangeoverEvents

Product or recipe changes as point events, and changeover windows that end after a fixed time or once process metrics are stable again, from a product/recipe signal.

`ts_shape.events.production.changeover` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/changeover.md)

## Example

```python
import ts_shape
from ts_shape.events.production.changeover import ChangeoverEvents

df = ts_shape.make_id_signal("line1:product", ["SKU-A", "SKU-B", "SKU-A"], hold=120, freq="1min")

co = ChangeoverEvents(df)
print(co.detect_changeover(product_uuid="line1:product", include_initial=False))
windows = co.changeover_window(product_uuid="line1:product", until="fixed_window",
                               config={"duration": "10min"})
metrics = co.changeover_quality_metrics(product_uuid="line1:product")
```

## Methods

| Method | Returns |
|---|---|
| `detect_changeover(product_uuid, min_duration='0s', include_initial=True)` | point events: `systime`, `new_value` |
| `changeover_window(product_uuid, until='fixed_window', config=None, fallback=None)` | one window per change: `start`, `end`, `duration_seconds`, `method`, `completed` |
| `changeover_quality_metrics(product_uuid)` | per product: `changeover_count`, `avg_`/`min_`/`max_`/`std_time_between_seconds` |

## Notes

- `include_initial=True` (default) emits the first reading as a changeover. Pass `False` to report only real value changes.
- `min_duration`: the new product must persist at least this long to count (`config["min_duration"]` for `changeover_window`). The former name `min_hold` still works and warns.
- `until="fixed_window"` ends the window at `start + config["duration"]` (default `10min`). `until="stable_band"` ends it when every metric in `config["metrics"]` (`{"uuid", "value_column", "band", "hold"}`) stays within `band` of its reference for `hold`; the reference is `config["reference_method"]`: `expanding_median`, `rolling_mean` (`rolling_window` points), `ewma` (`ewma_span`) or `target_value` (`target_values`).
- If the end cannot be found, `fallback` applies (default `{"default_duration": "10min", "completed": False}`), and its `completed` value is copied to the output.
- `changeover_quality_metrics` measures how often and how far apart changeovers to each product occur, not how long they take; it is empty with fewer than two changeovers.
