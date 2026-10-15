# DesignSpaceEvents

Fits a multivariate design space (the region of critical process parameters that qualification runs showed to deliver acceptable quality, as in ICH Q8) as a per-axis box or a convex hull, and flags where production operation leaves it or comes close to its edge. For a single parameter use [ProcessWindowEvents](../engineering/process-windows.md) or [OperatingRangeEvents](../engineering/operating-range.md).

`ts_shape.events.development.design_space` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/development/design_space.md)

## Example

```python
import ts_shape
from ts_shape.events.development.design_space import DesignSpaceEvents

cpps = ["gran1:inlet_air_temp", "gran1:spray_rate", "gran1:atomization_bar"]
qualification = ts_shape.make_timeseries(cpps, n_points=500, freq="1min", noise=2.0, seed=1)
operation = ts_shape.make_timeseries(cpps, n_points=500, freq="1min", noise=2.0, drift=8.0, seed=2)

ds = DesignSpaceEvents(qualification, cpps).fit_box(quantiles=(0.05, 0.95))
print(ds.detect_excursions(operation).head()[["start", "end", "excursion_mode"]])
near_edge = ds.boundary_proximity(operation, warn_margin=0.1)
hull_excursions = ds.fit_hull().detect_excursions(operation)
```

## Methods

| Method | Returns |
|---|---|
| `fit_box(quantiles=(0.05, 0.95))` | `self`, with per-axis bounds at those quantiles of the qualification data |
| `fit_hull()` | `self`, with a convex hull around the qualification points |
| `detect_excursions(operation_df)` | intervals outside the space: `start`, `end`, `duration_seconds`, `excursion_mode` (`box`/`hull`) |
| `boundary_proximity(operation_df, warn_margin=0.1)` | samples inside the box but near an edge: `systime`, `signed_margin`, `closest_axis` |

## Notes

- The constructor's DataFrame is the qualification data; production data goes to the detection methods. Both are pivoted to wide form on the CPP timestamps and forward-filled, so the CPPs should share a time base or be sampled densely.
- At least two `cpp_uuids` are required (`ValueError` otherwise). Call `fit_box()` or `fit_hull()` before detecting (`RuntimeError` otherwise); the last fit wins.
- `quantiles=(0.0, 1.0)` uses the absolute min/max; the default trims outliers in the qualification data and is conservative.
- `fit_hull()` needs scipy and at least `len(cpp_uuids) + 1` qualification points in general position. It captures correlation between CPPs that a box cannot.
- `boundary_proximity` works only after `fit_box()`. The margin per axis is the distance to the nearer bound as a fraction of the axis span; a sample is reported when its smallest margin is between 0 and `warn_margin`. Samples outside the box are left to `detect_excursions`.
- Excursion `start`/`end` are the first and last outside samples, so a single outside sample has zero duration.
