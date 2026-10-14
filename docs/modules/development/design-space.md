# DesignSpaceEvents

> Fit a multivariate design space (qualified operating window) from development data and flag when commercial operation leaves it or comes close to its edge.

**Module:** `ts_shape.events.development.design_space`

---

## When to Use

The design space is the region of critical process parameters (CPPs) that development or characterization runs showed to deliver acceptable quality (ICH Q8 in pharma; equally useful in any process where several parameters interact). Use this class to:

- fit that region from qualification data, either as a per-axis box or as a convex hull that also captures correlation between the CPPs;
- find the intervals in production data where operation left it;
- warn about samples still inside the box but close to an edge.

For a single parameter use [Process Windows](../engineering/process-windows.md) instead.

---

## Quick Example

```python
from ts_shape.events.development.design_space import DesignSpaceEvents

cpps = ["gran1:inlet_air_temp", "gran1:spray_rate", "gran1:atomization_pressure"]

ds = DesignSpaceEvents(qualification_df, cpps)
ds.fit_box(quantiles=(0.05, 0.95))

excursions = ds.detect_excursions(operation_df)
near_edge = ds.boundary_proximity(operation_df, warn_margin=0.1)

# Tighter region that follows correlated CPPs
ds.fit_hull()
hull_excursions = ds.detect_excursions(operation_df)
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `fit_box(quantiles)` | static | `self`, with per-axis bounds at the given quantiles of the qualification data |
| `fit_hull()` | static | `self`, with a convex hull around the qualification points |
| `detect_excursions(operation_df)` | interval | `start`, `end`, `duration_seconds`, `uuid`, `excursion_mode` (`box`/`hull`) |
| `boundary_proximity(operation_df, warn_margin)` | point | `systime`, `uuid`, `signed_margin`, `closest_axis` |

## Notes

- The constructor's DataFrame is the qualification data; production data is passed to the detection methods. Both are pivoted to wide form on the timestamps of the CPP signals and forward-filled, so the CPPs should share a time base or be sampled densely.
- At least two `cpp_uuids` are required (`ValueError` otherwise). Call `fit_box()` or `fit_hull()` before detecting (`RuntimeError` otherwise); the last fit wins.
- `fit_box(quantiles=(0.0, 1.0))` uses the absolute min/max. The default (0.05, 0.95) trims outliers in the qualification data and is conservative.
- `fit_hull()` needs scipy and at least `len(cpp_uuids) + 1` qualification points in general position.
- `boundary_proximity` works only after `fit_box()`. The margin per axis is the distance to the nearer bound as a fraction of that axis's span; a sample is reported when its smallest margin is between 0 and `warn_margin`. Samples outside the box are not reported here; use `detect_excursions`.
- Excursion `start`/`end` are the first and last outside samples, so a single outside sample has zero duration.
