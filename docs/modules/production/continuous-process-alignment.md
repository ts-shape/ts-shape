# ContinuousProcessAlignmentEvents

Puts readings from stations along a continuous line (extrusion, rolling, film, paper,
cable) onto the same piece of material by shifting each station's reading back by its
speed-dependent transport lag, and cuts the aligned data into pieces at each cut.

`ts_shape.events.production.continuous_process_alignment` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/continuous_process_alignment.md)

## Example

```python
import pandas as pd
from ts_shape.events.production.continuous_process_alignment import (
    ContinuousProcessAlignmentEvents,
)

t = pd.date_range("2024-01-01 08:00", periods=10, freq="10s")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line:speed", "value_double": 60.0}),  # m/min
    pd.DataFrame({"systime": t, "uuid": "gauge:thickness", "value_double": range(10)}),
    pd.DataFrame({"systime": t[[3, 7]], "uuid": "cutter:length", "value_double": [3.0, 4.0]}),
], ignore_index=True)
line = [{"name": "gauge", "offset": 10.0, "uuids": ["gauge:thickness"]},
        {"name": "cutter", "offset": 20.0, "uuids": ["cutter:length"]}]

cpa = ContinuousProcessAlignmentEvents(df, "line:speed", line)
aligned = cpa.align_to_reference(station_uuids=["gauge:thickness"])
pieces = cpa.segment_by_cut(aligned, cut_length_uuid="cutter:length")
print(pieces[["systime", "lag_seconds", "piece_id", "piece_length_m"]])
```

At 1 m/s the gauge sits 10 s and the cutter 20 s behind the reference point. The cut
at 08:00:30 is material that passed the reference at 08:00:10, so gauge readings
whose material passed up to then (08:00:00–08:00:20 on the gauge) form piece 1; the
cut at 08:01:10 finishes piece 2. Readings whose material has not been cut yet have
`piece_id` = `<NA>`.

## Methods

| Method | Returns |
|---|---|
| `align_to_reference(station_uuids, value_column)` | per reading: `material_ref_time`, `systime`, `uuid`, `component`, `position_offset_m`, `lag_seconds`, the value |
| `segment_by_cut(aligned_df, cut_length_uuid, part_counter_uuid, value_column_cut, handle_resets)` | `aligned_df` plus `piece_id` (nullable integer, from 1), `piece_length_m`, `piece_cut_ref_time` |
| `lag_profile(station_uuids, window)` | per window and station: `mean_speed_m_s`, `lag_seconds` (NaN for a window without speed readings) |
| `alignment_quality(station_uuids, window)` | per window: whether speed and every station have data |

## Notes

- `line_config` lists stations as `{"name", "offset" (meters from the reference
  point), "uuids"}`. Signals not listed are ignored.
- `speed_unit` is `"m/min"` (default), `"m/s"` or `"mm/s"`. Speed is clamped to at
  least `min_speed` (m/s) so a standstill does not give an infinite lag.
- Each reading uses the last valid speed reading before it; missing speed readings
  are skipped, and readings before the first speed sample use the first one. The lag
  assumes constant speed over the transport distance.
- A cut finishes the piece whose material passed the reference point since the
  previous cut: aligned rows with `material_ref_time` in (previous cut, this cut] get
  this cut's `piece_id`. Material up to the first cut is piece 1; material after the
  last cut has no piece yet. Cut times are shifted back by the cutter's own lag when
  the cut signal is in `line_config`.
- With `part_counter_uuid`, a True bool or a counter step of n is n cuts at once;
  their material goes to the first of them. The first counter reading is the
  baseline, missing readings are gaps, and a counter reset counts the value after
  it (`handle_resets=True`).
- `ref_uuid` has no effect and is deprecated.
