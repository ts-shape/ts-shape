# ContinuousProcessAlignmentEvents

> Line up readings from stations along a continuous line (extrusion, rolling, film, paper, cable) on the same piece of material, by compensating each station's speed-dependent transport lag.

**Module:** `ts_shape.events.production.continuous_process_alignment`  
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

On a continuous line, a point of material passes the extruder first and the thickness gauge, the cooling bath and the cutter seconds or minutes later. Comparing readings at the same timestamp compares different material. Use this class when you know where each station sits (meters from a reference point) and have a line-speed signal. It:

- shifts each station reading back by `offset / speed` to a common `material_ref_time`;
- optionally assigns every aligned reading to the cut piece it ended up in;
- shows how the lag per station varies with speed;
- checks that the speed and station signals have data in every window.

---

## Quick Example

```python
from ts_shape.events.production.continuous_process_alignment import (
    ContinuousProcessAlignmentEvents,
)

line_config = [
    {"name": "extruder", "offset": 0.0, "uuids": ["ext1:melt_temp"]},
    {"name": "gauge", "offset": 12.5, "uuids": ["ext1:wall_thickness"]},
    {"name": "cutter", "offset": 40.0, "uuids": ["ext1:cut_length"]},
]

cpa = ContinuousProcessAlignmentEvents(
    df,
    "ext1:line_speed",
    line_config,
    speed_unit="m/min",
)

aligned = cpa.align_to_reference(station_uuids=["ext1:melt_temp", "ext1:wall_thickness"])
pieces = cpa.segment_by_cut(aligned, cut_length_uuid="ext1:cut_length")

cpa.lag_profile(window="1min")
cpa.alignment_quality(window="1h")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `align_to_reference(station_uuids, value_column)` | point | per reading: `material_ref_time`, `systime`, `uuid`, `component`, `position_offset_m`, `lag_seconds`, the value column |
| `segment_by_cut(aligned_df, cut_length_uuid, part_counter_uuid, cut_value_column, handle_resets)` | point | `aligned_df` plus `piece_id` (1-based), `piece_length_m`, `piece_cut_ref_time` |
| `lag_profile(station_uuids, window)` | summary | per window and station: `start`, `uuid`, `component`, `position_offset_m`, `mean_speed_m_s`, `lag_seconds` |
| `alignment_quality(station_uuids, window)` | summary | per window: `start`, `speed_sample_count`, `has_speed_data`, `has_full_coverage`, `per_uuid_counts` (dict) |

## Notes

- `line_config` is a list of dicts with `name`, `offset` (meters from the reference point) and `uuids`. UUIDs not listed there are ignored by every method.
- `speed_unit` is `"m/min"` (default), `"m/s"` or `"mm/s"`; anything else raises `ValueError`. Speed is clamped to at least `min_speed` (m/s, default 0.01) so a standstill does not give an infinite lag.
- `align_to_reference` uses the most recent speed reading at each station sample, read from the constructor `value_column` (default `value_double`), the same column `lag_profile` and `segment_by_cut` read speed from. A constant-speed approximation over the transport distance: with large speed changes during transit the lag is approximate.
- `segment_by_cut` needs `cut_length_uuid` (or the constructor `cut_uuid`) or `part_counter_uuid` (else `ValueError`). With a counter, a boolean True row is one cut and a numeric counter step of `n` is `n` cuts; `handle_resets=True` counts the value after a counter reset as new pieces instead of dropping them. Cut times are shifted back by the cutter's own lag when its UUID is in `line_config`. Material before the first cut is assigned to the first piece.
- `cut_uuid` is the default `cut_length_uuid` for `segment_by_cut`. `ref_uuid` has no effect; passing it emits a `DeprecationWarning` and it will be removed in 1.0.
