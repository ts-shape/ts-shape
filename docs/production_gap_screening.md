# Production modules — functional gap screening

A systematic screen of the 33 modules under `ts_shape/events/production/` for
**functional gaps**: real-world correctness bugs where a plausible industrial
input produces a silently wrong result. The reference case is the counter that
**resets to zero** at a shift/part change or PLC restart — a naive
`last - first` then loses all production in the affected window. That gap was
already fixed in `PartProductionTracking` (`part_tracking.py`); this screen
found the **same class of gap recurring across the pack**, plus several
adjacent correctness issues.

Every finding below was reproduced with a concrete input. Fixes are
**backward-compatible**: new behaviour is opt-in behind a keyword argument and
default output (values *and* column sets) is unchanged.

## Status legend

- ✅ **Fixed** in this change.
- 🔜 **Lined out** — verified and specified, scheduled as a focused follow-up
  (behaviour-changing interval/semantic redesigns kept out of this change so
  each ships with its own targeted tests).

---

## Theme A — Counter resets (the reference gap) ✅

A monotonic counter (parts, scrap, rejects, cumulative total) is turned into a
quantity via `max(0, last - first)` / `iloc[-1] - iloc[0]` / `max().diff()`
with no reset handling. A reset inside the window makes the delta negative,
which clamps to `0` and drops the post-reset production.

**Fix:** a shared helper `ts_shape.utils.counters` (`counter_quantity`,
`counter_increments`, `count_resets`) and an opt-in `handle_resets=True` on
each affected method. When enabled, a decreasing counter is treated as a reset
and the post-reset value is counted as new production; a `resets` column is
added so the resets are visible. Default (`False`) is unchanged.

| Module | Method(s) | Sev | Status |
|---|---|---|---|
| `part_tracking.py` | `production_by_part`, `daily_production_summary`, `production_totals`, `detect_resets` | high | ✅ (reference; refactored onto shared helper) |
| `oee_calculator.py` | `calculate_performance`, `calculate_quality` (→ `calculate_oee`) | high | ✅ |
| `line_throughput.py` | `count_parts` (→ `throughput_oee`, `throughput_trends`) | high | ✅ |
| `target_tracking.py` | `compare_to_target`, `target_achievement_summary` | high | ✅ |
| `scrap_tracking.py` | `scrap_by_shift`, `scrap_by_reason`, `scrap_cost`, `scrap_trend` | high | ✅ |
| `rework_tracking.py` | `rework_by_shift`, `rework_by_reason`, `rework_rate`, `rework_cost`, `rework_trend` | high | ✅ |
| `quality_tracking.py` | `nok_by_shift`, `quality_by_part`, `nok_by_reason`, `daily_quality_summary` | high | ✅ |
| `batch_tracking.py` | `batch_yield` | high | ✅ |
| `operator_performance.py` | `production_by_operator`, `quality_by_operator` | high | ✅ |
| `continuous_process_alignment.py` | `segment_by_cut` (numeric counter path) | high | ✅ |
| `shift_reporting.py` | `shift_production` (→ targets/comparison) | high | ✅ |
| `shift_handover.py` | `_counter_by_shift` (→ `generate_report`) | high | ✅ |
| `period_summary.py` | `_daily_counter` (→ weekly/monthly/compare) | high | ✅ |

**Also fixed under this theme (interleaving over-count):** in
`quality_by_part`, `scrap_by_reason`, and `rework_by_reason` a category
(part/reason) is attached to a *global* counter, then grouped and reduced with
`last - first`. When categories alternate, a group's first/last straddle the
other category's production and over-count. With `handle_resets=True` the
quantity is computed from per-reading increments *before* grouping, which fixes
both the reset under-count and the interleaving over-count.

---

## Theme B — Interval duration measured as sample-span, not to the next transition 🔜

State intervals are built as `end = last sample carrying this state`, so the
time from the last same-state sample to the actual transition is never counted.
On dense periodic data this undercounts each interval by one sample period; on
**change-encoded (`is_delta=True`) signals** — one row per transition, the
library's own convention — each state is a single row, so `start == end` and
**duration collapses to 0**. Existing tests pass only because their fixtures
insert a redundant trailing same-state sample.

| Module | Method | Sev | Effect on delta signals |
|---|---|---|---|
| `machine_state.py` | `detect_run_idle` (→ `state_quality_metrics`) | high | every duration `0`; `min_duration` filter drops all intervals |
| `micro_stop_detection.py` | `_intervalize` (→ detect/frequency/impact) | high | `total_micro_stop_time = 0`, `availability_loss_pct = 0` |
| `long_downtime_events.py` | `detect_long_downtime` | high | a real multi-hour stop is a single row → `duration 0 < min_gap` → missed |
| `duty_cycle.py` | `on_off_intervals` (→ `excessive_cycling`) | medium | all on/off durations `0` |
| `order_traceability.py`, `routing_traceability.py`, `multi_process_traceability.py` | `_detect_intervals` / dwell stats | medium | per-station dwell `0` for single-arrival events |

**Fix approach (follow-up):** close each interval at the *next transition's*
timestamp (`end = next_start`), leaving the final interval open (optionally
bounded by a `data_end`/`now` argument). Offer behind an opt-in kwarg (e.g.
`interval_end="next_change"`) so bracketed-data users are unaffected. A shared
`utils.intervals` helper analogous to `utils.counters` keeps it consistent.

**Genuine bug in the same family — fixed now:** ✅
`downtime_tracking.downtime_by_reason` computed `next_time` *after* filtering to
stopped rows, so each stop's duration ran until the *next stop* — booking the
intervening running time as downtime for that reason (and dropping the last
stop). Now the per-interval duration is computed on the full state timeline
*before* filtering, mirroring the already-correct `downtime_by_shift`.

---

## Theme C — Durations span data gaps (no max-gap cap) 🔜

A duration taken as `next_time - time` (or `diff()` between events) blindly
bridges logging gaps and inter-cycle stoppages.

| Module | Method | Sev | Concrete effect |
|---|---|---|---|
| `downtime_tracking.py` | `downtime_by_shift`, `availability_trend` | medium | a logger outage between a `Running` sample and the next reading is booked as uptime, inflating `availability_pct` |
| `cycle_time_tracking.py` | `cycle_time_by_part` (→ stats/slow/hourly) | medium | a 2 h idle gap between completions is reported as one 7200 s "cycle", skewing max/avg and false-flagging slow cycles |
| `performance_loss.py` | `performance_by_shift`, `slow_periods`, `performance_trend` | medium | a lunch break becomes one giant "cycle", so a 100 %-speed machine reports ~11 % performance and large speed loss |

**Fix approach (follow-up):** opt-in `max_gap` / `max_cycle_gap` kwarg
(default `None` = current behaviour); intervals exceeding it are excluded as
"no data" rather than attributed to the preceding state/cycle.

---

## Theme D — Sample-count average instead of time-weighted 🔜

`resample(window).mean()` over a state signal averages *sample counts*, correct
only under uniform sampling. On change-encoded/irregular data the result is
driven by how many rows land in the window, not by time.

| Module | Method | Sev | Concrete effect |
|---|---|---|---|
| `bottleneck_detection.py` | `station_utilization` (→ detect/shifting/summary) | medium | a station emitting one `True` per hour reads 100 % utilization; mis-ranks the constraint |
| `duty_cycle.py` | `duty_cycle_per_window` | medium | `On@09:00, Off@09:59` reads 50 % duty (two samples) instead of ~98 % |

**Fix approach (follow-up):** integrate the step function over the window
(weight each state by `next_time - time`, clipped to window bounds), consistent
with the time-weighted approach already used in `runtime_accounting` and
`oee_calculator.calculate_availability`. Offer as `method="time_weighted"`.

---

## Theme E — Boundary / edge-event handling

| Module | Method | Sev | Issue | Status |
|---|---|---|---|---|
| `target_tracking.py` | `compare_to_target` | high | overnight shift (22:00–06:00) grouped by raw calendar date; the group spans 00:00–23:59 with a hole, so `last - first` attributes the **whole day** to the night shift (≈750 % achievement) | 🔜 |
| `shift_reporting.py`, `shift_handover.py` | `_assign_shift` + calendar-date grouping | medium | an overnight shift is split across two calendar dates into two rows, dropping the midnight-crossing delta and double-comparing against the target | 🔜 |
| `changeover.py` | `detect_changeover` | medium | `series.ne(series.shift())` flags the first reading (vs `NaN`) as a changeover, over-counting the initial product | ✅ (opt-in `include_initial`, default preserves behaviour) |
| `line_balancing.py` | `balance_metrics` | medium | a fully stopped station has no cycle in the window and vanishes from the denominator, so a dead station reads as a perfectly balanced line (`efficiency 100 %`) | 🔜 |
| `alarm_management.py` | `alarm_frequency` | low | an alarm already active at the window start is counted as an activation | 🔜 |

**Overnight-shift fix approach (follow-up):** attribute an overnight shift's
post-midnight readings to the shift's *start* date (a "production day") so each
shift is one contiguous group, then apply the same Theme-A reset handling.
Shared across `target_tracking`, `shift_reporting`, `shift_handover` (identical
`_assign_shift`).

---

## Theme F — Division-by-zero / degenerate denominators ✅ / 🔜

| Module | Method | Sev | Issue | Status |
|---|---|---|---|---|
| `quality_tracking.py` | `daily_quality_summary` | medium | a zero-total day yields `0/0 = NaN` for every rate, inconsistent with `nok_by_shift` (which reports 0) | ✅ (guarded with the reset work) |
| `line_throughput.py` | `throughput_oee` | medium | a flat/stopped counter makes `target_rate = quantile(0.95) = 0`, so `count/target_rate = NaN` poisons every `oee_score` | ✅ |
| `shift_reporting.py` | `shift_targets` | low | a `target = 0` yields `achievement_pct = inf` (unlike the guarded sibling in `shift_handover`) | ✅ |
| `line_throughput.py` | `cycle_quality_check` | low | `expected_cycle_time = 0` (explicit or degenerate median) → inf/NaN deviation | 🔜 |

---

## Modules found robust

`flow_metrics.py` and `runtime_accounting.py` derive everything from boolean
rising edges / per-sample durations (no counter subtraction) and guard their
divisions. `setup_time_tracking.py` uses next-transition interval logic and
guards its denominators. `alarm_management.py` (beyond the low note above) and
`machine_state.transition_events` handle intervals, empties, and chattering
correctly. Empty/missing-signal handling returns documented columns
consistently across the pack.

---

## Follow-up plan (lined out)

1. **Theme B** — `utils.intervals` helper + opt-in `interval_end="next_change"`
   across `machine_state`, `micro_stop_detection`, `long_downtime_events`,
   `duty_cycle`, and the traceability dwell methods.
2. **Theme E overnight** — production-day grouping across `target_tracking`,
   `shift_reporting`, `shift_handover`.
3. **Theme C** — `max_gap`/`max_cycle_gap` caps on `downtime_tracking`,
   `cycle_time_tracking`, `performance_loss`.
4. **Theme D** — time-weighted option on `bottleneck_detection` and
   `duty_cycle.duty_cycle_per_window`.
5. **Theme E/F remainder** — `line_balancing` missing-station denominator,
   `alarm_management` initial-active flag, `line_throughput.cycle_quality_check`
   guard.
