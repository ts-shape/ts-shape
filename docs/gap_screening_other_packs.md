# Quality, engineering, maintenance, energy, supply-chain, features & transform — functional gap screening

A follow-up to the [production-pack screen](production_gap_screening.md),
carrying the same lens to the packs it did not cover:
`events/quality`, `events/engineering`, `events/maintenance`,
`events/energy`, `events/supplychain`, `events/correlation`, `features/`
and `transform/`. The screen was run as the 2026-10-13 weekly maintenance
pass (correctness and edge cases, schema, test coverage).

Every finding below was reproduced with a concrete input. That input is
the regression test that ships with the fix.

Two kinds of fix:

- **Outright bugs** (no caller can rely on the old output): fixed directly.
- **Semantic improvements** (the old output is a defensible default on
  dense, uniform, single-signal data): **opt-in** behind a keyword
  argument, with default values *and* column sets unchanged — the same
  rule as the production screen.

The shared helpers from the production screen are reused, and two were added:

| Helper | Purpose |
|---|---|
| `utils.counters.counter_per_window` *(new)* | per-window counter quantity; legacy `max - min` by default, reset-aware with `handle_resets=True` |
| `utils.intervals.interval_end_times` *(new)* | per-row interval close time (`"last_sample"` / `"next_change"`) for code that builds its own segments |
| `utils.time_weighted.time_weighted_resample` *(new)* | zero-order-hold mean per window, optional `data_end` / `max_gap`, tz-aware |
| `utils.counters.counter_increments` | reset-aware per-reading increments |

## Status legend

- ✅ **Fixed** in this change.
- 📝 **Reviewed, left unchanged**, with a reason.
- 🔜 **Lined out**: verified, not fixed in this change.

---

## Phase 1 — outright bugs ✅

| Module | Method | Sev | Issue |
|---|---|---|---|
| `features/cycles/cycles_extractor.py` | every `process_*` | critical | `merge_asof` lost the end time in the join key: **every** cycle had `cycle_end = NaT`, `is_complete = False`. Also: persistent/trigger cycles opened on every True sample (now rising edges); same-row starts closed at themselves (zero-length); a held step opened several cycles; tz stripped via `.values`; other uuids leaked in. |
| `quality/tolerance_deviation.py` | `process_and_group_data_with_events` | high | data sorted newest-first before the forward-fill: each measurement got the **next, future** tolerance. Capability indices took the first-listed limits; a single tolerance gave `USL == LSL` (Cp ≡ 0); one measurement returned NaN. |
| `engineering/control_loop_health.py` | `_align` (all methods) | high | SP on the left of a `nearest` as-of join: a change-encoded SP dropped dense PV data (`error_integrals` empty) and PV could pick up a future setpoint. Now PV-left, SP held backward. |
| `engineering/disturbance_recovery.py` | baseline | high | same `nearest` look-ahead: samples just before a setpoint step read as deviations. |
| `maintenance/failure_prediction.py` | `remaining_useful_life` | high | an improving trend reported RUL **0** (imminent failure); NaN samples poisoned the next 10 fits. |
| `supplychain/lead_time_analysis.py` | `calculate_lead_times` | high | orders paired with deliveries by position: one out-of-order/missing delivery mismatched every later order. Now matched by order id when deliveries carry it; tz kept. |
| `energy/efficiency_tracking.py` | `efficiency_comparison` | high | summed raw cumulative counter readings (100, 101, 102 → 303 parts). |

## Phase 2 — timezone safety ✅

`series.values` on a tz-aware `systime` returns naive UTC `datetime64`, so outputs were shifted by the UTC offset and lost their tz. In some places this raised `TypeError` (`coverage_by_period`, `detect_level_shift`). Replaced with the tz-aware `.array` in
`data_gap_analysis`, `signal_quality`, `threshold_monitoring`,
`rate_of_change`, `warmup_analysis`, `degradation_detection`,
`inventory_monitoring`, and in production `flow_constraints`,
`part_tracking` and `continuous_process_alignment`.

| Module | Change |
|---|---|
| `transform/filter/datetime_filter.py` | naive bounds are interpreted in the column's tz (tz-aware columns raised `TypeError`); `inclusive=` on the between filters (default `"neither"` unchanged) |
| `transform/time_functions/timezone_shift.py` | `ambiguous=` / `nonexistent=` for DST wall-clock times (default `"raise"` unchanged); no longer mutates the caller's frame |
| `utils/base.py` | `DataQualityWarning` when the time column has NaT rows (rows kept — warn only) |

Tests use a new `berlin_signal` fixture (Europe/Berlin, DST days).

## Phase 3 — production themes applied to the other packs

### Theme A — counter resets / per-window deltas ✅ (opt-in `handle_resets`)

Per-window production was `max - min` of the readings inside the window. That loses the step across a window boundary, counts nothing for a single-reading window, and mis-counts a reset: `0, 50 | 70, 5, 25` gives 115 instead of 95.

`energy_per_unit`, `efficiency_trend`, `specific_energy_consumption`,
`enpi_by_window` / `enpi_vs_baseline` / `enpi_by_hierarchy`,
`carbon_intensity_per_unit`, `TimeGroupedStatistics(stat_method="diff")`.
Also `InventoryMonitoringEvents.consumption_rate(handle_refills=True)`: a refill no longer cancels the consumption (100 → 40 → 100 → 40 read as 0).

🔜 Energy meters are always summed as *interval* energy. A cumulative kWh register would need a `meter_type="cumulative"` mode on top of `counter_per_window`.

### Theme B — interval ends at the next change ✅ (opt-in `interval_end="next_change"`)

`multi_level_threshold`, `detect_out_of_range`, `detect_trend_degradation`,
`detect_variance_increase`, `detect_rms_exceedance`, `detect_steady_state`,
`detect_transient_periods`, `SegmentExtractor.extract_time_ranges`,
`detect_low_stock`, `correlation_breakdown`.
`time_above_threshold(interval_end="next_change")` measures the real time above the threshold: `0@10:00, 100@10:05, 0@10:55` gave 1200 s and now gives 3000 s.

### Themes C/D — gap caps and time weighting ✅

| Module | Change |
|---|---|
| `engineering/material_balance.py` | opt-in `time_weighted`; opt-in `missing_as_zero=False` (an outage is "unknown", not zero flow). **Fix:** zero input with output flowing read `balanced=True`; it is now `inf` / not balanced. |
| `energy/idle_energy_detection.py`, `energy/efficiency_tracking.py` | opt-in `time_weighted` state fraction with a left join. With a single change-encoded state row, 12 kWh had read as 4 kWh. |
| `engineering/control_loop_health.py` | opt-in `max_gap` on `error_integrals`: zero-order hold, and gaps longer than the cap are excluded. A 30-minute comms gap had been integrated as error. |

### Per-uuid grouping ✅

| Module | Change |
|---|---|
| `quality/outlier_detection.py` | opt-in `group_by_uuid`. **Fixes:** z-score ignores NaN (one NaN had disabled detection); when MAD = 0 it falls back to the mean absolute deviation instead of eps (eps flagged every value of a quantised sensor that differed from the median); IsolationForest skips NaN rows. |
| `transform/filter/boolean_filter.py`, `string_filter.py` | opt-in `group_by_uuid` on the edge and change filters: there was a false edge at uuid boundaries. |
| `features/time_stats/time_stats_numeric.py` | opt-in `group_by_uuid`. Docstrings now say `'h'` (pandas 3 rejects `'H'`). |

### Theme F — degenerate statistics ✅ / 📝

| Module | Method | Status |
|---|---|---|
| `quality/statistical_process_control.py` | `calculate_control_limits` | ✅ warns when σ is NaN or 0 |
| `quality/statistical_process_control.py` | `detect_cusum_shifts` | ✅ returns no shifts (with a warning) instead of flagging every deviation at `h·σ = 0` |
| `engineering/steady_state_detection.py` | `_rolling_std` | ✅ a window with fewer than 2 samples has no std; it had been filled with 0 and read as "steady" |
| `supplychain/demand_pattern.py` | `demand_by_period`, `seasonality_summary` | ✅ an empty period is no data, not zero demand; calendar periods (`'MS'`) no longer raise |
| `engineering/process_stability_index.py` | constant signal | 📝 a constant signal correctly scores 100 |
| `maintenance/degradation_detection.py` | σ = 0 baseline | 📝 penalties saturate, and noise after a constant baseline is a real change |

## Phase 4 — coverage (Pass 3) ✅

Tests were added for previously untested public API:
- SPC: rules 2/3/5–8, dynamic limits, `apply_rules_vectorized`, CUSUM, `interpret_violations`.
- Setpoint: rise time, decay rate, derivative settling, oscillation frequency and `control_quality_metrics`, checked against analytic step responses.
- Startup: multi-signal, adaptive, quality, phases and failed startups.

The coverage tests found three more bugs, all ✅:

| Module | Method | Issue |
|---|---|---|
| `engineering/setpoint_events.py` | `oscillation_frequency`, `overshoot_metrics` | a sample exactly on the setpoint (sign 0) counted a crossing twice, so a 40 s oscillation read as ~30 s |
| `engineering/startup_events.py` | `detect_startup_adaptive` | the baseline included the current sample and the test was `>=`, so a flat idle signal read as a startup at t0 and the real start was missed |
| `engineering/startup_events.py` | `track_startup_phases` | non-overlapping phases (ramp 10–90, run ≥ 90) never advanced |

## Lined out 🔜

- `startup_events.detect_startup_adaptive`: `baseline_window` is accepted but unused (the baseline is `lookback_periods` samples).
- `setpoint_events.overshoot_metrics`: `undershoot` is the minimum error, which on a rising step is always the initial deviation (100 %).
- `signal_quality.data_completeness`, `data_gap_analysis.coverage_by_period`: fixed window length on 23 h / 25 h DST days, and partial first/last windows.
- `signal_correlation.lag_correlation` / `rolling_correlation`: lags and windows count rows, not time, after `dropna()`.
- `anomaly_classification.detect_flatline`: starts one sample late; a deadband historian's single-row flatline is never detected.
- `gauge_repeatability.repeatability`: EV range taken across operators (AIAG uses per operator×part).
- `warmup_analysis._detect_ramps`: one flat or noisy sample splits a ramp.
- Energy `meter_type="cumulative"` (see Theme A).
