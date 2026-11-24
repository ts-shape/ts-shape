---
tags: [xes]
---

# CycleProfileEvents

Mapping-free cycle time profiling of one PLC. Give it every raw signal of a machine controller; it finds the cycle by itself, learns the standard sequence of a cycle, splits each cycle into segments and answers two questions per product variant:

1. **What must be optimized first to shorten the cycle time?** The contribution register ranks segments by their *potential* (median minus best demonstrated P10).
2. **What is unstable within a cycle, and how is the variation distributed?** The same register reports each segment's spread, skew, bimodality and share of the cycle time variance.

No role mapping, no master or planning data: signal names are used for labels only.

`ts_shape.events.production.cycle_profiling` · [Cycle time by part](cycle-time.md) · [API reference](../../reference/ts_shape/events/production/cycle_profiling.md)

## Example

```python
import ts_shape

# A simulated PLC: step chain, handshakes, counter, article, pressure, heartbeat.
frame, truth = ts_shape.make_plc_cycles(n_cycles=300, seed=7)

profile = ts_shape.CycleProfileEvents(frame)        # no uuid given
profile.evaluation_window()        # enough coherent cycles per variant?
profile.cycle_anchor_candidates()  # which event cuts the cycle, and why
profile.standard_view()            # learned sequence per variant
register = profile.contribution_register()
register.sort_values(["variant", "potential_rank"])          # question 1
register.sort_values("variance_contribution_factor", ascending=False)  # question 2
profile.detect_cycle_influencers()  # why long / short cycles happen
profile.cycle_records()             # one record per cycle, one timestamp
```

## From the loaders: two inputs, no mapping

`ts_shape.profile_cycles(timeseries, metadata=None)` takes exactly what the ts-shape loaders return:

- `timeseries`: the raw frame of `AzureBlobParquetLoader` (`load_by_time_range` or `load_files_by_time_range_and_uuids`; `systime` as Unix nanoseconds is fine);
- `metadata` (optional): the datapoint table of a metadata loader (`MetadataJsonLoader(...).to_df()`), used for **labels only** (node path, else display name, else label).

Signals matching `Worker` are dropped by default (personal data). The result holds every table and a summary of plain numbers; `save()` writes CSV + `summary.json`.

<!-- no-run -->
```python
from ts_shape import profile_cycles
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

loader = AzureBlobParquetLoader(sas_url=SAS_URL, prefix="<plc folder>/")
metadata = MetadataJsonLoader.from_file("datapoints.json").to_df()
raw = loader.load_files_by_time_range_and_uuids(start, end, list(metadata.index))
result = profile_cycles(raw, metadata)
result.save("profile/")
```

Command line (credentials from `--sas-url` / `AZURE_STORAGE_SAS_URL` or `AZURE_STORAGE_CONNECTION_STRING`):

```bash
python -m ts_shape.cycle_profile --container <name> --prefix <plc folder>/ --start 2026-11-17T00:00Z --end 2026-11-17T23:59:59Z --metadata-json datapoints.json --out profile/
```

Dashboard (numbers and tables of one result; sources: result folder, Azure, simulated PLC):

```bash
streamlit run examples/cycle_profiling/dashboard.py
```

## Pipeline

| Step | What happens | Method |
|---|---|---|
| 1. Signals → events | The kind inferred by `infer_spec` decides: boolean edges `uuid↑`/`uuid↓`, entry into a state or text value `uuid=v`, counter increment `uuid+1`, analog crossings of a 40/60 % hysteresis band of its P10..P90 range `uuid>`/`uuid<`. With `analog_levels=k` (or `"auto"`, 2–4 levels by the largest relative drop of the within-level variance) a 1-D k-means learns the operating levels and entering a level is the event `uuid≈180`. | `signal_events` |
| 2. Anchor | Every event with ≥ 3 occurrences is ranked by *support* (number of other signals with an event count within ±10 %), counter first, robust CV of its intervals, name. A heartbeat is regular but has no partners, so it never wins. `anchor_uuid` overrides. Cycles are cut with `CycleExtractor`. | `cycle_anchor_candidates` |
| 2a. Support | A partner supports a candidate when its count is within 0.8–1.5 × the candidate's and it is **phase locked** to it (mean resultant length of its phase within the candidate's intervals ≥ 0.7). A second machine on the same controller runs at a similar rate but drifts, so it does not count; a step skipped on an alternative route still does. Candidates covering < 50 % of the best candidate's time span (a burst at shift start) are ranked last (`coverage_ratio`). | `cycle_anchor_candidates` |
| 3. Variant | A text or integer-state signal with ≥ 2 values that changes in ≤ 20 % of the cycles; its value at cycle start. The candidate with the highest η² of log(duration) wins if η² ≥ 0.2 (computed on the bulk of the cycles, without pauses and outliers). | `evaluation_window` (`variant_uuid`) |
| 4. Classes | Per variant on log(duration), median *m* and scaled MAD *s* (floor 1 %): `pause` at ≥ `pause_factor` × median (or above `max_duration`), `long` at z > `n_sigma`, `short` at z < −`n_sigma` (or below `min_duration`), else `normal`. | `cycle_time_statistics`, `cycle_records` |
| 5. Readiness | Non-pause cycles are stitched; a variant is evaluable with ≥ `min_cycles` (50) of them. | `evaluation_window` |
| 6. Standard view | Events that occur exactly once in ≥ `presence_ratio` (90 %) of a variant's non-pause cycles, ordered by median phase. Events closer than the *time resolution* form one milestone (see Polled signals). | `standard_view` |
| 7. Segments | Anchor → first view event → … → last view event → next anchor, so the segments add up exactly to the cycle. A view event counts at its first occurrence; a missing or out-of-order view event makes a *sequence deviation* (no segments, counted in the register). | `segment_durations` |
| 8. Register | Per variant × segment: median, P10, P90, spread, MAD, mean, std, skewness, Sarle's bimodality coefficient (`is_bimodal` above 5/9), share of the median cycle, potential = median − P10 with its rank, variance contribution = Cov(segment, cycle) / Var(cycle). Both come with a 90 % bootstrap interval (200 cycle resamples, fixed seed: `potential_ci_*_seconds`, `variance_contribution_ci_*_factor`). | `contribution_register` |
| 9. Influencers | Per cycle: count of every event, mean and max of every analog signal, state of every boolean at cycle start. `long_vs_normal` and `short_vs_normal` are scored with the Mann-Whitney AUC: importance = \|AUC − 0.5\| × 2. Events that fire more than twice per cycle on average (a heartbeat) only measure the duration and are left out. Analog means and maxima are taken over the same window in every cycle (from the start for the variant's P10), so a cycle that only waits longer does not change them. Every feature gets a Mann-Whitney `p_value`, Benjamini-Hochberg `q_value` over all tested features and `is_significant` (q < 0.05). | `detect_cycle_influencers` |
| 10. Records | One point per cycle at its end: `start_ns`, `end_ns`, duration, class, variant, deviation from the variant median, the segment that lost the most time. | `cycle_records`, `detect_unusual_cycles` |

`ts_shape.cycle_profile.profile_cycles` adds the structure of the cycle on top:

| Table | What it holds |
|---|---|
| `cycle_events` | every event with its cycle, `offset_seconds` and `phase_ratio` (event heatmaps) |
| `recurring_events` | events present in ≥ 10 % of cycles at ≤ 2 per cycle: the rows of a heatmap |
| `sequence_variants`, `cycle_sequences` | the order of the recurring events per cycle (events within one time resolution grouped, events near the cycle boundary left out), counted as routes; `main_sequence_share_ratio` |
| `process_graph` | directly-follows graph of the main events (▶ cycle start … ■ cycle end) with count, share and median transition time |
| `parts_per_cycle` | per counter signal: increment per cycle (median, mean, share of cycles with the median) |
| `parallel_lanes` | regular events at the anchor's rate (0.8–1.25 ×) that are *not* phase locked to it (< 0.5): a second machine; profile it separately |
| `curve_bands`, `curve_scores`, `curve_flagged` | per analog signal the shape over the cycle on 64 phase points: median band, shift-tolerant deviation (±2 points) and a discord score (distance to the nearest other cycle); a cycle is flagged above a robust bound and an absolute floor |
| `cycle_time_check` | when the controller reports its own cycle time (found by name-free matching of a value signal to the cycle durations, units s/ms/min): agreement per cycle |

`ts_shape.cycle_profile.case_log` turns `cycle_events` and `cycle_records` into XES-style columns (`case:concept:name`, `concept:name`, `time:timestamp`, …) for pm4py and other process-mining tools.

`summary["confidence"]` collects the self-checks: anchor support, full-sequence share, cycle-time signal agreement, main route share, parts per cycle, independent lanes, shape anomalies, significant influencers.

## Quality: benchmark against ground truth

`benchmark_profiles(seeds, n_cycles)` profiles `BENCHMARK_SCENARIOS` of `make_plc_cycles` with known truth and returns one row per scenario and seed: base, polled (13 s), heartbeat, 4 parts per cycle, alternative route, curve-shape faults, a parallel lane, a reported cycle time. Seeds 11, 23, 37 at 250 cycles:

| Check | Result |
|---|---|
| anchor, route, parts per cycle, lane, cycle-time signal | correct in every scenario |
| cycles matched to the truth | 100 % |
| long / short / pause precision and recall | 1.0 |
| duration error | exact on change-encoded data, mean 0.7 s at 13 s polling |
| curve-shape faults | precision 0.92–1.0, recall 0.86–0.93 |

`tests/test_cycle_profile_quality.py` holds these as regression thresholds.

## Polled signals

First real run (Desma 2K press 356, one day, 124 signals polled every 13 s): the anchor is found without configuration (injection counter), the derived cycle durations match the press's own reported cycle time to a median of 4 s, the 6 breaks are kept as pauses, and 97 % of the cycle time variance lies in the segment from end of dosing to the next shot.



Many OPC UA clients poll instead of subscribing (the Desma pilot reads every signal every 13 s). Events of one poll then arrive in random order, and every duration is known only to about one poll. The profile handles this:

- `time_resolution` (default: the median sample period of the standard view's sampled signals, 0 for change-encoded data) is reported as `time_resolution_seconds` in `evaluation_window`.
- Standard-view events within one resolution form a **milestone** (`standard_view.milestone`) that counts at its earliest event; segments run between milestones, so poll jitter does not turn every cycle into a sequence deviation.
- `is_bimodal` needs a spread of more than two resolution steps: two values one poll apart are quantization, not two modes.
- A potential below the resolution is polling noise. For segment work, subscribe to the step and state signals on change.

The anchor ranking breaks ties on the interval CV at two decimals and then prefers the event with more occurrences: a shift counter that resets (and so skips an increment) loses to the part counter.

## Reading the register

- **`potential_seconds`** is time a segment loses against its own best demonstrated performance (P10). It is what can be gained without new equipment, so `potential_rank` 1 is where to start.
- **`variance_contribution_factor`** sums to exactly 1 per variant. A segment with a small median but a large share (a re-inspection, a waiting time) is where the cycle becomes unstable. A negative share means the segment compensates others (a buffer that absorbs upstream delays).
- **`is_bimodal`** marks a segment with two typical durations (a conveyor that is sometimes not ready, two process paths): look for the condition that switches between them.
- **`deviation_cycle_count`** cycles were left out because their sequence differed (aborted cycles, a variant with another step chain).

## Spec traceability

The requirements come from the cycle time profiling notes (OneNote exports in the work vault, consolidated in the *Cycle Time Profiling* spec note). Every ID has an acceptance test in `tests/test_cycle_profiling.py` (`test_<id>_...`).

| ID | Requirement | Implemented by |
|---|---|---|
| US-1 | Find the unknown cycle signal automatically, only from raw data. | `cycle_anchor_candidates`, `anchor_uuid=None` |
| US-2 | Detect short **and** long cycles statistically; KPIs transferable across plants. | robust log z-score per variant; `cycle_time_statistics`, `detect_unusual_cycles`; `CycleTimeEvents.detect_unusual_cycles`, relative `stable_band_ratio` |
| US-3 | Report long/short populations with state information that explains why. | `detect_cycle_influencers`, `dominant_segment` in `cycle_records` |
| DQ-1 | One start signal; its recurrence is the stop. | anchor event, `CycleExtractor.process_trigger_cycle` |
| DQ-2 | Pauses handled separately; subprocesses that ran differently distinguished. | `pause` class (kept, not dropped); sequence deviations |
| DQ-3 | 50–100 coherent cycles, otherwise optimize organizationally first; cycles may be stitched. | `evaluation_window` |
| DQ-4 | v1: boolean signals and user min/max cycle time. | **Extended** (decision): limits are derived; `min_duration` / `max_duration` are optional plausibility bounds; int, counter, text and analog signals are used too. |
| LL-1 | Classify signals (analog vs. nominal), enough data per variant. | `infer_spec` kinds in `signal_events`; variant detection; everything per variant |
| OUT-1 | All outputs of a cycle share one timestamp; top-10 influencers. | `cycle_records` (`systime` = cycle end); `detect_cycle_influencers(top_n=10)` |
| VIZ-1 | Time axis, pauses shown separately, target vs. actual. | `cycle_records` (classes incl. `pause`); `target_seconds` = P10 (best demonstrated) instead of a master-data target |
| VSM-1 | Station sequence, event pairing, bottleneck. | Event pairing within one PLC: `segment_durations`. The cross-station VSM is the next step (see Limits). |

## Limits

- **Parts per cycle** comes from counter increments; a machine without a part counter is profiled per cycle only.
- **Parallel lanes** are detected and listed, not split: profile each lane with `signal_uuids` or `anchor_uuid`.
- **Shape anomalies** are statistical: a flagged curve is unusual, not necessarily bad. Synthetic recall is ~0.9; on polled data a curve has few points per cycle.
- **A lone signal** (no partners) can still be the anchor (support 0); with several unrelated regular signals, name the cycle signal with `anchor_uuid`.
- **The whole frame** takes part by default: to profile one PLC out of a plant export, pass its uuids as `signal_uuids`.
- **Validation** on real machines so far: one Desma press (anchor and cycle time agree with the press's own cycle-time signal, r = 0.94). More machines with known truth are needed for a field accuracy figure.
- **VSM-1** (ordering stations by lead/lag of their anchors, bottleneck, buffers) builds on one profile per station and is not part of this class yet.

## Synthetic data

`ts_shape.make_plc_cycles(n_cycles, seed, ...)` simulates such a PLC with known disturbances and returns `(frame, truth)`: camera NOK with re-inspection (long cycles), aborted cycles (short), breaks (pauses), a slower article (variant) and a bimodal conveyor wait; optionally `parts_per_cycle`, `reported_cycle_time`, `alt_sequence_probability`, `shape_fault_probability`, `parallel_lane` and `poll_period`. Use it to try the profiler and as the ground truth of its tests.
