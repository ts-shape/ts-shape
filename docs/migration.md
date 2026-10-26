# Migration and deprecations

Renames, deprecated aliases and behavior changes. Deprecated
names still work and emit a `DeprecationWarning`; they are removed only in a
future major release. To find every deprecated name your code still uses, run
your tests with `-W error::DeprecationWarning`. The dated record is the
[Changelog](changelog.md).

## October 2026, round 24: materials that change once a day

Results change for `identify_objects` types of materials and their
neighbours, and for nesting (`parent_uuid`); screening is unchanged.

- A parent that changes only one to three times in the child's span (a
  material changed once a day in a continuous process) nests a batch or lot
  when every one of its changes meets a value change of the child (within
  5 s, on the windows the sampling leaves open), the product of the
  changes' chances times the number of child-parent pairs tried is at most
  1 % (a Bonferroni count over the pairs that could reach it, Tarone's
  correction), no batch straddles one of its changes, and the child has
  twice its distinct values and 6+ changes. Before, at least 3 matched
  changes were needed.
  - It also counts when the material reads as no type, or as a lot or batch.
  - It is found through the work order the batch lies in when the batch is
    polled too coarsely.
  - A finer id that changes at every change of a recipe, just as unlikely by
    chance, takes the recipe's place.
- An id whose every change falls on a shift boundary or midnight (within
  5 s, local wall clock) nests nothing: a day order or date code meets a
  batch made per shift or day by the clock.
- No production parent for a tiny pool with at most 3 changes, nor for an
  id that another order or batch nests in (a work order read as a batch).
- Types: a unique id over two levels of finer unique ids (a material over
  its work orders and batches) +0.45 for material, article and recipe; the
  reported parent of a batch or lot +0.35 for them and -0.30 to -0.40 for
  tool, fixture, work order, batch, lot, order and package (no longer
  `unrelated`); a plain number of 6+ digits -0.25 for recipe, tool and
  fixture; a batch that holds a batch -0.40. New facts `tops_unique_chain`,
  `holds_batches`, `long_number` in `type_scores` and `explain()`.
- Still undecided: a material that never changes (a constant), parts
  directly in a material, and one change over coarsely polled batches.
- Benchmark: type accuracy held-out 0.891 -> 0.898, never-seen 0.890 ->
  0.903; nesting held-out 0.95 / 0.95 -> 0.98 / 0.95 (never-seen 0.99 /
  0.94 unchanged), materials polled every 30 min 0.82 / 0.82 -> 0.84 /
  0.82. With materials changed about once a day (new opt-in generator
  option `material_changes_per_day`) the edges into a material go from
  0.83 / 0.68 to 0.93 / 0.93 held-out; the new benchmark test requires
  0.85 / 0.85. New opt-in option `continuous_process` ("lot", "serial").
  Default generator data are unchanged.

## October 2026, round 23: nesting for coarsely polled materials

Results change for `identify_objects` nesting (`parent_uuid`,
`packing_parent_uuid`) only; screening, types and `attribute_of` are
unchanged.

- Each change is also tested on the window its sampling leaves open: a value
  seen at `t` changed after the sample before it. A parent change meets a
  child change when the two windows overlap (within 5 s); each change's chance
  is the share of time a window as wide as its own would meet one, and the
  matches are tested against those chances (at least 3 matches, lift 0.6+,
  at most 1 % by chance).
  - A value written on change and re-sent every few minutes or hours
    (keep-alive): a read off the re-send grid is a change seen when it
    happened. Before, its change was taken to be up to 15 min late.
  - A material polled every 15 to 60 minutes: its change, seen up to that
    long after the batch change, still nests the batch when the windows meet
    far more often than chance. The containment test reads the same windows.
- A parent that holds the child in the strong sense needs only 5 % by chance:
  no child presence straddles one of its changes, and the child has four
  times as many distinct values (a material pool over its batches).
- Of a serial's equally preferred parents (batch, lot), one that itself nests
  in a material or recipe comes first: a package or a lot on its own clock no
  longer beats the batch for being finer.
- Still undecided: a material polled every 30 min over batches shorter than
  that (any half hour holds a batch change), and a material that changes only
  once or twice in the recording.
- Benchmark nesting precision / recall: held-out 0.93 / 0.93 -> 0.95 / 0.95,
  never-seen 0.96 / 0.89 -> 0.99 / 0.94; with every material polled every 30
  min (new opt-in generator option `material_poll`) held-out 0.54 / 0.50 ->
  0.82 / 0.82. The benchmark test now requires 0.92 / 0.92 (was 0.90 / 0.90)
  and 0.80 / 0.80 on the coarse-material split.

## October 2026, round 23: traceability leftovers

Results change for numeric item ids and MultiProcess handovers where noted.

- **Numeric ids:** an item id that is an integral number renders without `.0`: `1001.0` is `"1001"` in `RoutingTraceabilityEvents` (`item_id` of `build_routing_timeline` / `lead_time`, `oid` of `to_intervals`) and in `MultiProcessTraceabilityEvents.handover_log`. Before, these gave `"1001.0"` while `to_intervals` of the other classes gave `"1001"`. A float `1001.0` and the text `"1001"` on one Routing ID signal are now one item. Event ids of these rows change. The `item_id` / `identifier` columns that keep the signal's dtype (Value and MultiProcess timelines, lead times, `parallel_activity`) are unchanged.
  - In `to_event_log`, the `serial` objects of every traceability log are `"1001"` (before: `"1001.0"` except for `to_intervals`). Event ids are unchanged.
- **Parallel cells:** a handover can name the cell it leaves, `{"uuid": ..., "from_station": "Weld", "to_station": ..., "from_id_uuid": "w1"}` (or a list of cells). Without it, a `from_station` shared by several cells takes the latest reading of any cell, as before, and `handover_log()` raises a `DataQualityWarning` naming the station. A `from_id_uuid` that is not a cell of `from_station` raises `ValueError`.
- **Unknown stations:** a handover whose `from_station` is not in `processes` raises a `DataQualityWarning` (once per call); it still gives no rows.
- **`source_uuid`:** added as the last column of MultiProcess `build_timeline` (the `id_uuid`), `lead_time` (the cells' `id_uuid` in visit order, comma-joined), `parallel_activity` (the two visits' `id_uuid`, comma-joined) and `handover_log` (the handover signal). `handover_log` events now link the handover signal as `asset`. Event ids are unchanged: the new `LabelRule.eid_omit` leaves the column out of the id.

## October 2026, round 22: business partners and nesting recall

Results change for `identify_objects` typing of order attributes and for its
nesting (`parent_uuid`); screening is unchanged.

**Business partners**

- A recurring pool that changes only with an order outside production is that
  order's partner, also when the order is held for minutes or hours (before,
  only one-sample stamps or orders arriving at random counted). The host must
  read as an order (`sales_order`, `purchase_order` or the `order` group), 0.15
  ahead of every other type. The pool gets +0.40 for `business_partner` and
  -0.30 for recipe, material and article.
  - A sales order 0.15 ahead of a purchase order adds +0.20 for `customer`, a
    purchase order +0.20 for `supplier`. Otherwise the group
    `business_partner` is reported, as before.
  - Not for a pool with production structure of its own (it holds batches or
    serials, or changes with a production id), one that changes at cycle rate,
    or one that changes at fewer than a quarter of the order's changes (a tool).
  - Before, such a pool (e.g. `CUST-…` on `SO-…` orders) was typed `material`
    or `recipe`. A work order with no batch or serial signal that reads as an
    order keeps its recipe a recipe unless the order type is clear.
- New business-partner prefixes: customer `CU-`, `KND`, `KUNDE`, `DEB`;
  supplier `SUPP`, `SUPPL`, `VDR`, `V-`, `LIEF`, `KRED` (`CUST`, `KD`, `SUP`,
  `VEND`, `LF` as before). `CU-` and `V-` count only with a separator after the
  letters. The weight is 0.25, as for every prefix.
- `type_scores` and `explain()` show the new facts (`attribute_of_order`,
  `attribute_of_sales_order`, `attribute_of_purchase_order`).
- Benchmark type accuracy: 0.871 → 0.891 held-out, 0.880 → 0.890 never-seen.

**Nesting**

- A batch's material that is mistyped on its own values now nests the batch.
  - A tool or fixture that nests the batch both ways counts as its material or
    recipe (a tiny pool held for days).
  - An order, package, batch or lot that nests the batch both ways and holds
    another such id is the material (it holds the work order).
  - Neither applies to an id with another order or batch inside it.
- Of parents of the same level for a batch, after the finer id, the one whose
  values read like a material wins: its format and prefix evidence, a number
  of 6+ digits against a number of up to 3 digits or a `_`-segmented name.
  Before, a material and a recipe that changed together one to one were
  separated by presence count and uuid. The package-first rule no longer
  applies to batch parents.
- A serial that holds other serials (4× the presences, both ways) and nests in
  a material, recipe or order (lift 0.7+) is a batch for the nesting: batches
  shorter than 30 minutes now hold their serials. Its reported type is
  unchanged.
- A serial whose batch is polled too coarsely to show the alignment goes into
  that batch when the batch nests in another id the serial verifiably lies in
  (a farther level, a setup resource, a package, or a mistyped order or
  material). Before, only a verified production parent served this way.
- Benchmark nesting precision / recall: held-out 0.83 / 0.77 → 0.93 / 0.93,
  never-seen 0.93 / 0.77 → 0.96 / 0.89. The benchmark test now requires 0.90 /
  0.90 (was 0.80 / 0.75), and type accuracy 0.87 (was 0.85).

## October 2026, round 21: traceability

Results change for the traceability classes where noted.

- **Event logs:** `MultiProcessTraceabilityEvents.lead_time`, `parallel_activity` and `handover_log` now convert with `to_event_log`. `lead_time` and `parallel_activity` get `start`, `end` and `duration_seconds`; `handover_log` gets `systime`. These are copies of `first_seen` / `last_seen` / `lead_time_seconds`, `overlap_*` and `timestamp`, added at the end; the other columns are unchanged.
  - `parallel_activity` and `handover_log` link the serial and both stations; `lead_time` links the serial.
  - `LabelRule.object_columns` takes a tuple of columns.
- **Lead time:** `lead_time()` of `ValueTraceabilityEvents` and `MultiProcessTraceabilityEvents` ends at the latest visit end. Before, it used the end of the last-started visit. X at A [0, 20] and B [5, 10] now has 1200 s, not 600 s. This now matches `object_kpis`.
- **Routing gaps:** in `RoutingTraceabilityEvents`, a step sample with no item (a missing or empty id) or no step value is no visit, and it ends the visit before it.
  - Before, the visits around such a sample merged.
  - Before, `""` was an item and a missing step value was the station `"State nan"`.
- **Handovers:** `MultiProcessTraceabilityEvents.handover_log` takes the item read just before the trigger. An id written at the trigger instant is the next item arriving, not the transferred one. A missing handover reading holds the previous value, so `1, NaN, 1` is one transfer.
- **Timestamps:** rows without a timestamp no longer crash the Routing timeline or `handover_log`; they are left out. Readings at the same timestamp keep their input order.
- **Timeline log attributes:** the `MultiProcessTraceabilityEvents.build_timeline` event log carries the station in `ts_shape:lifecycle_state` and `step_sequence` in `ts_shape:direction`. Before, these were the literal texts `station_name` / `station_sequence`. Event ids are unchanged.
- **Missing signals:** `MultiProcessTraceabilityEvents` warns about `processes` and `handovers` signals that are not in the data.
- **Misreads (opt-in):** new `exclude=` on the three classes, the same as `ObjectSpec.exclude` (`"misreads"`, values, or a callable). The default `None` keeps every reading. `"misreads"` is planned to become the default in the next major release.

## October 2026, round 20 (part N): object analytics (new)

New API; results of existing calls are unchanged except where noted.

- **Hand-off from `identify_objects`:**
  - `report.to_event_log(dataframe, events=...)` runs the whole chain in one call: specs, then `detect_objects` with `report.hierarchy()`, then `attach_objects`.
  - `report.hierarchy()` returns signal-level parent edges for `detect_objects(hierarchy=...)`.
  - `to_object_specs(namespace=True, attributes=True)` gives `"{type}:{value}"` ids and fills `ObjectSpec.attributes`.
  - New report column `attribute_of`, appended at the end of the table: the identifier a signal describes, e.g. the customer of a sales order.
- **Presence events:** `presence_events(intervals)` gives enter/leave events per object presence. Convert them with `to_event_log(..., detector="eventlog.presence_events")`, so a serial-centric XES log has one trace per serial.
- **Object KPIs (`ts_shape.eventlog`):**
  - `object_kpis`: lead time, dwell per station, throughput per window.
  - `wip_over_time`: mean and max WIP per window, DST-safe.
  - `carrier_loops`: loop count and loop times.
  - `attach_counts`: counter increments per object, split pro rata over presences.
- **Genealogy:**
  - `where_used`, `where_from` and `lineage`, as functions and as `EventLog` methods.
  - `EventLog.filter_by_object_type` and `EventLog.filter_by_time`, which return sub-logs with no dangling references.
  - New `ts_shape.ObjectNotFoundError` (a `KeyError`).
  - The `o2o` table has no validity time, so time windows use object presence from the lifecycle rows.
- **Traceability classes:**
  - Event logs from `ValueTraceabilityEvents`, `RoutingTraceabilityEvents` and `MultiProcessTraceabilityEvents` now link each event to its serial and station objects. Event ids are unchanged; `objects={"serial": None}` turns the links off.
  - New `to_intervals()` on all three classes returns presences in the `object_intervals` columns plus `station`, so they feed `object_kpis`, `presence_events` and the other analytics.
  - New `interval_end="last_sample" | "next_change"` on the timeline methods. The default, `last_sample`, keeps today's results; `next_change` holds a visit until the next reading, like `object_intervals`.
  - `MultiProcessTraceabilityEvents.build_timeline` output now converts with `to_event_log` (before, it raised).
- **Behaviour change:** the `detect_objects` lifecycle no longer merges touching presences at different stations. A serial that leaves st1 at 10 and enters st2 at 10 now has `active` 0, `active` 10, `released` 20; before, it had `active` 0, `released` 20. Overlapping presences, and touching ones at the same station, are still merged.

## October 2026, round 20: correct object handling

Results change for event logs with objects, for `identify_objects` nesting
and for text run states.

**Event to object links** (`attach_objects`, `link_events_to_objects`,
`relate`)

- An interval event links to every object present during part of it
  (`ts_shape:start_timestamp` to `ocel:timestamp`, half-open). Before, only
  the objects present at the event's end were linked. A run spanning two
  batches now links to both; an object that starts exactly at the event's
  end is no longer linked. Point events link as before.
- New `scope={signal_uuid: label}`: an event links only to objects whose
  signal has the same label (line, station) as the event's source signal.
- Without `scope`, a `DataQualityWarning` fires when an event links to more
  than one object of a type present at the same time, which is typical for
  data from several lines or stations. Pass `scope=` or
  `fanout_threshold=None` to silence it.
- `link_events_to_objects` and `relate` warn when one table is tz-naive and
  the other tz-aware (the naive side is read as UTC, as before).
- Temporal E2O rows are ordered by event, then object.

**Misread ids** (`ObjectSpec.exclude`, default `"misreads"`)

- PLC misreads are no longer objects: blank text, text without letters or
  digits (`????????`), PLC defaults (`0`, `-1`, `32767`, `65535`, …, as
  numbers or text), `NOREAD`, `ERR`, `#N/A`, `NULL`, `NONE`, filler like
  `XXXXXXXX`, and short truncated reads. The filter is
  `ts_shape.utils.id_codes.is_misread`.
- A misread between two readings of the same id no longer splits the
  presence (`A, NOREAD, A` is one presence).
- Pass `exclude=None` for the old behaviour, or a collection of values or a
  callable for your own rule.

**Object-to-object relations** (`detect_objects`, `derive_o2o`)

- Without a hierarchy the default o2o is empty; before, every overlapping
  pair of types got a `co_occurs` row. Use `infer_o2o="co_occurs"` (or
  `derive_o2o(..., co_occurs=True)`) to get it back.
- `hierarchy=` also takes signal-level edges `{child_uuid: parent_uuid}`
  or `{child_uuid: [parent_uuid, ...]}`, so a serial can sit in a batch and a
  handling unit, and two lines stay apart.

**`identify_objects` nesting**

- `parent_uuid` is the production parent and is set only for serials,
  batches and lots: a serial's parent is its batch (before often its handling
  unit), a batch's parent is its material or recipe. Work orders, materials
  and recipes no longer get a parent.
- New column `packing_parent_uuid` (between `parent_uuid` and `evidence`):
  the handling unit or container a serial is packed in. `explain()` and
  `summary()` say "packed in …".
- Nesting on the benchmark: precision/recall 0.83/0.77 on held-out seeds and
  0.93/0.77 on never-seen seeds, up from below 0.2. Screening and typing are
  unchanged.
- Ids like `T3-0000123` no longer crash `identify_objects` (an out-of-range
  date parse is "not a date").

**`MachineStateEvents`**

- New `run_states={...}`: the values of a text state column that mean
  running. A text state column without `run_states` or `value_range` raises a
  `DataQualityWarning`, because the bool cast counts every non-empty text,
  including `"IDLE"`, as running. Results without `run_states` are unchanged.

## October 2026, round 19 follow-up: `setup` group

- `identify_objects` reports `setup` when a recipe, a tool and a fixture
  score within 0.15 of each other. This happens for a tiny pool of codes held
  for hours that changes with no other id: without a work order or a format
  cue, the data cannot tell a recipe from a tool or a fixture.
- Before, the best guess among the three was reported with a low confidence.
- A recipe tied to a work order, and a tool or fixture with a format cue,
  are still reported as their own type.
- `hints` and `register_identifier_pattern` separate them as before.

## October 2026, round 19: object-identifier signals (new)

New, plus pandas 3.1 compatibility (below); no other result changes.
`SignalSpec`, `infer_spec` and `profile()` are unchanged. See
[Object identification](guides/object-identification.md).

- **pandas 3.1: timestamp resolution.** The energy loaders (`systime` from
  epoch numbers or text), `parse_utc_times` and
  `TimestampConverter.convert_to_datetime` now always return
  `datetime64[ns, tz]`, as documented. pandas infers the resolution itself
  and it differs by version (2.2: ns; 3.0: the epoch unit, `us` for text;
  3.1: at least `us`). Unchanged on pandas 2.2; on 3.0 the unit was `s`,
  `ms` or `us`. Values are unchanged.
- **pandas 3.1: sub-microsecond seconds.** pandas 3.1 keeps nanoseconds in
  `Timedelta.total_seconds()` (earlier versions drop them). The vectorized
  `duration_seconds` (and similar) columns follow the installed pandas, so
  they equal the per-row results on every version. With pandas 3.1, for
  nanosecond timestamps only, they can differ from 3.0 by under 1 µs.

- `ts_shape.identify_objects(dataframe)` returns an `ObjectReport` with one
  row per uuid:
  - **Stage 1, screening:** the `role` (identifier, state, free text,
    measurement, counter, timestamp, constant, insufficient data) and an
    `identifier_score`.
  - **Stage 2, identification:** the `object_type`, with `type_confidence`
    and a runner-up type.
  - **Supporting columns:** the screening features, the detected
    `parent_uuid` (nesting), and an evidence trail. `explain(signal_uuid)`
    prints the evidence for one signal.
- **What it looks at.** It reads only the values and how they behave over
  time: formats with check digits (GS1, GTIN, SSCC, EPC, SAP widths, date
  codes), cardinality, holds, revisits, recurrence, and relations between
  signals (nesting, co-changes, alignment with counters and shifts). It never
  reads uuid names or metadata. `hints={uuid: type}` pins a type.
- **Types.** Types that cannot be told apart without a format cue are
  reported as a group: `order` (sales or purchase order) and
  `business_partner` (customer or supplier). A clear identifier of unknown
  type is reported as `identifier`.
- **Extending it.** `register_identifier_type` and
  `register_identifier_pattern` add site-specific types and formats.
- **Into the event log.** `ObjectReport.to_object_specs()` returns the
  `ObjectSpec`s for `detect_objects` / `to_event_log`.
- **Benchmark.** `ts_shape.make_object_signals(seed=...)` generates a
  realistic labelled frame. `tests/test_identify_benchmark.py` scores the
  classifier on the held-out seeds.

## October 2026, round 18: Databricks open ranges, memory

### Results that change

- `DatabricksUnityParquetLoader.fetch_data_as_dataframe` with only `start`
  or only `end` floors the bound to the folder granularity of
  `hour_pattern`: the hour with `{H}`, otherwise the day, month or year.
  With a `{Y}/{m}/{d}/` layout, a mid-day `start` now reads that day's
  folder, which was skipped before. Closed ranges and the Azure loaders are
  unchanged.

### Memory and speed (results unchanged)

- **Detectors that read only their own signals** keep just those rows in
  `detector.dataframe`. These are the classes that read only the signals
  named in their constructor: `MachineStateEvents`,
  `ThresholdMonitoringEvents`, `SignalQualityEvents` and 33 others.
  - **What stays the same:** results, warnings, `get_dataframe()` and
    `repr()`. `tests/invariants/test_signal_rows_equivalence.py` checks all
    330 registry methods.
  - **Savings:** on 10M rows with 5 signals, the memory used on top of the
    input drops from about 0.42 to about 0.25 GB on pandas 3, and the time
    roughly halves.
  - **If you read `detector.dataframe` directly** on these classes, it now
    holds only the detector's own signals. Use `get_dataframe()` for the
    whole processed frame. It is built from your input on the first call,
    so changes you make to that input after construction show up in it.
  - **Data that keeps the full-frame path:** some data still goes through
    the previous path, with the same memory use as before:
    - text numbers;
    - nullable or pyarrow numeric/boolean columns;
    - a time column that is not datetime;
    - uuids that are not text;
    - a requested uuid that is missing from the data.
- **`to_event_log`** builds event ids and activities in chunks, and
  `schema.validate` checks Arrow-string ids without a hash table.
  - **What stays the same:** event ids, activities and every table are
    byte-identical, including dtypes.
  - **100M-row frame:** with the full 6-column frame, run/idle or
    transition events (about 3M events) peak at about 7.4 GB, down from
    7.8–9.0 GB.
  - **Speed:** large logs are about 1.5–2 times faster.
  - **One event per input row:** a log with one event per input row still
    needs about 300 B per event, mostly in the output's string columns.

## October 2026, round 17 (part J): loaders, transforms, features

Each change is pinned by a hand-computed test in
`tests/reference/test_round17_production_power.py`,
`tests/reference/test_round17_features_filters.py` and
`tests/reference/test_round17_loaders.py`.

### Performance loss and energy

- `PerformanceLossEvents.slow_periods`, `performance_trend` and
  `performance_by_shift` book each cycle, from the previous count to this
  one, to the windows or shifts it ran in. Its time is cut at the boundaries
  and its parts are shared in proportion to that time.
  - `run_seconds` never exceeds the window.
  - `actual_parts` can be fractional.
  - The false slow period after a long cycle is gone.
- `PowerIntegrator.energy_by_window`, `segment_energy` and
  `to_energy_deltas` use true calendar bounds for `'W'`, `'MS'` and `'ME'`.
  `'ME'` and `'W'` used to give negative kWh. `'1D'` is the real local day.
- `PowerIntegrator.energy_by_window(total=True)` counts a meter with no data
  in a window as coverage 0.

### Transforms

- `DataHarmonizer.resample_to_uniform` and `merge_multi_signals`:
  - New keyword `sampling="first"|"interpolate"`.
    - `"first"` is the default, as before: the first sample in each bin,
      labeled with the bin start.
    - `"interpolate"` gives the value at each grid timestamp.
  - Interpolation no longer extends a signal before its first or after its
    last sample. Those rows are now NaN.
- `IntegerFilter.filter_value_integer_not_match` and
  `StringFilter.filter_value_string_not_match` keep missing values for
  nullable dtypes, as documented.
- `TimezoneShift.revert_to_original_timezone` on a naive column raises a
  `ValueError` that points to `shift_timezone`.

### Features

- `TimeWindowedFeatureTable`:
  - `compute_long` reports true window bounds for right-labeled windows. The
    `'ME'` window labeled 2024-01-31 is now `[2024-01-01, 2024-02-01)`.
  - `compute` has a row for every window between the first and last sample,
    NaN-filled where no data is available.
- `CycleExtractor`: a cycle already running at the first non-missing sample
  has `is_complete=False`.
- `CrossSignalAnalytics.granger_causality`:
  - New `p_value_adjusted`, the Bonferroni-adjusted minimum p-value.
  - `is_causal` now uses it. On independent noise the false-positive rate
    drops from about 15 % to about 2 %.
  - `p_value` is unchanged.
- `NumericStatistics.column_sum` and `column_range` on integer columns no
  longer wrap around int64.

### Loaders

- **Naive bounds:** `trim_to_time_range` and every loader that trims rows
  read naive bounds as UTC, as the folder selection always did. To select
  by local time, pass aware bounds.
- **`S3ProxyDataAccess`:** reads floored UTC hour folders and drops rows
  outside `[start, end]`. With no files it returns an empty frame.
- **`ParquetLoader`:** `load_by_time_range` and
  `load_files_by_time_range_and_uuids` drop rows outside `[start, end]`.
  Pass `trim_to_range=False` to get whole hour folders.
- **Hour patterns without `{H}`:** an `hour_pattern` without `{H}` no longer
  returns each row once per hour. This covers the Azure parquet and flexible
  loaders and the Databricks Unity and Spark loaders.
- **`AzureBlobEnergyLoader` and `DatabricksUnityEnergyLoader`:**
  - They drop rows outside a `start`/`end` pair; a date-only `end` keeps that
    whole day.
  - Azure `load_by_series_ids` accepts a single open bound.
  - Databricks day folders are UTC days.
- **`value_double`:** always `float64`, in the energy loaders and in
  `EnergyAPILoader`.
- **`parse_utc_times`:** infers the epoch unit from magnitude (seconds below
  1e11, milliseconds below 1e14, microseconds below 1e17, else nanoseconds),
  or takes `unit=`.
- **`TimescaleDBDataAccess.fetch_data_as_parquet`:** no longer writes an
  `hour` column.
- **`MetadataJsonLoader`:** raises in strict mode when the index maps' keys
  differ. With `strict=False` missing fields are filled. Mixed digit and text
  keys sort with the digits first.
- **`DatapointAPI`:** accepts datapoints without `enabled` or `config`. An
  unknown device gets an empty frame.

## October 2026, round 17 (part I): event log, pipeline, signals

Each change is pinned by a hand-computed test in
`tests/reference/test_round17_eventlog.py` and
`tests/reference/test_round17_pipeline_signals.py`.

### Event log

- Event ids now include the objects bound with `objects=`. The same detector
  row bound to asset M1 and to asset M2 gives two events, so `concat` of
  per-asset logs no longer merges them. Ids of logs built without `objects=`
  are unchanged; ids of logs built with `objects=` change.
- `to_event_log_xes(lifecycle="two_row")` puts `start` before `complete` when
  both have the same timestamp.
- `concat` keeps object changes that differ only in `ocel:type`, and
  relations that differ only in `ocel:qualifier`. `filter_by_pack` and
  `filter_by_object` keep only the changes of the `(oid, type)` objects they
  keep.
- Severity follows each detector's own scale. The new
  `LabelRule.severity_thresholds` sets the warn and critical levels.
  - `detect_outliers_isolation_forest` is warn from a score of 0.6 and
    critical from 0.7. Before, it was always `info`.
  - `detect_outliers_iqr` is warn from 1.5 and critical from 3.0 IQR beyond
    the fence. Before, the levels were 3.0 and 4.5.
- `severity_level` ignores case and surrounding blanks, so "High" maps like
  "high".
- Rows with a missing event timestamp are dropped with a
  `DataQualityWarning`. The timestamp is `systime` for point rows and `end`
  otherwise.
- `to_event_log_xes` on an empty log returns an empty XES frame instead of
  raising.

### Context

- `ValueMapper` reads mapping-file keys as text. "007" and "7" are different
  keys, and a file key "1.0" no longer matches the code 1. Integer,
  integral-float and text codes still match the key "1".

### Pipeline

- `init__<name>=` and `method__<name>=` send an argument to the constructor
  only or to the method only. A plain name that both accept still goes to
  both. Watch `LongDowntimeEvents.count_events_between_gaps` and
  `ContinuousProcessAlignmentEvents.align_to_reference`, where `value_column`
  means different things on each side.
- Class-method and static-method detect steps carry a `"Class.method"`
  identity and appear in `to_event_log()`. Unknown arguments on them now
  raise `ValueError` when the pipeline is built.
- A plain-callable step whose arguments name its first parameter (for
  example `dataframe="$prev"`) gets the frame by keyword.
- Every detect step gets a private copy of the working frame, and every use
  of `$input` is a fresh copy. A step that changes its input in place no
  longer affects later steps.
- `Pipeline.run_stream`: each detect step sees the data transformed only by
  the transforms before it, as in `run()`. Before, it saw the data after all
  transforms.

### Signals

- Inference decides sampled or change-encoded by the share of intervals
  within ±10 % of the median. At least 80 % means sampled. This replaces the
  mean/std CV, so a regular signal with a few long gaps stays sampled and
  `profile()` reports its gaps.
- A slow monotonic counter with at least 3 increases is a counter. Two-level
  and constant signals never are. A slow rise over 16 or fewer integer levels
  stays a state; pass `kind="counter"` to override.
- `windows()` with `"W"` or `"ME"` gives real calendar windows: Monday to
  Monday weeks, and months from the 1st. The first window contains `start`.
- `intervals()` no longer holds the last value to `data_end` across more than
  `max_gap`. The last interval then ends at its last sample, as in
  `per_window`.
- The primitives take `uuid_column=` (default `"uuid"`). A frame without that
  column whose timestamps repeat now raises `ValueError`.

## October 2026, round 16: DST-day windows, more chunkable methods

Each change is pinned by a hand-computed test in
`tests/reference/test_round16_dst.py`. `tests/invariants/test_dst_windows.py`
checks every summary method on both DST changes.

### Results that change (DST days on tz-aware data only)

On a DST change a daily window is 23 h (82800 s, spring) or 25 h (90000 s,
fall) long. These methods measured it as a fixed 24 h; they now use the real
window. Windows that do not cross a DST change are unchanged, bit for bit.

- Time shares, percentages and rates:
  - `ThresholdMonitoringEvents.time_above_threshold`: `time_above`,
    `time_above_seconds`, `pct_above` and `time_above_pct`, in both
    `interval_end` modes. A signal that is always above the threshold on
    2024-03-31 in Europe/Berlin reads 82800 s and 100 %. Before, it read
    86400 s, or 95.8 % when the data ended at midnight.
  - `ThresholdMonitoringEvents.threshold_exceedance_trend`:
    `total_duration(_seconds)`.
  - `MicroStopEvents.micro_stop_frequency`: `pct_of_window` and
    `share_of_window_pct`.
  - `DutyCycleEvents.duty_cycle_by_window(time_weighted=False)`: `on_time`
    and `off_time`.
  - `InventoryMonitoringEvents.consumption_rate`: the rate per hour.
  - `FlowMetricsEvents.throughput` and `flow_summary`:
    `throughput_per_hour`.
  - `DisturbanceRecoveryEvents.disturbance_frequency`: `pct_time_disturbed`
    and `time_disturbed_pct`.
- `end` is the next local midnight and `duration_seconds` the real length,
  instead of start + 24 h. Affected:
  - `FlowMetricsEvents.throughput`, `flow_summary` and `wip_over_time`
  - `BottleneckDetectionEvents.detect_bottleneck`
  - `AssetHealthIndexEvents.health_index`
  - `MultiSensorValidationEvents.detect_disagreement`
  - `SensorDriftEvents.detect_zero_drift`
  - `MaterialBalanceEvents.detect_balance_exceedance`
  - `LineBalancingEvents.station_cycle_times` and `balance_metrics`
  - `PerformanceLossEvents.slow_periods`
  - `ControlLoopHealthEvents.detect_oscillation`
- Membership follows the real day:
  - `SensorDriftEvents.detect_zero_drift`, `detect_span_drift` and
    `calibration_health` look up the reference over the real day. A 23 h
    day no longer includes reference samples from the next day's first
    hour.
  - `ControlLoopHealthEvents.loop_health_summary` counts oscillation
    episodes in the real window (`oscillation_count`).
  - `DutyCycleEvents.excessive_cycling` averages the intervals in the real
    window (`avg_on_duration` / `avg_off_duration`).
- `DisturbanceRecoveryEvents.disturbance_frequency` also assigns
  disturbances to windows by the real day.

### Chunked processing

- `ts_shape.stream.CONDITIONAL` (`"Class.method" -> Condition(argument,
  reason)`) lists methods that run chunked only with one argument:
  - `DataGapAnalysisEvents.detect_gaps` and `coverage_by_period` need
    `min_gap=...`.
  - `AnomalyClassificationEvents.detect_flatline` needs
    `interval_end="last_sample"`.
  - `ChangeoverEvents.changeover_window` works, but not with
    `until="stable_band"`.

  Without it, `NotChunkableError` names the argument to pass.
- Calendar chunks: `chunk="W"` (Monday to Monday; also `"W-<DAY>"`) and
  `chunk="MS"`, cut at local midnight.
  `PeriodSummaryEvents.weekly_summary` and `monthly_summary` are chunkable
  with `chunk="W"` and `chunk="MS"`.
- `MicroStopEvents.micro_stop_frequency` is chunkable. Empty windows between
  slices are filled in on merge.
- `iter_chunked` yields all `detect_gaps` rows, and the last
  `coverage_by_period` window, after the last slice. They need the whole
  run's sample count or median sampling step.
- Fix: chunked `ChangeoverEvents.detect_changeover` with `min_duration`,
  `changeover_window` with a `config` `min_duration`, and
  `MicroStopEvents.micro_stop_impact` now match the in-memory result when an
  event crosses a slice border.
- `DataGapAnalysisEvents.interpolation_candidates` is no longer chunkable.
  Its `safe_to_interpolate` uses the whole signal's standard deviation, so
  run it in memory.

## October 2026, round 15: chunked processing

New only; no existing result changes. See
[Chunked processing](guides/chunked-processing.md).

- `ts_shape.run_chunked(source, Target, "method", chunk="1D", **kwargs)` runs
  a detector slice by slice and returns the same frame as
  `Target(all_data, ...).method(...)`. Events and windows that cross a slice
  border are stitched: open intervals are carried into the next slice, and
  overnight shift windows belong to the day they start. `iter_chunked` yields
  finished rows slice by slice. `time_range_source(load, start, end)` loads
  one slice per call, for databases. A source can be any iterable of frames
  or `(name, frame)` pairs in time order, such as a loader's
  `stream_by_time_range`.
- `Pipeline.run_stream(source, chunk="1D")` runs a pipeline the same way.
  Transforms must be row-local. `keep_data=True` keeps the transformed data.
- `ts_shape.NotChunkableError` (a `ValueError`) is raised for methods that
  need the whole signal: whole-signal statistics, model fits, first-window
  baselines and static summaries. `ts_shape.stream.SUPPORT` lists the
  chunkable methods, each checked against the in-memory result by
  `tests/test_stream_equivalence.py`.
- On data in a time zone with DST, chunked runs reject windows of several
  hours that are not whole days (for example `"8h"`). Use a window that
  divides an hour, whole days, or UTC data.

## October 2026, round 14: audit leftovers

Each change is pinned by a hand-computed test in
`tests/reference/test_round14_*.py`.

### Results that change

- `DemandPatternEvents.demand_by_period` keeps periods without orders as zero
  demand (`empty_periods="zero"`, the new default): every period from the
  first to the last one with data is returned, empty ones with
  `total_demand = 0`, `peak_demand = 0`, `order_count = 0` and `avg_demand`
  NaN. New `order_count` column. `detect_demand_spikes` and
  `seasonality_summary` use the zero-filled totals, which lowers the spike
  baseline and the weekday averages (orders of 10 every other day: Monday
  average 10 → 5). `empty_periods="drop"` restores the former output.
  Cumulative counters must be differenced into per-order quantities first.
- `LeadTimeAnalysisEvents.lead_times` with order identifiers pairs orders and
  deliveries one-to-one, first in first out, within each identifier. One
  delivery could close several orders with the same identifier and later
  deliveries were ignored (two orders, deliveries on Jan 5 and Jan 6: 96 h and
  72 h → 96 h and 96 h). Unpaired orders and deliveries raise a
  `DataQualityWarning` on this path too. Multi-line orders need line-level
  identifiers. `lead_time_statistics` and `detect_lead_time_anomalies`
  follow.
- `DataGapAnalysisEvents.coverage_by_period`: the data covers
  `[first sample, last sample + median sampling step)`; time of the first and
  last window outside that span is missing (in `gap_seconds`; `gap_count`
  still counts internal gaps). 1 s data from 10:30 to 11:29:59 in hourly
  windows reads 50 % / 50 % (was 100 % / 100 %), as `data_completeness`.
- `SamplingResolutionEvents.detect_aliasing` takes the sampling rate from the
  timestamps of non-missing values (every other value NaN on a 1 s grid:
  1.0 Hz → 0.5 Hz; dominant frequency and `aliasing_risk` follow).
- `DegradationDetectionEvents` and `AssetHealthIndexEvents`: the baseline is
  the half-open `[first, first + window)`, as long as the rolling windows, so
  a stationary signal has a variance ratio of 1. `detect_variance_increase`
  does not flag samples inside the baseline window.
- `ControlLoopHealthEvents.detect_oscillation` and
  `AnomalyClassificationEvents.detect_oscillation` / `classify_anomalies`: a
  zero carries the last non-zero sign, so `1, 0, -1` is one crossing (was two)
  and touching zero is none. The control-loop period doubles where zeros sat
  between the signs.

### New and clarified

- `RateOfChangeEvents.rate_statistics`: `mean_abs_rate`, `std_abs_rate`,
  `max_abs_rate`, `min_abs_rate`. The `*_rate` columns are statistics of the
  absolute rate today and become statistics of the signed rate in the next
  major release; read `*_abs_rate` for the absolute rate.
- `CalibrationValidationEvents.calibration_curve`: a `power` column after
  `coefficient_index`. Coefficients are highest power first and describe
  measured = f(reference).
- Canonical twins: `OperatingStateEvents.state_duration_statistics`
  `count` → `interval_count`; `MicroStopEvents.micro_stop_frequency`
  `count` → `micro_stop_count`.
- `signals={uuid: ...}` arguments (`AssetHealthIndexEvents`,
  `StartupDetectionEvents.detect_startup_multi_signal`) get the
  uuid-not-found warning; `AssetHealthIndexEvents` warns once per signal whose
  values are in another `value_*` column, also when the index is computed
  from the others.
- `signals.primitives.windows` (docs): sub-daily windows are fixed lengths
  laid out like `resample`, so their wall-clock start moves after a DST
  change; use `shift_windows` for wall-clock shifts.

## October 2026, round 13: audit fixes

Fixes from an audit of the engineering, quality and maintenance packs. Each
one is pinned by a hand-computed test in `tests/reference/test_round13_*.py`.

### Engineering: time-weighted results

- `ControlLoopHealthEvents.error_integrals`: IAE, ISE and ITAE integrate the
  held error over each window, split at the window edges. Time that crossed a
  window edge was lost: a constant error of 1 sampled every 30 min gave IAE
  1800 per hour instead of 3600. ITAE measures time from the window start.
  A window covered by a held value but without samples is now reported.
- `ControlLoopHealthEvents.output_saturation`: the `pct_time_at_*` columns
  are shares of observed time, not of samples (33.3 % → 90.9 % on an output
  held at 100 for 50 of 55 minutes); `longest_saturation_seconds` is the
  held run length (a single sample held for 50 min was 0 s).
- `ThresholdMonitoringEvents.time_above_threshold` (`next_change`): windows
  covered by a held value are reported even without samples, and a signal
  that starts above the threshold counts one exceedance.
- `SteadyStateDetectionEvents.detect_steady_state`: the statistics use the
  steady samples only; the first sample after the steady period was
  included (20 zeros then 100: mean 5.0 → 0.0). `steady_state_statistics`
  returns NaN, not 0.0, when there is no steady period or no time span.
- `WarmUpCoolDownEvents.time_to_target`: overshoot is measured until the next
  ramp starts, not over the rest of the data.
- `RateOfChangeEvents.detect_rapid_change` and
  `StartupDetectionEvents.detect_startup_by_slope`: an interval starts at the
  sample before the first fast step (a single jump lasts one sample
  interval, not 0 s).
- `StartupDetectionEvents.assess_startup_quality`: the stability tail has at
  least two samples; a short startup ending flat scores 1.0 (was 0.0).

### Quality

- `ToleranceDeviationEvents`: `tolerance_lag` delays the limits, not the
  reported event times. All limit rows are used, also with `is_delta=False`,
  and a frame without an `is_delta` column works. `capability_indices` for a
  missing signal returns an empty frame with a warning instead of raising.
- `AnomalyClassificationEvents`: `detect_flatline` ends a run when the value
  leaves the tolerance around the run's first value (a slow ramp was reported
  as a stuck sensor). Windows are half-open and the last partial window is
  evaluated (a drift in the last 30 min of 90 min of data was missed).
- `SamplingResolutionEvents` and `ControlLoopHealthEvents`: sampling
  intervals are no longer truncated to whole milliseconds (1.5 ms sampling
  read as 1000 Hz; 0.5 ms sampling returned nothing).
- `DataGapAnalysisEvents`: a gap is an interval *longer* than `min_gap`, as
  documented. The default `min_gap=None` is twice the median sampling
  interval (as in `coverage_by_period`), so regular data has no gaps.
  `gap_summary` reports the real data span and NaN shortest/mean gap when
  there are no gaps.
- Calendar windows such as `"MS"` work in
  `CalibrationValidationEvents.detect_recalibration_need`,
  `calibration_stability` and
  `EnvironmentalCompensationEvents.detect_env_exceedance` (they raised).

### Maintenance, transforms, severity

- `AssetHealthIndexEvents.health_events`: a recovery to a better band is
  `direction="above"`.
- `IntegerCalc.power_column`: negative exponents and results beyond int64 are
  float (`2 ** 64` wrapped to 0).
- One severity table: the event log's `ts_shape:severity` equals the
  detector's `severity_level` column. Changed in the event log: tolerance
  `major` → critical (was warn), CUSUM `high` → warn (was critical),
  flow-constraint `severe` → critical (was missing). `minor` is `info` and
  threshold `alarm` is `critical` in `severity_level`.

### Robustness

- A signal whose values are in another `value_*` column than the one the
  method reads now emits a `DataQualityWarning` naming the column to pass
  (`value_column=...`). Results are unchanged.
- `MachineStateEvents` and `DutyCycleEvents`: a missing state (`NaN`,
  `pd.NA`) is unknown time, not idle/off; ratios use the known time. Averages
  over no intervals are NaN. `duty_cycle_by_window` and `cycle_count` extend
  to the constructor's `data_end`.
- `MicroStopEvents` and `DutyCycleEvents.on_off_intervals` accept a naive or
  string `data_end` on tz-aware data (local wall time; it raised).
- A `value_column` that does not exist raises `ColumnNotFoundError` in
  `PerformanceLossEvents` and in the state-based detectors (it fell back
  silently). `ProductionScheduleAdherenceEvents` raises
  `ColumnNotFoundError` (a `ValueError`) for missing schedule columns.
- Variance ratios and `shift_sigma` are NaN, not `inf`, when the reference
  spread is undefined (`ProcessWindowEvents`, `DisturbanceRecoveryEvents`,
  `ChangePointEvents`).
- `CrossSignalAnalytics` and `FeatureMatrixExporter` accept numbers stored as
  object dtype.

### Results that change (names and intervals)

- `OperatingStateEvents` intervals end at the next state's first sample
  (`interval_end="next_change"`, the library default), so consecutive states
  cover the signal without gaps. At 30 s sampling a 15-minute phase is 900 s
  (was 870 s). `interval_end="last_sample"` restores the former intervals.
- `MicroStopEvents.micro_stop_patterns(hour_grouping=False)` names shifts
  `shift_1` / `shift_2` / `shift_3` (was `morning` / `afternoon` / `night`;
  pass `shift_definitions=` for your own) and groups a shift that crosses
  midnight by its start date: two stops at 23:00 and 02:00 in a 22–06 shift
  are one group (`avg_count` 2.0, was two groups of 1.0).
- `LineThroughputEvents.cycle_quality_check`: `tolerance_pct` is a percent
  (default `10.0`) like every other `*_pct`; a value below 1 is still read as
  the former fraction, with a `DeprecationWarning`. `deviation_pct` is 0–100
  (was 0–1); the fraction is in the new `deviation_ratio`. `quality_flag`
  bins follow the tolerance (good ≤ tol, warning ≤ 2.5 × tol); at the
  default they are the former fixed bins.
- `GoldenBatchDeviationEvents` works with tz-aware timestamps (it raised).
- `Pipeline` steps accept deprecated keyword names (with the warning) instead
  of rejecting them as unknown.

### Renamed parameters

| Where | Old | New |
|---|---|---|
| `ChangeoverEvents.detect_changeover`, `SetpointChangeEvents.detect_setpoint_steps` | `min_hold` | `min_duration` |
| `ChangeoverEvents.changeover_window` config key | `min_hold` | `min_duration` |
| `SetpointChangeEvents.detect_setpoint_changes` | `min_hold` | `min_step_duration` (its `min_duration` is the ramp duration) |
| `OutlierDetectionEvents.detect_outliers_zscore`, `detect_outliers_mad` | `threshold` | `n_sigma` |
| `ToleranceDeviationEvents` (argument and attribute) | `tolerance_column`, `actual_column` | `value_column_tolerance`, `value_column_actual` |
| `ContextEnricher.enrich_with_tolerances` | `low_col`, `high_col` | `lower_limit_column`, `upper_limit_column` |
| `ContextEnricher.enrich_with_mapping` | `raw_value_col`, `mapped_value_col` | `value_column_raw`, `value_column_mapped` |
| `GoldenBatchDeviationEvents` | `reference_df` | `dataframe` |
| `CycleDataProcessor` (argument and attribute) | `cycles_df`, `values_df` | `dataframe_cycles`, `dataframe_values` |
| `PatternRecognition` methods | `uuid` | `signal_uuid` |
| `LineThroughputEvents.cycle_quality_check` | `tolerance_pct` as a fraction | `tolerance_pct` in percent, or `tolerance_ratio` |

### Renamed output columns (old column kept as a twin)

Reading the old column does not warn; it is removed in the next major release.

| Where | Old | New |
|---|---|---|
| `ToleranceDeviationEvents.capability_indices` | `Cp`, `Cpk`, `Pp`, `Ppk`, `Cpm` | `cp`, `cpk`, `pp`, `ppk`, `cpm` |
| `ContextEnricher.enrich_with_tolerances` | `low_limit`, `high_limit` | `lower_limit`, `upper_limit` |
| `DutyCycleEvents` | `on_time`, `off_time`, `avg_on_duration`, `avg_off_duration` | `*_seconds` |
| `MachineStateEvents.state_quality_metrics` | `avg_run_duration`, `avg_idle_duration`, `run_idle_ratio` | `*_seconds`, `run_idle_factor` |
| `MicroStopEvents` | `total_run_time`, `total_micro_stop_time`, `total_lost_time`, `avg_lost_time` | `*_seconds` |
| `OperatingStateEvents.state_duration_statistics` | `mean/std/min/max/total_duration` | `*_duration_seconds` |
| `ThresholdMonitoringEvents` | `time_above`, `total_duration` | `time_above_seconds`, `total_duration_seconds` |
| `SignalQualityEvents.sampling_regularity` | `mean/std/min/max_interval` | `*_interval_seconds` (and `gap_duration_seconds` beside `gap_duration`) |
| `CycleTimeEvents` | `count`, `moving_avg`, `avg/min/max_cycle_time` | `cycle_count`, `moving_avg_seconds`, `*_cycle_time_seconds` |
| `LineBalancingEvents` | `cycle_time_mean/median/std`, `bottleneck_cycle_time`, `n_stations` | `*_seconds`, `station_count` |
| `LineThroughputEvents` | `count`, `expected_time` | `part_count`, `expected_time_seconds` |
| `WarmUpCoolDownEvents.warmup_consistency` | `deviation_from_median_duration` | `deviation_from_median_duration_seconds` |
| `LeadTimeAnalysisEvents`, `SetupTimeEvents`, `BatchTrackingEvents` | `count` | `order_count`, `setup_count`, `batch_count` |
| `ProcessWindowEvents.windowed_statistics`, `OperatingRangeEvents.value_distribution` | `count` | `sample_count` |
| `ReliabilityEvents.weibull_fit` | `n_failures`, `n_censored` | `failure_count`, `censored_count` |
| `SamplingResolutionEvents` | `n_unique_values`, `resolution_ratio` | `unique_value_count`, `resolution_factor` |
| `ValueDistributionEvents.detect_mode_changes` | `n_modes_detected` | `mode_count` |
| `DesignOfExperimentsEvents.effects` | `n_runs` | `run_count` |
| `DegradationDetectionEvents`, `DisturbanceRecoveryEvents`, `ProcessWindowEvents` | `variance_ratio` | `variance_factor` |
| `FlowMetricsEvents.flow_summary` | `consistency_ratio` | `consistency_factor` |
| `StatisticalProcessControlEvents.detect_violations` | `triggered_rule`, `value_double` | `rule`, `value` (as `detect_rule_violations`) |

`docs/naming.md` now spells out the rules behind these: durations carry a
unit suffix, counts are `<thing>_count`, and an unbounded quotient is
`*_factor` (a `*_ratio` stays within 0–1).

## October 2026, round 12: oscillations, EnPI baseline, windowed features

### Results that change

- `SetpointChangeEvents.control_quality_metrics`: `oscillation_count` and
  `oscillation_freq_hz` count a crossing of the setpoint only once the error
  leaves a band around it on the other side. The band (`deadband=`) defaults
  to the settle band, `tol` or `settle_pct` of each step, so noise a settled
  response shows inside the band is no longer counted as oscillation. A step
  that swings three times and then sits in ±0.1 noise with `tol=0.5`:
  82 → 3. With the default `tol=0.0` nothing changes; `deadband=0` restores
  the former count. `overshoot_metrics` and `oscillation_frequency` take the
  same `deadband=` and default to `0` (unchanged).
- `EnergyPerformanceIndicatorEvents.enpi_vs_baseline(baseline="regression")`
  leaves baseline windows without output (zero or missing units) out of the
  `energy = a + b * units` fit and names them in a `DataQualityWarning`. A
  window with energy but no units (the counter's first window, a stop day)
  formerly pulled the fit: with a 600 kWh idle day before nine days on
  `200 + 2 * units` the intercept was ~330 instead of 200. Results change only
  when the baseline period has such windows.
- `ts_shape.utils.frames.window_offset` with a `Timedelta` (or
  `datetime.timedelta` / `numpy.timedelta64`) window returns a fixed
  duration on pandas 2.2 as on pandas 3: `Timedelta("2D")` → `<48 * Hours>`
  (pandas 2.2 formerly gave a two-calendar-day offset, 47 h across the spring
  DST change). Window ends of every method that takes a `Timedelta` window
  are affected on pandas 2.2 only. Offset strings like `"1D"` stay calendar
  days.

### New

- `DescriptiveFeatures.compute_windowed(window="1s", metrics=None)` (also
  `compute(output_format="windowed", window=...)`): one row per time window,
  `start`, `end`, `duration_seconds`, then `{uuid}__{feature}` for every
  signal. Every window from the first to the last timestamp is present; a
  signal without data in a window is NaN, its counts (`total_rows`,
  `true_count`, ...) are 0. `metrics=` selects features by name. See
  [Features for ML](guides/ml-features.md).

## October 2026, round 11: typed API, docs that run

No result changes except the fixes listed under *Fixes that change output*.

### New and wider imports

- `from ts_shape import X` is fully typed for mypy, pyright and IDEs (it was
  `object`). Deprecated classes, methods and attributes are reported by type
  checkers (mypy: `--enable-error-code deprecated`); runtime warnings are
  unchanged.
- New top-level exports: every transform and feature class, `ValueMapper`,
  `PipelineResult`, and `TsShapeWarning`, `DataQualityWarning`,
  `PerformanceWarning`, `LoaderConfigWarning`, `ColumnNotFoundError`,
  `LoaderError`. `ts_shape.events` re-exports every detector;
  `ts_shape.context` has `__all__`; `ts_shape.loader` also exports the Azure
  flexible-file, Databricks and energy API loaders and `ContextEnricher`.
- `list_detectors()` lists every top-level public class, now including the
  transform and feature classes, and no longer imports the modules.

### Import time

`from ts_shape import OutlierDetectionEvents` takes about 0.35 s (was
1.2–1.9 s): scipy and scikit-learn load only in the methods that use them.
`outlier_detection.IsolationForest` is no longer a module attribute
(`SKLEARN_AVAILABLE` is kept). The z-score is computed in numpy, identical to
current scipy; on scipy older than 1.18 the last bits can differ, nearly
constant data gives NaN scores, and scipy's "Precision loss" warning is no
longer emitted.

### PostgreSQL loaders

`TimescaleDBDataAccess` and `DatapointDB` build their URL with the psycopg2
driver named explicitly (works on SQLAlchemy 2.1) and escape the password.
New settings: `db_port` (default 5432) and `db_driver` (default
`"psycopg2"`), as keys of `TimescaleDBDataAccess`'s `db_config` and as
keyword arguments of `DatapointDB`, which also takes `db_name` (default
`"config_repository"`). Both
have a `.url`; without psycopg2 an ImportError names `ts-shape[postgres]`.
Loader `repr()` shows the settings with passwords, keys, tokens and
connection strings hidden.

### Fixes that change output

- `PatternRecognition.discover_motifs(top_k > 1)` no longer returns trivial
  matches: a later motif never starts within `exclusion_zone` (default
  `window_size // 2`) of an earlier one, so ranks 2+ are different shapes.
  `top_k=1` is unchanged.
- `StartupDetectionEvents.detect_startup_by_slope` works with repeated index
  labels (it raised `IndexError`).
- `detect_startup_multi_signal` returns its full column set when nothing is
  found, for `logic="any"` and `"all"` (it returned a frame with no columns).
- `DatapointDB` gives an empty frame for a device without data points (it
  raised `KeyError: 'uuid'`).

### Names follow docs/naming.md

Old names keep working until the next major release: a renamed keyword or attribute warns with a `DeprecationWarning` naming the replacement; a renamed output column stays as a twin with the same values. A canonical column sits right after its legacy twin; window bounds (`start`, `end`, `duration_seconds`) lead every summary.

### Renamed parameters and attributes

| Old | New | Where |
|---|---|---|
| `state_uuid` | `run_state_uuid` | `DowntimeEvents` (`downtime_by_shift`, `downtime_by_reason`, `top_downtime_reasons`, `availability_trend`), `SetupTimeEvents` (`setup_durations`, `setup_by_product`, `setup_statistics`, `setup_trend`), `ShiftHandoverEvents` (`generate_report`, `highlight_issues`), `IdleEnergyDetectionEvents` (`idle_energy_by_window`, `idle_energy_by_shift`, `idle_energy_trend`), `EnergyEfficiencyEvents.idle_energy_waste`, `LongDowntimeEvents`, `ReliabilityEvents` |
| `.state_uuid` (attribute) | `.run_state_uuid` | `LongDowntimeEvents`, `ReliabilityEvents` |
| `run_value_column` | `value_column_state` | `OEEEvents.performance` |
| `cut_value_column` | `value_column_cut` | `ContinuousProcessAlignmentEvents.segment_by_cut` |
| `id_value_column` (argument and attribute) | `value_column_id` | `MultiProcessTraceabilityEvents`, `RoutingTraceabilityEvents` |
| `handover_value_column` (argument and attribute) | `value_column_handover` | `MultiProcessTraceabilityEvents` |
| `routing_value_column` (argument and attribute) | `value_column_routing` | `RoutingTraceabilityEvents` |
| `df` | `dataframe` | `to_event_log`, `object_intervals`, `detect_objects`, `attach_objects`, `run_backtest`, `evaluate` and `LambdaDetector.evaluate` / `.to_event_log` (lambda rules), `EnergyConsumptionEvents.normalize`, `EnergyEfficiencyEvents.normalize`, `StatisticalProcessControlEvents.rule_1` .. `rule_8` |
| `data` | `dataframe` | `CycleDataProcessor.group_by_cycle_uuid` |
| `column` | `group_column` | `CycleDataProcessor.split_dataframes_by_group` |
| `id_column` | `uuid_column` | `EnergyConsumptionEvents.normalize`, `EnergyEfficiencyEvents.normalize` |
| `expected_freq` | `expected_interval` | `SignalQualityEvents.detect_missing_data`, `data_completeness` |
| `expected_frequency` | `expected_frequency_hz` | `SamplingResolutionEvents.detect_aliasing` (a frequency in Hz, not an interval) |
| `uuid_a` / `uuid_b` | `signal_uuid_a` / `signal_uuid_b` | `SignalCorrelationEvents.rolling_correlation`, `correlation_breakdown`, `lag_correlation` |
| `period1` / `period2` | `period_a` / `period_b` | `PeriodSummaryEvents.compare_periods` |
| `parquet_only` | `only_parquet` | `AzureBlobParquetLoader.list_structure`, `DatabricksUnityParquetLoader.list_structure` |

### Renamed output columns (old column kept as a twin)

| Old | New | Where |
|---|---|---|
| `window_start` only | `start`, `end`, `duration_seconds` lead; `window_start` kept | `SamplingResolutionEvents` (all four methods), `EnvironmentalCompensationEvents.detect_correlation` / `sensitivity` / `detect_env_exceedance` (also keeps `window_end`), `MeasurementUncertaintyEvents.type_a_uncertainty` / `combined_uncertainty`, `CalibrationValidationEvents.detect_recalibration_need` / `calibration_stability`, `TimeToThresholdEvents.time_to_threshold_windows`. The event-log taxonomy now treats them as summaries (they were static), so their events carry the window's time instead of the default timestamp. |
| `time` | `start`, `end` (= `time`), `duration_seconds` (of `duration`) | `RateOfChangeEvents.detect_step_changes` |
| `mean_interval_ms`, `std_interval_ms`, `max_gap_ms` | `mean_interval_seconds`, `std_interval_seconds`, `max_gap_seconds` | `SamplingResolutionEvents.sampling_jitter` |
| `n_samples` | `sample_count` | `CapabilityTrendingEvents.capability_over_time`, `MeasurementUncertaintyEvents.type_a_uncertainty`, `ValueDistributionEvents.detect_bimodal` / `normality_windows` / `percentile_tracking` |
| `pct_of_total` | `share_of_total_pct` | `DowntimeEvents.downtime_by_reason` / `top_downtime_reasons`, `QualityEvents.nok_by_reason`, `ReworkEvents.rework_by_reason`, `ScrapEvents.scrap_by_reason`, `MaterialBalanceEvents.contribution_breakdown` |
| `pct_of_best` | `share_of_best_pct` | `OperatorPerformanceEvents.operator_comparison`, `ProcessStabilityIndexEvents.stability_comparison` |
| `pct_of_window` | `share_of_window_pct` | `MicroStopEvents.micro_stop_frequency` |
| `pct_of_available_time` | `share_of_available_time_pct` | `SetupTimeEvents.setup_statistics` |
| `pct_above` | `time_above_pct` | `ThresholdMonitoringEvents.time_above_threshold` |
| `pct_time_at_high` / `pct_time_at_low` / `pct_time_saturated` | `time_at_high_pct` / `time_at_low_pct` / `time_saturated_pct` | `ControlLoopHealthEvents.output_saturation` |
| `pct_saturated` | `saturated_pct` | `ControlLoopHealthEvents.loop_health_summary` |
| `pct_time_disturbed` | `time_disturbed_pct` | `DisturbanceRecoveryEvents.disturbance_frequency` |
| `pct_GRR` / `pct_PV` / `pct_GRR_tolerance` | `grr_pct` / `pv_pct` / `grr_tolerance_pct` | `GaugeRepeatabilityEvents.gauge_rr_summary` |
| `pct_contribution` | `contribution_pct` | `MeasurementUncertaintyEvents.uncertainty_budget` |
| `availability`, `performance`, `oee_score` | `availability_ratio`, `performance_ratio`, `oee_ratio` | `LineThroughputEvents.throughput_oee` |
| `period1_value`, `period2_value` | `period_a_value`, `period_b_value` | `PeriodSummaryEvents.compare_periods` |

## October 2026, round 10: fast at 10 and 100 million rows

Results are unchanged except where listed below; event ids from
`to_event_log` are byte-identical.

- `ChangePointEvents.detect_change_points` and `segment_statistics` default to
  `method="auto"`: exact PELT up to 20,000 samples (results unchanged), binary
  segmentation above that (approximate; results on long series can differ and
  the slowness warning no longer fires). Pass `method="pelt"` for the exact
  search on long series.
- Rolling window moments (degradation and health scores) and matrix-profile
  distances (motif and discord discovery) differ from before by at most 1e-9
  relative; motif and discord positions are unchanged.
- Setpoint KPIs no longer fail on an actual signal with duplicate index
  labels.
- A missing uuid no longer matches the text `"nan"` or `"None"` when
  selecting signals.

## October 2026, round 9: KPIs match their textbook definitions

Several KPIs gave wrong numbers. They now follow the published definitions
(ISO 22400-2, Montgomery *Introduction to Statistical Quality Control*,
Ogata *Modern Control Engineering*, ISO 50006, Nakajima), and each one is
pinned by a hand-computed reference test in `tests/reference/`. **Results
change**; no code change is needed unless you depended on the old values.
Where an option keeps the old behavior, it is named below.

### Time and parts are booked to the period they belong to

Durations used to be booked to the shift/hour/day of the reading that started
them, and counter increments to the reading that reported them. Both are now
split at period boundaries (wall-clock, DST-safe), via the new
`ts_shape.utils.time_accounting` helpers.

Affected: `DowntimeEvents.downtime_by_shift` / `availability_trend`,
`ShiftHandoverEvents`, `RuntimeAccountingEvents.runtime_by_window` /
`operating_hours_meter`, `MicroStopEvents.micro_stop_impact`, `OEEEvents`
(availability, performance, daily OEE) and `QualityTrackingEvents.nok_by_shift`.

| Probe | Old | New |
|---|---|---|
| Stop 07:00–15:00, shift_1 (06–14) availability | 0 % (480 min down) | 12.5 % (60 up / 420 down) |
| Same stop, `availability_trend(window="1h")`, 07:00 hour | 480 min down | 60 min down (one row per hour) |
| Night shift 22–06 running at 100 %, OEE day 1 / day 2 | 12.5 % / 0 % | 100 % / 100 % |
| 30 parts counted 13:50–14:10, shift_1 / shift_2 | 0 / 30 | 15 / 15 |

- A counter increment is spread over `(previous reading, reading]` in
  proportion to time; a reset counts in the period of the reading that shows
  it. `handle_resets=False` keeps the per-period `max(0, last - first)`.
- Part counts in OEE (`actual_parts`, `total_parts`, `reject_parts`,
  `good_parts`) and shift-handover quantities are floats now.
- Shift and trend outputs list every period that holds observed time, not only
  periods that contain a reading.
- A day without state rows has NaN availability, performance and OEE (was:
  the counter span as run time, availability 0).
- Performance or availability above 100 % is no longer clipped; the value is
  kept and a `DataQualityWarning` names the period.
- `micro_stop_frequency` still counts a stop in the window it starts in.

### Production

- `PerformanceLossEvents` counts parts (counter increments), not counter rows;
  cycle time = run time / parts. Counter +2 per minute for 60 min at a 30 s
  ideal cycle: performance 50 % → 100 %, loss 1770 s → 0 s. A new
  `run_seconds` column carries the basis.
- `PeriodSummaryEvents.from_daily_data` recomputes KPIs from summed bases
  (quality = Σgood/Σtotal, availability = Σrun/Σplanned, OEE = A×P×Q,
  MTTR = Σrepair/Σfailures, EnPI = Σenergy/Σunits, ...). Ratios without a
  basis column are weighted means (plain mean with a `DataQualityWarning`);
  numeric columns with no rule are dropped with a warning instead of summed.
  A new `aggregations=` argument overrides the rule per column. Quality of
  900/1000 and 0/10: 45.0 % → 89.1 %; MTTR 1 h and 3 h: 4.0 → 2.0.
- `MicroStopEvents.micro_stop_impact` gains `performance_loss_pct` (minor
  stops are a performance loss); `availability_loss_pct` stays with the same
  value.

### Setpoint response

- `time_to_settle` is the time after which the response stays in the band
  until the end of the window (last entry, not first). The window now also ends
  at the next setpoint change. New `settled_within_window` column;
  `t_settle_seconds` is NaN when the response does not settle. `settled` means
  settled within the window with a final in-band stretch of at least `hold`.
  Underdamped step ζ=0.2, ωn=0.1, 2 % band: 18 s → 197 s.
- `overshoot_pct` / `undershoot_pct` are percent (0–100) like every other
  `*_pct` column: 0.5266 → 52.66. The fraction is in the new
  `overshoot_ratio` / `undershoot_ratio` columns.

### Statistical process control and capability

`StatisticalProcessControlEvents(..., sigma_method="moving_range")` (default)
estimates σ as MR̄/1.128 (individuals chart); `sigma_method="std"` brings back
the sample standard deviation.

| Probe | Old | New |
|---|---|---|
| Reference 1..20: control limits | −7.25 / 28.25 | 7.84 / 13.16 |

- The rule 1–8 zones and `detect_cusum_shifts` use the same σ̂.
- `dynamic_control_limits(method="moving_range")` tests each point against the
  `window_size` points before it (center = their mean, σ̂ = their MR̄/1.128);
  the first `window_size` points have NaN limits.
- `dynamic_control_limits(method="ewma")` is an EWMA chart:
  z_i = λx_i + (1−λ)z_{i−1}, limits μ0 ± L·σ̂·√(λ/(2−λ)·[1−(1−λ)^{2i}]).
  `ewma` is z_i, `mean` is μ0; new `ewma_lambda=` (default 2/(window_size+1))
  and `n_sigma=` (L) set the new `lcl` / `ucl` columns. Both methods add a
  `value` column.
- Cp/Cpk use the within σ (MR̄/1.128) and Pp/Ppk the overall σ (AIAG), in
  `CapabilityTrendingEvents` (`sigma_method=` on the constructor) and
  `ToleranceDeviationEvents.capability_indices` (`sigma_method=` on the
  method). `sigma_method="std"` makes Cp/Cpk equal Pp/Ppk again. New
  `std_within` column. Per-window Pp/Ppk use the window's own σ (they used the
  whole signal's). Example x = 10.0, 10.4, 10.2, 10.6, 9.8, specs 9–11:
  Cp 1.054 → 0.836, Pp unchanged 1.054.

### Energy and supply chain

- The EnPI rolling baseline, `EnergyEfficiencyEvents.sec_trend` and
  `efficiency_trend` `rolling_avg_efficiency` are Σnumerator/Σdenominator over
  the window (ISO 50006), not the mean of ratios: EnPI 1 and 3 over 130 kWh /
  110 units: 2.0 → 1.182.
- `InventoryMonitoringEvents.stockout_prediction` uses the refill-aware
  consumption rate (`handle_refills=True`, new); a refill no longer cancels
  consumption. `handle_refills=False` keeps first-minus-last.

### Documented, unchanged

- `SignalCorrelationEvents.lag_correlation`: positive `lag_periods` means `uuid_b`
  leads `uuid_a` (b(t) = a(t−5) gives −5). The docstring now says so.

## October 2026, round 8: legacy code removed

### Requirements

ts-shape now needs **pandas 2.2.2 or later** and **numpy 2.0 or later** (was
pandas 2.1, numpy 1.26). A CI job tests the oldest supported versions, and
results are the same on pandas 2.2 and pandas 3: daily windows are calendar
days (23 h / 25 h on a DST change) on both.

`setup.py`, `MANIFEST.in` and the `requirements*.txt` files are gone;
`pyproject.toml` is the only place dependencies are declared.

### Deprecation warnings you can see

Deprecation warnings point at your own line now. Before, warnings from
detector methods pointed inside ts-shape, and Python's default filters hid
them, so a deprecated name could be used without a hint. Run your tests with
`-W error::DeprecationWarning` once to find them.

Newly deprecated (old name still works and warns):

Loader time ranges are `start` / `end` everywhere; positional calls are
unchanged.

| Old | New |
|---|---|
| recipe spec key `duration_s` | `duration_seconds` |
| `ShiftHandoverEvents.from_shift_data(downtime_df=...)` with only `downtime_minutes` | a `downtime_seconds` column |
| `CycleDataProcessor.cycle_uuid_col` / `.systime_col` (attributes) | `.cycle_uuid_column` / `.time_column` |
| `RuntimeAccountingEvents.run_uuid` (attribute) | `.run_state_uuid` |
| `MultiSensorValidationEvents.sensor_uuids` (attribute) | `.signal_uuids` |
| `start_timestamp=` / `end_timestamp=` (Azure Blob, Databricks Unity and Spark parquet loaders, `TimescaleDBDataAccess`, `S3ProxyDataAccess`) | `start=` / `end=` |
| `start_time=` / `end_time=` (`ParquetLoader.load_by_time_range`, `load_files_by_time_range_and_uuids`) | `start=` / `end=` |
| `TimescaleDBDataAccess.start_timestamp` / `.end_timestamp`, same on `S3ProxyDataAccess` (attributes) | `.start` / `.end` |

### Behavior changes

- **Time-named numeric columns stay numbers.** A class without its time
  column used to convert every column whose name contained "time" or "date"
  (`cycle_time_mean`, `update_rate`) to datetimes. Only datetime or text
  columns are used now, with a `DataQualityWarning`.
- **Mixed time units.** `SegmentProcessor.apply_ranges` and
  `ContinuousProcessAlignmentEvents.segment_by_cut` compare times in one
  unit; ranges in seconds against data in nanoseconds matched no rows.
- **Hour folders are UTC in every loader** (Azure, Databricks Unity,
  Parquet): a tz-aware bound is converted to UTC, a naive bound is UTC.
  Before, some loaders used local wall-clock hours.
- `CycleDataProcessor.split_dataframes_by_group` raises
  `ColumnNotFoundError` for a frame without the group column (it was logged
  and skipped). `CriticalParameterRankingEvents.rank` raises
  `ColumnNotFoundError` for a missing outcome column.
- Problems that were only logged are `DataQualityWarning`s: degenerate SPC
  limits, a singular MSPC covariance, ignored or overlapping segment ranges,
  invalid or overlapping cycles, empty feature tables. Text in numeric
  columns is reported in pattern recognition, feature export, segment
  statistics, cycles and continuous-process alignment.
- `StatisticalProcessControlEvents.detect_cusum_shifts` counts a repeated
  sample (same time, same value) once, with a `DataQualityWarning`. A
  duplicated row was added to the running sum twice and could report a shift
  on its own.
- `ProfileComparison.cluster(n_clusters=...)` always returns the requested
  number of clusters (SciPy's `maxclust` cut could return fewer).
- New exports: `ts_shape.events.production` lists `FlowMetricsEvents`,
  `LineBalancingEvents`, `RuntimeAccountingEvents`; `ts_shape.features` lists
  `PatternRecognition`, `DescriptiveFeatures`, `TimeWindowedFeatureTable`.
- The API reference no longer has pages for private modules, and package
  pages have clean URLs (`.../events/production/` instead of
  `.../events/production/__init__/`).

## October 2026, round 7: every function robust and consistent

Every public function, not only the detectors, now handles empty frames,
timezones, nullable and pyarrow dtypes, text time columns and missing
columns the same way. Contract tests check this for every public callable.

### Renamed parameters

The old names work and warn.

| Old | New | Where |
|---|---|---|
| `column_name` (a value column) | `value_column` | filters, `IntegerCalc`, `LambdaProcessor.apply_function`, `UnitConverter.convert_column`, `Numeric/Boolean/StringStatistics` |
| `column_name` (the time column) | `time_column` | `DateTimeFilter`, `TimestampStatistics`, `ValueMapper` |
| `freq` | `window` | `PowerIntegrator`, `DataHarmonizer.resample_to_uniform` / `merge_multi_signals`, `TimeGroupedStatistics` |
| `df`, `uuid_col`, `value_cols`, `group_col` | `dataframe`, `uuid_column`, `value_columns`, `group_column` | `FeatureMatrixExporter.to_feature_matrix` |
| `cycle_uuid_col`, `systime_col` | `cycle_uuid_column`, `time_column` | `CycleDataProcessor` |
| `metric` | `value_column` | `CycleDataProcessor.compare_cycles`, `identify_golden_cycles` |
| `metadata_uuid_col`, `tolerance_uuid_col`, `mapping_uuid_col` | `*_uuid_column` | `ContextEnricher` |

`DateTimeFilter` takes the bound right after the frame:
`filter_after_date(df, "2024-01-02", time_column="systime")`. The old order
still works and warns.

### Results that were wrong

- `CrossSignalAnalytics.lead_lag` reported the leader backwards. A positive
  `optimal_lag` now means `signal_a` leads. All its methods sort by time and
  use only rows where both signals have a value.
- `PatternRecognition` sorts by time, keeps the timezone, and needs `uuid=`
  when the frame holds several signals. `normalize=False` now really uses
  the raw distance.
- `BooleanFilter` edges are found per uuid in time order; a missing value
  holds the previous state. `StringFilter.detect_changes_in_string` always
  reports the first row of each uuid, also for `string` dtype.
- `DataHarmonizer.fill_gaps(max_gap=)` measures the gap between the valid
  samples around it; signals without values are kept as NaN columns.
- `LambdaProcessor.apply_function` and
  `TimezoneShift.revert_to_original_timezone` no longer change your frame.
- `FeatureMatrixExporter` uses the sample standard deviation (ddof=1) and
  aggregates only `value_*` columns by default.
- `ReliabilityEvents`, `InventoryMonitoringEvents.stockout_prediction`:
  NaN instead of `inf`. `LeadTimeAnalysisEvents.lead_times` pairs each order
  with the first later delivery, never a negative lead time.
- A zero-length step (duplicate timestamp) is no data gap and starts no
  flatline. Duplicated rows no longer duplicate outlier and tolerance events.
- `object_intervals(min_duration=)` no longer loses an id that comes back
  after a short blip.

### Event ids

`to_event_log` builds `ocel:eid` from the detector, `source_uuid`, the
timestamps, the activity and the row's values, no longer from the row
position. The same detector on two machines no longer collides, and `concat`
no longer drops events. Event ids from earlier releases change.

Also in `to_event_log`: a string in `objects=` that is not a column is used
as the object id for every row; a NaN in an activity template gives
`unknown` with a warning; interval rows with `end < start` are dropped with a
warning; the static shape uses `timestamp=` (default 1970-01-01 UTC), not
the current time.

### No data, unknown codes

- Percentages of no data (`true_percentage`, `uppercase_percentage`, ...)
  are NaN, not 0. A mode of no data is `None`.
- `CrossSignalAnalytics` with too little data gives NaN, not 0.0 / 1.0.
- `ValueMapper` keeps codes the mapping table does not know, with a
  `DataQualityWarning` (they were NaN). Duplicate keys raise `ValueError`.
- `ContextEnricher` calls build on each other; `get_enriched_dataframe()`
  returns the cumulative result. Duplicate keys and colliding column names
  raise `ValueError`.
- `ToleranceDeviationEvents.capability_indices` and `MultivariateSPCEvents`
  with too little data give NaN / an empty result with a warning instead of
  raising.

### Errors

A missing column raises `ColumnNotFoundError` (a `ValueError`) in every
public function; it was often a `KeyError`.

### Loaders, signals, rules

- `DatabricksSparkParquetLoader.build_paths` includes the hour of a
  mid-hour start and builds folder hours in UTC. `ParquetLoader` matches
  uuids to whole file names. `EnergyAPILoader` returns UTC times.
- `signals.align` handles mixed time units and names the time column after
  `time_column`; `per_window` handles booleans with missing values.
  `SignalSpec` validates its fields.
- Lambda rule expressions reject `and`, `or`, `not`, `in` and chained
  comparisons at compile time (use `&`, `|`, `~`, `.isin`). Interval rules
  end at the next sample after the last true one, like `signals.intervals`.
- `Pipeline` passes arrays and frames as step arguments and never changes
  your input frame.

### New

- `max_gap` on `detect_flatline`, `threshold_with_hysteresis` and
  `on_off_intervals`.
- `DataHarmonizer.detect_gaps` adds `start`, `end`, `duration_seconds`.
- `TimeGroupedStatistics.calculate_custom_func(group_by_uuid=True)`.

## October 2026, round 6: consistent outputs

### Summary methods return a DataFrame

Like `best_and_worst_shifts` (round 5), these methods take `as_frame: bool = True`. The default returns a DataFrame whose columns are the old keys, plus `start` / `end` / `duration_seconds` for the data the summary covers. `as_frame=False` returns the old dict or list and emits a `DeprecationWarning`; that form will be removed in the next major release. To keep old code working unchanged, pass `as_frame=False`. Alternatively, read a one-row result with `.iloc[0]`, which supports the same `["key"]` access.

| Method | Default DataFrame |
|---|---|
| `SteadyStateDetectionEvents.steady_state_statistics` | one row; the dict keys plus `uuid`, `source_uuid` |
| `MachineStateEvents.state_quality_metrics` | one row; the dict keys plus `uuid`, `source_uuid` |
| `TargetEvents.target_hit_rate` | one row; `start` / `end` span the analysed days |
| `ToleranceDeviationEvents.capability_indices` | one row; `Cpm` is always a column (NaN without `target_value`), plus `sample_count` |
| `BottleneckDetectionEvents.throughput_constraint_summary` | one row per station: `bottleneck_count`, `bottleneck_pct`, `avg_utilization_pct`, `most_frequent_bottleneck` |
| `FlowConstraintEvents.flow_constraint_analytics` | one row per `constraint_type` (`blocked`, `starved`): `event_count`, `total_duration_seconds`, `avg_duration_seconds`, `minor_count`, `moderate_count`, `severe_count`, `time_alignment_quality`. The event lists come from `blocked_events` / `starved_events`. |
| `ShiftHandoverEvents.highlight_issues` | one row per issue, with the report row's shift window and `date` |

### KPI percentages carry their basis

New columns; none were removed:

| Method | New columns |
|---|---|
| `OEEEvents.oee` | `run_seconds`, `planned_seconds`, `excluded_seconds`, `actual_parts`, `ideal_parts`, `total_parts`, `reject_parts`, `good_parts` (NaN without quality counters), `resets` |
| `OEEEvents.availability` | `excluded_seconds` (time `max_gap` removed as no data) |
| `LineThroughputEvents.throughput_oee` | `run_seconds`, `planned_seconds`, `resets` (with `handle_resets=True`) |
| `DowntimeEvents.downtime_by_shift` / `availability_trend` | `excluded_seconds` |
| `PerformanceLossEvents` (all three methods) | `run_seconds` |

### Durations in seconds

Each `*_minutes` or `*_s` duration column now has a `*_seconds` twin right after it: `total_seconds`, `downtime_seconds`, `uptime_seconds`, `avg_seconds` (downtime), `avg_cycle_time_seconds`, `target_cycle_time_seconds`, `loss_seconds` (performance loss), `avg_seconds`, `min_seconds`, `max_seconds`, `median_seconds`, `std_seconds`, `avg_setup_seconds`, `total_setup_seconds` (setup time), `downtime_seconds` (shift handover, also in `OUTPUT_COLUMNS`). The old columns stay as a convenience. Their removal would be a breaking change and is left for a future major release; new code should read the `*_seconds` columns. In recipe-phase specs `duration_seconds` is the new spelling of the `duration_s` criterion; `duration_s` still works and, since round 8, warns.

### Run state as string or boolean

`DowntimeEvents` and `ShiftHandoverEvents` read either a string state signal (`running_value="Running"` in `value_string`) or a boolean run signal (`value_bool`, True = running). The boolean signal is used when the state rows carry no `value_string`, or when `running_value` / `stopped_value` is a bool. Both forms give the same downtime. Before this change, a boolean run signal was compared with `"Running"` and therefore reported 100 % downtime.

### Fixes

`FlowConstraintEvents.flow_constraint_analytics` applies its `warning_threshold` / `critical_threshold` to the severity counts and to the event lists. Before, it accepted them but classified with the 5 s / 30 s defaults.

### Same numbers for any data source, totals that add up

- **Dtypes no longer change results.** Value columns with pandas nullable,
  pyarrow or text-number dtypes are converted once when a detector is built.
  Text that is not a number is reported with a `DataQualityWarning` at
  construction, not when a method runs. Counter totals on nullable or pyarrow
  data were too high (the first reading counted as production): they now
  match NumPy data.
- **Group totals add up to the counter total.** Per shift, day, operator,
  part, reason, batch and schedule period, the part made between the last
  reading of one group and the first reading of the next now counts in the
  later group (`utils.counters.counter_by_group`). Quantities can grow by one
  increment per group boundary; `resets` can grow by one when a reset falls on
  a group boundary. Per-operator/part/reason totals no longer double count
  when those groups alternate. `shift_production(date=...)` includes the
  increment into the day's first reading. `batch_yield` books production
  between two batches to the earlier batch. `handle_resets=False` is
  unchanged.
- **`FlowConstraintEvents`** intervals end at the next change
  (`interval_end="next_change"`); pass `interval_end="last_sample"` for the
  previous, shorter durations.
- **`MultiSensorValidationEvents.detect_outlier_sensor`** needs three sensors
  to name an outlier; with two it skips the window and warns.
- **`ToleranceDeviationEvents.capability_indices`** on empty data returns an
  empty frame with the full schema; `CapabilityTrendingEvents` uses the spec
  limits in force in each window.
- **Energy methods with `window="W"` / `"ME"`** report the real week or month
  (`end` was `start` + 1 day).
- **Shift labels:** a row with a missing timestamp gets the shift `"unknown"`.
- **Degenerate inputs:** duplicate timestamps give `NaN` rates (with a
  warning) instead of `inf`; `detect_oscillation(window<=0)` raises.

### Cycles, segments and objects

These were verified against hand-built scenarios; the fixes change results:

- **Cycles** (`CycleExtractor`):
  - back-to-back cycles sharing a timestamp come out correctly;
  - a second start while a cycle is open aborts it (`is_complete=False`) instead of
    producing overlapping cycles;
  - a held start pulse is one start;
  - value/state-change cycles read only `start_uuid`;
  - a cycle already running at the first sample has `is_complete=False`;
  - `cycle_uuid` is deterministic (same data, same id).
- **Cycle data** (`CycleDataProcessor`): every method uses `start <= t < end`. A
  sample exactly at a cycle end belongs to the next cycle, not to both; open cycles
  get no samples. `identify_golden_cycles(method="target_value")` needs `target=`.
- **Segments** (`SegmentExtractor` / `SegmentProcessor`):
  - the last segment runs to the end of the data (it had zero length);
  - `apply_ranges` cuts half-open, so boundary samples are no longer duplicated;
  - the first sample of an empty-id gap is no longer given to the previous order.
- **`segment_by_cut`:** material is assigned to the piece the next cut finishes. Before,
  it was one piece off and piece 1 got two pieces' worth. Material after the last
  cut has `piece_id` `<NA>`.
- **Objects** (`object_intervals` / `detect_objects`):
  - an empty or NaN reading ends a presence;
  - a change-encoded id stays present to the end of the data;
  - at a hand-over instant only the incoming object gets the event;
  - a child is `part_of` only the parent it overlaps longest;
  - `min_duration` measures the presence, not first-to-last sample;
  - use `to_event_log_ocel(log, pm4py=True)` for pm4py export.
- **`production_dates`:** correct after a DST change (it was one day off).

### Severity levels and small fixes

- **`severity_level`.** Every result with a categorical `severity` also has
  `severity_level` (`info` / `warn` / `critical`), right after `severity`. The
  `severity` labels are unchanged. Code that selects columns by position sees one
  more column.
- **`TimeWindowedFeatureTable`:** `freq` is renamed `window`. `freq` still works and
  warns.
- **`CriticalParameterRankingEvents()`:** the long-format frame is optional and
  `rank()` never used it, so a time column is no longer needed.
- **`ContinuousProcessAlignmentEvents.lag_profile`:** a window without speed
  readings has NaN speed and lag. Before, it reported the lag at `min_speed`:
  thousands of seconds.

## October 2026, round 5: correct by default

The correct behavior is now the default, and results are returned at full
precision. No name was removed, so your code keeps running, but **results
change** wherever your data has counter resets, change-encoded signals, uneven
sampling or several signals in one frame. Each old behavior is one argument
away.

### Defaults that changed

| Parameter | Old default | New default | What it fixes | Where |
|---|---|---|---|---|
| `handle_resets` | `False` | `True` | A counter that resets (shift start, rollover) no longer makes a window report too little, often 0. Results carry a `resets` column. | production (OEE, parts, quality, scrap, rework, operator, target, period summary, shift report and handover, batch yield, line throughput, process alignment), energy (EnPI, efficiency, consumption, carbon), `utils.counters`, `TimeGroupedStatistics` (`'diff'`) |
| `interval_end` | `"last_sample"` | `"next_change"` | An interval lasts until the signal changes. A change-encoded (deadband) signal's interval is no longer cut to zero length. | machine state, duty cycle, micro stops, long downtime, steady state, threshold monitoring, degradation, vibration, recipe phases, inventory, signal quality, flatline, correlation breakdown, `SegmentExtractor`, `utils.intervals` |
| `time_weighted` | `False` | `True` | Shares and averages are weighted by time, not by sample count, so irregular sampling no longer biases them. | bottleneck detection, duty cycle by window, operating range, material balance, idle energy, efficiency idle waste |
| `group_by_uuid` | `False` | `True` | Statistics are computed per signal instead of pooling unrelated signals. | `OutlierDetectionEvents`, `TimeGroupedStatistics`, `BooleanFilter` edges, `StringFilter.detect_changes_in_string` |
| `slope_per` | `"sample"` | `"second"` | Drift slopes are per second, so irregular sampling no longer distorts them; `min_slope` is per second too. | `AnomalyClassificationEvents.detect_drift`, `classify_anomalies` |
| `missing_as_zero` | `True` | `False` | A missing flow signal makes the balance NaN instead of a false imbalance. | `MaterialBalanceEvents` |
| `handle_refills` | `False` | `True` | A refill no longer turns into negative consumption. | `InventoryMonitoringEvents.consumption_rate` |
| `include_severity` | `False` | `True` | Violations carry `triggered_rule` and `severity`. | `StatisticalProcessControlEvents.detect_violations` |
| `regex` | `True` | `False` | Substring filters match literally (`"A.1"` no longer matches `"AB1"`). | `StringFilter.filter_string_contains`, `StringStatistics.contains_substring_count` |
| `trim_to_range` | `False` | `True` | Loaders return only rows inside the requested time range. | Azure Blob and Databricks Unity parquet loaders |
| `as_frame` | `False` | `True` | `best_and_worst_shifts` returns one DataFrame with a `rank_type` column; `as_frame=False` (a dict) is deprecated. | `ShiftReportEvents` |
| `layout` (new) | — | `"long"` | `batch_transition_matrix` returns `from_state`, `to_state`, `transition_count`, and an empty result keeps that schema. `layout="wide"` gives the old matrix. | `BatchTrackingEvents` |

To compare old and new numbers, pass the old default explicitly, for example `oee(..., handle_resets=False)`.

### Other result changes

- **Full precision.** Detectors no longer round results (they used 0 to 8 decimals). Round when you display: `df.round(2)`.
- **`machine_running_pct` is a percentage (0–100).** In `IdleEnergyDetectionEvents.idle_energy_by_window` and `EnergyEfficiencyEvents.idle_energy_waste` it held a 0–1 ratio. The ratio is `machine_running_ratio`.
- **Flatlines:** with `"next_change"`, a single sample counts as a flatline only when the gap after it is longer than the signal's usual sampling interval.
- **`ValueDistributionEvents.detect_bimodal`** returns a unimodal row for a constant signal (it was empty or had a `None` mode).
- **`CriticalParameterRankingEvents.rank`** raises `ValueError` for an unknown candidate column (it was skipped).
- **`TimezoneShift.list_available_timezones`** returns a sorted list (it was a set).
- **`make_timeseries`** no longer rounds `value_double`.

### Nothing is skipped silently

- **Loaders:** an unreadable blob or file used to disappear from the result. Multi-object Azure Blob and Databricks Unity methods take `errors="warn"` (default: skip it and name it in one `DataQualityWarning`) or `errors="raise"` (`LoaderError`). Series metadata that exists but cannot be read raises `LoaderError`.
- **Non-numeric values** in numeric inputs are still treated as missing, but a `DataQualityWarning` now says how many there were.
- **A missing value column** (`value_column`, `value_column_<role>`, ...) raises `ColumnNotFoundError` naming it and listing the columns, instead of a bare `KeyError`.
- Malformed recipe-phase specs, untestable normality windows, failed fits and event-log object bindings to a missing column now warn instead of being dropped.
- `lambda_rules.load_yaml` without PyYAML raises `ImportError` pointing to `ts-shape[yaml]`.

### Newly deprecated

| Old | New |
|---|---|
| `OrderTraceabilityEvents` | `ValueTraceabilityEvents` |
| `ToleranceDeviationEvents(tolerance_uuid=...)` alone | `upper_tolerance_uuid` and/or `lower_tolerance_uuid` (either one alone is a one-sided spec) |
| `RoutingTraceabilityEvents(station_map=...)` | `state_map` |
| `eventlog.adapters.adapt(legacy=...)` | `frame=` |
| integer `window` in `SignalCorrelationEvents` | `window_size` |
| `best_and_worst_shifts(as_frame=False)` | the default DataFrame |

### New

- `OutlierDetectionEvents(signal_uuids=[...])` restricts the analysis to some signals.
- `pip install ts-shape[parquet]` installs pyarrow (also pulled in by `[s3]` and `[azure]`); `[all]` now installs every optional feature.

## Renamed parameters (old keywords deprecated)

Parameter names follow [Naming & Output Conventions](naming.md). The old keyword still works, but emits a `DeprecationWarning`.

| Old | New | Where |
|---|---|---|
| `column_name` (the time column) | `time_column` | every detector constructor (`Base`) |
| `freq`, `period`, `resample` | `window` | data-gap coverage, value distribution, demand patterns, period summary, correlation and root-cause methods |
| integer `window` (a row count) | `window_size` | `SignalCorrelationEvents.rolling_correlation` / `correlation_breakdown`, `StatisticalProcessControlEvents.dynamic_control_limits` |
| `z_threshold`, `threshold_sigma` | `n_sigma` | anomaly classification, anomaly correlation, disturbance recovery |
| `low`/`high`, `lower`/`upper`, `low_limit`/`high_limit`, `lower_bound`/`upper_bound` | `lower_limit` / `upper_limit` | threshold hysteresis, operating range, control-loop saturation, time-to-threshold |
| `minor_threshold` / `moderate_threshold` | `warning_threshold` / `critical_threshold` | `FlowConstraintEvents.flow_constraint_analytics` |
| `include_first`, `count_initial_active` | `include_initial` | reliability (Weibull), alarm frequency |
| `significant_only`, `on_crossing_only` | `only_significant`, `only_crossings` | root-cause ranking, asset-health events |
| `sensor_uuids` | `signal_uuids` | `MultiSensorValidationEvents` |
| `run_uuid` | `run_state_uuid` | `RuntimeAccountingEvents` |
| `counter_column`, `energy_column`, `state_column` | `value_column_counter`, `value_column_energy`, `value_column_state` | energy pack |

Optional arguments are keyword-only. Passing them positionally still works, but warns, for example `detect_outliers_zscore(3.0)` instead of `detect_outliers_zscore(threshold=3.0)`.

The default `event_uuid` values now all follow `"<pack>:<name>"`:
- `outlier_event` → `quality:outlier`
- `startup_event` → `eng:startup`
- `setpoint_change_event` → `eng:setpoint_change`
- `FlowMetricsEvents`: `prod:flow` → `prod:flow_metrics` (it used to collide with `FlowConstraintEvents`)

If you filter results by these values, update the filters.

## Renamed classes and methods (old names deprecated)

Detector classes all end in `Events`. Event classes no longer use `calculate_` / `compute_` / `process_`: methods that find events start with `detect_`, and metrics are nouns (see [Naming & Output Conventions](naming.md)).

The old names still work. They emit a `DeprecationWarning`, and the event log accepts them too: for example, `to_event_log(df, "OEECalculator.calculate_oee")` is resolved to `OEEEvents.oee`.

| Old | New |
|---|---|
| `CycleTimeTracking` | `CycleTimeEvents` |
| `DowntimeTracking` | `DowntimeEvents` |
| `OperatorPerformanceTracking` | `OperatorPerformanceEvents` |
| `PartProductionTracking` | `PartProductionEvents` |
| `PerformanceLossTracking` | `PerformanceLossEvents` |
| `QualityTracking` | `QualityEvents` |
| `ReworkTracking` | `ReworkEvents` |
| `ScrapTracking` | `ScrapEvents` |
| `SetupTimeTracking` | `SetupTimeEvents` |
| `TargetTracking` | `TargetEvents` |
| `OEECalculator` (`calculate_availability`, `calculate_performance`, `calculate_quality`, `calculate_oee`) | `OEEEvents` (`availability`, `performance`, `quality`, `oee`) |
| `PeriodSummary` | `PeriodSummaryEvents` |
| `ShiftReporting` | `ShiftReportEvents` |
| `ShiftHandoverReport` | `ShiftHandoverEvents` |
| `ProcessStabilityIndex` | `ProcessStabilityIndexEvents` |
| `StatisticalProcessControlRuleBased` (`process`, `apply_rules_vectorized`, `calculate_control_limits`, `calculate_dynamic_control_limits`) | `StatisticalProcessControlEvents` (`detect_violations`, `detect_rule_violations`, `control_limits`, `dynamic_control_limits`) |
| `ToleranceDeviationEvents.process_and_group_data_with_events` / `.compute_capability_indices` | `.detect_deviations` / `.capability_indices` |
| `DataGapAnalysisEvents.find_gaps` | `.detect_gaps` |
| `MultiSensorValidationEvents.identify_outlier_sensor` | `.detect_outlier_sensor` |
| `LeadTimeAnalysisEvents.calculate_lead_times` | `.lead_times` |
| `EnvironmentalCompensationEvents.estimate_sensitivity` | `.sensitivity` |
| `DesignOfExperimentsEvents.compute_effects` | `.effects` |
| `*.alarm_duration_stats`, `*.batch_duration_stats`, `*.state_duration_stats` | `*_statistics` |
| `DutyCycleEvents.duty_cycle_per_window`, `RuntimeAccountingEvents.runtime_per_window` | `*_by_window` |

## Behavior fixes that change results

These change output because the old output was wrong or contradicted the documentation. Each one has a regression test.

- **SPC rules now follow Nelson (1984) / Western Electric** (`StatisticalProcessControlEvents`):
  - rule 8 flags 8 points in a row with **none** within 1σ (it used to flag 8 points *within* 1σ);
  - rule 3 is 6 points steadily rising or falling (was 7), and rule 4 is 14 points alternating up and down (was 15);
  - rules 5 and 6 count points beyond the next zone too, per side;
  - rule 2 measures sides against the chart center line;
  - the rules no longer add columns to your frame, and the vectorized and per-rule paths flag the same points.
- **`ToleranceDeviationEvents.detect_deviations`** keeps the measured value and reports single-sample deviations. A missing `actual_uuid` raises instead of returning nothing. `capability_indices` adds `Cpm` when `target_value` is given.
- **`LineThroughputEvents.takt_adherence`:**
  - a cycle exactly equal to takt is no longer a violation;
  - a decrease in a numeric counter (a reset) is no longer a cycle.
- **`MachineStateEvents.state_quality_metrics`:**
  - follows the signal's spec like `detect_run_idle`;
  - `run_idle_ratio` is NaN, not 0.0, when there is no idle time.
- **`ShiftHandoverEvents.highlight_issues`** without `report_df` or the four uuids raises instead of returning `[]`.
- **`PeriodSummaryEvents.from_daily_data`** honors anchored weeks (`'W-FRI'` …).
- **`GoldenBatchDeviationEvents.phase_breakdown`** compares each phase with the same phase of the golden batch.
- **`EnvironmentalCompensationEvents`:**
  - constructing without `env_uuid` raises;
  - `compensated_signal`'s `uuid` is the event uuid.
- **`SetpointChangeEvents`:** two setpoint changes at the same instant no longer multiply the KPI rows.
- **`ProcessWindowEvents`** uses the strict `>` its docstrings describe.
- **Hour-folder loaders** no longer drop the hour that contains the range end. Databricks loaders apply a single `start` or `end` bound.

Parameters that had no effect now warn when passed: `RateOfChangeEvents.detect_rapid_change(window=...)`, `MeasurementUncertaintyEvents(part_uuid=...)`, `ContinuousProcessAlignmentEvents(ref_uuid=...)`. When both `settle_pct` and `tol` are passed to a setpoint KPI, a warning says `settle_pct` wins. When both `state_map` and `station_map` are passed, a warning says `state_map` wins.

## Every detector output carries its time bounds

Each detector returns the canonical columns of its shape (see [Naming & Output Conventions](naming.md)):
- **intervals:** `start`, `end`, `duration_seconds`, `uuid`, `source_uuid`;
- **summaries:** `start`, `end`, `duration_seconds`, also when they are keyed by `date`, `(date, shift)` or `period`;
- **points:** `systime`, `uuid`, `source_uuid`.

The new columns were added next to the old ones; none was removed or renamed. Two consequences:

- **Merging two summaries on `date` or `(date, shift)`** now finds `start`, `end` and `duration_seconds` on both sides, and pandas suffixes them `_x` / `_y`. Merge on the window instead, or select your columns first:

```python
import numpy as np
import pandas as pd
from ts_shape import PartProductionEvents, QualityEvents

t = pd.date_range("2024-01-01", periods=96, freq="h")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:ok", "value_integer": np.arange(96) * 9}),
    pd.DataFrame({"systime": t, "uuid": "line1:nok", "value_integer": np.arange(96)}),
    pd.DataFrame({"systime": t, "uuid": "line1:part", "value_string": "P-100"}),
    pd.DataFrame({"systime": t, "uuid": "line1:count", "value_integer": np.arange(96) * 10}),
], ignore_index=True)

quality = QualityEvents(df).daily_quality_summary("line1:ok", "line1:nok")
production = PartProductionEvents(df).daily_production_summary("line1:part", "line1:count")
daily = quality.merge(production, on=["start", "end", "duration_seconds"])
```

- **Night shifts are one row.** An overnight shift (e.g. 22:00 to 06:00) is attributed to the day it starts (`production_day=True` is now the default). Before, grouping by calendar date split one night across two rows, and each row mixed two different nights. Pass `production_day=False` to get the old grouping.

## Removed exporter aliases

The event-log exporters were renamed into a consistent `to_event_log_*` family.
The old aliases (`to_flat_df`, `to_ocel_tables`) have now been **removed** —
update any remaining imports:

| Removed | Use instead |
|---|---|
| `to_flat_df(...)` | `to_event_log_xes(...)` |
| `to_ocel_tables(...)` | `to_event_log_ocel(...)` |

<!-- no-run -->

```python
# Before
from ts_shape.eventlog import to_flat_df, to_ocel_tables

# After
from ts_shape.eventlog import to_event_log_xes, to_event_log_ocel
```

## `to_event_log_ocel` now returns five tables

To fully model OCEL 2.0, `to_event_log_ocel(log)` returns an `OCEL2Tables`
bundle (object-to-object relations and time-varying object attributes were
added) instead of a 3-tuple. Access the frames by attribute:

<!-- no-run -->

```python
# Before
events_df, objects_df, relations_df = to_event_log_ocel(log)

# After
tables = to_event_log_ocel(log)
tables.events, tables.objects, tables.relations          # as before
tables.o2o, tables.object_changes                        # new OCEL 2.0 tables
```

## `org:resource` in XES export

`to_event_log_xes` previously copied the case object id into `org:resource`.
Per the XES standard `org:resource` is the performing resource, so it is now
populated only from an `operator`-type object relation, and **omitted** when
the log has no operators. Supply one via `objects={"operator": ...}` if you
need it.

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
