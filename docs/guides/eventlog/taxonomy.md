# Event Log: labelling standard & taxonomy

This page documents the rules that fill `ocel:activity` and the standard
attributes every detector emits. It is the reference companion to the
[Event Log overview](index.md) (the canonical schema and exports) and
[Adapter anatomy](adapters.md) (how a detector's DataFrame is normalized).

---

## Activity-name taxonomy

The full set of rules is codified in [Event labelling standard](#event-labelling-standard). At a glance:

| Detector method | `ocel:activity` |
|---|---|
| `OutlierDetectionEvents.detect_outliers_zscore` | `quality.outlier.zscore` |
| `StatisticalProcessControlRuleBased.process` | `quality.spc.rule_violation` |
| `MachineStateEvents.detect_run_idle` | `production.machine_state.{state}` |
| `MachineStateEvents.transition_events` | `production.machine_state.transition_{transition}` |
| `SetpointChangeEvents.detect_setpoint_steps` | `engineering.setpoint.step_{change_type}` |
| `DegradationDetectionEvents.detect_trend_degradation` | `maintenance.degradation.trend` |

The full registry lives in `ts_shape.eventlog.taxonomy.REGISTRY` and is enforced by `tests/eventlog/test_adapter_coverage.py` — adding a new detector method without registering a label rule fails CI.

---

## Event labelling standard

These are the rules every `LabelRule` in
`ts_shape.eventlog.taxonomy.REGISTRY` follows. Adhering to them keeps
the output of any two detectors directly compatible without per-pack
glue code in downstream tools.

### Activity-name format

```
<pack>.<family>.<specifier>[.<subtype>]
```

- All segments **lowercase, snake_case**, separated by `.`.
- `pack` is one of the seven fixed packs (see below).
- `family` is the conceptual category within the pack (e.g. `outlier`,
  `machine_state`, `setpoint`).
- `specifier` distinguishes between methods/algorithms inside the
  family (e.g. `zscore`, `iqr`, `mad` for `quality.outlier`).
- `subtype` is optional and almost always **templated** — a placeholder
  like `{state}` or `{change_type}` substituted from the legacy row.
- Templated segments use `{column_name}` syntax. At adapter time the
  value of that column is dropped in. **Missing values render as
  `unknown` rather than raising.**

### Pack vocabulary (fixed)

| Pack | What belongs here |
|---|---|
| `quality` | Per-measurement quality findings: outliers, SPC violations, tolerance breaches, sensor health. |
| `production` | Shop-floor state and KPIs: machine state, alarms, batches, OEE, traceability, shift reports. |
| `engineering` | Process-engineering analytics: setpoint behavior, control-loop health, steady state, thresholds. |
| `maintenance` | Equipment health: degradation, failure prediction, vibration. |
| `supplychain` | Inventory, demand, lead-time signals. |
| `energy` | Energy consumption, efficiency, carbon intensity, EnPI. |
| `correlation` | Cross-signal analytics that don't naturally belong to one asset (signal correlation, anomaly co-occurrence). |

### Family vocabulary (extensible)

| Pack | Standard families |
|---|---|
| `quality` | `outlier`, `spc`, `tolerance`, `sensor_drift`, `signal`, `data_gap`, `gauge_rr`, `multi_sensor`, `capability`, `distribution`, `anomaly` |
| `production` | `machine_state`, `alarm`, `batch`, `bottleneck`, `changeover`, `cycle_time`, `downtime`, `duty_cycle`, `flow`, `long_downtime`, `micro_stop`, `oee`, `operator`, `order`, `part`, `performance`, `period`, `quality`, `rework`, `routing`, `scrap`, `setup`, `shift`, `target`, `throughput`, `traceability`, `value_trace`, `alignment` |
| `engineering` | `setpoint`, `startup`, `threshold`, `rate_of_change`, `steady_state`, `signal_comparison`, `operating_range`, `thermal`, `process_window`, `control_loop`, `disturbance`, `material_balance`, `stability` |
| `maintenance` | `degradation`, `failure`, `vibration`, `health` |
| `supplychain` | `inventory`, `demand`, `lead_time` |
| `energy` | `consumption`, `efficiency`, `enpi`, `carbon`, `idle` |
| `correlation` | `signal`, `anomaly` |

When a new detector class lands in an existing pack, prefer reusing
one of the families above. Add a new family only when none fits.

### Specifier conventions

Use a **literal specifier** when the method always emits the same
activity:

```python
("OutlierDetectionEvents", "detect_outliers_zscore"):
    LabelRule(template="quality.outlier.zscore", ...)
```

Use a **templated specifier** when one method emits multiple
semantically distinct activities, distinguished by a categorical column:

```python
("MachineStateEvents", "detect_run_idle"):
    LabelRule(template="production.machine_state.{state}", ...)
# emits both production.machine_state.run and production.machine_state.idle
```

Recommended placeholders (use the legacy column name verbatim):

| Placeholder | From column | Examples |
|---|---|---|
| `{state}` | `state` | `run`, `idle`, `setup`, `down` |
| `{transition}` | `transition` | `run_to_idle`, `idle_to_run` |
| `{change_type}` | `change_type` | `step_up`, `step_down`, `ramp` |
| `{phase}` | `phase` | `warmup`, `nominal`, `cooldown` |
| `{anomaly_class}` | `anomaly_class` | `drift`, `flatline`, `oscillation` |

Severity is **never** a templated segment — it lives in
`ts_shape:severity` (see below).

### Attribute-naming rule

| Prefix | Origin | Examples |
|---|---|---|
| `ocel:` | OCEL 2.0 spec | `ocel:eid`, `ocel:activity`, `ocel:timestamp`, `ocel:oid`, `ocel:type`, `ocel:qualifier` |
| `ts_shape:` | ts-shape canonical fields with no OCEL counterpart | `ts_shape:start_timestamp`, `ts_shape:duration_s`, `ts_shape:detector`, `ts_shape:pack`, `ts_shape:severity`, `ts_shape:value` |
| `<pack>:` | Detector-specific legacy columns | `production:state`, `quality:rule_violated`, `engineering:overshoot`, `maintenance:health_score` |
| `concept:`, `time:`, `case:`, `lifecycle:`, `org:` | XES spec — added only by `to_event_log_xes` | `concept:name`, `time:timestamp`, `case:concept:name`, `lifecycle:transition`, `org:resource` |

The pack prefix prevents any clash between detectors (e.g. two packs
that both use a `state` column become `production:state` and
`quality:state`).

### Severity bucket thresholds

When a `LabelRule` declares a `severity_field`, the numeric value is
bucketed into a string:

| Numeric range | `ts_shape:severity` |
|---|---|
| `< 3.0` | `info` |
| `3.0 ≤ v < 4.5` | `warn` |
| `v ≥ 4.5` | `critical` |
| `NaN` / non-numeric / missing | `<NA>` |

The thresholds match the numeric `severity` column emitted by
`OutlierDetectionEvents` and other detectors. If a DataFrame already
carries a literal `severity` column with one of `info`/`warn`/`critical`,
that string value is passed through verbatim.

### Object-type vocabulary

The 16 standard types in `STANDARD_OBJECT_TYPES`, grouped by what they
represent. Extend with `register_object_type("name")` when none fits.

| Group | Types | Used for |
|---|---|---|
| Physical | `asset`, `tool`, `sensor`, `signal`, `station` | Equipment and instrumentation. `asset` is the default auto-extracted from `source_uuid`. |
| Process | `cycle`, `batch`, `lot`, `recipe`, `work_order`, `shift` | The process context an event happens in. |
| Product | `material`, `part`, `serial`, `article` | What's being made. |
| People | `operator` | The person responsible. |

### Qualifier vocabulary

Recommended values for `ocel:qualifier` (the role of the object in the
event). Free text is permitted, but stick to these for cross-pack
consistency:

| Qualifier | Object type | Meaning |
|---|---|---|
| `produced_on` | `asset` | The event happened on this asset. |
| `during_batch` | `batch` | The event occurred while this batch was running. |
| `during_cycle` | `cycle` | The event occurred during this cycle. |
| `during_shift` | `shift` | The event occurred during this shift. |
| `made_of` | `material` | The product being processed contained this material. |
| `identified_by` | `serial` | The product carries this serial number. |
| `operated_by` | `operator` | The operator on duty. |
| `measured_by` | `sensor` | The sensor producing the reading. |
| `governed_by` | `recipe` | The recipe in effect. |

### Standard attribute extension

In addition to the canonical event columns, every method's `LabelRule`
declares a `standard_attrs` mapping that pins detector-specific values to
a **fixed vocabulary** of attribute keys. This is what makes
cross-detector aggregation possible — two detectors that conceptually
emit the same thing emit it under the same column name.

The full vocabulary (defined in `ts_shape.eventlog.schema.STANDARD_ATTR_KEYS`):

| Key | Type | Used for |
|---|---|---|
| `ts_shape:method` | string | Algorithm name. Always literal. e.g. `"zscore"`, `"iqr"`, `"western_electric_rule_1"`, `"cusum"`. |
| `ts_shape:baseline` | float | Expected / nominal value (SPC centerline, setpoint target, baseline mean). |
| `ts_shape:threshold_low` | float | Lower bound. `NaN` if one-sided. |
| `ts_shape:threshold_high` | float | Upper bound. `NaN` if one-sided. |
| `ts_shape:deviation` | float | Signed `value - baseline`. |
| `ts_shape:deviation_pct` | float | `(value - baseline) / baseline`. |
| `ts_shape:direction` | string | `above` / `below` / `up` / `down` / `outside` / `inside` / `lead` / `lag` / `shift`. |
| `ts_shape:confidence` | float | 0..1, for ML / probabilistic detectors. |
| `ts_shape:sample_count` | int | Number of underlying observations rolled into this row. |
| `ts_shape:outcome` | string | Categorical outcome: `ok` / `nok` / `rework` / `scrap` / `pass` / `fail`, or a normalized reason code. |
| `ts_shape:lifecycle_state` | string | XES-style: `raised` / `cleared` / `predicted` / state names (`run`, `idle`). |
| `ts_shape:lifecycle_pair_id` | string | Pairs raise/clear into a single occurrence. |

Each entry in `standard_attrs` maps a key to either:

* a **legacy column name** (string matching a column in the detector
  output) — the adapter renames it and coerces to the declared type,
* a **literal scalar** (string / float / int) — broadcast to every
  row. This is the common case for `ts_shape:method = "zscore"`.
* `None` — explicitly declares the attribute is not applicable for this
  method (used for archetype-required keys when the detector has no
  natural source).

Example for `OutlierDetectionEvents.detect_outliers_zscore`:

```python
LabelRule(
    template="quality.outlier.zscore",
    pack="quality",
    shape="point",
    severity_field="severity_score",
    standard_attrs={
        "ts_shape:method": "zscore",          # literal — always "zscore"
        "ts_shape:direction": "outside",      # literal
    },
)
```

And for an aggregate KPI like `CycleTimeTracking.cycle_time_statistics`:

```python
LabelRule(
    template="production.cycle_time.statistics",
    pack="production",
    shape="summary",
    standard_attrs={
        "ts_shape:sample_count": "count",     # rename legacy `count` column
    },
)
```

### Required keys per archetype

The coverage test `test_required_standard_attrs_per_archetype` enforces
this mapping at CI time — every method in `REGISTRY` must populate at
least its archetype's required keys.

| Archetype | Required keys | Typical optional keys |
|---|---|---|
| `threshold` | `method`, `direction` | `baseline`, `threshold_low`, `threshold_high`, `deviation`, `deviation_pct`, `confidence` |
| `interval` | `lifecycle_state` | `lifecycle_pair_id`, `sample_count`, `direction` |
| `aggregate` | `sample_count` | `baseline`, `threshold_low`, `threshold_high`, `method` |
| `outcome` | `outcome` | `sample_count`, `method` |
| `static` | `method` | `sample_count`, `baseline`, `threshold_low`, `threshold_high` |
| `trace` | `lifecycle_state`, `direction` | `sample_count` |
| `forecast` | `method`, `confidence` | `baseline`, `threshold_low`, `threshold_high` |
| `correlation` | `method` | `direction`, `confidence`, `sample_count` |

The archetype assignment for every detector method lives in
`ts_shape.eventlog.archetypes.ARCHETYPE_BY_METHOD` and is enforced by
`test_archetype_assignment_is_complete` — every entry in `REGISTRY` has
exactly one archetype.

### Why this matters — cross-detector aggregation

Once every event log emits `ts_shape:method`, `ts_shape:direction`,
`ts_shape:deviation_pct`, `ts_shape:sample_count`, `ts_shape:outcome`,
queries like these become trivial:

```python
# All threshold violations grouped by algorithm.
log.events.groupby("ts_shape:method")["ocel:eid"].count()

# All "above-threshold" events with > 10% deviation.
log.events.query(
    "`ts_shape:direction` == 'above' and `ts_shape:deviation_pct` > 0.10"
)

# Pareto by outcome reason across scrap, rework, NOK.
log.events.groupby("ts_shape:outcome")["ts_shape:sample_count"].sum().sort_values()
```

No per-detector dispatch — the column names are stable across all 290+
methods.

### Adding a new detector method

1. Add a `LabelRule` entry to `REGISTRY` in
   `src/ts_shape/eventlog/taxonomy.py`. Pick `pack`, `family`, and
   `specifier` following the conventions above. Pick `shape` based on
   what the method returns (`point` / `interval` / `summary` /
   `static`).
2. If the detector emits multiple activities, use a templated
   specifier (`{column_name}`) and make sure the column is present in
   the legacy DataFrame.
3. If the legacy output uses a non-standard column name for severity
   or value, set `severity_field=` / `value_field=` on the
   `LabelRule`.
4. Classify the method in `ts_shape.eventlog.archetypes.ARCHETYPE_BY_METHOD`
   (one of `threshold`, `interval`, `aggregate`, `outcome`, `static`,
   `trace`, `forecast`, `correlation`). Populate the
   [required `standard_attrs`](#standard-attribute-extension) keys for
   that archetype.
5. Run the coverage test:

    ```bash
    pytest tests/eventlog/test_adapter_coverage.py -q
    ```

    It enforces:

    * every detector method has a rule, and every rule maps to a method
      (no orphans),
    * every key in `standard_attrs` is in the fixed vocabulary,
    * every method has an archetype, and the archetype's required keys
      are populated.
6. If your detector's shape is exotic (multiple events per row,
   nested data, runtime-dependent objects), [register a custom
   adapter](adapters.md#custom-adapters) instead of fighting the generic one.
