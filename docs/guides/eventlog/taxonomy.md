# Labeling and taxonomy

The rules that fill `ocel:activity` and the standard attributes of every event.
They keep the events of any two detectors comparable without per-detector
code downstream. Each detector method has one `LabelRule` in
`ts_shape.eventlog.taxonomy.REGISTRY`; `tests/eventlog/test_adapter_coverage.py`
fails when a public detector method has no rule.

## Activity names

```
<pack>.<family>.<specifier>[.<subtype>]
```

- All segments are lowercase snake_case, separated by dots.
- `pack` is one of `quality`, `production`, `engineering`, `maintenance`,
  `energy`, `correlation`, `supplychain`, `development`.
- `family` is the topic within the pack (`outlier`, `machine_state`,
  `setpoint`, ...). Reuse an existing family where one fits.
- `specifier` distinguishes methods within the family (`zscore`, `iqr`, `mad`
  for `quality.outlier`).
- A segment may be a `{column}` placeholder, filled from the detector's
  output row. A missing value renders as `unknown`.

| Detector method | `ocel:activity` |
|---|---|
| `OutlierDetectionEvents.detect_outliers_zscore` | `quality.outlier.zscore` |
| `StatisticalProcessControlEvents.detect_violations` | `quality.spc.rule_violation` |
| `MachineStateEvents.detect_run_idle` | `production.machine_state.{state}` (`...run`, `...idle`) |
| `MachineStateEvents.transition_events` | `production.machine_state.transition_{transition}` |
| `SetpointChangeEvents.detect_setpoint_steps` | `engineering.setpoint.step_{change_type}` |
| `DegradationDetectionEvents.detect_trend_degradation` | `maintenance.degradation.trend` |

The families in use can be listed from the registry:

```python
from ts_shape.eventlog.taxonomy import REGISTRY

families = sorted({rule.template.split(".")[1] for rule in REGISTRY.values() if rule.pack == "quality"})
```

Severity is never part of the activity name; it goes into `ts_shape:severity`.

## Attribute prefixes

| Prefix | Source | Examples |
|---|---|---|
| `ocel:` | OCEL 2.0 | `ocel:eid`, `ocel:activity`, `ocel:timestamp`, `ocel:oid`, `ocel:type`, `ocel:qualifier` |
| `ts_shape:` | ts-shape fields without an OCEL counterpart | `ts_shape:start_timestamp`, `ts_shape:duration_s`, `ts_shape:detector`, `ts_shape:severity` |
| `<pack>:` | the detector's own output columns | `production:state`, `quality:triggered_rule`, `engineering:magnitude` |
| `concept:`, `time:`, `case:`, `lifecycle:`, `org:` | XES, added by `to_event_log_xes` only | `concept:name`, `time:timestamp`, `case:concept:name` |

The pack prefix keeps columns of different packs apart: two packs that both
output a `state` column produce `production:state` and `quality:state`.

## Severity

When a rule names a numeric `severity_field`, its value is bucketed:

| Value | `ts_shape:severity` |
|---|---|
| `< 3.0` | `info` |
| `3.0` to `< 4.5` | `warn` |
| `>= 4.5` | `critical` |
| missing or not numeric | `<NA>` |

The thresholds match the z-score style `severity` column of
`OutlierDetectionEvents`. A text column that already holds `info`, `warn` or
`critical` is passed through.

## Object types and qualifiers

`asset` is created automatically from `source_uuid`. The other standard object
types are `tool`, `sensor`, `signal`, `station` (equipment), `cycle`, `batch`,
`lot`, `recipe`, `work_order`, `shift` (process context), `material`, `part`,
`serial`, `article` (product) and `operator`. Add your own with
`register_object_type("name")`.

Recommended values for `ocel:qualifier` (free text is allowed):

| Qualifier | Object type | Meaning |
|---|---|---|
| `produced_on` | `asset` | the event happened on this asset |
| `during_batch` | `batch` | the event happened while this batch ran |
| `during_cycle` | `cycle` | the event happened during this cycle |
| `during_shift` | `shift` | the event happened during this shift |
| `made_of` | `material` | the product contained this material |
| `identified_by` | `serial` | the product carries this serial number |
| `operated_by` | `operator` | the operator on duty |
| `measured_by` | `sensor` | the sensor that produced the reading |
| `governed_by` | `recipe` | the recipe in effect |

## Standard attribute extension

Besides the core columns, each `LabelRule` maps detector output to a fixed set
of attribute keys (`ts_shape.eventlog.schema.STANDARD_ATTR_KEYS`). Two
detectors that report the same kind of thing then use the same column.

| Key | Type | Content |
|---|---|---|
| `ts_shape:method` | string | algorithm, for example `zscore`, `western_electric`, `cusum` |
| `ts_shape:baseline` | float | expected value (center line, setpoint, baseline mean) |
| `ts_shape:threshold_low` | float | lower bound; `NaN` if one-sided |
| `ts_shape:threshold_high` | float | upper bound; `NaN` if one-sided |
| `ts_shape:deviation` | float | `value - baseline` |
| `ts_shape:deviation_pct` | float | `(value - baseline) / baseline` |
| `ts_shape:direction` | string | `above`, `below`, `up`, `down`, `outside`, `inside`, `lead`, `lag`, `shift` |
| `ts_shape:confidence` | float | 0 to 1, for probabilistic detectors |
| `ts_shape:sample_count` | int | number of observations behind the row |
| `ts_shape:outcome` | string | `ok`, `nok`, `rework`, `scrap`, `pass`, `fail` or a reason code |
| `ts_shape:lifecycle_state` | string | `raised`, `cleared`, `predicted` or a state name (`run`, `idle`) |
| `ts_shape:lifecycle_pair_id` | string | pairs a raise and a clear into one occurrence |

Each entry in `standard_attrs` is either the name of a detector output column
(renamed and converted to the declared type), a literal broadcast to every row,
or `None` (not applicable). Two registry entries as examples:

```python
from ts_shape.eventlog.taxonomy import LabelRule

LabelRule(
    template="quality.outlier.zscore",
    pack="quality",
    shape="point",
    severity_field="severity",
    standard_attrs={"ts_shape:method": "zscore", "ts_shape:direction": "outside"},  # literals
)
LabelRule(
    template="production.cycle_time.statistics",
    pack="production",
    shape="static",
    standard_attrs={"ts_shape:sample_count": "count"},  # taken from the "count" column
)
```

With these keys, questions across detectors need no per-detector code:

```python
import ts_shape
from ts_shape.eventlog import concat, to_event_log

df = ts_shape.make_timeseries(["sensor:temp"], n_points=1000, n_outliers=5)
detector = ts_shape.OutlierDetectionEvents(df, value_column="value_double")
log = concat(
    to_event_log(detector.detect_outliers_zscore(), detector="OutlierDetectionEvents.detect_outliers_zscore"),
    to_event_log(detector.detect_outliers_mad(), detector="OutlierDetectionEvents.detect_outliers_mad"),
)
log.events.groupby(["ts_shape:method", "ts_shape:direction"])["ocel:eid"].count()
```

### Required keys per archetype

Every method is assigned one archetype in
`ts_shape.eventlog.archetypes.ARCHETYPE_BY_METHOD`, and must fill that
archetype's required keys (checked by `test_required_standard_attrs_per_archetype`):

| Archetype | Required | Typical optional |
|---|---|---|
| `threshold` | `method`, `direction` | `baseline`, `threshold_low`, `threshold_high`, `deviation`, `deviation_pct`, `confidence` |
| `interval` | `lifecycle_state` | `lifecycle_pair_id`, `sample_count`, `direction` |
| `aggregate` | `sample_count` | `baseline`, `threshold_low`, `threshold_high`, `method` |
| `outcome` | `outcome` | `sample_count`, `method` |
| `static` | `method` | `sample_count`, `baseline`, `threshold_low`, `threshold_high` |
| `trace` | `lifecycle_state`, `direction` | `sample_count` |
| `forecast` | `method`, `confidence` | `baseline`, `threshold_low`, `threshold_high` |
| `correlation` | `method` | `direction`, `confidence`, `sample_count` |

## Adding a new detector method

1. Add a `LabelRule` to `REGISTRY` in `src/ts_shape/eventlog/taxonomy.py`,
   following the naming rules above. Set `shape` to what the method returns
   (`point`, `interval`, `summary` or `static`).
2. For several activities from one method, use a `{column}` placeholder and
   make sure the column is in the output.
3. If severity or the main value are in non-standard columns, set
   `severity_field` or `value_field`.
4. Add the method to `ARCHETYPE_BY_METHOD` in
   `src/ts_shape/eventlog/archetypes.py` and fill the archetype's required
   `standard_attrs`.
5. Run `pytest tests/eventlog/test_adapter_coverage.py -q`. It checks that
   every method has a rule and an archetype, that no rule is orphaned, that
   every `standard_attrs` key is in the vocabulary and that the required keys
   are filled.
6. If the output does not fit the generic adapter (several events per row,
   nested data), [register a custom adapter](adapters.md#custom-adapters).
