# Lambda rules

A lambda rule is a detector declared as data (YAML or a dict) instead of a
Python class: a boolean trigger expression over the columns of a DataFrame,
plus the label rule that names its events. Registering it adds it to the same
`REGISTRY` as the built-in detectors, so its output goes through the same
event-log conversion, severity mapping, object extraction and export.

Use a lambda rule for a plant-specific threshold or condition that no built-in
detector covers. For anything that needs state across rows beyond contiguous
runs, write a detector class.

## Example data

Two assets with torque, a run state, a precomputed severity score and a
bearing temperature, sampled every 15 seconds. A lambda rule works on any
columns of a wide frame like this one; it does not use the long
`uuid` / `value_*` format.

```python
import pandas as pd

t = pd.date_range("2026-05-07 08:00", periods=8, freq="15s", tz="UTC")
df = pd.DataFrame({
    "systime": list(t) * 2,
    "source_uuid": ["asset-A"] * 8 + ["asset-B"] * 8,
    "torque": [72, 74, 79.5, 86, 73, 95, 68, 70, 70, 71, 72, 70, 69, 77, 71, 70],
    "state": ["run"] * 6 + ["idle"] * 2 + ["run"] * 8,
    "severity_score": [0, 0, 3.2, 4.6, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, 3.1, 0, 0],
    "bearing_temp_c": [82, 86, 88, 87, 84, 80, 80, 80, 89, 83, 90, 91, 92, 92, 84, 80],
})
```

## A point rule

One event per row where torque is above 75 while the machine runs. The
severity comes from `severity_score` (below 3 `info`, below 4.5 `warn`, else
`critical`), the event value from `torque`, and the `asset` object from
`source_uuid`.

```python
from ts_shape.eventlog import RuleSpec, TriggerSpec, register_lambda_rule

high_torque = register_lambda_rule(RuleSpec(
    id="high_torque",
    class_name="LambdaToolWear",
    method_name="high_torque",
    pack="maintenance",
    shape="point",
    archetype="threshold",
    template="maintenance.tool.high_torque",
    trigger=TriggerSpec(expression="(torque > 75) & (state == 'run')"),
    severity_field="severity_score",
    value_field="torque",
    standard_attrs={
        "ts_shape:method": "lambda_threshold",
        "ts_shape:direction": "above",
        "ts_shape:threshold_high": 75.0,
    },
))
```

## An interval rule

One event per contiguous run of rows where the bearing is above 85 °C, per
asset, kept only if it lasts at least 30 seconds. Each event carries `start`,
`end` and the mean of `value_field` over the run. The same rule as YAML-style
dicts, loaded with `load_dicts`:

```python
from ts_shape.eventlog import load_dicts

[bearing_hot] = load_dicts([{
    "id": "bearing_hot",
    "class_name": "LambdaBearing",
    "method_name": "hot_window",
    "pack": "maintenance",
    "shape": "interval",
    "archetype": "interval",
    "template": "maintenance.bearing.hot",
    "value_field": "bearing_temp_c",
    "trigger": {"expression": "bearing_temp_c > 85", "min_duration_s": 30, "group_by": ["source_uuid"]},
    "standard_attrs": {"ts_shape:lifecycle_state": "hot"},
}])
```

In a YAML file the rules are a list under `rules:`; see
[`examples/lambda_rules_demo.yaml`](https://github.com/ts-shape/ts-shape/blob/main/examples/lambda_rules_demo.yaml).
Load it with `load_yaml("rules.yaml")` (needs `ts-shape[yaml]`).

## Running rules

```python
from ts_shape.eventlog import concat, run_backtest

frame = high_torque.evaluate(df)     # the detector output, before conversion
log = concat(high_torque.to_event_log(df), bearing_hot.to_event_log(df))

result = run_backtest(high_torque, df)   # a BacktestResult
result.hit_count      # 4
result.by_severity    # {'warn': 2, 'critical': 2}
result.by_asset       # {'asset-A': 3, 'asset-B': 1}
result.event_log      # the full EventLog
```

`run_backtest` returns a `BacktestResult` (`ts_shape.eventlog`): the
`detector` name, `hit_count`, `by_severity` (hits per `ts_shape:severity`,
`"unset"` for none), `by_asset` (hits per `asset` object) and the full
`event_log`. It counts hits only; there is no label matching or
precision/recall.

Rules live in the current process only.
`unregister_lambda_rule("LambdaToolWear", "high_torque")` removes one again,
for example between tests or when reloading a notebook.

## Expression language

Trigger expressions are a small subset of Python, parsed with
`ast.parse(mode="eval")` and checked against an allowlist:

- comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`;
- element-wise logic: `&` (and), `|` (or), `~` (not);
- membership: `column.isin(['run', 'setup'])`;
- arithmetic: `+`, `-`, `*`, `/`, `%`, unary `-` and `+`;
- constants and column names;
- the function `abs`, and `min` / `max` over constants.

Anything else (imports, other attribute access, indexing, other calls,
lambdas, comprehensions) raises `UnsafeExpression` when the rule is compiled.
So do Python's `and`, `or`, `not`, `in` / `not in`, chained comparisons and
`min` / `max` over columns: on a column they are not element-wise, and the
error message points to `&`, `|`, `~` and `.isin(...)`. An expression that
does not give True / False per row (for example `torque * 2`) is rejected
too, and a name that is not a column of the frame raises
`ColumnNotFoundError` when the rule runs. Rules are compiled when they are
registered; `load_dicts` registers a batch all or nothing.

`&` and `|` bind tighter than comparisons, as in `DataFrame.query`, so put
each comparison in parentheses: `(torque > 75) & (state == 'run')`, not
`torque > 75 & state == 'run'` (rejected as a chained comparison).

Interval rules join consecutive True rows (per `group_by`) into one event
that lasts until the next sample after the last True row, half-open like
`ts_shape.signals.primitives.intervals`; a run at the end of the data ends at
its last sample.

## RuleSpec fields

| Field | Content |
|---|---|
| `id` | rule id |
| `class_name`, `method_name` | the name the rule is registered under; `to_event_log(frame, detector="LambdaToolWear.high_torque")` also works |
| `pack`, `template`, `shape`, `produces_objects`, `severity_field`, `value_field`, `standard_attrs` | as in a `LabelRule`; see [Adapters](eventlog/adapters.md#labelrule-fields) |
| `archetype` | one of the [archetypes](eventlog/taxonomy.md#required-keys-per-archetype); its required `standard_attrs` must be given |
| `trigger.expression` | the boolean expression |
| `trigger.min_duration_s` | interval rules: shortest run kept, in seconds |
| `trigger.group_by` | interval rules: columns to find runs in separately, for example per asset |

A complete script is
[`examples/lambda_rules_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/lambda_rules_demo.py).
