# Pipeline

> Chain transforms and detectors into one reusable, declarative definition.

**Module:** `ts_shape.pipeline` — top-level export: `ts_shape.Pipeline`

---

## Why a Pipeline?

A typical ts-shape workflow chains several classes by hand: call a transform,
pass its output to a detector constructor, call the detector method, repeat.
Intermediate variables pile up and the wiring is re-written for every dataset.

`Pipeline` captures that wiring **once** and re-runs it on any DataFrame:

```python
from ts_shape import Pipeline
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

pipe = (
    Pipeline(name="sensor-quality")
    .transform(IntegerCalc, "scale_column", column_name="value_double", factor=0.1)
    .detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers", value_column="value_double", threshold=3.0)
)

result = pipe.run(df)          # reusable — call .run() on many DataFrames
result.data                    # final transformed signal
result.events["outliers"]      # detector output
result.to_event_log()          # normalized, combined OCEL event log
```

---

## Two kinds of step

The pipeline is **linear single-channel**. You declare each step's role —
it is never inferred:

| Method | What it does |
|--------|--------------|
| `.transform(...)` | Output **replaces** the working signal — the signal flows on. |
| `.detect(...)` | Output is **stored** under a name in `result.events`; the working signal is left untouched (detectors branch off). |

---

## Step targets

A step target can be supplied two ways:

**Plain callable** — anything `df -> df`:

```python
pipe.transform(lambda df: df[df["uuid"].isin(["temperature", "pressure"])],
               name="select_signals")
```

**`(class, "method")` pair** — the pipeline inspects the class and does the
right thing:

- a `@classmethod` / `@staticmethod` (stateless transforms like `IntegerCalc`,
  the filters) is called directly on the class;
- an instance method (detectors, `DataHarmonizer`, …) instantiates the class
  first.

Keyword arguments are routed between the constructor and the method
automatically by parameter name, so you pass them all flat:

```python
pipe.detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers",
            value_column="value_double",   # -> OutlierDetectionEvents.__init__
            threshold=3.0)                 # -> detect_outliers_zscore(...)
```

An unknown keyword argument raises a `ValueError` that lists what the
constructor and the method accept.

---

## Wiring two DataFrames — sentinels

Most steps just pass the working DataFrame forward. Some need a *second*
DataFrame. Any keyword-argument value may be a sentinel string:

- `"$input"` — the DataFrame originally passed to `run()`;
- `"$prev"` — the current working DataFrame.

```python
pipe = (
    Pipeline(name="segmented")
    .transform(SegmentExtractor, "extract_time_ranges", segment_uuid="order_number")
    .transform(SegmentProcessor, "apply_ranges",
               dataframe="$input",      # the original raw data
               time_ranges="$prev")     # the ranges from the previous step
)
```

If a kwarg names the step's own DataFrame parameter (like `dataframe` above),
that value is used instead of the auto-injected working signal. Unknown
`$`-sentinels are rejected at build time.

---

## Running and debugging

```python
print(pipe.describe())     # preview every step (and its kwargs) without running

result = pipe.run(df)      # -> PipelineResult(.data, .events, .to_event_log())

steps = pipe.run_steps(df) # dict of every intermediate DataFrame:
steps["input"]             #   the original frame
steps["scale_column"]      #   the signal after that transform
steps["outliers"]          #   that detector's events
```

If a step raises, the error names the failing step (index and name). A step
that returns a non-DataFrame raises a `TypeError`.

---

## Event-log export

Because the pipeline records each detector step's `"Class.method"` identity,
`PipelineResult.to_event_log()` normalizes every detector output into one
canonical OCEL `EventLog` (via `ts_shape.eventlog.to_event_log` + `concat`).
Pass `concat=False` for a per-step `dict`.

---

## Next steps

- [Feature Extraction](feature-extraction.md) — cycles vs segments
- [Event Log (XES & OCEL)](eventlog.md) — what `to_event_log()` produces
- [API Reference](../reference/ts_shape/pipeline.md) — full parameter docs
