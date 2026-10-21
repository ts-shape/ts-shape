# Chunked processing

Some datasets do not fit in memory: a year of 1 Hz data from a few hundred
signals is billions of rows. `ts_shape.run_chunked` processes such data **slice
by slice** -- one day at a time by default. It loads a slice, runs the detector
on it, keeps only what the next slice needs, and moves on, so memory holds
about one slice instead of the whole range.

Events and windows that cross a slice border are **stitched**: a stop that
starts at 23:50 and ends at 00:20 comes out as one stop, not two halves, and a
shift from 22:00 to 06:00 is one row. The result is the same as running the
detector on all the data at once:

```text
run_chunked(days, Target, "method", **kwargs)
    ==  Target(all_data, **init_kwargs).method(**method_kwargs)
```

(up to float rounding and row order). Methods that can only work on the whole
signal at once raise `NotChunkableError` instead of returning something
slightly different.

## When to use it

Use it when the data of the time range you analyse is larger than the memory
you have. A long-format frame takes about 38-59 bytes per row (measured with
`benchmarks/run.py`), and a detector needs a few times its input while it runs.
Size the slice so that two or three slices fit:

| Data | Rows per day | Memory per day |
|---|---|---|
| 10 signals at 1 Hz | 864 000 | 33-51 MB |
| 100 signals at 1 Hz | 8.6 million | 330-510 MB |
| 1 000 signals every 10 s | 8.6 million | 330-510 MB |

A detector reads only the signals named in its arguments
(`run_state_uuid=...`, `signal_uuid=...`), and chunked processing drops the
other signals as each frame arrives, so what counts is usually the size of the
signals the detector uses, not of the whole frame.

When the data fits in memory, run the detector directly: it is simpler and
works for every method.

## A first run

Ten days of a machine's run state at one-minute resolution, delivered one day
at a time. The chunked result equals the in-memory result:

```python
import numpy as np
import pandas as pd
from ts_shape import MachineStateEvents, run_chunked

t = pd.date_range("2024-03-25", "2024-04-04", freq="1min", inclusive="left",
                  tz="Europe/Berlin")
rng = np.random.default_rng(7)
state = np.repeat(rng.random(len(t) // 97 + 1) < 0.7, 97)[: len(t)]
df = pd.DataFrame({"systime": t, "uuid": "press:run", "value_bool": state})


def days():
    """Yield the data one local day at a time (as a loader would)."""
    for _, day in df.groupby(df["systime"].dt.normalize()):
        yield day


chunked = run_chunked(days(), MachineStateEvents, "detect_run_idle",
                      run_state_uuid="press:run", min_duration="2h")
in_memory = MachineStateEvents(df, "press:run").detect_run_idle(min_duration="2h")

pd.testing.assert_frame_equal(chunked, in_memory.reset_index(drop=True))
print(chunked[["start", "end", "state"]].head())
```

Keyword arguments go to the constructor or the method by name, exactly as in
[Pipeline.detect](pipeline-builder.md). The data crosses the change to summer
time on 31 March: that daily slice lasts 23 hours, and a run that spans
midnight, or several days, comes out once with its full length, so
`min_duration` filters it correctly.

`iter_chunked` takes the same arguments and yields the finished rows slice by
slice instead of collecting them, for example to write one result file per
day. Every row is yielded exactly once:

```python
from ts_shape import iter_chunked

parts = list(iter_chunked(days(), MachineStateEvents, "detect_run_idle",
                          run_state_uuid="press:run"))
print(len(parts), sum(len(p) for p in parts))
```

## Sources

`run_chunked` reads from any iterable of long-format frames. The frames must
come **in time order at slice granularity**: the rows of one day may arrive in
any order and in several frames, but once a row of a later day has been seen,
a row of an earlier day raises `ValueError` (that day has already been
processed).

- **Frames or `(name, frame)` pairs.** The parquet loaders'
  `stream_by_time_range` yields `(path, frame)` one hourly folder at a time, in
  time order, so it can be passed directly:

<!-- no-run -->

```python
from ts_shape import MicroStopEvents, run_chunked
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

loader = AzureBlobParquetLoader(container_name="historian", connection_string="...")
stops = run_chunked(
    loader.stream_by_time_range("2024-01-01", "2024-12-31 23:59:59"),
    MicroStopEvents, "detect_micro_stops", run_state_uuid="line1:run",
)
```

- **`time_range_source(load, start, end, chunk="1D")`** calls `load(slice_start,
  slice_end)` once per slice, for databases and any loader with a time-range
  query. Rows outside the half-open `[slice_start, slice_end)` are dropped, so
  a query with an inclusive end does not deliver a row twice:

<!-- no-run -->

```python
from ts_shape import DowntimeEvents, run_chunked, time_range_source
from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess


def load(start, end):
    return TimescaleDBDataAccess(
        start=str(start), end=str(end), uuids=["line1:state"], db_config=db_config
    ).fetch_data_as_dataframe()


shifts = run_chunked(
    time_range_source(load, "2024-01-01", "2025-01-01"),
    DowntimeEvents, "downtime_by_shift", run_state_uuid="line1:state",
)
```

```python
from ts_shape import time_range_source


def load(start, end):
    return df[(df["systime"] >= start) & (df["systime"] <= end)]


source = time_range_source(load, t[0], t[-1] + pd.Timedelta("1min"))
from_db = run_chunked(source, MachineStateEvents, "detect_run_idle",
                      run_state_uuid="press:run", min_duration="2h")
pd.testing.assert_frame_equal(from_db, chunked)
```

Slices are aligned to the wall clock of the data's time zone (`chunk="1D"` cuts
at local midnight; a DST day lasts 23 or 25 hours). Naive timestamps work the
same way. `chunk` can be any fixed length (`"12h"`, `"2D"`) or a calendar
period, see [Calendar chunks](#calendar-chunks).

## Calendar chunks

`chunk="W"` cuts the data into calendar weeks, Monday 00:00 to Monday 00:00
local time (pandas' weeks ending Sunday, also written `"W-SUN"`; `"W-<DAY>"`
gives weeks ending on another day). `chunk="MS"` cuts it into calendar months.
The cuts are made at local midnight like daily slices, so the week of a DST
change lasts 167 or 169 hours and February its real length.

`PeriodSummaryEvents.weekly_summary` and `monthly_summary` run chunked only
with the matching chunk -- every week or month is then exactly one slice --
and raise `ValueError` with any other chunk. Methods with a window that
divides a day (`"1h"`, `"1D"`) and interval methods run with any calendar
chunk. The source may still deliver daily frames; slices collect them:

```python
from ts_shape import PeriodSummaryEvents

ct = pd.date_range("2024-02-01", "2024-05-01", freq="30min", inclusive="left",
                   tz="Europe/Berlin")
counter = pd.DataFrame({"systime": ct, "uuid": "press:count",
                        "value_integer": np.arange(len(ct)) * 3})
counter_days = (g for _, g in counter.groupby(counter["systime"].dt.normalize()))

weekly = run_chunked(counter_days, PeriodSummaryEvents, "weekly_summary",
                     chunk="W", counter_uuid="press:count")
expected = PeriodSummaryEvents(counter).weekly_summary("press:count")
pd.testing.assert_frame_equal(weekly, expected.reset_index(drop=True))
print(weekly[["start", "end", "total_production"]].head(3))
```

`time_range_source(load, start, end, chunk="W")` loads one week per call on
the same grid.

## How the borders are stitched

Each slice runs on its own rows plus, for every signal, the last reading before
the slice (a state that was set yesterday still holds) and the first reading
after it (the last interval of the slice closes where it really closes, and a
counter increment over midnight is shared correctly). Then, by the shape of
the method's output:

- **Point events** are kept when their time lies in the slice.
- **Intervals** still open at the end of the slice are not emitted yet: their
  rows are carried into the next slice, which computes them again with more
  data. An interval that spans three days is emitted once, by the slice where
  it ends. `max_carry` (default 7 days) bounds how long an interval may stay
  open; a longer one raises `ValueError`.
- **Window summaries** (per hour, per day, per shift) are kept when the window
  starts in the slice; a window that ends after the slice, such as a 22:00-06:00
  night shift, is carried like an open interval. The chunk must be a whole
  multiple of the method's `window` (`"1D"` with `window="1h"` or `"15min"`),
  and a whole number of days for per-shift reports.

Rolling methods get history before each slice: `overlap` defaults to the
method's window-like argument (`window`, `baseline_window`, `lookback`, ...),
and can be passed explicitly.

Some events are only settled by data after them, and wait for it in the same
way: a changeover whose new product must hold for `min_duration` (also the
`min_duration` of `changeover_window`'s `config`), and a stop that counts as
a micro-stop only up to `max_duration` -- a stop still running at midnight is
classified by its full length, in the windows of both days.

## Pipelines

`Pipeline.run_stream(source, chunk="1D")` runs a [pipeline](pipeline-builder.md)
the same way: every detect step goes through the chunked engine and returns
the events `run()` would return. As in `run()`, each detect step sees the
slice transformed only by the transform steps before it. Transforms run on
each slice separately, so
they must be **row-local** -- each output row depends only on its own input
row, as unit conversions, scaling and value mappings do. `result.data` is
empty by default (the point is not to hold all rows); pass `keep_data=True` to
get the transformed data of every slice.

```python
from ts_shape import IntegerCalc, Pipeline, ThresholdMonitoringEvents

temp = pd.DataFrame({"systime": t, "uuid": "press:temp",
                     "value_double": 60 + 10 * np.sin(np.arange(len(t)) / 300)})
data = pd.concat([df, temp], ignore_index=True)

pipe = (
    Pipeline(name="press")
    .transform(IntegerCalc, "scale_column", value_column="value_double", factor=0.1)
    .detect(MachineStateEvents, "detect_run_idle", name="runs",
            run_state_uuid="press:run")
    .detect(ThresholdMonitoringEvents, "time_above_threshold", name="hot",
            signal_uuid="press:temp", threshold=6.5, window="1h")
)
streamed = pipe.run_stream(g for _, g in data.groupby(data["systime"].dt.normalize()))
full = pipe.run(data)
for name in ("runs", "hot"):
    pd.testing.assert_frame_equal(streamed.events[name],
                                  full.events[name].reset_index(drop=True))
print(streamed)
```

## Supported methods

`ts_shape.stream.SUPPORT` lists the methods that run chunked. The list is
data-driven: a test runs every detector method in memory and in daily slices on
a 3-day dataset across a DST change (and on variants with skewed statistics
and shifted borders; the weekly and monthly summaries in weekly and monthly
slices of a 9-week dataset) and keeps the methods whose results are equal and
whose code uses no whole-signal statistics.

```python
from collections import defaultdict

from ts_shape.stream import SUPPORT

by_class = defaultdict(list)
for name in sorted(SUPPORT):
    cls, method = name.split(".")
    by_class[cls].append(method)
for cls, methods in by_class.items():
    print(f"{cls}: {', '.join(methods)}")
```

A few results need the whole run; the engine completes them at the end of
the stream, so they still equal the in-memory result:

- `DataGapAnalysisEvents.detect_gaps` counts `samples_before_gap` and
  `samples_after_gap` over the whole signal: a running count per signal
  offsets each slice, and `samples_after_gap` is filled in once the total is
  known. `iter_chunked` therefore yields all of its rows together, after the
  last slice.
- `DataGapAnalysisEvents.coverage_by_period` counts one median sampling step
  of the whole signal after its last sample as covered: the engine counts
  every sampling step and corrects the last window at the end (`iter_chunked`
  yields that window last).
- `MicroStopEvents.micro_stop_frequency` reports every window from the first
  to the last micro-stop, with zeros in between: the zero windows between two
  slices are inserted when they are merged, on the method's window grid.

## Conditional support

Some methods run chunked only when one argument is passed explicitly: their
default takes something from the whole signal, which an explicit value
replaces. `ts_shape.stream.CONDITIONAL` maps each of them to the argument and
the reason:

| Method | Pass | Because the default |
|---|---|---|
| `DataGapAnalysisEvents.detect_gaps` | `min_gap=...` | is twice the whole signal's median sampling step |
| `DataGapAnalysisEvents.coverage_by_period` | `min_gap=...` | is twice the whole signal's median sampling step |
| `AnomalyClassificationEvents.detect_flatline` | `interval_end="last_sample"` | `"next_change"` keeps a one-sample run only when the gap after it is longer than the whole signal's median sampling step |
| `ChangeoverEvents.changeover_window` | `until="fixed_window"` (the default) | `"stable_band"` compares against an expanding median over all data after each change |

Without the argument the call raises `NotChunkableError`, which names it:

```python
from ts_shape import DataGapAnalysisEvents, NotChunkableError
from ts_shape.stream import CONDITIONAL

print(CONDITIONAL["DataGapAnalysisEvents.detect_gaps"].argument)
try:
    run_chunked(days(), DataGapAnalysisEvents, "detect_gaps",
                signal_uuid="press:run")
except NotChunkableError as exc:
    print(exc)

gaps = run_chunked(days(), DataGapAnalysisEvents, "detect_gaps",
                   signal_uuid="press:run", min_gap="2min")
```

## Limitations

Any other method raises `NotChunkableError` (a `ValueError`) with the reason
and the in-memory alternative:

```python
from ts_shape import NotChunkableError, OutlierDetectionEvents

try:
    run_chunked(days(), OutlierDetectionEvents, "detect_outliers_zscore",
                value_column="value_double")
except NotChunkableError as exc:
    print(exc)
```

These need the whole signal at once:

- **whole-signal statistics** -- z-score, IQR, MAD and isolation-forest
  outliers, SPC control limits and rules, Hotelling T² without a reference
  period, process stability scores;
- **fits over all data** -- change points (PELT / BIC), Weibull fits,
  regressions over the whole range, rolling trends over earlier windows;
- **baselines from the first window** -- asset health, degradation and sensor
  drift baselines;
- **unbounded look-back** -- cycle times measured from the previous cycle,
  "time since the last transition", operating-hour meters from the start of
  the data;
- **static summaries** -- one table for the whole range (`*_statistics`,
  `*_by_reason`, rankings).

Run those in memory on a range that fits, or on a pre-aggregated frame.

Two more rules:

- On data in a time zone with daylight saving time, a window of several hours
  that is not a whole number of days (`"8h"`, `"4h"`) raises `ValueError`:
  pandas places such bins in absolute time, so after a DST change they start
  at a different hour than in a slice that starts later. Use a window that
  divides one hour, whole days, or convert the data to UTC.
- A signal whose next reading is far ahead (a change-encoded setpoint that
  changes once a week) makes the run wait for that reading, up to
  `max_carry`; after that its last reading is treated as its last one and a
  `DataQualityWarning` names it.
