# Features for ML

How to turn long-format signals into inputs for a model or a comparison:
summary statistics per signal, one feature row per order or cycle, profiles that
compare orders with each other, recurring and unusual patterns in one signal, and
the relations between signals. The classes live in `ts_shape.features`.

## Example data

A press over twelve hours, one sample per minute: the mold temperature and the
hydraulic pressure, which follows the temperature three minutes later, and an
order signal that changes every two hours. During order PO-4 the pressure
surges for half an hour.

```python
import numpy as np
import pandas as pd

t = pd.date_range("2024-01-01 06:00", periods=720, freq="1min")
rng = np.random.default_rng(7)
temp = 180 + 3 * np.sin(np.arange(720) / 15) + rng.normal(0, 0.3, 720)
pressure = 4.0 + 0.1 * (np.roll(temp, 3) - 180) + rng.normal(0, 0.02, 720)
pressure[400:430] += 1.5
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "press:temp", "value_double": temp}),
    pd.DataFrame({"systime": t, "uuid": "press:pressure", "value_double": pressure}),
    pd.DataFrame({"systime": t[::120], "uuid": "press:order",
                  "value_string": [f"PO-{i}" for i in range(1, 7)]}),
], ignore_index=True)
signals = df[df["uuid"] != "press:order"]
```

## Statistics per signal

`DescriptiveFeatures` describes every signal in the frame: row count and time
span, and the statistics of each value column (numeric, string or boolean).

```python
from ts_shape.features.stats.feature_table import DescriptiveFeatures

stats = DescriptiveFeatures(signals).compute()
temp_stats = stats["press:temp"]["value_double"]["numeric_stats"]
print(round(temp_stats["mean"], 2), round(temp_stats["std"], 2), round(temp_stats["iqr"], 2))
```

- `compute()` returns a nested dict: per uuid an `overall` entry (`total_rows`,
  `total_time`, `is_delta_*`) and one entry per value column with its
  `numeric_stats`, `string_stats` or `boolean_stats`.
- `output_format="dataframe"` flattens the dict to one row per uuid with
  columns `<uuid>::<column>::<group>::<statistic>`; each row fills only the
  columns of its own uuid.
- Numeric statistics include mean, median, std, quantiles, IQR, MAD,
  skewness, kurtosis and the 10th and 90th percentiles.

### Statistics per time window

`compute_windowed(window)` (or `compute(output_format="windowed", window=...)`)
puts the same statistics on a time grid: one row per window, starting with
`start`, `end` and `duration_seconds`, and in that row every feature of every
signal as `{uuid}__{feature}`.

```python
hourly = DescriptiveFeatures(df).compute_windowed(
    "1h", metrics=["total_rows", "mean", "most_frequent"]
)
cols = ["start", "press:temp__mean", "press:order__most_frequent"]
print(hourly[cols].round({"press:temp__mean": 2}))
```

- The feature names are those of `compute()`: the `overall` keys
  (`total_rows`, `total_time`, `is_delta_*`, plus `total_time_seconds`) and
  the keys of `numeric_stats`, `boolean_stats` or `string_stats`, chosen by
  the value column each signal fills (`value_double` / `value_integer`,
  `value_bool`, `value_string`). Each cell equals `compute()` run on that
  window's rows of that signal.
- `metrics=` picks features; a name that does not fit a signal's kind is
  skipped for that signal (above, the order signal has no `mean`), an unknown
  name raises `ValueError` listing the valid ones.
- Every window from the first to the last timestamp is a row. A window
  without rows of a signal gives NaN for that signal (the order signal
  changes every two hours), except for counts such as `total_rows`,
  `true_count` or `count_null`, which are 0.
- Windows are those of `pd.Grouper(freq=window)`, wall-clock windows in the
  data's time zone: a `"1D"` window on a DST day lasts 23 or 25 hours.
- All statistics are grouped aggregations, so a million rows in 1 s windows
  take a few seconds.

**Which to pick:** `TimeWindowedFeatureTable` (see
[Cycles and segments](feature-extraction.md)) works on the output of
`SegmentProcessor.apply_ranges`: it splits each window by segment, reads one
numeric column (`value_double`) and drops groups with fewer than 2 values.
`compute_windowed` describes every signal of the frame by its own kind
(numeric, boolean or string), keeps every window and has no segments.

## One feature row per order

Cut the signals into orders (see [Cycles and segments](feature-extraction.md)) and
pivot them into a matrix with one row per order and one column per signal and
statistic: the shape scikit-learn and XGBoost expect.

```python
from ts_shape.features.export import FeatureMatrixExporter
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor

orders = SegmentExtractor.extract_time_ranges(df, "press:order")
cut = SegmentProcessor.apply_ranges(df, orders, target_uuids=["press:temp", "press:pressure"])

X = FeatureMatrixExporter.to_feature_matrix(cut, group_column="segment_value")
print(X[["press:pressure__value_double__mean", "press:pressure__value_double__max"]].round(2))
```

- Columns are named `{uuid}__{value_column}__{agg}`. The default aggregations
  are `mean`, `std`, `min` and `max`; pass `agg_funcs={"p90": lambda s: s.quantile(0.9)}`
  (names to pandas aggregation strings or callables) for others.
- Without `group_column` the whole frame is one row.
- Without `value_columns`, the numeric `value_*` columns are used; a
  `value_*` column stored as `object` (a JSON or CSV load) that holds numbers
  counts too and is coerced with `pd.to_numeric(errors="coerce")`.
- For features per time window and order, see `TimeWindowedFeatureTable`
  ([Cycles and segments](feature-extraction.md)).

## Comparing orders

`SegmentProcessor.compute_metric_profiles` gives one statistics vector per
signal and order. `ProfileComparison` measures the distance between those
vectors to find similar orders, outliers and shifts:

```python
from ts_shape.features.segment_analysis.profile_comparison import ProfileComparison

profiles = SegmentProcessor.compute_metric_profiles(cut)
pressure_profiles = profiles[profiles["uuid"] == "press:pressure"]

dist = ProfileComparison.compute_distance_matrix(pressure_profiles, group_column="segment_value")
print(ProfileComparison.detect_anomalous(dist, threshold=1.5))
groups = ProfileComparison.cluster(dist, n_clusters=2)
nearest = ProfileComparison.find_similar(dist, "PO-1", top_k=2)
changes = ProfileComparison.detect_changes(profiles)   # per uuid: change_score from one order to the next
```

PO-4, the order with the pressure surge, is far from all the others.

- `compute_distance_matrix` z-normalizes each statistic first
  (`normalize=True`), so statistics on different scales weigh the same.
  `distance_metric` is `euclidean`, `cosine` or `manhattan`; several rows per
  group are averaged.
- `detect_anomalous` flags items whose mean distance to the others has a
  z-score above `threshold`. With few items, a z-score cannot grow large: use a
  lower threshold than you would for a long series.
- `cluster` is hierarchical clustering (`linkage_method`: `average`,
  `complete`, `single`, `ward`); `distance_threshold=` cuts the tree at a
  distance instead of a cluster count.
- `find_similar_pairs` ranks the closest (uuid, order) pairs across signals.

## Patterns in one signal

`PatternRecognition` works on the samples of one signal (pass `signal_uuid=` when the
frame holds several). Window sizes and positions count samples:

```python
from ts_shape.features.pattern_recognition import PatternRecognition

discords = PatternRecognition.discover_discords(df, window_size=30, top_k=1, signal_uuid="press:pressure")
print(discords)
motif = PatternRecognition.discover_motifs(df, window_size=30, top_k=1, signal_uuid="press:temp")

template = temp[:94]   # one temperature cycle (2 * pi * 15 minutes)
hits = PatternRecognition.template_match(df, template, signal_uuid="press:temp", threshold=3.0)
occurrence = (hits["start_index"].diff() > 1).cumsum()   # neighboring starts are one match
best = hits.loc[hits.groupby(occurrence)["distance"].idxmin()]
print(best[["start_index", "distance", "start_time"]])
```

The most unusual 30-minute window starts with the pressure surge, and the
temperature cycle recurs every 94 minutes.

- `discover_motifs` returns the closest pair of windows (`index_a`,
  `index_b`, `distance`); `discover_discords` the windows whose nearest
  neighbor is farthest away. Both use the matrix profile with z-normalized
  windows, so they compare shapes, not levels. A window is never matched
  with one that overlaps it by `exclusion_zone` samples or less (default half
  the window), and the discords returned are that far apart. So are motif
  pairs: with `top_k > 1`, no window of a later motif starts within
  `exclusion_zone` samples of a window of an earlier one, so the next motifs
  are other recurring shapes, not the first one shifted by a sample or two.
- `similarity_search` ranks windows by dynamic time warping distance to a
  query; `template_match` returns every window start closer to a template
  than `threshold` (default: mean minus two standard deviations of all
  distances), so one occurrence yields several neighboring starts. On a
  periodic signal the default threshold admits little more than the template
  itself; pass a distance.
- A window that contains a missing value is skipped.

## Relations between signals

`CrossSignalAnalytics` takes a wide frame with one column per signal on a
common time index, for example from `DataHarmonizer.pivot_to_wide` or
`resample_to_uniform` ([Transforms](transforms.md)):

```python
from ts_shape.features.cross_signal import CrossSignalAnalytics
from ts_shape.transform.harmonization import DataHarmonizer

wide = DataHarmonizer(signals).pivot_to_wide().loc[:"2024-01-01 12:00"]   # before the surge
cross = CrossSignalAnalytics(wide)
lag = cross.lead_lag("press:temp", "press:pressure", max_lag=10)
print(lag["leader"], lag["optimal_lag"], round(lag["correlation_at_lag"], 2))
granger = cross.granger_causality("press:temp", "press:pressure", max_lag=5)
print(granger["is_causal"], granger["optimal_lag"])
lags = cross.lead_lag_matrix(max_lag=10)          # lags.loc[A, B] > 0: A leads B
te = cross.pairwise_transfer_entropy(lag=3)        # te.loc[A, B]: information from A to B
```

The temperature leads the pressure by three samples (three minutes).

- Lags count rows of the wide frame, so resample irregular signals to a
  uniform grid first.
- Numbers stored as `object` or text are signals too: they are coerced with
  `pd.to_numeric(errors="coerce")` (other values become NaN, with a
  `DataQualityWarning`). A column without any number (labels, ids) is not a
  signal.
- `lead_lag` uses cross-correlation; `granger_causality` an F-test of
  whether past values of one signal improve the prediction of the other.
  `transfer_entropy` measures directed information flow in bits, also for
  non-linear relations. `synchronization_index` gives phase locking
  (`method="phase"`) or envelope correlation (`"amplitude"`) from 0 to 1.
- Only rows where both signals have a value are used.

Next: [Pipelines](pipeline-builder.md).
