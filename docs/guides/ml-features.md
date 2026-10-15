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

`PatternRecognition` works on the samples of one signal (pass `uuid=` when the
frame holds several). Window sizes and positions count samples:

```python
from ts_shape.features.pattern_recognition import PatternRecognition

discords = PatternRecognition.discover_discords(df, window_size=30, top_k=1, uuid="press:pressure")
print(discords)
motif = PatternRecognition.discover_motifs(df, window_size=30, top_k=1, uuid="press:temp")

template = temp[:94]   # one temperature cycle (2 * pi * 15 minutes)
hits = PatternRecognition.template_match(df, template, uuid="press:temp", threshold=3.0)
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
  the window), and the discords returned are that far apart. Motif pairs are
  not: with `top_k > 1`, the next motifs can be the first one shifted by a
  sample or two.
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
- `lead_lag` uses cross-correlation; `granger_causality` an F-test of
  whether past values of one signal improve the prediction of the other.
  `transfer_entropy` measures directed information flow in bits, also for
  non-linear relations. `synchronization_index` gives phase locking
  (`method="phase"`) or envelope correlation (`"amplitude"`) from 0 to 1.
- Only rows where both signals have a value are used.

Next: [Pipelines](pipeline-builder.md).
