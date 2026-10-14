# Transforms and statistics

Filters, time conversions, calculations and harmonization take a long-format
frame and return one. Statistics classes summarize a column. Most of these are
stateless classmethods; plain pandas works just as well where it is shorter,
and the examples say so.

The examples on this page use this frame:

```python
import numpy as np
import ts_shape

df = ts_shape.make_timeseries(
    ["line1:temp", "line1:pressure", "line2:temp"], n_points=1440, freq="1min", start="2024-06-03 00:00"
)
```

## Filter rows

```python
from ts_shape.transform.filter.numeric_filter import DoubleFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.filter.string_filter import StringFilter
from ts_shape.transform.filter.custom_filter import CustomFilter

valid = DoubleFilter.filter_value_double_between(df, "value_double", min_value=95, max_value=105)
valid = DoubleFilter.filter_nan_value_double(valid, "value_double")

early_shift = DateTimeFilter.filter_between_datetimes(
    df, "2024-06-03 06:00:00", "2024-06-03 14:00:00", time_column="systime"
)

line1 = StringFilter.filter_string_contains(df, "line1:", value_column="uuid")   # literal substring
temp2 = StringFilter.filter_value_string_match(df, "line2:temp", value_column="uuid")

hot = CustomFilter.filter_custom_conditions(df, "value_double > 101 and uuid == 'line1:temp'")

# Plain pandas is fine too
selected = df[df["uuid"].isin(["line1:temp", "line2:temp"])]
```

Also available: `IntegerFilter`, `BooleanFilter`, `IsDeltaFilter`
(`ts_shape.transform.filter`), and `DateTimeFilter.filter_after_date`,
`filter_before_date`, `filter_between_dates`. The value or time column of a
filter is `value_column=` / `time_column=` (`column_name=` is deprecated). Edge
and change filters (`BooleanFilter`, `StringFilter.detect_changes_in_string`)
compare each sample with the previous sample of the same uuid in time order; a
missing boolean holds the previous state.

## Time zones and timestamps

```python
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift

local = TimezoneShift.shift_timezone(
    df, time_column="systime", input_timezone="UTC", target_timezone="Europe/Berlin"
)
```

`TimestampConverter.convert_to_datetime(df, columns=["systime"], unit="ms")`
converts epoch integers to tz-aware datetimes.

## Calculations

```python
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.transform.functions.lambda_func import LambdaProcessor

scaled = IntegerCalc.scale_column(df, "value_double", factor=0.1)
clipped = LambdaProcessor.apply_function(df, "value_double", lambda x: np.clip(x, 98, 102))

# Rolling values per signal: plain pandas
g = df.sort_values("systime").groupby("uuid")["value_double"]
rolling_mean = g.transform(lambda s: s.rolling(10, min_periods=1).mean())
```

`ts_shape.transform.calculator` also contains `PowerIntegrator` (kW to kWh)
and `UnitConverter` (needs `ts-shape[units]`).

## Harmonize several signals

`DataHarmonizer` turns signals sampled at different times into a common grid.

```python
from ts_shape.transform.harmonization import DataHarmonizer

line1_signals = df[df["uuid"].str.startswith("line1:")]
harmonizer = DataHarmonizer(line1_signals)
wide = harmonizer.pivot_to_wide()                       # one column per uuid
grid = harmonizer.resample_to_uniform(window="5min")    # wide, on a 5-minute grid
gaps = harmonizer.detect_gaps(threshold="5min")         # start, end, duration_seconds per gap
filled = harmonizer.fill_gaps(strategy="interpolate", max_gap="1h")
```

`align_asof(left_uuid, right_uuid, tolerance=..., direction="backward")`
attaches the latest value of one signal to each sample of another (an unknown
uuid warns). `fill_gaps(max_gap=...)` measures a gap from the valid sample
before it to the valid sample after it.

## Descriptive statistics

```python
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics

temp = df[df["uuid"] == "line1:temp"]
NumericStatistics.summary_as_dataframe(temp, "value_double")   # min, max, mean, percentiles, ...
NumericStatistics.column_quantile(temp, "value_double", 0.95)

hourly = TimeGroupedStatistics.calculate_statistic(
    df, time_column="systime", value_column="value_double", window="1h", stat_method="mean"
)   # one row per uuid and hour
```

`NumericStatistics` pools all rows it is given; filter to one signal first or
use `df.groupby("uuid")`. The same package has `StringStatistics`,
`BooleanStatistics` and `TimestampStatistics`.

Cycle and segment extraction, feature tables and pattern recognition are
covered in [Cycles and segments](feature-extraction.md).
