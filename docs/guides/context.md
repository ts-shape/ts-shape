# Context and enrichment

Raw signals carry a `uuid` and a value. Reports and detectors usually need more:
a description and unit per signal, tolerance limits, readable names for state
codes. This page joins that context onto the long-format frame with
`ContextEnricher` (lookup tables you already have in memory) and `ValueMapper`
(a mapping file on disk). To join signal metadata while loading, see
[Combining timeseries with metadata](loading.md#combining-timeseries-with-metadata).

## Example data

Two process signals and a machine state that the PLC logs as a numeric code
in `value_string`, one row per minute:

```python
import numpy as np
import pandas as pd

t = pd.date_range("2024-01-01 06:00", periods=20, freq="1min")
rng = np.random.default_rng(42)
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "temp:zone_a", "value_double": rng.normal(85, 3, 20)}),
    pd.DataFrame({"systime": t, "uuid": "press:zone_a", "value_double": rng.normal(2.1, 0.2, 20)}),
    pd.DataFrame({"systime": t, "uuid": "state:machine1", "value_string": rng.choice(["0", "1", "2"], 20)}),
], ignore_index=True)
```

## Metadata, tolerances and code tables

`ContextEnricher` joins lookup tables onto the data by `uuid`. Each
`enrich_*` call builds on the previous one, and `get_enriched_dataframe()`
returns the result of all of them:

```python
from ts_shape.loader.context.context_enricher import ContextEnricher

metadata = pd.DataFrame({
    "uuid": ["temp:zone_a", "press:zone_a", "state:machine1"],
    "description": ["Zone A temperature", "Zone A pressure", "Machine 1 state"],
    "unit": ["degC", "bar", "code"],
    "area": ["hall 1", "hall 1", "assembly"],
})
tolerances = pd.DataFrame({
    "uuid": ["temp:zone_a", "press:zone_a"],
    "low_limit": [80.0, 1.8],
    "high_limit": [90.0, 2.4],
})
codes = pd.DataFrame({
    "uuid": ["state:machine1"] * 3,
    "raw_value": ["0", "1", "2"],
    "mapped_value": ["idle", "running", "error"],
})

enricher = ContextEnricher(df)
enricher.enrich_with_metadata(metadata, columns=["description", "unit"])
enricher.enrich_with_tolerances(tolerances)
enricher.enrich_with_mapping(codes, value_column="value_string")
enriched = enricher.get_enriched_dataframe()

process = enriched[enriched["uuid"] != "state:machine1"]
outside = process[(process["value_double"] < process["low_limit"])
                  | (process["value_double"] > process["high_limit"])]
print(outside[["systime", "description", "value_double", "low_limit", "high_limit"]].head())

states = enriched[enriched["uuid"] == "state:machine1"]
print(states[["systime", "value_string", "mapped_value"]].head(3))
```

- `enrich_with_metadata` adds the named `columns` (default: all columns but
  the uuid). Signals without a metadata row get missing values.
- `enrich_with_tolerances` adds `low_limit` and `high_limit` (rename with
  `low_col=` and `high_col=`).
- `enrich_with_mapping` adds `mapped_value` for the rows whose
  `value_column` matches a `raw_value` of their uuid.
- Keys are compared as text, so a numeric uuid or code matches a string in
  the lookup table.
- A lookup table with duplicate keys, or a column that already exists in the
  data, raises `ValueError` instead of silently picking a row or creating
  `_x` / `_y` columns.

The limit columns suit row-wise checks like the one above. For deviation
events and capability indices, `ToleranceDeviationEvents` takes the limits as
signals in the same frame, so they can change over time; see
[Tolerance deviation](../modules/quality/tolerance-deviation.md).

## Mapping values from a file

`ValueMapper` maps one column through a CSV or JSON table on disk and keeps
every other column. Use it for code tables maintained outside the code, for
example an alarm-code list exported from the MES:

```python
import tempfile
from pathlib import Path

from ts_shape.context.value_mapping import ValueMapper

table = Path(tempfile.mkdtemp()) / "machine_states.csv"   # your exported file
pd.DataFrame({"code": [0, 1, 2], "state": ["idle", "running", "error"]}).to_csv(
    table, index=False
)

codes = df[df["uuid"] == "state:machine1"]
mapped = ValueMapper(
    codes,
    mapping_file=str(table),
    map_column="value_string",
    mapping_key_column="code",
    mapping_value_column="state",
).map_values()
print(mapped[["systime", "value_string"]].head(3))
```

- The mapped values replace the codes in `map_column`.
- Keys are compared as text: `1`, `1.0` and `"1"` match the same row.
- Codes the table does not contain keep their value, and a
  `DataQualityWarning` counts them.
- `file_type="json"` reads a JSON table; `sep=` and `encoding=` apply to CSV.

## Which to use

| Situation | Use |
|---|---|
| Descriptions, units, areas per signal | `ContextEnricher.enrich_with_metadata` |
| Limits per signal for tolerance checks | `ContextEnricher.enrich_with_tolerances` |
| State or alarm codes, different per signal | `ContextEnricher.enrich_with_mapping` |
| One code table in a CSV or JSON file | `ValueMapper` |
| Metadata from JSON while loading | `MetadataJsonLoader` with `DataIntegratorHybrid` ([Loading data](loading.md)) |

Next: [Features for ML](ml-features.md).
