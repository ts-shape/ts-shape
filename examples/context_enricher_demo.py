"""Context Enricher Demo

Demonstrates how to enrich timeseries data with contextual metadata:
descriptions, units, tolerance limits, and value mappings.

Run: python examples/context_enricher_demo.py
"""

import numpy as np
import pandas as pd

from ts_shape.loader.context.context_enricher import ContextEnricher


def create_sample_data():
    """Create sample timeseries, metadata, tolerances, and mappings."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")

    # Timeseries data
    timeseries = pd.DataFrame(
        {
            "systime": (
                [base + pd.Timedelta(minutes=i) for i in range(20)]
                + [base + pd.Timedelta(minutes=i) for i in range(20)]
                + [base + pd.Timedelta(minutes=i) for i in range(20)]
            ),
            "uuid": (
                ["temp:zone_a"] * 20 + ["press:zone_a"] * 20 + ["state:machine1"] * 20
            ),
            "value_double": (
                list(np.random.normal(85, 3, 20))
                + list(np.random.normal(2.1, 0.2, 20))
                + [None] * 20
            ),
            "value_string": ([None] * 40 + list(np.random.choice(["0", "1", "2"], 20))),
            "is_delta": [True] * 60,
        }
    )

    # Metadata
    metadata = pd.DataFrame(
        {
            "uuid": ["temp:zone_a", "press:zone_a", "state:machine1"],
            "description": ["Zone A Temperature", "Zone A Pressure", "Machine 1 State"],
            "unit": ["°C", "bar", "code"],
            "area": ["production_hall", "production_hall", "assembly"],
        }
    )

    # Tolerances
    tolerances = pd.DataFrame(
        {
            "uuid": ["temp:zone_a", "press:zone_a"],
            "low_limit": [75.0, 1.5],
            "high_limit": [95.0, 3.0],
        }
    )

    # Value mappings
    mappings = pd.DataFrame(
        {
            "uuid": ["state:machine1"] * 3,
            "raw_value": ["0", "1", "2"],
            "mapped_value": ["idle", "running", "error"],
        }
    )

    return timeseries, metadata, tolerances, mappings


if __name__ == "__main__":
    print("=" * 70)
    print("CONTEXT ENRICHER DEMO")
    print("=" * 70)

    ts, meta, tol, maps = create_sample_data()
    print(f"\nTimeseries: {len(ts)} rows, {ts['uuid'].nunique()} signals")
    print(f"Metadata: {len(meta)} signal descriptions")
    print(f"Tolerances: {len(tol)} tolerance entries")
    print(f"Mappings: {len(maps)} value mappings")

    # -----------------------------------------------------------------------
    # 1. Enrich with metadata
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. ENRICH WITH METADATA")
    print("-" * 70)

    enricher = ContextEnricher(ts)
    enriched = enricher.enrich_with_metadata(meta)
    print(f"\nColumns after metadata enrichment: {list(enriched.columns)}")
    print("\nSample (temp:zone_a with metadata):")
    print(
        enriched[enriched["uuid"] == "temp:zone_a"][
            ["systime", "uuid", "value_double", "description", "unit", "area"]
        ].head()
    )

    # -----------------------------------------------------------------------
    # 2. Enrich with tolerances
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. ENRICH WITH TOLERANCES")
    print("-" * 70)

    enricher2 = ContextEnricher(ts)
    with_tol = enricher2.enrich_with_tolerances(tol)
    print(f"\nColumns after tolerance enrichment: {list(with_tol.columns)}")
    temp_with_tol = with_tol[with_tol["uuid"] == "temp:zone_a"][
        ["systime", "uuid", "value_double", "low_limit", "high_limit"]
    ]
    print("\nTemperature with tolerance limits:")
    print(temp_with_tol.head())

    # Check which values are outside tolerance
    out_of_tol = temp_with_tol[
        (temp_with_tol["value_double"] < temp_with_tol["low_limit"])
        | (temp_with_tol["value_double"] > temp_with_tol["high_limit"])
    ]
    print(f"\nOut-of-tolerance readings: {len(out_of_tol)} of {len(temp_with_tol)}")

    # -----------------------------------------------------------------------
    # 3. Enrich with value mappings
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. ENRICH WITH VALUE MAPPINGS")
    print("-" * 70)

    enricher3 = ContextEnricher(ts)
    with_maps = enricher3.enrich_with_mapping(maps)
    state_rows = with_maps[with_maps["uuid"] == "state:machine1"][
        ["systime", "uuid", "value_string", "mapped_value"]
    ]
    print("\nMachine state with mapped values:")
    print(state_rows.head(10))

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)
