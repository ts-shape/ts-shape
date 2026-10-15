"""
Demonstration of data loader utilities in ts-shape.

This script shows how to use:
1. ParquetLoader (constructor and usage patterns -- no actual files needed)
2. MetadataJsonLoader (load metadata from JSON strings)
3. DataIntegratorHybrid (combine timeseries + metadata DataFrames)
"""

import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ts_shape.loader.combine.integrator import DataIntegratorHybrid
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader


def create_timeseries_dataframe():
    """Create a synthetic timeseries DataFrame (simulating loaded parquet data)."""
    np.random.seed(42)
    start_time = datetime(2024, 1, 1, 8, 0, 0)
    rows = []

    uuids = ["sensor-temp-001", "sensor-press-002", "sensor-flow-003"]

    for uid in uuids:
        for i in range(50):
            t = start_time + timedelta(minutes=i * 5)
            rows.append(
                {
                    "systime": t,
                    "uuid": uid,
                    "value_double": np.random.normal(100 if "temp" in uid else 50, 5),
                    "value_string": None,
                    "value_bool": None,
                    "is_delta": True,
                }
            )

    return pd.DataFrame(rows)


def create_metadata_json_string():
    """Create a JSON string representing sensor metadata."""
    metadata = {
        "uuid": {
            "0": "sensor-temp-001",
            "1": "sensor-press-002",
            "2": "sensor-flow-003",
        },
        "label": {
            "0": "Temperature Sensor A",
            "1": "Pressure Sensor B",
            "2": "Flow Meter C",
        },
        "config": {
            "0": {
                "unit": "celsius",
                "location": "reactor_1",
                "min_range": 0,
                "max_range": 200,
            },
            "1": {
                "unit": "bar",
                "location": "pipeline_3",
                "min_range": 0,
                "max_range": 10,
            },
            "2": {
                "unit": "m3/h",
                "location": "inlet_valve",
                "min_range": 0,
                "max_range": 500,
            },
        },
    }
    return json.dumps(metadata, indent=2)


def demo_parquet_loader():
    """Demo 1: ParquetLoader usage patterns."""
    print("\n" + "=" * 70)
    print("DEMO 1: ParquetLoader (Usage Patterns)")
    print("=" * 70)

    print("""
    ParquetLoader loads timeseries data from Parquet files organized in a
    directory structure: base_path/YYYY/MM/DD/HH/<uuid>.parquet

    NOTE: This demo shows the API patterns without loading actual files.

    --- Constructor ---
        loader = ParquetLoader(base_path='/data/timeseries/')

    --- Load All Files ---
        df = ParquetLoader.load_all_files('/data/timeseries/')
        # Returns a single DataFrame with all parquet data concatenated

    --- Load by Time Range ---
        df = ParquetLoader.load_by_time_range(
            base_path='/data/timeseries/',
            start=pd.Timestamp('2024-01-01'),
            end=pd.Timestamp('2024-01-02'),
        )
        # Filters files whose directory path (YYYY/MM/DD/HH) falls in range

    --- Load by UUID List ---
        df = ParquetLoader.load_by_uuid_list(
            base_path='/data/timeseries/',
            uuid_list=['sensor-temp-001', 'sensor-press-002'],
        )
        # Filters files whose filenames contain one of the specified UUIDs

    --- Load by Time Range + UUIDs ---
        df = ParquetLoader.load_files_by_time_range_and_uuids(
            base_path='/data/timeseries/',
            start=pd.Timestamp('2024-01-01'),
            end=pd.Timestamp('2024-01-02'),
            uuid_list=['sensor-temp-001'],
        )
        # Combines both filters: time range AND uuid matching
    """)

    # Show what the resulting DataFrame looks like
    print("--- Simulated Output (using synthetic data) ---")
    df = create_timeseries_dataframe()
    print(f"\n  Shape: {df.shape}")
    print(f"  UUIDs: {df['uuid'].unique().tolist()}")
    print(f"  Time range: {df['systime'].min()} to {df['systime'].max()}")
    print("\n  Sample rows:")
    print(df.head(6).to_string(index=False))


def demo_metadata_json_loader():
    """Demo 2: MetadataJsonLoader from JSON string."""
    print("\n" + "=" * 70)
    print("DEMO 2: MetadataJsonLoader")
    print("=" * 70)

    json_str = create_metadata_json_string()
    print("\n--- Input JSON ---")
    print(json_str)

    # Load from string
    print("\n--- Loading from JSON string ---")
    loader = MetadataJsonLoader.from_str(json_str)

    # View the DataFrame
    print("\n--- Metadata DataFrame ---")
    df = loader.to_df()
    print(df.to_string())

    # List UUIDs and labels
    print("\n--- UUIDs ---")
    print(f"  {loader.list_uuids()}")

    print("\n--- Labels ---")
    print(f"  {loader.list_labels()}")

    # Lookup by UUID
    print("\n--- Lookup by UUID ---")
    result = loader.get_by_uuid("sensor-temp-001")
    if result:
        print("  sensor-temp-001:")
        for key, value in result.items():
            print(f"    {key}: {value}")

    # Lookup by label
    print("\n--- Lookup by Label ---")
    result = loader.get_by_label("Pressure Sensor B")
    if result:
        print("  Pressure Sensor B:")
        for key, value in result.items():
            print(f"    {key}: {value}")

    # Filter by UUID set
    print("\n--- Filter by UUIDs ---")
    filtered = loader.filter_by_uuid(["sensor-temp-001", "sensor-flow-003"])
    print(filtered.to_string())

    # Filter by label
    print("\n--- Filter by Labels ---")
    filtered = loader.filter_by_label(["Temperature Sensor A"])
    print(filtered.to_string())

    # Load from list-of-records format
    print("\n--- Alternative: Load from list of records ---")
    records = [
        {
            "uuid": "valve-001",
            "label": "Inlet Valve",
            "config": {"type": "butterfly", "size": "DN100"},
        },
        {
            "uuid": "valve-002",
            "label": "Outlet Valve",
            "config": {"type": "gate", "size": "DN150"},
        },
    ]
    loader2 = MetadataJsonLoader(records)
    print(loader2.to_df().to_string())

    # Load from columnar lists
    print("\n--- Alternative: Load from column lists ---")
    col_data = {
        "uuid": ["pump-01", "pump-02"],
        "label": ["Main Pump", "Backup Pump"],
        "config": [
            {"power_kw": 75, "type": "centrifugal"},
            {"power_kw": 55, "type": "positive_displacement"},
        ],
    }
    loader3 = MetadataJsonLoader(col_data)
    print(loader3.to_df().to_string())


def demo_data_integrator():
    """Demo 3: DataIntegratorHybrid combining timeseries and metadata."""
    print("\n" + "=" * 70)
    print("DEMO 3: DataIntegratorHybrid")
    print("=" * 70)

    # Create timeseries data
    ts_df = create_timeseries_dataframe()
    print("\n--- Timeseries Data ---")
    print(f"  Shape: {ts_df.shape}")
    print(f"  UUIDs: {ts_df['uuid'].unique().tolist()}")

    # Create metadata
    metadata_df = pd.DataFrame(
        {
            "uuid": ["sensor-temp-001", "sensor-press-002", "sensor-flow-003"],
            "label": ["Temperature Sensor A", "Pressure Sensor B", "Flow Meter C"],
            "unit": ["celsius", "bar", "m3/h"],
            "location": ["reactor_1", "pipeline_3", "inlet_valve"],
        }
    )
    print("\n--- Metadata ---")
    print(metadata_df.to_string(index=False))

    # Combine timeseries + metadata
    print("\n--- Combined Data (left join on uuid) ---")
    combined = DataIntegratorHybrid.combine_data(
        timeseries_sources=[ts_df],
        metadata_sources=[metadata_df],
        join_key="uuid",
        merge_how="left",
    )
    print(f"  Combined shape: {combined.shape}")
    print(f"  Columns: {list(combined.columns)}")
    print("\n  Sample rows:")
    print(combined.head(6).to_string(index=False))

    # Filter by specific UUIDs
    print("\n--- Combined + Filtered by UUIDs ---")
    filtered = DataIntegratorHybrid.combine_data(
        timeseries_sources=[ts_df],
        metadata_sources=[metadata_df],
        uuids=["sensor-temp-001"],
        join_key="uuid",
        merge_how="left",
    )
    print(f"  Filtered shape: {filtered.shape}")
    print(f"  UUIDs in result: {filtered['uuid'].unique().tolist()}")
    print(filtered.head(5).to_string(index=False))

    # Combine multiple timeseries sources
    print("\n--- Combining Multiple Timeseries Sources ---")
    ts_df_2 = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                [
                    datetime(2024, 1, 2, 8, 0) + timedelta(minutes=i * 5)
                    for i in range(10)
                ]
            ),
            "uuid": "sensor-temp-001",
            "value_double": np.random.normal(105, 3, 10),
            "is_delta": True,
        }
    )
    multi_combined = DataIntegratorHybrid.combine_data(
        timeseries_sources=[ts_df, ts_df_2],
        metadata_sources=[metadata_df],
        join_key="uuid",
        merge_how="left",
    )
    print(f"  Combined from 2 sources: {multi_combined.shape}")
    temp_data = multi_combined[multi_combined["uuid"] == "sensor-temp-001"]
    print(
        f"  sensor-temp-001 rows: {len(temp_data)} (50 from source 1 + 10 from source 2)"
    )

    # Integration with MetadataJsonLoader
    print("\n--- Integration with MetadataJsonLoader ---")
    json_str = create_metadata_json_string()
    loader = MetadataJsonLoader.from_str(json_str)
    meta_from_loader = loader.to_df().reset_index()  # bring uuid from index to column

    combined_with_loader = DataIntegratorHybrid.combine_data(
        timeseries_sources=[ts_df],
        metadata_sources=[meta_from_loader],
        join_key="uuid",
        merge_how="left",
    )
    print(f"  Shape: {combined_with_loader.shape}")
    print(f"  Columns: {list(combined_with_loader.columns)}")
    print(combined_with_loader.head(3).to_string(index=False))


def main():
    """Run all loader usage demonstrations."""
    print("\n" + "=" * 70)
    print("Loader Usage Demonstration")
    print("=" * 70)

    try:
        demo_parquet_loader()
        demo_metadata_json_loader()
        demo_data_integrator()

        print("\n" + "=" * 70)
        print("All demonstrations completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
