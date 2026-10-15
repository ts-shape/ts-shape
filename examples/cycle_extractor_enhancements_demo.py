"""
Demonstration of CycleExtractor enhancements.

This script shows how to use the new features added to CycleExtractor:
1. Cycle validation
2. Overlapping cycle detection
3. Incomplete cycle handling
4. Method selection helper
5. Cycle extraction statistics
6. Value change significance threshold
"""

import sys
from datetime import datetime, timedelta

import pandas as pd

from ts_shape.features.cycles.cycles_extractor import CycleExtractor


def create_sample_data():
    """Create sample data for demonstration."""
    # Generate timestamps
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * 10) for i in range(100)]

    # Create sample cycle data with some incomplete cycles
    data = []
    for i, ts in enumerate(timestamps):
        # Create a pattern: True for 3 records, False for 2 records
        value_bool = i % 5 < 3
        # Add some numeric values that change
        value_double = 100.0 + (i % 10) * 5.5
        value_integer = i % 10

        data.append(
            {
                "systime": ts,
                "uuid": "test-uuid-001",
                "value_bool": value_bool,
                "value_double": value_double,
                "value_integer": value_integer,
                "value_string": f"state_{i % 5}",
            }
        )

    return pd.DataFrame(data)


def demo_basic_extraction():
    """Demo 1: Basic cycle extraction with incomplete cycle tracking."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Cycle Extraction with Incomplete Cycle Tracking")
    print("=" * 70)

    df = create_sample_data()
    extractor = CycleExtractor(df, start_uuid="test-uuid-001")

    # Extract cycles using persistent cycle method
    cycles_df = extractor.process_persistent_cycle()

    print(f"\nExtracted {len(cycles_df)} cycles")
    print(f"Complete cycles: {cycles_df['is_complete'].sum()}")
    print(f"Incomplete cycles: {(~cycles_df['is_complete']).sum()}")

    print("\nFirst 5 cycles:")
    print(cycles_df.head())

    print("\nIncomplete cycles (if any):")
    incomplete = cycles_df[~cycles_df["is_complete"]]
    if not incomplete.empty:
        print(incomplete)
    else:
        print("No incomplete cycles found")

    return extractor, cycles_df


def demo_validation():
    """Demo 2: Cycle validation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Cycle Validation")
    print("=" * 70)

    df = create_sample_data()
    extractor = CycleExtractor(df, start_uuid="test-uuid-001")
    cycles_df = extractor.process_persistent_cycle()

    # Validate cycles with duration constraints
    validated_df = extractor.validate_cycles(
        cycles_df, min_duration="5s", max_duration="2m", warn=True
    )

    print("\nValidation results:")
    print(f"Total cycles: {len(validated_df)}")
    print(f"Valid cycles: {validated_df['is_valid'].sum()}")
    print(f"Invalid cycles: {(~validated_df['is_valid']).sum()}")

    print("\nValidation issues breakdown:")
    print(validated_df["validation_issue"].value_counts())

    print("\nSample of validated cycles:")
    print(
        validated_df[
            [
                "cycle_start",
                "cycle_end",
                "cycle_duration",
                "is_valid",
                "validation_issue",
            ]
        ].head(10)
    )

    return validated_df


def demo_overlap_detection():
    """Demo 3: Overlapping cycle detection."""
    print("\n" + "=" * 70)
    print("DEMO 3: Overlapping Cycle Detection")
    print("=" * 70)

    # Create data with overlapping cycles
    start_time = datetime(2024, 1, 1, 10, 0, 0)

    # Manual cycle data with overlaps
    cycles_data = [
        {
            "cycle_start": start_time,
            "cycle_end": start_time + timedelta(seconds=30),
            "cycle_uuid": "cycle-1",
            "is_complete": True,
        },
        {
            "cycle_start": start_time + timedelta(seconds=20),
            "cycle_end": start_time + timedelta(seconds=50),
            "cycle_uuid": "cycle-2",
            "is_complete": True,
        },  # Overlaps with cycle-1
        {
            "cycle_start": start_time + timedelta(seconds=60),
            "cycle_end": start_time + timedelta(seconds=80),
            "cycle_uuid": "cycle-3",
            "is_complete": True,
        },
        {
            "cycle_start": start_time + timedelta(seconds=70),
            "cycle_end": start_time + timedelta(seconds=95),
            "cycle_uuid": "cycle-4",
            "is_complete": True,
        },  # Overlaps with cycle-3
    ]

    cycles_df = pd.DataFrame(cycles_data)

    # Create a dummy extractor just to use the overlap detection
    df = create_sample_data()
    extractor = CycleExtractor(df, start_uuid="test-uuid-001")

    print("\nOriginal cycles:")
    print(cycles_df[["cycle_start", "cycle_end", "cycle_uuid"]])

    # Detect overlaps (flag only)
    flagged_df = extractor.detect_overlapping_cycles(cycles_df, resolve="flag")
    print(f"\nCycles with overlaps: {flagged_df['has_overlap'].sum()}")
    print(flagged_df[["cycle_uuid", "has_overlap"]])

    # Resolve by keeping first
    resolved_df = extractor.detect_overlapping_cycles(cycles_df, resolve="keep_first")
    print(f"\nAfter resolving (keep_first): {len(resolved_df)} cycles remain")
    print(resolved_df[["cycle_uuid", "has_overlap"]])

    # Resolve by keeping longest
    resolved_df = extractor.detect_overlapping_cycles(cycles_df, resolve="keep_longest")
    print(f"\nAfter resolving (keep_longest): {len(resolved_df)} cycles remain")
    print(resolved_df[["cycle_uuid", "has_overlap"]])


def demo_method_suggestion():
    """Demo 4: Method selection helper."""
    print("\n" + "=" * 70)
    print("DEMO 4: Method Selection Helper")
    print("=" * 70)

    df = create_sample_data()
    extractor = CycleExtractor(df, start_uuid="test-uuid-001")

    # Get method suggestions
    suggestions = extractor.suggest_method()

    print("\nData characteristics:")
    for key, value in suggestions["data_characteristics"].items():
        print(f"  {key}: {value}")

    print("\nRecommended methods:")
    for i, (method, reason) in enumerate(
        zip(suggestions["recommended_methods"], suggestions["reasoning"]), 1
    ):
        print(f"  {i}. {method}")
        print(f"     Reason: {reason}")


def demo_extraction_stats():
    """Demo 5: Cycle extraction statistics."""
    print("\n" + "=" * 70)
    print("DEMO 5: Cycle Extraction Statistics")
    print("=" * 70)

    df = create_sample_data()
    extractor = CycleExtractor(df, start_uuid="test-uuid-001")

    # Perform extraction
    cycles_df = extractor.process_persistent_cycle()
    print(f"\nExtracted {len(cycles_df)} cycles")

    # Get statistics
    stats = extractor.get_extraction_stats()

    print("\nExtraction Statistics:")
    print(f"  Total cycles: {stats['total_cycles']}")
    print(f"  Complete cycles: {stats['complete_cycles']}")
    print(f"  Incomplete cycles: {stats['incomplete_cycles']}")
    print(f"  Unmatched starts: {stats['unmatched_starts']}")
    print(f"  Unmatched ends: {stats['unmatched_ends']}")
    print(f"  Overlapping cycles: {stats['overlapping_cycles']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")

    print("\nConfiguration:")
    for key, value in stats["configuration"].items():
        print(f"  {key}: {value}")

    if stats["warnings"]:
        print("\nWarnings:")
        for warning in stats["warnings"]:
            print(f"  - {warning}")
    else:
        print("\nNo warnings generated")


def demo_value_change_threshold():
    """Demo 6: Value change significance threshold."""
    print("\n" + "=" * 70)
    print("DEMO 6: Value Change Significance Threshold")
    print("=" * 70)

    df = create_sample_data()

    # Extract with default threshold (0.0)
    print("\nExtracting with threshold = 0.0 (all changes detected):")
    extractor1 = CycleExtractor(
        df, start_uuid="test-uuid-001", value_change_threshold=0.0
    )
    cycles1 = extractor1.process_value_change_cycle()
    print(f"  Extracted {len(cycles1)} cycles")

    # Extract with higher threshold (only significant changes)
    print("\nExtracting with threshold = 10.0 (only significant changes):")
    extractor2 = CycleExtractor(
        df, start_uuid="test-uuid-001", value_change_threshold=10.0
    )
    cycles2 = extractor2.process_value_change_cycle()
    print(f"  Extracted {len(cycles2)} cycles")

    print(
        f"\nDifference: {len(cycles1) - len(cycles2)} fewer cycles with higher threshold"
    )


def demo_complete_workflow():
    """Demo 7: Complete workflow combining all features."""
    print("\n" + "=" * 70)
    print("DEMO 7: Complete Workflow")
    print("=" * 70)

    df = create_sample_data()

    # Step 1: Get method suggestions
    print("\nStep 1: Analyzing data and getting method suggestions...")
    extractor = CycleExtractor(
        df, start_uuid="test-uuid-001", value_change_threshold=5.0
    )
    suggestions = extractor.suggest_method()
    print(f"  Recommended: {suggestions['recommended_methods'][0]}")

    # Step 2: Extract cycles
    print("\nStep 2: Extracting cycles...")
    cycles_df = extractor.process_persistent_cycle()
    print(f"  Extracted {len(cycles_df)} cycles")

    # Step 3: Validate cycles
    print("\nStep 3: Validating cycles...")
    validated_df = extractor.validate_cycles(
        cycles_df, min_duration="10s", max_duration="1m", warn=False
    )
    print(f"  Valid cycles: {validated_df['is_valid'].sum()}/{len(validated_df)}")

    # Step 4: Check for overlaps
    print("\nStep 4: Checking for overlaps...")
    final_df = extractor.detect_overlapping_cycles(validated_df, resolve="flag")
    overlaps = final_df["has_overlap"].sum()
    print(f"  Overlapping cycles: {overlaps}")

    # Step 5: Get final statistics
    print("\nStep 5: Final statistics...")
    stats = extractor.get_extraction_stats()
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Complete cycles: {stats['complete_cycles']}")
    print(f"  Incomplete cycles: {stats['incomplete_cycles']}")

    print("\nFinal DataFrame columns:")
    print(f"  {list(final_df.columns)}")

    return final_df


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("CycleExtractor Enhancements Demonstration")
    print("=" * 70)

    try:
        # Run all demos
        demo_basic_extraction()
        demo_validation()
        demo_overlap_detection()
        demo_method_suggestion()
        demo_extraction_stats()
        demo_value_change_threshold()
        final_df = demo_complete_workflow()

        print("\n" + "=" * 70)
        print("All demonstrations completed successfully!")
        print("=" * 70)

        print("\nFinal cycles DataFrame shape:", final_df.shape)
        print("\nSample of final result:")
        print(final_df.head())

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
