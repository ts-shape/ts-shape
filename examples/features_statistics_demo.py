"""
Features and Statistics Demo for ts-shape
==========================================

Demonstrates all feature-extraction and statistics classes:

Stats:
  - NumericStatistics    (numeric_stats)
  - BooleanStatistics    (boolean_stats)
  - StringStatistics     (string_stats)
  - TimestampStatistics  (timestamp_stats)
  - DescriptiveFeatures  (feature_table)
  - Missing-value handling across the column-type statistics

Time-grouped stats:
  - TimeGroupedStatistics (time_stats_numeric)

Cycle analysis:
  - CycleExtractor       (cycles_extractor)
  - CycleDataProcessor   (cycle_processor)

Scenario: A semiconductor wafer-fab chamber collects temperature, pressure,
gas-flow, recipe labels, and chamber-active status over several production
cycles.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ts_shape.features.cycles.cycle_processor import CycleDataProcessor
from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.features.stats.boolean_stats import BooleanStatistics
from ts_shape.features.stats.feature_table import DescriptiveFeatures
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.features.stats.string_stats import StringStatistics
from ts_shape.features.stats.timestamp_stats import TimestampStatistics
from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics

# ---------------------------------------------------------------------------
# Helper: build realistic wafer-fab chamber data
# ---------------------------------------------------------------------------


def create_chamber_data(n_cycles: int = 5, points_per_cycle: int = 120) -> pd.DataFrame:
    """
    Simulate semiconductor chamber sensor data across multiple process cycles.

    Each cycle lasts ~10 minutes (one reading per 5 seconds).  Between cycles
    there is a 2-minute idle gap.

    Signals:
      - 'chamber_temp'    : double, target 350 C with noise
      - 'chamber_pressure': double, target 2.5 Torr
      - 'gas_flow_rate'   : integer, sccm (standard cubic centimetres per min)
      - 'recipe_name'     : string, the recipe loaded for the cycle
      - 'chamber_active'  : boolean, True while processing, False while idle
    """
    np.random.seed(2025)
    rows = []
    recipes = ["OXIDE_DEP_A", "NITRIDE_DEP_B", "ETCH_C", "OXIDE_DEP_A", "NITRIDE_DEP_B"]
    t = datetime(2025, 8, 15, 7, 0, 0)

    for cycle_idx in range(n_cycles):
        recipe = recipes[cycle_idx % len(recipes)]

        # --- Active phase ---
        for i in range(points_per_cycle):
            temp = (
                350.0 + np.random.normal(0, 1.2) + (cycle_idx * 0.3)
            )  # slight drift per cycle
            pressure = 2.5 + np.random.normal(0, 0.05)
            gas_flow = int(200 + np.random.normal(0, 5))

            # Temperature row
            rows.append(
                {
                    "systime": t,
                    "uuid": "chamber_temp",
                    "value_bool": None,
                    "value_integer": None,
                    "value_double": round(temp, 3),
                    "value_string": None,
                    "is_delta": True,
                }
            )

            # Pressure row
            rows.append(
                {
                    "systime": t + timedelta(seconds=1),
                    "uuid": "chamber_pressure",
                    "value_bool": None,
                    "value_integer": None,
                    "value_double": round(pressure, 4),
                    "value_string": None,
                    "is_delta": True,
                }
            )

            # Gas flow row
            rows.append(
                {
                    "systime": t + timedelta(seconds=2),
                    "uuid": "gas_flow_rate",
                    "value_bool": None,
                    "value_integer": gas_flow,
                    "value_double": None,
                    "value_string": None,
                    "is_delta": True,
                }
            )

            # Recipe label (only on first sample of each cycle)
            if i == 0:
                rows.append(
                    {
                        "systime": t + timedelta(seconds=3),
                        "uuid": "recipe_name",
                        "value_bool": None,
                        "value_integer": None,
                        "value_double": None,
                        "value_string": recipe,
                        "is_delta": True,
                    }
                )

            # Chamber active status
            rows.append(
                {
                    "systime": t + timedelta(seconds=3),
                    "uuid": "chamber_active",
                    "value_bool": True,
                    "value_integer": None,
                    "value_double": None,
                    "value_string": None,
                    "is_delta": True,
                }
            )

            t += timedelta(seconds=5)

        # --- Idle gap (chamber off) ---
        for i in range(24):  # 2 minutes at 5-second intervals
            rows.append(
                {
                    "systime": t,
                    "uuid": "chamber_active",
                    "value_bool": False,
                    "value_integer": None,
                    "value_double": None,
                    "value_string": None,
                    "is_delta": True,
                }
            )
            t += timedelta(seconds=5)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_numeric_statistics(df: pd.DataFrame):
    """Demonstrate NumericStatistics on chamber temperature data."""
    print("=" * 72)
    print("1. NUMERIC STATISTICS")
    print("=" * 72)

    temp_df = df[df["uuid"] == "chamber_temp"].copy()
    col = "value_double"
    print(f"\nChamber temperature: {len(temp_df)} readings")

    print(f"  Mean:     {NumericStatistics.column_mean(temp_df, col):.3f} C")
    print(f"  Median:   {NumericStatistics.column_median(temp_df, col):.3f} C")
    print(f"  Std dev:  {NumericStatistics.column_std(temp_df, col):.4f} C")
    print(f"  Variance: {NumericStatistics.column_variance(temp_df, col):.4f}")
    print(f"  Min:      {NumericStatistics.column_min(temp_df, col):.3f} C")
    print(f"  Max:      {NumericStatistics.column_max(temp_df, col):.3f} C")
    print(f"  Range:    {NumericStatistics.column_range(temp_df, col):.3f} C")
    print(f"  IQR:      {NumericStatistics.column_iqr(temp_df, col):.4f}")
    print(f"  Skewness: {NumericStatistics.column_skewness(temp_df, col):.4f}")
    print(f"  Kurtosis: {NumericStatistics.column_kurtosis(temp_df, col):.4f}")
    print(f"  Q1 (25%): {NumericStatistics.column_quantile(temp_df, col, 0.25):.3f}")
    print(f"  Q3 (75%): {NumericStatistics.column_quantile(temp_df, col, 0.75):.3f}")
    print(f"  SEM:      {NumericStatistics.standard_error_mean(temp_df, col):.5f}")
    print(f"  CV:       {NumericStatistics.coefficient_of_variation(temp_df, col):.6f}")

    # Full summary as DataFrame
    print("\n  Full describe():")
    desc = NumericStatistics.describe(temp_df[[col]])
    print(desc.to_string())

    # Summary as dictionary
    print("\n  summary_as_dataframe() (first few columns):")
    summary_df = NumericStatistics.summary_as_dataframe(temp_df, col)
    print(summary_df.to_string(index=False))
    print()


def demo_boolean_statistics(df: pd.DataFrame):
    """Demonstrate BooleanStatistics on chamber active status."""
    print("=" * 72)
    print("2. BOOLEAN STATISTICS")
    print("=" * 72)

    active_df = df[df["uuid"] == "chamber_active"].copy()
    col = "value_bool"
    print(f"\nChamber active status: {len(active_df)} readings")

    print(f"  True count:      {BooleanStatistics.count_true(active_df, col)}")
    print(f"  False count:     {BooleanStatistics.count_false(active_df, col)}")
    print(f"  Null count:      {BooleanStatistics.count_null(active_df, col)}")
    print(f"  Not-null count:  {BooleanStatistics.count_not_null(active_df, col)}")
    print(
        f"  True %:          {BooleanStatistics.true_percentage(active_df, col):.1f}%"
    )
    print(
        f"  False %:         {BooleanStatistics.false_percentage(active_df, col):.1f}%"
    )
    print(f"  Mode:            {BooleanStatistics.mode(active_df, col)}")
    print(f"  Is balanced?     {BooleanStatistics.is_balanced(active_df, col)}")

    # Summary as DataFrame
    print("\n  summary_as_dataframe():")
    summary_df = BooleanStatistics.summary_as_dataframe(active_df, col)
    print(summary_df.to_string(index=False))
    print()


def demo_string_statistics(df: pd.DataFrame):
    """Demonstrate StringStatistics on recipe names."""
    print("=" * 72)
    print("3. STRING STATISTICS")
    print("=" * 72)

    # Collect recipe rows plus some chamber_active rows to have more string variety
    recipe_df = df[df["uuid"] == "recipe_name"].copy()
    col = "value_string"
    print(f"\nRecipe name entries: {len(recipe_df)}")

    print(f"  Unique values:      {StringStatistics.count_unique(recipe_df, col)}")
    print(f"  Most frequent:      {StringStatistics.most_frequent(recipe_df, col)}")
    print(
        f"  Count most freq:    {StringStatistics.count_most_frequent(recipe_df, col)}"
    )
    print(f"  Null count:         {StringStatistics.count_null(recipe_df, col)}")
    print(
        f"  Avg string length:  {StringStatistics.average_string_length(recipe_df, col):.1f}"
    )
    print(f"  Longest string:     {StringStatistics.longest_string(recipe_df, col)}")
    print(f"  Shortest string:    {StringStatistics.shortest_string(recipe_df, col)}")
    print(
        f"  Uppercase %:        {StringStatistics.uppercase_percentage(recipe_df, col):.1f}%"
    )
    print(
        f"  Contains digit:     {StringStatistics.contains_digit_count(recipe_df, col)}"
    )
    print(
        f"  Starts with 'OXIDE': {StringStatistics.starts_with_count(recipe_df, 'OXIDE', col)}"
    )
    print(
        f"  Ends with 'B':      {StringStatistics.ends_with_count(recipe_df, 'B', col)}"
    )
    print(
        f"  Contains 'DEP':     {StringStatistics.contains_substring_count(recipe_df, 'DEP', col)}"
    )

    # Top N most common
    print("\n  Top 3 most common recipes:")
    top3 = StringStatistics.most_common_n_strings(recipe_df, n=3, value_column=col)
    for name, count in top3.items():
        print(f"    {name}: {count}")

    # String length summary
    print("\n  String length summary:")
    len_summary = StringStatistics.string_length_summary(recipe_df, col)
    print(len_summary.to_string(index=False))

    # Full summary
    print("\n  summary_as_dataframe():")
    summary_df = StringStatistics.summary_as_dataframe(recipe_df, col)
    print(summary_df.to_string(index=False))
    print()


def demo_timestamp_statistics(df: pd.DataFrame):
    """Demonstrate TimestampStatistics on the systime column."""
    print("=" * 72)
    print("4. TIMESTAMP STATISTICS")
    print("=" * 72)

    # Use the full DataFrame for timestamp analysis
    ts_df = df.copy()
    ts_df["systime"] = pd.to_datetime(ts_df["systime"])
    col = "systime"
    print(f"\nTotal rows: {len(ts_df)}")

    print(f"  Earliest:          {TimestampStatistics.earliest_timestamp(ts_df, col)}")
    print(f"  Latest:            {TimestampStatistics.latest_timestamp(ts_df, col)}")
    print(f"  Range:             {TimestampStatistics.timestamp_range(ts_df, col)}")
    print(f"  Median:            {TimestampStatistics.median_timestamp(ts_df, col)}")
    print(f"  Null count:        {TimestampStatistics.count_null(ts_df, col)}")
    print(f"  Not-null count:    {TimestampStatistics.count_not_null(ts_df, col)}")
    print(f"  Avg time gap:      {TimestampStatistics.average_time_gap(ts_df, col)}")
    print(
        f"  Std dev of gaps:   {TimestampStatistics.standard_deviation_timestamps(ts_df, col)}"
    )
    print(
        f"  Most frequent day: {TimestampStatistics.most_frequent_day(ts_df, col)} (0=Mon)"
    )
    print(f"  Most frequent hr:  {TimestampStatistics.most_frequent_hour(ts_df, col)}")

    # Distributions
    print("\n  Hour distribution (top 5):")
    hour_dist = TimestampStatistics.hour_distribution(ts_df, col)
    for hour, count in hour_dist.head(5).items():
        print(f"    Hour {hour:02d}: {count} rows")

    # Quartiles
    print("\n  Timestamp quartiles:")
    quartiles = TimestampStatistics.timestamp_quartiles(ts_df, col)
    for q, val in quartiles.items():
        print(f"    Q{q}: {val}")

    # Days with most activity
    print("\n  Top 3 days with most activity:")
    top_days = TimestampStatistics.days_with_most_activity(ts_df, col, n=3)
    for day, count in top_days.items():
        print(f"    {day}: {count} rows")
    print()


def demo_descriptive_features(df: pd.DataFrame):
    """Demonstrate DescriptiveFeatures (feature table) grouped by UUID."""
    print("=" * 72)
    print("5. DESCRIPTIVE FEATURES (Feature Table)")
    print("=" * 72)

    # Use a subset to keep output manageable
    subset = df[df["uuid"].isin(["chamber_temp", "chamber_active"])].copy()
    subset["systime"] = pd.to_datetime(subset["systime"])
    print(f"\nUsing {len(subset)} rows (chamber_temp + chamber_active)")

    feat = DescriptiveFeatures(subset)

    # Compute as dictionary
    print("\n  Compute as dict (showing keys per UUID):")
    result_dict = feat.compute(output_format="dict")
    for uuid_key, stats in result_dict.items():
        print(f"\n  UUID: {uuid_key}")
        if isinstance(stats, dict):
            for section, content in stats.items():
                print(f"    Section: {section}")
                if isinstance(content, dict):
                    for k, v in list(content.items())[:3]:
                        print(f"      {k}: {v}")
                    if len(content) > 3:
                        print(f"      ({len(content)} keys in total)")
                else:
                    print(f"      {content}")

    # Compute as DataFrame
    print("\n  Compute as DataFrame (first 5 columns):")
    result_df = feat.compute(output_format="dataframe")
    cols = list(result_df.columns)[:5]
    print(result_df[cols].to_string(index=False))
    print(f"  ({len(result_df.columns)} columns in total)")
    print()


def demo_time_grouped_statistics(df: pd.DataFrame):
    """Demonstrate TimeGroupedStatistics with time-binned aggregations."""
    print("=" * 72)
    print("6. TIME-GROUPED STATISTICS")
    print("=" * 72)

    temp_df = df[df["uuid"] == "chamber_temp"][["systime", "value_double"]].copy()
    temp_df["systime"] = pd.to_datetime(temp_df["systime"])
    print(f"\nChamber temperature: {len(temp_df)} readings")

    # Single statistic: 10-minute mean
    print("\n--- 10-minute mean temperature ---")
    mean_10m = TimeGroupedStatistics.calculate_statistic(
        temp_df,
        time_column="systime",
        value_column="value_double",
        window="10min",
        stat_method="mean",
    )
    print(mean_10m.head(8).to_string())

    # Multiple statistics at once
    print("\n--- 10-minute mean, min, max, range ---")
    multi = TimeGroupedStatistics.calculate_statistics(
        temp_df,
        time_column="systime",
        value_column="value_double",
        window="10min",
        stat_methods=["mean", "min", "max", "range"],
    )
    print(multi.head(8).to_string())

    # Hourly difference (last - first)
    print("\n--- Hourly difference (last - first) ---")
    diff_h = TimeGroupedStatistics.calculate_statistic(
        temp_df,
        time_column="systime",
        value_column="value_double",
        window="h",
        stat_method="diff",
    )
    print(diff_h.to_string())

    # Custom aggregation function (coefficient of variation per 10-min bin)
    print("\n--- 10-minute coefficient of variation (custom func) ---")

    def cv_func(x):
        return x.std() / x.mean() if x.mean() != 0 else 0

    cv_10m = TimeGroupedStatistics.calculate_custom_func(
        temp_df,
        time_column="systime",
        value_column="value_double",
        window="10min",
        func=cv_func,
    )
    print(cv_10m.head(8).to_string())
    print()


def demo_cycle_extraction(df: pd.DataFrame):
    """Demonstrate CycleExtractor with persistent-cycle and step-sequence methods."""
    print("=" * 72)
    print("7. CYCLE EXTRACTION")
    print("=" * 72)

    # --- Persistent cycle: chamber_active True/False transitions ---
    active_df = df[df["uuid"] == "chamber_active"].copy()
    active_df["systime"] = pd.to_datetime(active_df["systime"])
    print(f"\nChamber active rows: {len(active_df)}")

    extractor = CycleExtractor(
        dataframe=active_df,
        start_uuid="chamber_active",
    )

    # Method suggestion
    print("\n--- Method suggestion ---")
    suggestion = extractor.suggest_method()
    for method, reason in zip(
        suggestion["recommended_methods"], suggestion["reasoning"]
    ):
        print(f"  {method}: {reason}")

    # Extract persistent cycles (True -> False transitions)
    print("\n--- Persistent cycles (active=True -> active=False) ---")
    cycles = extractor.process_persistent_cycle()
    print(f"  Extracted {len(cycles)} cycles")
    if not cycles.empty:
        print(
            cycles[["cycle_start", "cycle_end", "is_complete"]].to_string(index=False)
        )

    # Extraction statistics
    stats = extractor.get_extraction_stats()
    print("\n  Extraction stats:")
    print(f"    Total:      {stats['total_cycles']}")
    print(f"    Complete:   {stats['complete_cycles']}")
    print(f"    Incomplete: {stats['incomplete_cycles']}")
    print(f"    Success rate: {stats['success_rate']:.1%}")

    # Validate cycles (min 30s, max 15min)
    print("\n--- Validate cycles ---")
    validated = extractor.validate_cycles(
        cycles, min_duration="30s", max_duration="15m"
    )
    valid_count = validated["is_valid"].sum()
    invalid_count = (~validated["is_valid"]).sum()
    print(f"  Valid: {valid_count}, Invalid: {invalid_count}")
    if "validation_issue" in validated.columns:
        issues = validated[~validated["is_valid"]]
        if not issues.empty:
            print("  Issues found:")
            for _, row in issues.iterrows():
                print(
                    f"    {row['cycle_start']} -> {row.get('validation_issue', 'unknown')}"
                )

    # Detect overlapping cycles
    print("\n--- Overlap detection ---")
    overlap_result = extractor.detect_overlapping_cycles(cycles, resolve="flag")
    overlaps = overlap_result["has_overlap"].sum()
    print(f"  Cycles with overlap: {overlaps}")
    print()


def demo_cycle_processor(df: pd.DataFrame):
    """Demonstrate CycleDataProcessor: split, merge, statistics, compare, golden cycles."""
    print("=" * 72)
    print("8. CYCLE DATA PROCESSOR")
    print("=" * 72)

    # First extract cycles
    active_df = df[df["uuid"] == "chamber_active"].copy()
    active_df["systime"] = pd.to_datetime(active_df["systime"])
    extractor = CycleExtractor(dataframe=active_df, start_uuid="chamber_active")
    cycles = extractor.process_persistent_cycle()

    # Filter to complete cycles only
    complete_cycles = cycles[cycles["is_complete"]].copy()
    print(f"\nComplete cycles: {len(complete_cycles)}")

    if complete_cycles.empty:
        print("  No complete cycles found -- skipping processor demo")
        return

    # Prepare values DataFrame (temperature readings)
    temp_df = df[df["uuid"] == "chamber_temp"].copy()
    temp_df["systime"] = pd.to_datetime(temp_df["systime"])
    print(f"Temperature readings: {len(temp_df)}")

    processor = CycleDataProcessor(
        cycles_df=complete_cycles,
        values_df=temp_df,
        cycle_uuid_column="cycle_uuid",
        time_column="systime",
    )

    # --- Merge values with cycles ---
    print("\n--- Merge temperature data with cycles ---")
    merged = processor.merge_dataframes_by_cycle()
    print(f"  Merged rows: {len(merged)}")
    print(f"  Cycles represented: {merged['cycle_uuid'].nunique()}")

    # --- Split by cycle ---
    print("\n--- Split data by cycle ---")
    split_data = processor.split_by_cycle()
    for cycle_uuid, cycle_df in list(split_data.items())[:3]:
        print(
            f"  Cycle {cycle_uuid[:12]}...: {len(cycle_df)} readings, "
            f"mean temp = {cycle_df['value_double'].mean():.2f} C"
        )

    # --- Cycle statistics ---
    print("\n--- Cycle statistics ---")
    cycle_stats = processor.compute_cycle_statistics()
    if not cycle_stats.empty:
        display_cols = [
            c
            for c in [
                "cycle_uuid",
                "duration_seconds",
                "value_count",
                "mean_value_double",
                "std_value_double",
            ]
            if c in cycle_stats.columns
        ]
        # Truncate cycle_uuid for display
        stats_display = cycle_stats[display_cols].copy()
        stats_display["cycle_uuid"] = stats_display["cycle_uuid"].str[:12] + "..."
        print(stats_display.to_string(index=False))

    # --- Compare cycles against a reference ---
    print("\n--- Compare cycles against first cycle ---")
    ref_uuid = complete_cycles["cycle_uuid"].iloc[0]
    comparison = processor.compare_cycles(
        reference_cycle_uuid=ref_uuid, value_column="value_double"
    )
    if not comparison.empty:
        comp_display = comparison[
            [
                "cycle_uuid",
                "is_reference",
                "mean_value",
                "deviation_from_ref",
                "deviation_pct",
            ]
        ].copy()
        comp_display["cycle_uuid"] = comp_display["cycle_uuid"].str[:12] + "..."
        print(comp_display.to_string(index=False))

    # --- Identify golden cycles ---
    print("\n--- Identify golden cycles (low variability) ---")
    golden = processor.identify_golden_cycles(
        value_column="value_double", method="low_variability", top_n=2
    )
    for i, gc in enumerate(golden):
        print(f"  #{i+1}: {gc[:12]}...")

    # --- Group by cycle UUID ---
    print("\n--- Group by cycle UUID ---")
    grouped = processor.group_by_cycle_uuid(merged)
    print(f"  Number of groups: {len(grouped)}")
    if grouped:
        print(f"  Rows in first group: {len(grouped[0])}")

    # --- Split groups further by uuid (signal type) ---
    print("\n--- Split groups by signal uuid ---")
    sub_groups = processor.split_dataframes_by_group(grouped[:2], column="uuid")
    print(f"  Sub-groups from first 2 cycles: {len(sub_groups)}")
    print()


def demo_missing_values():
    """Show how the column statistics treat missing values in mixed columns."""
    print("=" * 72)
    print("9. STATISTICS WITH MISSING VALUES")
    print("=" * 72)

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01 08:00", periods=n, freq="5min"),
            "value_double": rng.normal(100.0, 15.0, n),
            "value_string": rng.choice(
                ["PRODUCT_A", "PRODUCT_B", "PRODUCT_C", "PRODUCT_D", None],
                n,
                p=[0.35, 0.30, 0.20, 0.10, 0.05],
            ),
            "value_bool": rng.choice([True, False, None], n, p=[0.6, 0.35, 0.05]),
        }
    )
    print(f"\nMixed-type frame: {len(df)} rows with some missing labels and flags")

    col = "value_double"
    print(
        f"  P90 of value_double:   {NumericStatistics.column_quantile(df, col, 0.90):.3f}"
    )
    print(f"  Null strings:          {StringStatistics.count_null(df, 'value_string')}")
    print(f"  Null booleans:         {BooleanStatistics.count_null(df, 'value_bool')}")
    print(
        f"  True % (non-null):     {BooleanStatistics.true_percentage(df, 'value_bool'):.1f}%"
    )

    print("\n  StringStatistics.summary_as_dict():")
    for key, value in StringStatistics.summary_as_dict(df, "value_string").items():
        print(f"    {key}: {value}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = create_chamber_data(n_cycles=5, points_per_cycle=120)
    print(f"Generated {len(df)} rows of wafer-fab chamber data")
    print(f"UUIDs: {df['uuid'].unique().tolist()}\n")

    demo_numeric_statistics(df)
    demo_boolean_statistics(df)
    demo_string_statistics(df)
    demo_timestamp_statistics(df)
    demo_descriptive_features(df)
    demo_time_grouped_statistics(df)
    demo_cycle_extraction(df)
    demo_cycle_processor(df)
    demo_missing_values()

    print("=" * 72)
    print("Features and statistics demo complete.")
    print("=" * 72)
