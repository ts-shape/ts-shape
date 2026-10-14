"""Features

Feature extraction and summarization utilities for shaped timeseries.

Classes:
- NumericStatistics: Compute descriptive statistics for numeric columns.
  - column_mean: Mean of a column.
  - column_median: Median of a column.
  - column_std: Standard deviation of a column.
  - column_variance: Variance of a column.
  - column_min: Minimum value.
  - column_max: Maximum value.
  - column_sum: Sum of values.
  - column_kurtosis: Kurtosis of values.
  - column_skewness: Skewness of values.
  - column_quantile: Quantile of a column.
  - column_iqr: Interquartile range.
  - column_range: Range (max - min).
  - column_mad: Mean absolute deviation.
  - coefficient_of_variation: Standard deviation divided by mean (guarded).
  - standard_error_mean: Standard error of the mean.
  - describe: Pandas describe wrapper.
  - summary_as_dict: Comprehensive numeric summary as dict.
  - summary_as_dataframe: Comprehensive numeric summary as DataFrame.

- StringStatistics: String-based statistics for categorical/text columns.
  - count_unique: Number of unique strings.
  - most_frequent: Most frequent string.
  - count_most_frequent: Count of the most frequent string.
  - count_null: Number of nulls.
  - average_string_length: Average length of non-null strings.
  - longest_string: Longest string.
  - shortest_string: Shortest string.
  - string_length_summary: Summary of lengths.
  - most_common_n_strings: Top-N most frequent strings.
  - contains_substring_count: Count of strings containing a substring.
  - starts_with_count: Count of strings starting with a prefix.
  - ends_with_count: Count of strings ending with a suffix.
  - uppercase_percentage: Percentage of uppercase strings.
  - lowercase_percentage: Percentage of lowercase strings.
  - contains_digit_count: Count of strings containing digits.
  - summary_as_dict: Comprehensive string summary as dict.
  - summary_as_dataframe: Comprehensive string summary as DataFrame.

- BooleanStatistics: Boolean column statistics.
  - count_true: Count of True values.
  - count_false: Count of False values.
  - count_null: Count of nulls.
  - count_not_null: Count of non-nulls.
  - true_percentage: Percentage True.
  - false_percentage: Percentage False.
  - mode: Most common boolean value.
  - is_balanced: Whether distribution is 50/50.
  - summary_as_dict: Summary as dict.
  - summary_as_dataframe: Summary as DataFrame.

- TimestampStatistics: Timestamp distributions and ranges.
  - count_null: Count of null timestamps.
  - count_not_null: Count of non-null timestamps.
  - earliest_timestamp: Earliest timestamp.
  - latest_timestamp: Latest timestamp.
  - timestamp_range: Time range (latest - earliest).
  - most_frequent_timestamp: Most frequent timestamp.
  - count_most_frequent_timestamp: Count of the modal timestamp.
  - year_distribution: Distribution by year.
  - month_distribution: Distribution by month.
  - weekday_distribution: Distribution by weekday.
  - hour_distribution: Distribution by hour.
  - most_frequent_day: Most frequent weekday.
  - most_frequent_hour: Most frequent hour.
  - average_time_gap: Average gap between consecutive timestamps.
  - median_timestamp: Median timestamp.
  - standard_deviation_timestamps: Standard deviation of consecutive differences.
  - timestamp_quartiles: 25th/50th/75th percentiles.
  - days_with_most_activity: Top-N active days.

- TimeGroupedStatistics: Time-windowed aggregations for numeric series.
  - calculate_statistic: Single aggregation per window (mean/sum/min/max/diff/range).
  - calculate_statistics: Multiple aggregations merged.
  - calculate_custom_func: Apply a custom aggregation per window.

- CycleExtractor: Build cycles from state/step/value changes.
  - process_persistent_cycle: a cycle runs from a rising edge to the next
    falling edge.
  - process_trigger_cycle: a cycle runs from one rising edge to the next.
  - process_separate_start_end_cycle: start and end events on two signals.
  - process_step_sequence: entering a start step opens a cycle, entering an
    end step closes it.
  - process_state_change_cycle: every sample is a boundary.
  - process_value_change_cycle: a change above a threshold is a boundary.

- CycleDataProcessor: Split/merge/group by cycle windows.
  - split_by_cycle: Split values by cycle ranges.
  - merge_dataframes_by_cycle: Annotate values with cycle UUIDs.
  - group_by_cycle_uuid: Group values by cycle key.
  - split_dataframes_by_group: Further split by column groupings.

- CrossSignalAnalytics: Cross-signal analytics for multi-signal timeseries.
  - granger_causality: Test if one signal Granger-causes another.
  - transfer_entropy: Estimate information transfer between signals.
  - pairwise_transfer_entropy: Transfer entropy for all directed pairs.
  - synchronization_index: Phase or amplitude synchronization.
  - pairwise_synchronization: Synchronization for all pairs.
  - lead_lag: Detect lead-lag relationships via cross-correlation.
  - lead_lag_matrix: Lead-lag for all pairs.

- PatternRecognition: Pattern discovery for univariate timeseries.
  - discover_motifs: Find top-k recurring subsequence patterns.
  - discover_discords: Find top-k anomalous subsequences.
  - similarity_search: Find subsequences most similar to a query (DTW).
  - template_match: Find all occurrences of a reference template.
  - compute_distance_profile: Distance from query to every subsequence.

- SegmentExtractor: Extract time ranges from categorical signals (order/part number).
  - extract_time_ranges: Detect value transitions and extract segment boundaries.

- SegmentProcessor: Apply time ranges to process data and compute metric profiles.
  - apply_ranges: Filter data by time ranges, annotate with segment info.
  - compute_metric_profiles: Compute statistical metrics per UUID per segment.

- ProfileComparison: Distance, clustering, similarity, anomaly on metric profiles.
  - compute_distance_matrix: Pairwise distance matrix between groups.
  - cluster: Hierarchical clustering by metric similarity.
  - find_similar: Top-K most similar items to a target.
  - detect_anomalous: Flag items with unusual metric profiles.
  - detect_changes: Track metric shifts across consecutive segments.
  - find_similar_pairs: Find similar (UUID, segment) pairs across all data.

- FeatureMatrixExporter: Convert long-format timeseries to wide ML-ready feature matrices.
  - to_feature_matrix: Pivot by uuid × value_col × agg into {uuid}__{col}__{agg} columns.
    Supports optional group_col (cycle, batch, segment) as row index.
"""

import importlib as _importlib

# The classes listed above, importable from the pack. They are imported on
# first access, so a class whose optional dependency is missing only fails
# when it is used.
_CLASSES = {
    "BooleanStatistics": "ts_shape.features.stats.boolean_stats",
    "CrossSignalAnalytics": "ts_shape.features.cross_signal",
    "CycleDataProcessor": "ts_shape.features.cycles.cycle_processor",
    "CycleExtractor": "ts_shape.features.cycles.cycles_extractor",
    "FeatureMatrixExporter": "ts_shape.features.export",
    "NumericStatistics": "ts_shape.features.stats.numeric_stats",
    "ProfileComparison": "ts_shape.features.segment_analysis.profile_comparison",
    "SegmentExtractor": "ts_shape.features.segment_analysis.segment_extractor",
    "SegmentProcessor": "ts_shape.features.segment_analysis.segment_processor",
    "StringStatistics": "ts_shape.features.stats.string_stats",
    "TimeGroupedStatistics": "ts_shape.features.time_stats.time_stats_numeric",
    "TimestampStatistics": "ts_shape.features.stats.timestamp_stats",
}

__all__ = sorted(_CLASSES)


def __getattr__(name: str) -> object:
    if name in _CLASSES:
        return getattr(_importlib.import_module(_CLASSES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASSES))
