"""Segment Analysis

Modular segment-based process analysis for production timeseries.

Detects time ranges from categorical signals (order numbers, part numbers)
and applies them to process parameter UUIDs for per-segment analysis.

Classes:
- SegmentExtractor: Detect transitions in a categorical signal and extract time ranges.
  - extract_time_ranges: Detect value changes, return segment start/end/value/duration.

- SegmentProcessor: Apply extracted time ranges to process data and compute profiles.
  - apply_ranges: Filter process data by time ranges, annotate with segment info.
  - compute_metric_profiles: Compute statistical metrics per UUID per segment.

- ProfileComparison: Distance, clustering, similarity, and anomaly detection on profiles.
  - compute_distance_matrix: Pairwise distance matrix between groups.
  - cluster: Hierarchical clustering of items by metric similarity.
  - find_similar: Top-K most similar items to a target.
  - detect_anomalous: Flag items with unusual metric profiles.
  - detect_changes: Track metric shifts across consecutive segments per UUID.
  - find_similar_pairs: Find most similar (UUID, segment) pairs across all data.

- TimeWindowedFeatureTable: Build ML-ready feature tables from segmented data.
  - compute_long: Metrics per (time_window, uuid, segment) in long format.
  - compute: Wide-format table with one row per time window, columns = {uuid}__{metric}.

To chain these classes into a reusable pipeline, use ``ts_shape.Pipeline``.
"""
