import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, List

from scipy.spatial.distance import pdist, squareform  # type: ignore
from scipy.cluster.hierarchy import linkage, fcluster  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ProfileComparison(Base):
    """Distance, clustering, similarity, and anomaly detection on metric profiles.

    Operates on the output of SegmentProcessor.compute_metric_profiles.
    All methods are classmethods working on DataFrames.

    Methods:
    - compute_distance_matrix: Pairwise distance matrix between groups.
    - cluster: Hierarchical clustering of items by metric similarity.
    - find_similar: Top-K most similar items to a target.
    - detect_anomalous: Flag items with unusual metric profiles.
    - detect_changes: Track metric shifts across consecutive segments per UUID.
    - find_similar_pairs: Find most similar (UUID, segment) pairs across all data.
    """

    @classmethod
    def _get_metric_columns(
        cls,
        df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None,
    ) -> List[str]:
        """Identify numeric metric columns from a profiles DataFrame."""
        non_metric = {
            "uuid",
            "segment_value",
            "segment_index",
            "window_start",
            "window_end",
            "sample_count",
        }
        available = [
            c
            for c in df.columns
            if c not in non_metric and pd.api.types.is_numeric_dtype(df[c])
        ]
        if metric_columns is not None:
            missing = set(metric_columns) - set(available)
            if missing:
                raise ValueError(f"Columns not found or not numeric: {missing}")
            return metric_columns
        return available

    @classmethod
    def compute_distance_matrix(
        cls,
        metric_profiles: pd.DataFrame,
        group_column: str = "uuid",
        metric_columns: Optional[List[str]] = None,
        distance_metric: str = "euclidean",
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Compute pairwise distance matrix between metric profile vectors.

        Can compare UUIDs (group_column='uuid') or segments
        (group_column='segment_value'). When multiple rows exist per group,
        metrics are averaged.

        Args:
            metric_profiles: Output from SegmentProcessor.compute_metric_profiles.
            group_column: Column to group by ('uuid' or 'segment_value').
            metric_columns: Which metric columns to use. None auto-detects.
            distance_metric: 'euclidean', 'cosine', or 'manhattan'.
            normalize: Z-normalize metrics before computing distances.

        Returns:
            Square DataFrame indexed by group values with pairwise distances.
        """
        cols = cls._get_metric_columns(metric_profiles, metric_columns)
        agg = metric_profiles.groupby(group_column)[cols].mean()
        labels = agg.index.tolist()
        matrix = agg.values.astype(float)
        matrix = np.nan_to_num(matrix, nan=0.0)

        if normalize and matrix.shape[0] > 1:
            col_std = matrix.std(axis=0)
            col_std[col_std < 1e-10] = 1.0
            matrix = (matrix - matrix.mean(axis=0)) / col_std

        metric_map = {
            "euclidean": "euclidean",
            "cosine": "cosine",
            "manhattan": "cityblock",
        }
        if distance_metric not in metric_map:
            raise ValueError(
                f"Unknown distance_metric: {distance_metric}. "
                f"Use 'euclidean', 'cosine', or 'manhattan'."
            )

        condensed = pdist(matrix, metric=metric_map[distance_metric])
        dist_matrix = squareform(condensed)
        return pd.DataFrame(dist_matrix, index=labels, columns=labels)

    @classmethod
    def cluster(
        cls,
        distance_matrix: pd.DataFrame,
        n_clusters: int = 3,
        distance_threshold: Optional[float] = None,
        linkage_method: str = "average",
    ) -> pd.DataFrame:
        """Group items by metric similarity using hierarchical clustering.

        Args:
            distance_matrix: Square distance matrix from compute_distance_matrix.
            n_clusters: Number of clusters. Ignored if distance_threshold is set.
            distance_threshold: Cut dendrogram at this distance. Overrides n_clusters.
            linkage_method: 'average', 'complete', 'single', or 'ward'.

        Returns:
            DataFrame with columns [label, cluster].
        """
        labels = distance_matrix.index.tolist()

        if len(labels) < 2:
            return pd.DataFrame({"label": labels, "cluster": [1] * len(labels)})

        condensed = squareform(distance_matrix.values, checks=False)
        Z = linkage(condensed, method=linkage_method)

        if distance_threshold is not None:
            clusters = fcluster(Z, t=distance_threshold, criterion="distance")
        else:
            clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

        return pd.DataFrame({"label": labels, "cluster": clusters.astype(int)})

    @classmethod
    def find_similar(
        cls,
        distance_matrix: pd.DataFrame,
        target: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Find items most similar to a target based on metric profiles.

        Args:
            distance_matrix: Square distance matrix from compute_distance_matrix.
            target: Item label to find similarities for.
            top_k: Number of similar items to return.

        Returns:
            DataFrame with columns [label, distance, rank] sorted by distance.
        """
        if target not in distance_matrix.index:
            raise ValueError(f"'{target}' not found in distance matrix.")

        distances = distance_matrix.loc[target].drop(target)
        sorted_dists = distances.sort_values().head(top_k)
        return pd.DataFrame(
            {
                "label": sorted_dists.index,
                "distance": sorted_dists.values,
                "rank": range(1, len(sorted_dists) + 1),
            }
        ).reset_index(drop=True)

    @classmethod
    def detect_anomalous(
        cls,
        distance_matrix: pd.DataFrame,
        threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Detect items with unusual metric profiles.

        Computes mean distance from each item to all others. Items whose
        z-score exceeds the threshold are flagged as anomalous.

        Args:
            distance_matrix: Square distance matrix from compute_distance_matrix.
            threshold: Z-score threshold for anomaly detection.

        Returns:
            DataFrame with columns [label, anomaly_score, z_score, is_anomalous].
        """
        labels = distance_matrix.index.tolist()
        n = len(labels)
        mean_dists = distance_matrix.values.sum(axis=1) / max(n - 1, 1)

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        if global_std < 1e-10:
            z_scores = np.zeros(n)
        else:
            z_scores = (mean_dists - global_mean) / global_std

        return pd.DataFrame(
            {
                "label": labels,
                "anomaly_score": mean_dists,
                "z_score": z_scores,
                "is_anomalous": z_scores > threshold,
            }
        )

    @classmethod
    def detect_changes(
        cls,
        metric_profiles: pd.DataFrame,
        uuid_column: str = "uuid",
        group_column: str = "segment_index",
        metric_columns: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Track how each UUID's metrics change across consecutive segments.

        Computes Euclidean distance between consecutive segment metric vectors
        for each UUID. Large change scores indicate process shifts.

        Args:
            metric_profiles: Output from SegmentProcessor.compute_metric_profiles.
            uuid_column: Column identifying each timeseries.
            group_column: Column ordering the segments (e.g. 'segment_index').
            metric_columns: Which metric columns to use. None auto-detects.
            normalize: Z-normalize metrics before computing change scores.

        Returns:
            DataFrame with columns [uuid, <group_column>, change_score].
        """
        cols = cls._get_metric_columns(metric_profiles, metric_columns)
        rows = []

        for uuid_val, group in metric_profiles.groupby(uuid_column):
            group = group.sort_values(group_column)
            matrix = group[cols].values.astype(float)
            matrix = np.nan_to_num(matrix, nan=0.0)

            if normalize and matrix.shape[0] > 1:
                col_std = matrix.std(axis=0)
                col_std[col_std < 1e-10] = 1.0
                matrix = (matrix - matrix.mean(axis=0)) / col_std

            segments = group[group_column].values
            for i in range(1, len(matrix)):
                dist = float(np.linalg.norm(matrix[i] - matrix[i - 1]))
                rows.append(
                    {
                        uuid_column: uuid_val,
                        group_column: segments[i],
                        "change_score": dist,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=[uuid_column, group_column, "change_score"])
        return pd.DataFrame(rows).reset_index(drop=True)

    @classmethod
    def find_similar_pairs(
        cls,
        metric_profiles: pd.DataFrame,
        uuid_column: str = "uuid",
        group_column: str = "segment_value",
        metric_columns: Optional[List[str]] = None,
        normalize: bool = True,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Find the most similar (UUID, segment) pairs across all data.

        Useful for finding which process parameters behave similarly across
        different orders or part numbers.

        Args:
            metric_profiles: Output from SegmentProcessor.compute_metric_profiles.
            uuid_column: Column identifying each timeseries.
            group_column: Column identifying each segment.
            metric_columns: Which metric columns to use. None auto-detects.
            normalize: Z-normalize metrics before computing distances.
            top_k: Number of closest pairs to return.

        Returns:
            DataFrame with columns [uuid_a, group_a, uuid_b, group_b, distance, rank].
        """
        cols = cls._get_metric_columns(metric_profiles, metric_columns)
        matrix = metric_profiles[cols].values.astype(float)
        matrix = np.nan_to_num(matrix, nan=0.0)

        if normalize and matrix.shape[0] > 1:
            col_std = matrix.std(axis=0)
            col_std[col_std < 1e-10] = 1.0
            matrix = (matrix - matrix.mean(axis=0)) / col_std

        uuids = metric_profiles[uuid_column].values
        groups = metric_profiles[group_column].values

        condensed = pdist(matrix, metric="euclidean")
        n = len(matrix)

        k = min(top_k, len(condensed))
        if k == 0:
            return pd.DataFrame(
                columns=[
                    "uuid_a",
                    "group_a",
                    "uuid_b",
                    "group_b",
                    "distance",
                    "rank",
                ]
            )

        if len(condensed) <= top_k:
            sorted_indices = np.argsort(condensed)
        else:
            sorted_indices = np.argpartition(condensed, k)[:k]
            sorted_indices = sorted_indices[np.argsort(condensed[sorted_indices])]

        rows = []
        for rank, idx in enumerate(sorted_indices, 1):
            i, j = cls._condensed_to_square(idx, n)
            rows.append(
                {
                    "uuid_a": uuids[i],
                    "group_a": groups[i],
                    "uuid_b": uuids[j],
                    "group_b": groups[j],
                    "distance": float(condensed[idx]),
                    "rank": rank,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _condensed_to_square(idx: int, n: int):
        """Convert a condensed distance matrix index to (row, col) pair."""
        i = int(n - 2 - np.floor(np.sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
        j = int(idx + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
        return i, j
