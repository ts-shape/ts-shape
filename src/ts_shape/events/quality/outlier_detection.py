import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import zscore

from ts_shape.events._output import (
    drop_duplicate_events,
    empty_event_df,
    finalize_point_df,
)
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only

logger = logging.getLogger(__name__)

# Try to import sklearn for IsolationForest
try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class OutlierDetectionEvents(Base):
    """
    Processes time series data to detect outliers based on specified statistical methods.

    Example:
        >>> import ts_shape
        >>> df = ts_shape.make_timeseries(["temp"], n_outliers=3)
        >>> out = ts_shape.OutlierDetectionEvents(df, "value_double").detect_outliers_zscore()
        >>> out[["systime", "value_double", "severity"]].round(
        ...     {"value_double": 1, "severity": 1}
        ... )
                      systime  value_double  severity
        0 2025-01-01 01:34:30          89.7       8.9
        1 2025-01-01 03:21:00          88.8       9.6
        2 2025-01-01 05:23:00          88.6       9.8
        3 2025-01-01 07:54:30          96.4       3.1
    """

    #: ``severity`` is a numeric score; no ``severity_level`` is added.
    _severity_levels = None

    @keyword_only(3)
    def __init__(
        self,
        dataframe: pd.DataFrame,
        value_column: str,
        event_uuid: str = "quality:outlier",
        time_threshold: str = "5min",
        group_by_uuid: bool = True,
        *,
        signal_uuids: list[str] | None = None,
    ) -> None:
        """
        Initializes the OutlierDetectionEvents with specific attributes for outlier detection.

        Args:
            dataframe (pd.DataFrame): The input time series DataFrame.
            value_column (str): The name of the column containing the values for outlier detection.
            event_uuid (str): A UUID or identifier for detected outlier events.
            time_threshold (str): The time threshold to group close events together.
            group_by_uuid (bool): Compute the statistics (mean/std, quartiles,
                median/MAD, forest) and group events per ``uuid`` (default
                ``True``), so a frame holding e.g. a temperature (~80) and a
                pressure (~2) is scored per signal. ``False`` pools every row
                into one population (one mean/std across all signals).
            signal_uuids: Signals to analyze. ``None`` (default) analyzes
                every ``uuid`` in the frame.
        """
        super().__init__(dataframe)
        self.signal_uuids = list(signal_uuids) if signal_uuids is not None else None
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_threshold = time_threshold
        self.group_by_uuid = group_by_uuid

    def _group_outliers(
        self, outliers_df: pd.DataFrame, include_singles: bool = True
    ) -> pd.DataFrame:
        """
        Groups detected outliers that are close in time and prepares the final events DataFrame.

        Args:
            outliers_df (pd.DataFrame): DataFrame containing detected outliers.
            include_singles (bool): Whether to include single outliers in the output. Default is True.

        Returns:
            pd.DataFrame: Canonical ``point``-shape event DataFrame
            (``systime``, ``uuid``, ``source_uuid``) plus ``<value_column>``,
            ``severity`` and ``is_delta``. ``uuid`` carries the event id and
            ``source_uuid`` the originating signal id.
        """
        # Carry the input columns too, so an empty result has the same
        # schema as a non-empty one (which keeps every input column).
        dropped = {"systime", "uuid", "source_uuid", "outlier", "group_id"}
        extra_cols = [c for c in outliers_df.columns if c not in dropped] + [
            self.value_column,
            "severity",
            "is_delta",
        ]
        if outliers_df.empty:
            return empty_event_df("point", extra_cols=extra_cols)

        # Grouping outliers that are close to each other in terms of time
        outliers_df["group_id"] = (
            outliers_df["systime"].diff().abs() > pd.to_timedelta(self.time_threshold)
        ).cumsum()

        # Prepare events DataFrame
        events_data = []

        group_sizes = outliers_df.groupby("group_id").size()
        multi_groups = group_sizes[group_sizes > 1].index
        single_groups = group_sizes[group_sizes == 1].index

        if len(multi_groups) > 0:
            multi_df = outliers_df[outliers_df["group_id"].isin(multi_groups)]
            firsts = multi_df.loc[multi_df.groupby("group_id")["systime"].idxmin()]
            lasts = multi_df.loc[multi_df.groupby("group_id")["systime"].idxmax()]
            events_data.append(pd.concat([firsts, lasts]).sort_values("systime"))

        if include_singles and len(single_groups) > 0:
            singles_df = outliers_df[outliers_df["group_id"].isin(single_groups)]
            events_data.append(singles_df)

        # Convert list of DataFrame slices to a single DataFrame
        if not events_data:
            return empty_event_df("point", extra_cols=extra_cols)
        events_df = pd.concat(events_data)

        # Ensure consistent schema
        events_df["is_delta"] = True
        if "severity" not in events_df.columns:
            events_df["severity"] = np.nan

        # Preserve the originating signal id as source_uuid before the
        # canonical helper broadcasts the event id into the uuid column.
        if "source_uuid" in events_df.columns:
            events_df = events_df.drop(columns=["uuid"], errors="ignore")
        elif "uuid" in events_df.columns:
            events_df = events_df.rename(columns={"uuid": "source_uuid"})
        events_df = events_df.drop(columns=["outlier", "group_id"], errors="ignore")

        # Duplicated input samples would report the same outlier twice.
        return drop_duplicate_events(
            finalize_point_df(events_df, uuid=self.event_uuid, source_uuid=None)
        )

    def _detect(
        self,
        scorer: Callable[[pd.Series], tuple[pd.Series, pd.Series]],
        include_singles: bool,
    ) -> pd.DataFrame:
        """Score values (pooled or per uuid), keep outliers and group them.

        ``scorer`` maps a value Series to an index-aligned boolean outlier
        mask and a severity Series.
        """
        df = self.dataframe
        if self.signal_uuids is not None:
            df = df[df["uuid"].isin(self.signal_uuids)]
        df = df.copy()

        # Convert 'systime' to datetime and sort the DataFrame by 'systime'
        df["systime"] = pd.to_datetime(df["systime"])
        # Unique labels: masks and severities are aligned by index.
        df = df.sort_values(by="systime", ascending=True).reset_index(drop=True)

        if not (self.group_by_uuid and "uuid" in df.columns):
            mask, severity = scorer(df[self.value_column])
            outliers_df = df.loc[mask].copy()
            outliers_df["severity"] = severity[mask]
            return self._group_outliers(outliers_df, include_singles=include_singles)

        parts = []
        for _, group in df.groupby("uuid", sort=True):
            mask, severity = scorer(group[self.value_column])
            outliers = group.loc[mask].copy()
            outliers["severity"] = severity[mask]
            if not outliers.empty:
                parts.append(
                    self._group_outliers(outliers, include_singles=include_singles)
                )
        if not parts:
            return self._group_outliers(df.iloc[0:0], include_singles=include_singles)
        return pd.concat(parts, ignore_index=True).sort_values("systime")

    @keyword_only(1)
    def detect_outliers_zscore(
        self, threshold: float = 3.0, include_singles: bool = True
    ) -> pd.DataFrame:
        """
        Detects outliers using the Z-score method.

        NaN values are ignored when computing the mean and standard deviation
        (one NaN used to turn every z-score into NaN, so nothing was found).

        Args:
            threshold (float): The Z-score threshold for detecting outliers.
            include_singles (bool): Whether to include single outliers in the output. Default is True.

        Returns:
            pd.DataFrame: A DataFrame of detected outliers and grouped events.
        """

        def scorer(values: pd.Series) -> tuple[pd.Series, pd.Series]:
            z = pd.Series(
                np.abs(zscore(values.astype(float), nan_policy="omit")),
                index=values.index,
            )
            return z > threshold, z

        return self._detect(scorer, include_singles)

    @keyword_only(1)
    def detect_outliers_iqr(
        self, threshold: tuple = (1.5, 1.5), include_singles: bool = True
    ) -> pd.DataFrame:
        """
        Detects outliers using the IQR method.

        Args:
            threshold (tuple): The multipliers for the IQR range for detecting outliers (lower, upper).
            include_singles (bool): Whether to include single outliers in the output. Default is True.

        Returns:
            pd.DataFrame: A DataFrame of detected outliers and grouped events.
        """

        def scorer(values: pd.Series) -> tuple[pd.Series, pd.Series]:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold[0] * iqr
            upper_bound = q3 + threshold[1] * iqr
            mask = (values < lower_bound) | (values > upper_bound)
            # Severity: normalized distance from bounds in terms of IQR
            if iqr > 0:
                severity = np.maximum(0, (lower_bound - values) / iqr) + np.maximum(
                    0, (values - upper_bound) / iqr
                )
            else:
                severity = pd.Series(0.0, index=values.index)
            return mask, severity

        return self._detect(scorer, include_singles)

    @keyword_only(1)
    def detect_outliers_mad(
        self, threshold: float = 3.5, include_singles: bool = True
    ) -> pd.DataFrame:
        """
        Detects outliers using the Median Absolute Deviation (MAD) method.
        This method is more robust to outliers than z-score.

        When more than half of the values are identical the MAD is 0. The
        modified z-score then falls back to the mean absolute deviation
        (``(x - median) / (1.253314 * MeanAD)``); if that is 0 too, no point
        is an outlier. (Previously MAD was replaced by machine epsilon, which
        flagged every value that differed from the median at all.)

        Args:
            threshold (float): The MAD threshold for detecting outliers. Default is 3.5.
            include_singles (bool): Whether to include single outliers in the output. Default is True.

        Returns:
            pd.DataFrame: A DataFrame of detected outliers and grouped events.
        """

        def scorer(values: pd.Series) -> tuple[pd.Series, pd.Series]:
            values = values.astype(float)
            median = values.median()
            deviation = values - median
            mad = deviation.abs().median()
            if mad > 0:
                modified_z = 0.6745 * deviation / mad
            else:
                mean_ad = deviation.abs().mean()
                if not mean_ad > 0:
                    none = pd.Series(False, index=values.index)
                    return none, pd.Series(0.0, index=values.index)
                modified_z = deviation / (1.253314 * mean_ad)
            severity = modified_z.abs()
            return severity > threshold, severity

        return self._detect(scorer, include_singles)

    @keyword_only(1)
    def detect_outliers_isolation_forest(
        self,
        contamination: float = 0.1,
        include_singles: bool = True,
        random_state: int | None = 42,
    ) -> pd.DataFrame:
        """
        Detects outliers using sklearn's IsolationForest algorithm.
        Falls back gracefully if sklearn is not available.

        Rows with a NaN value are not scored.

        Args:
            contamination (float): The proportion of outliers in the dataset. Default is 0.1.
            include_singles (bool): Whether to include single outliers in the output. Default is True.
            random_state (Optional[int]): Random state for reproducibility. Default is 42.

        Returns:
            pd.DataFrame: A DataFrame of detected outliers and grouped events.

        Raises:
            ImportError: If sklearn is not installed.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is not available. Please install it with: pip install scikit-learn"
            )

        def scorer(values: pd.Series) -> tuple[pd.Series, pd.Series]:
            mask = pd.Series(False, index=values.index)
            severity = pd.Series(np.nan, index=values.index)
            valid = values.dropna()
            if valid.empty:
                return mask, severity
            X = valid.to_numpy(dtype=float).reshape(-1, 1)
            iso_forest = IsolationForest(
                contamination=contamination, random_state=random_state
            )
            # IsolationForest returns -1 for outliers, 1 for inliers
            predictions = iso_forest.fit_predict(X)
            # More negative scores are more anomalous: severity = -score
            mask.loc[valid.index] = predictions == -1
            severity.loc[valid.index] = -iso_forest.score_samples(X)
            return mask, severity

        return self._detect(scorer, include_singles)


# Example usage:
# outlier_detector = OutlierDetectionEvents(dataframe=df, value_column='value')
# detected_outliers_zscore = outlier_detector.detect_outliers_zscore(threshold=3.0)
# detected_outliers_iqr = outlier_detector.detect_outliers_iqr(threshold=(1.5, 1.5))
# detected_outliers_mad = outlier_detector.detect_outliers_mad(threshold=3.5)
# detected_outliers_iforest = outlier_detector.detect_outliers_isolation_forest(contamination=0.1)
