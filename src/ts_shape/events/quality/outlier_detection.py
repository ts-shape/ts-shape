import logging
import pandas as pd  # type: ignore
import numpy as np
from scipy.stats import zscore
from typing import Optional
from ts_shape.utils.base import Base

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
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        value_column: str,
        event_uuid: str = "outlier_event",
        time_threshold: str = "5min",
    ) -> None:
        """
        Initializes the OutlierDetectionEvents with specific attributes for outlier detection.

        Args:
            dataframe (pd.DataFrame): The input time series DataFrame.
            value_column (str): The name of the column containing the values for outlier detection.
            event_uuid (str): A UUID or identifier for detected outlier events.
            time_threshold (str): The time threshold to group close events together.
        """
        super().__init__(dataframe)
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_threshold = time_threshold

    def _group_outliers(
        self, outliers_df: pd.DataFrame, include_singles: bool = True
    ) -> pd.DataFrame:
        """
        Groups detected outliers that are close in time and prepares the final events DataFrame.

        Args:
            outliers_df (pd.DataFrame): DataFrame containing detected outliers.
            include_singles (bool): Whether to include single outliers in the output. Default is True.

        Returns:
            pd.DataFrame: A DataFrame of grouped outlier events.
        """
        if outliers_df.empty:
            # Return empty DataFrame with consistent schema
            empty_df = pd.DataFrame(
                columns=[
                    "systime",
                    self.value_column,
                    "is_delta",
                    "uuid",
                    "severity_score",
                ]
            )
            return empty_df

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
        if events_data:
            events_df = pd.concat(events_data)
        else:
            # Create empty DataFrame with consistent schema
            events_df = pd.DataFrame(
                columns=[
                    "systime",
                    self.value_column,
                    "is_delta",
                    "uuid",
                    "severity_score",
                ]
            )
            return events_df

        # Ensure consistent schema
        events_df["is_delta"] = True
        events_df["uuid"] = self.event_uuid

        # Preserve severity_score if it exists, otherwise set to NaN
        if "severity_score" not in events_df.columns:
            events_df["severity_score"] = np.nan

        return events_df.drop(["outlier", "group_id"], axis=1, errors="ignore")

    def detect_outliers_zscore(
        self, threshold: float = 3.0, include_singles: bool = True
    ) -> pd.DataFrame:
        """
        Detects outliers using the Z-score method.

        Args:
            threshold (float): The Z-score threshold for detecting outliers.
            include_singles (bool): Whether to include single outliers in the output. Default is True.

        Returns:
            pd.DataFrame: A DataFrame of detected outliers and grouped events.
        """
        df = self.dataframe.copy()

        # Convert 'systime' to datetime and sort the DataFrame by 'systime'
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", ascending=True)

        # Calculate z-scores and detect outliers
        z_scores = np.abs(zscore(df[self.value_column]))
        df["outlier"] = z_scores > threshold

        # Filter to keep only outliers
        outliers_df = df.loc[df["outlier"]].copy()

        # Add severity score (absolute z-score value)
        outliers_df["severity_score"] = z_scores[df["outlier"]]

        # Group and return the outliers
        return self._group_outliers(outliers_df, include_singles=include_singles)

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
        df = self.dataframe.copy()

        # Convert 'systime' to datetime and sort the DataFrame by 'systime'
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", ascending=True)

        # Detect outliers using the IQR method
        Q1 = df[self.value_column].quantile(0.25)
        Q3 = df[self.value_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold[0] * IQR
        upper_bound = Q3 + threshold[1] * IQR
        df["outlier"] = (df[self.value_column] < lower_bound) | (
            df[self.value_column] > upper_bound
        )

        # Filter to keep only outliers
        outliers_df = df.loc[df["outlier"]].copy()

        # Add severity score (normalized distance from bounds in terms of IQR)
        if IQR > 0:
            lower_distance = np.maximum(
                0, (lower_bound - outliers_df[self.value_column]) / IQR
            )
            upper_distance = np.maximum(
                0, (outliers_df[self.value_column] - upper_bound) / IQR
            )
            outliers_df["severity_score"] = lower_distance + upper_distance
        else:
            outliers_df["severity_score"] = 0.0

        # Group and return the outliers
        return self._group_outliers(outliers_df, include_singles=include_singles)

    def detect_outliers_mad(
        self, threshold: float = 3.5, include_singles: bool = True
    ) -> pd.DataFrame:
        """
        Detects outliers using the Median Absolute Deviation (MAD) method.
        This method is more robust to outliers than z-score.

        Args:
            threshold (float): The MAD threshold for detecting outliers. Default is 3.5.
            include_singles (bool): Whether to include single outliers in the output. Default is True.

        Returns:
            pd.DataFrame: A DataFrame of detected outliers and grouped events.
        """
        df = self.dataframe.copy()

        # Convert 'systime' to datetime and sort the DataFrame by 'systime'
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", ascending=True)

        # Calculate MAD
        median = df[self.value_column].median()
        mad = np.median(np.abs(df[self.value_column] - median))

        # Handle case where MAD is 0 (all values are the same)
        if mad == 0:
            # Use a small constant to avoid division by zero
            mad = np.finfo(float).eps

        # Detect outliers using the MAD method
        modified_z_scores = 0.6745 * (df[self.value_column] - median) / mad
        df["outlier"] = np.abs(modified_z_scores) > threshold

        # Filter to keep only outliers
        outliers_df = df.loc[df["outlier"]].copy()

        # Add severity score (absolute modified z-score)
        outliers_df["severity_score"] = np.abs(modified_z_scores[df["outlier"]])

        # Group and return the outliers
        return self._group_outliers(outliers_df, include_singles=include_singles)

    def detect_outliers_isolation_forest(
        self,
        contamination: float = 0.1,
        include_singles: bool = True,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Detects outliers using sklearn's IsolationForest algorithm.
        Falls back gracefully if sklearn is not available.

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

        df = self.dataframe.copy()

        # Convert 'systime' to datetime and sort the DataFrame by 'systime'
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", ascending=True)

        # Prepare data for IsolationForest (needs 2D array)
        X = df[[self.value_column]].values

        # Detect outliers using IsolationForest
        iso_forest = IsolationForest(
            contamination=contamination, random_state=random_state
        )
        predictions = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)

        # Mark outliers (IsolationForest returns -1 for outliers, 1 for inliers)
        df["outlier"] = predictions == -1

        # Filter to keep only outliers
        outliers_df = df.loc[df["outlier"]].copy()

        # Add severity score (inverse of anomaly score, normalized to positive values)
        # More negative scores indicate more anomalous points
        outliers_df["severity_score"] = -anomaly_scores[df["outlier"]]

        # Group and return the outliers
        return self._group_outliers(outliers_df, include_singles=include_singles)


# Example usage:
# outlier_detector = OutlierDetectionEvents(dataframe=df, value_column='value')
# detected_outliers_zscore = outlier_detector.detect_outliers_zscore(threshold=3.0)
# detected_outliers_iqr = outlier_detector.detect_outliers_iqr(threshold=(1.5, 1.5))
# detected_outliers_mad = outlier_detector.detect_outliers_mad(threshold=3.5)
# detected_outliers_iforest = outlier_detector.detect_outliers_isolation_forest(contamination=0.1)
