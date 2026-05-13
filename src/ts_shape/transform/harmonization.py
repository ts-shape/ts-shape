import logging
import warnings
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, List

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DataHarmonizer(Base):
    """Data Harmonization for multi-signal timeseries.

    Provides utilities to pivot, resample, align, and fill gaps across
    multiple UUID-keyed signals stored in long (stacked) format.

    Methods:
    - pivot_to_wide: Pivot long-format to wide-format (one column per UUID).
    - resample_to_uniform: Resample to a uniform time grid with interpolation.
    - detect_gaps: Identify time gaps per UUID exceeding a threshold.
    - fill_gaps: Fill detected gaps using various strategies.
    - align_asof: Align two UUID signals using merge_asof.
    - merge_multi_signals: End-to-end harmonization pipeline.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.value_column = value_column

    def pivot_to_wide(self, aggfunc: str = "first") -> pd.DataFrame:
        """Pivot long-format DataFrame to wide-format with one column per UUID.

        Args:
            aggfunc: Aggregation function for duplicate timestamps ('first', 'mean', 'last').

        Returns:
            DataFrame with systime as index and one column per UUID.
        """
        dupes = self.dataframe.duplicated(
            subset=[self.time_column, self.uuid_column], keep=False
        )
        if dupes.any():
            warnings.warn(
                f"Found {dupes.sum()} duplicate (time, uuid) entries; "
                f"aggregating with '{aggfunc}'.",
                DataQualityWarning,
                stacklevel=2,
            )
        return self.dataframe.pivot_table(
            index=self.time_column,
            columns=self.uuid_column,
            values=self.value_column,
            aggfunc=aggfunc,
        )

    def resample_to_uniform(
        self,
        freq: str,
        method: str = "linear",
        fill_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Resample to a uniform time grid with interpolation.

        Args:
            freq: Pandas frequency string (e.g. '1s', '100ms', '1min').
            method: Interpolation method ('linear', 'time', 'nearest', 'quadratic', 'cubic').
            fill_limit: Maximum number of consecutive NaNs to fill.

        Returns:
            DataFrame with uniform DatetimeIndex.
        """
        wide = self.pivot_to_wide()
        resampled = wide.resample(freq).first()
        return resampled.interpolate(method=method, limit=fill_limit)

    def detect_gaps(self, threshold: str = "10s") -> pd.DataFrame:
        """Identify time gaps per UUID exceeding the threshold.

        Args:
            threshold: Minimum gap duration as a pandas Timedelta string.

        Returns:
            DataFrame with columns: uuid, gap_start, gap_end, gap_duration.
        """
        threshold_td = pd.Timedelta(threshold)
        rows = []

        for uid, group in self.dataframe.groupby(self.uuid_column):
            times = group[self.time_column].sort_values().reset_index(drop=True)
            diffs = times.diff()
            gap_positions = diffs[diffs > threshold_td].index

            for pos in gap_positions:
                gap_start = times.iloc[pos - 1]
                gap_end = times.iloc[pos]
                rows.append(
                    {
                        "uuid": uid,
                        "gap_start": gap_start,
                        "gap_end": gap_end,
                        "gap_duration": gap_end - gap_start,
                    }
                )

        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=["uuid", "gap_start", "gap_end", "gap_duration"])
        )

    def fill_gaps(
        self,
        strategy: str = "interpolate",
        max_gap: Optional[str] = None,
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """Fill gaps in the wide-format data using the specified strategy.

        Args:
            strategy: One of 'interpolate', 'ffill', 'bfill', 'constant'.
            max_gap: Maximum gap size to fill (pandas Timedelta string). Larger gaps remain NaN.
            fill_value: Value used when strategy='constant'.

        Returns:
            Wide-format DataFrame with gaps filled.
        """
        wide = self.pivot_to_wide()

        if strategy == "interpolate":
            filled = wide.interpolate(method="time", limit_area="inside")
        elif strategy == "ffill":
            filled = wide.ffill()
        elif strategy == "bfill":
            filled = wide.bfill()
        elif strategy == "constant":
            filled = wide.fillna(fill_value)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Use 'interpolate', 'ffill', 'bfill', or 'constant'."
            )

        if max_gap is not None:
            max_gap_td = pd.Timedelta(max_gap)
            nan_mask = wide.isna()
            for col in wide.columns:
                col_nans = nan_mask[col]
                if not col_nans.any():
                    continue
                groups = (col_nans != col_nans.shift()).cumsum()
                for _, grp_idx in col_nans[col_nans].groupby(groups):
                    start = grp_idx.index[0]
                    end = grp_idx.index[-1]
                    if (end - start) > max_gap_td:
                        filled.loc[grp_idx.index, col] = np.nan

        return filled

    def align_asof(
        self,
        left_uuid: str,
        right_uuid: str,
        tolerance: str = "1s",
        direction: str = "nearest",
    ) -> pd.DataFrame:
        """Align two UUID signals using merge_asof with configurable tolerance.

        Args:
            left_uuid: UUID of the left (reference) signal.
            right_uuid: UUID of the right signal.
            tolerance: Maximum time difference for matching.
            direction: One of 'nearest', 'backward', 'forward'.

        Returns:
            DataFrame with systime, value_left, value_right.
        """
        left = (
            self.dataframe[self.dataframe[self.uuid_column] == left_uuid][
                [self.time_column, self.value_column]
            ]
            .sort_values(self.time_column)
            .rename(columns={self.value_column: "value_left"})
        )
        right = (
            self.dataframe[self.dataframe[self.uuid_column] == right_uuid][
                [self.time_column, self.value_column]
            ]
            .sort_values(self.time_column)
            .rename(columns={self.value_column: "value_right"})
        )

        return pd.merge_asof(
            left,
            right,
            on=self.time_column,
            tolerance=pd.Timedelta(tolerance),
            direction=direction,
        )

    def merge_multi_signals(
        self,
        uuids: Optional[List[str]] = None,
        freq: Optional[str] = None,
        method: str = "linear",
    ) -> pd.DataFrame:
        """End-to-end harmonization: pivot, filter, resample, interpolate.

        Args:
            uuids: Optional list of UUIDs to include. None means all.
            freq: Optional resample frequency. None means no resampling.
            method: Interpolation method for resampling.

        Returns:
            Wide-format DataFrame ready for cross-signal analytics.
        """
        df = self.dataframe
        if uuids is not None:
            df = df[df[self.uuid_column].isin(uuids)]

        wide = df.pivot_table(
            index=self.time_column,
            columns=self.uuid_column,
            values=self.value_column,
            aggfunc="first",
        )

        if freq is not None:
            wide = wide.resample(freq).first()
            wide = wide.interpolate(method=method)

        return wide
