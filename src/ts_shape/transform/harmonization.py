import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs
from ts_shape.utils.frames import as_datetime, require_columns

logger = logging.getLogger(__name__)

Sampling = Literal["first", "interpolate"]


def _to_grid(
    wide: pd.DataFrame,
    window: str,
    *,
    sampling: Sampling,
    method: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """``wide`` on the ``resample(window)`` grid (see :meth:`resample_to_uniform`)."""
    if sampling not in ("first", "interpolate"):
        raise ValueError(f"sampling must be 'first' or 'interpolate', got {sampling!r}")
    binned = wide.resample(window).first()
    if sampling == "first":
        return binned.interpolate(method=method, limit=limit, limit_area="inside")
    grid = binned.index
    union = wide.index.union(grid)
    # 'linear' on a DatetimeIndex ignores the timestamps; 'time' is linear in
    # time, which is what a value at a grid instant needs.
    how = "time" if method == "linear" else method
    dense = wide.reindex(union).interpolate(
        method=how, limit=limit, limit_area="inside"
    )
    return dense.reindex(grid)


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
        """Keep a time-sorted copy of ``dataframe``.

        Args:
            dataframe: Long-format input.
            time_column: Timestamp column. Text and pyarrow timestamps are
                converted; an empty untyped column becomes ``datetime64``.
            uuid_column: Signal identifier column.
            value_column: Column holding the values to harmonize.

        Raises:
            ColumnNotFoundError: If one of the three columns is missing.
        """
        require_columns(
            dataframe,
            [time_column, uuid_column, value_column],
            where="DataHarmonizer",
        )
        super().__init__(dataframe, time_column=time_column)
        self.dataframe[time_column] = as_datetime(
            self.dataframe[time_column], where="DataHarmonizer"
        )
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.value_column = value_column

    def _pivot(self, dataframe: pd.DataFrame, aggfunc: str) -> pd.DataFrame:
        """Wide frame with a DatetimeIndex; signals without values are kept."""
        if dataframe.empty:
            index = pd.DatetimeIndex(dataframe[self.time_column], name=self.time_column)
            return pd.DataFrame(index=index[:0], columns=pd.Index([], dtype=object))
        wide: pd.DataFrame = dataframe.pivot_table(
            index=self.time_column,
            columns=self.uuid_column,
            values=self.value_column,
            aggfunc=aggfunc,
            dropna=False,
        )
        return wide

    def pivot_to_wide(self, aggfunc: str = "first") -> pd.DataFrame:
        """Pivot long-format DataFrame to wide-format with one column per UUID.

        Memory: the wide frame is dense, one row per *distinct* timestamp of
        any signal and one column per UUID, so it holds
        ``timestamps x signals`` cells (8 bytes each for float values)
        however sparse the long frame is. Signals sampled at their own
        instants barely share rows: 100 such signals of 100,000 samples each
        (10 million long rows, about 0.5 GB) pivot to 10 million x 100 cells,
        8 GB, mostly NaN. Select the signals you need and resample or round
        the timestamps onto a common grid before pivoting large data
        (:meth:`resample_to_uniform` pivots too, then resamples).

        Args:
            aggfunc: Aggregation function for duplicate timestamps ('first', 'mean', 'last').

        Returns:
            DataFrame with systime as index and one column per UUID. A signal
                without any value is kept as an all-NaN column.
        """
        dupes = self.dataframe.duplicated(
            subset=[self.time_column, self.uuid_column], keep=False
        )
        if dupes.any():
            warnings.warn(
                f"Found {dupes.sum()} duplicate (time, uuid) entries; "
                f"aggregating with '{aggfunc}'.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        return self._pivot(self.dataframe, aggfunc)

    @renamed_kwargs(freq="window")
    def resample_to_uniform(
        self,
        window: str,
        method: str = "linear",
        fill_limit: int | None = None,
        *,
        sampling: Sampling = "first",
    ) -> pd.DataFrame:
        """Resample to a uniform time grid with interpolation.

        Pivots first, so the full wide frame of :meth:`pivot_to_wide` (one
        row per distinct timestamp, see its memory note) is built before it
        is reduced to the grid.

        The grid is the ``resample(window)`` bin starts. ``sampling`` sets
        what a grid row holds:

        - ``"first"`` (default): the *first sample in the bin*
          ``[t, t + window)``, labeled with the bin start ``t``. The value is
          not the signal at ``t``: a sample at ``t + 0.5 s`` in a 1 s bin is
          reported at ``t``. Empty bins are then interpolated with
          ``method``.
        - ``"interpolate"``: the signal's value *at* ``t``, interpolated
          between the raw samples on either side (linear in time for
          ``method="linear"``).

        Either way a signal is only filled *between* its samples
        (``limit_area="inside"``): grid rows before its first or after its
        last sample stay NaN rather than repeat the edge value.

        Args:
            window: Pandas frequency string (e.g. '1s', '100ms', '1min').
            method: Interpolation method ('linear', 'time', 'nearest',
                'quadratic', 'cubic').
            fill_limit: Maximum number of consecutive NaNs to fill.
            sampling: ``"first"`` (default) or ``"interpolate"``, see above.

        Returns:
            DataFrame with uniform DatetimeIndex.
        """
        wide = self.pivot_to_wide()
        return _to_grid(
            wide, window, sampling=sampling, method=method, limit=fill_limit
        )

    def detect_gaps(self, threshold: str = "10s") -> pd.DataFrame:
        """Identify time gaps per UUID exceeding the threshold.

        Args:
            threshold: Minimum gap duration as a pandas Timedelta string.

        Returns:
            DataFrame with one row per gap: ``uuid``, ``gap_start``,
                ``gap_end``, ``gap_duration`` (Timedelta) and the canonical
                ``start``, ``end``, ``duration_seconds`` (the same gap). Empty
                with these columns, typed, when there is no gap.
        """
        threshold_td = pd.Timedelta(threshold)
        time_dtype = self.dataframe[self.time_column].dtype
        uuids: list[object] = []
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []

        for uid, group in self.dataframe.groupby(self.uuid_column, sort=True):
            times = group[self.time_column].dropna().sort_values()
            diffs = times.diff()
            is_gap = (diffs > threshold_td).to_numpy()
            positions = np.flatnonzero(is_gap)
            uuids.extend([uid] * len(positions))
            starts.extend(times.iloc[positions - 1])
            ends.extend(times.iloc[positions])

        start = pd.Series(starts, dtype=time_dtype)
        end = pd.Series(ends, dtype=time_dtype)
        duration = end - start
        return pd.DataFrame(
            {
                "uuid": pd.Series(uuids, dtype=object),
                "gap_start": start,
                "gap_end": end,
                "gap_duration": duration,
                "start": start,
                "end": end,
                "duration_seconds": duration.dt.total_seconds().astype("float64"),
            }
        )

    def fill_gaps(
        self,
        strategy: str = "interpolate",
        max_gap: str | None = None,
        fill_value: float | None = None,
    ) -> pd.DataFrame:
        """Fill gaps in the wide-format data using the specified strategy.

        Args:
            strategy: One of 'interpolate', 'ffill', 'bfill', 'constant'.
            max_gap: Maximum gap size to fill (pandas Timedelta string),
                measured from the valid sample before a run of missing values
                to the valid sample after it. Larger gaps remain NaN. ``None``
                (default) fills every gap the strategy can fill, however long.
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
            times = wide.index
            n = len(times)
            for col in wide.columns:
                missing = wide[col].isna().to_numpy()
                if not missing.any():
                    continue
                # Runs of consecutive missing rows: [first, last] positions.
                edges = np.diff(np.r_[0, missing.astype(np.int8), 0])
                firsts = np.flatnonzero(edges == 1)
                lasts = np.flatnonzero(edges == -1) - 1
                for first, last in zip(firsts, lasts):
                    # The gap spans from the valid sample before the run to
                    # the valid sample after it.
                    before = times[first - 1] if first > 0 else times[first]
                    after = times[last + 1] if last + 1 < n else times[last]
                    if (after - before) > max_gap_td:
                        filled.iloc[first : last + 1, filled.columns.get_loc(col)] = (
                            np.nan
                        )

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

        Warns:
            DataQualityWarning: When a uuid is not in the (non-empty) frame;
                the result then has no matches for it.
        """
        present = set(self.dataframe[self.uuid_column].dropna().unique())
        for name, uid in (("left_uuid", left_uuid), ("right_uuid", right_uuid)):
            if uid not in present and not self.dataframe.empty:
                warnings.warn(
                    f"DataHarmonizer.align_asof: {name} {uid!r} not found. "
                    f"Available uuids: {sorted(str(u) for u in present)}",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )
        left = (
            self.dataframe[self.dataframe[self.uuid_column] == left_uuid][
                [self.time_column, self.value_column]
            ]
            .dropna(subset=[self.time_column])
            .sort_values(self.time_column)
            .rename(columns={self.value_column: "value_left"})
        )
        right = (
            self.dataframe[self.dataframe[self.uuid_column] == right_uuid][
                [self.time_column, self.value_column]
            ]
            .dropna(subset=[self.time_column])
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

    @renamed_kwargs(freq="window")
    def merge_multi_signals(
        self,
        uuids: list[str] | None = None,
        window: str | None = None,
        method: str = "linear",
        *,
        sampling: Sampling = "first",
    ) -> pd.DataFrame:
        """End-to-end harmonization: pivot, filter, resample, interpolate.

        The selected signals are pivoted before resampling: the memory of
        :meth:`pivot_to_wide` applies to them (pass ``uuids`` to bound it).

        Args:
            uuids: Optional list of UUIDs to include. None means all.
            window: Optional resample frequency. None means no resampling.
            method: Interpolation method for resampling.
            sampling: With ``window``, ``"first"`` (default: the first sample
                in each bin, labeled with the bin start) or ``"interpolate"``
                (the value at each grid timestamp); see
                :meth:`resample_to_uniform`. A signal is not extended before
                its first or after its last sample.

        Returns:
            Wide-format DataFrame ready for cross-signal analytics.
        """
        df = self.dataframe
        if uuids is not None:
            df = df[df[self.uuid_column].isin(uuids)]

        wide = self._pivot(df, "first")

        if window is not None:
            wide = _to_grid(wide, window, sampling=sampling, method=method)

        return wide
