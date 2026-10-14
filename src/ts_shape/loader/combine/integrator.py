import logging
import warnings
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from ts_shape.errors import DataQualityWarning
from ts_shape.loader._utils import empty_signal_frame

logger = logging.getLogger(__name__)


def _warn(message: str) -> None:
    warnings.warn(message, DataQualityWarning, stacklevel=4)


class DataIntegratorHybrid:
    """
    A flexible utility class to integrate data from various sources, including:
    - API instances (e.g., DatapointAPI)
    - Direct raw data (e.g., UUID list, metadata, timeseries DataFrame)
    - Hybrid approaches (combination of instances and raw data)

    Problems (an unsupported source, a missing join key, timeseries sources
    with different timezone handling) are reported as
    :class:`~ts_shape.errors.DataQualityWarning`.
    """

    @classmethod
    def combine_data(
        cls,
        timeseries_sources: list[pd.DataFrame | object] | None = None,
        metadata_sources: list[pd.DataFrame | object] | None = None,
        uuids: list[str] | None = None,
        join_key: str = "uuid",
        merge_how: str = "left",
    ) -> pd.DataFrame:
        """Combine timeseries and metadata from various sources.

        Timeseries sources are concatenated. When their time columns differ
        in timezone handling (some tz-aware, some naive, or different
        zones), every source is converted to UTC -- naive times are read as
        UTC -- with a :class:`~ts_shape.errors.DataQualityWarning`.
        Metadata sources are combined per ``join_key``: several sources may
        describe the same uuid with different columns, but two different
        values for the same uuid and column raise ``ValueError``.

        Args:
            timeseries_sources: List of timeseries sources (DataFrame or instances
                with `fetch_data_as_dataframe`).
            metadata_sources: List of metadata sources (DataFrame or instances with
                `fetch_metadata`).
            uuids: Optional list of UUIDs to keep; applied whether or not
                metadata is given (compared as text).
            join_key: Key column to use for merging, default is "uuid".
            merge_how: Merge strategy ('left', 'inner', etc.), default is "left".

        Returns:
            A combined DataFrame. Without timeseries data, an empty frame with
            the standard columns (``systime``, ``uuid``, ``value_*``,
            ``is_delta``).

        Raises:
            ValueError: When metadata sources give conflicting rows for one
                uuid.
        """
        # Retrieve and combine timeseries data
        timeseries_data = cls._combine_timeseries(timeseries_sources, join_key)

        if timeseries_data.empty:
            _warn("DataIntegratorHybrid: no timeseries data found")
            return empty_signal_frame()

        if uuids is not None and join_key in timeseries_data.columns:
            wanted = {str(u) for u in uuids}
            keep = timeseries_data[join_key].astype(str).isin(wanted)
            timeseries_data = timeseries_data[keep.to_numpy()].reset_index(drop=True)
        elif uuids is not None:
            _warn(
                f"DataIntegratorHybrid: cannot filter by uuids, join key "
                f"{join_key!r} is missing in the timeseries data"
            )

        # Retrieve and combine metadata
        metadata = cls._combine_metadata(metadata_sources, join_key)

        if metadata.empty:
            logger.info("No metadata found.")
            return timeseries_data

        missing_timeseries_key = join_key not in timeseries_data.columns
        missing_metadata_key = join_key not in metadata.columns

        if missing_timeseries_key or missing_metadata_key:
            missing_parts = []
            if missing_timeseries_key:
                missing_parts.append("timeseries data")
            if missing_metadata_key:
                missing_parts.append("metadata")
            _warn(
                f"DataIntegratorHybrid: cannot merge because join key "
                f"{join_key!r} is missing in {', '.join(missing_parts)}"
            )
            return timeseries_data

        # Merge timeseries data with metadata
        return pd.merge(timeseries_data, metadata, on=join_key, how=merge_how)

    @classmethod
    def _combine_timeseries(
        cls, sources: list[pd.DataFrame | object] | None, join_key: str
    ) -> pd.DataFrame:
        """Combine timeseries data from multiple sources.

        Args:
            sources: List of sources (DataFrame or instances with
                `fetch_data_as_dataframe`).
            join_key: Key column to use for merging.

        Returns:
            A combined timeseries DataFrame.
        """
        if not sources:
            return pd.DataFrame()

        frames = []
        for source in sources:
            if isinstance(source, pd.DataFrame):
                frames.append(cls._ensure_join_key_column(source, join_key))
            elif hasattr(source, "fetch_data_as_dataframe"):
                df = source.fetch_data_as_dataframe()
                frames.append(cls._ensure_join_key_column(df, join_key))
            else:
                _warn(
                    "DataIntegratorHybrid: unsupported timeseries source "
                    f"{type(source).__name__}; it is skipped"
                )

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        frames = cls._same_timezone_handling(frames)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _same_timezone_handling(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Convert time columns to UTC when sources disagree on timezones."""
        time_columns = {
            c for f in frames for c in f.columns if ptypes.is_datetime64_any_dtype(f[c])
        }
        out = [f.copy() for f in frames]
        for col in sorted(time_columns, key=str):
            zones = {
                str(f[col].dt.tz)
                for f in out
                if col in f.columns and ptypes.is_datetime64_any_dtype(f[col])
            }
            if len(zones) <= 1:
                continue
            _warn(
                f"DataIntegratorHybrid: time column {col!r} differs between "
                f"sources ({sorted(zones)}); all are converted to UTC, naive "
                "times are read as UTC"
            )
            for f in out:
                if col in f.columns and ptypes.is_datetime64_any_dtype(f[col]):
                    times = f[col]
                    f[col] = (
                        times.dt.tz_localize("UTC")
                        if times.dt.tz is None
                        else times.dt.tz_convert("UTC")
                    )
        return out

    @classmethod
    def _combine_metadata(
        cls, sources: list[pd.DataFrame | object] | None, join_key: str
    ) -> pd.DataFrame:
        """Combine metadata from multiple sources.

        Args:
            sources: List of sources (DataFrame or instances with
                `fetch_metadata`).
            join_key: Key column to use for merging.

        Returns:
            A combined metadata DataFrame, one row per ``join_key``.

        Raises:
            ValueError: When two rows give different values for the same key
                and column.
        """
        if not sources:
            return pd.DataFrame()

        frames = []
        for source in sources:
            if isinstance(source, pd.DataFrame):
                frames.append(cls._ensure_join_key_column(source, join_key))
            elif hasattr(source, "fetch_metadata"):
                df = source.fetch_metadata()
                frames.append(cls._ensure_join_key_column(df, join_key))
            else:
                _warn(
                    "DataIntegratorHybrid: unsupported metadata source "
                    f"{type(source).__name__}; it is skipped"
                )

        if not frames:
            return pd.DataFrame()
        metadata = pd.concat(frames, ignore_index=True)
        if join_key not in metadata.columns or metadata.empty:
            return metadata
        return cls._one_row_per_key(metadata, join_key)

    @staticmethod
    def _one_row_per_key(metadata: pd.DataFrame, join_key: str) -> pd.DataFrame:
        """Collapse rows of the same key; conflicting values raise."""
        if metadata[join_key].is_unique:
            return metadata
        conflicts: list[str] = []
        for col in metadata.columns:
            if col == join_key:
                continue
            values: dict[Any, set[str]] = {}
            for key, value in zip(metadata[join_key], metadata[col]):
                if value is None or (not isinstance(value, str) and pd.isna(value)):
                    continue
                values.setdefault(key, set()).add(repr(value))
            bad = [k for k, v in values.items() if len(v) > 1]
            if bad:
                conflicts.append(f"{col!r} for {join_key} {bad[:3]}")
        if conflicts:
            raise ValueError(
                "DataIntegratorHybrid: metadata sources disagree: "
                + "; ".join(conflicts)
                + ". Keep one metadata row per uuid."
            )
        return (
            metadata.groupby(join_key, sort=False, dropna=False).first().reset_index()
        )

    @staticmethod
    def _ensure_join_key_column(df: pd.DataFrame, join_key: str) -> pd.DataFrame:
        """
        Ensure the join key exists as a column. If it lives in the index, bring it into the columns.
        """
        if join_key in df.columns:
            return df

        if isinstance(df.index, pd.MultiIndex):
            index_names = list(df.index.names)
            if join_key in index_names:
                return df.reset_index(level=join_key)
        else:
            index_names = [df.index.name]
            if df.index.name == join_key:
                return df.reset_index()

        _warn(
            f"DataIntegratorHybrid: join key {join_key!r} not found in columns or "
            f"index. Columns: {list(df.columns)}; index names: {index_names}"
        )
        return df
