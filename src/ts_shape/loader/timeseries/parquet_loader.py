import logging
import warnings
from functools import partial
from pathlib import Path

import pandas as pd

from ts_shape.errors import LoaderConfigWarning
from ts_shape.loader._utils import retry_call, same_tz_bounds, validate_local_path
from ts_shape.utils.deprecation import caller_stacklevel

logger = logging.getLogger(__name__)


def _folder_bounds(
    start_time: str | pd.Timestamp, end_time: str | pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Naive UTC bounds to compare ``YYYY/MM/DD/HH`` folder times with.

    The start is floored to the hour, so a mid-hour start still reads the
    folder that contains it. tz-aware bounds are converted to UTC; naive
    bounds are taken as UTC (the folder layout's timezone).
    """
    start, end = same_tz_bounds(start_time, end_time)
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
        end = end.tz_convert("UTC").tz_localize(None)
    return start.floor("h"), end


class ParquetLoader:
    """
    This class provides class methods to load parquet files from a specified directory structure.
    """

    def __init__(self, base_path: str):
        """
        Initialize the ParquetLoader with the base directory path.

        Args:
            base_path (str): The base directory where parquet files are stored.

        Raises:
            LoaderError: If ``base_path`` does not exist or is not a directory.
        """
        self.base_path = validate_local_path(base_path, must_be_dir=True)

    @classmethod
    def _get_parquet_files(cls, base_path: Path) -> list:
        """
        Recursively finds all parquet files in the directory structure.

        Args:
            base_path (Path): The base directory path.

        Returns:
            list: A list of paths to all found parquet files.
        """
        # Use rglob to recursively find all .parquet files in the directory
        return list(base_path.rglob("*.parquet"))

    @classmethod
    def _read_concat(cls, files: list, base_path: Path) -> pd.DataFrame:
        """Read parquet ``files`` (each with retry) and concatenate them.

        Warns with :class:`LoaderConfigWarning` and returns an empty DataFrame
        when ``files`` is empty -- a likely sign of a misconfigured path or
        filter rather than a genuine error.
        """
        if not files:
            warnings.warn(
                f"No parquet files matched under {base_path}. "
                "Check the path and any time-range/uuid filters.",
                LoaderConfigWarning,
                stacklevel=caller_stacklevel(),
            )
            return pd.DataFrame()
        dataframes = [
            retry_call(
                partial(pd.read_parquet, file),
                exclude=(FileNotFoundError,),
                description=f"read_parquet({file})",
            )
            for file in files
        ]
        return pd.concat(dataframes, ignore_index=True)

    @classmethod
    def load_all_files(cls, base_path: str) -> pd.DataFrame:
        """
        Loads all parquet files in the specified base directory into a single pandas DataFrame.

        Args:
            base_path (str): The base directory where parquet files are stored.

        Returns:
            pd.DataFrame: A DataFrame containing all the data from the parquet files.
        """
        root = validate_local_path(base_path, must_be_dir=True)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(root)
        return cls._read_concat(parquet_files, root)

    @classmethod
    def load_by_time_range(
        cls, base_path: str, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Loads parquet files that fall within a specified time range based on the directory structure.

        The directory structure is expected to be in the format YYYY/MM/DD/HH
        (UTC). The hour folder that contains ``start_time`` is included;
        tz-aware bounds are converted to UTC, naive ones are taken as UTC.

        Args:
            base_path (str): The base directory where parquet files are stored.
            start_time (pd.Timestamp): The start timestamp.
            end_time (pd.Timestamp): The end timestamp.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files within the time range.
        """
        root = validate_local_path(base_path, must_be_dir=True)
        first_hour, last = _folder_bounds(start_time, end_time)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(root)
        valid_files = []

        for file in parquet_files:
            try:
                # Extract the timestamp from the file's relative path
                folder_parts = file.relative_to(root).parts[
                    :4
                ]  # Extract YYYY/MM/DD/HH parts
                folder_time_str = "/".join(folder_parts)
                file_time = pd.to_datetime(folder_time_str, format="%Y/%m/%d/%H")

                # Check if the file's timestamp falls within the specified time range
                if first_hour <= file_time <= last:
                    valid_files.append(file)
            except ValueError:
                # Skip files that do not follow the expected folder structure
                continue

        return cls._read_concat(valid_files, root)

    @classmethod
    def load_by_uuid_list(cls, base_path: str, uuid_list: list) -> pd.DataFrame:
        """
        Loads parquet files that match any UUID in the specified list.

        A file matches when its name without ``.parquet`` equals a UUID
        (exact match, not a substring).

        Args:
            base_path (str): The base directory where parquet files are stored.
            uuid_list (list): A list of UUIDs to filter the files.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files with matching UUIDs.
        """
        root = validate_local_path(base_path, must_be_dir=True)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(root)
        valid_files = []

        wanted = {str(u) for u in uuid_list}
        for file in parquet_files:
            # The file name (without extension) is the uuid: exact match only,
            # so "temp" does not also load "temp_raw".
            if file.stem in wanted:
                valid_files.append(file)

        return cls._read_concat(valid_files, root)

    @classmethod
    def load_files_by_time_range_and_uuids(
        cls,
        base_path: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        uuid_list: list,
    ) -> pd.DataFrame:
        """
        Loads parquet files that fall within a specified time range and match any UUID in the list.

        The directory structure is expected to be in the format YYYY/MM/DD/HH
        (UTC; the hour folder that contains ``start_time`` is included), and a
        file matches when its name without ``.parquet`` equals a UUID.

        Args:
            base_path (str): The base directory where parquet files are stored.
            start_time (pd.Timestamp): The start timestamp.
            end_time (pd.Timestamp): The end timestamp.
            uuid_list (list): A list of UUIDs to filter the files.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files that meet both criteria.
        """
        root = validate_local_path(base_path, must_be_dir=True)
        first_hour, last = _folder_bounds(start_time, end_time)
        wanted = {str(u) for u in uuid_list}
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(root)
        valid_files = []

        for file in parquet_files:
            try:
                # Extract the timestamp from the file's relative path
                folder_parts = file.relative_to(root).parts[
                    :4
                ]  # Extract YYYY/MM/DD/HH parts
                folder_time_str = "/".join(folder_parts)
                file_time = pd.to_datetime(folder_time_str, format="%Y/%m/%d/%H")

                # Check if the file's timestamp falls within the specified time range
                if first_hour <= file_time <= last and file.stem in wanted:
                    valid_files.append(file)
            except ValueError:
                # Skip files that do not follow the expected folder structure
                continue

        return cls._read_concat(valid_files, root)
