import logging
import pandas as pd  # type: ignore
from pathlib import Path

logger = logging.getLogger(__name__)


class ParquetLoader:
    """
    This class provides class methods to load parquet files from a specified directory structure.
    """

    def __init__(self, base_path: str):
        """
        Initialize the ParquetLoader with the base directory path.

        Args:
            base_path (str): The base directory where parquet files are stored.
        """
        self.base_path = Path(base_path)

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
    def load_all_files(cls, base_path: str) -> pd.DataFrame:
        """
        Loads all parquet files in the specified base directory into a single pandas DataFrame.

        Args:
            base_path (str): The base directory where parquet files are stored.

        Returns:
            pd.DataFrame: A DataFrame containing all the data from the parquet files.
        """
        # Convert base path to a Path object
        base_path = Path(base_path)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(base_path)
        # Load all files into pandas DataFrames
        dataframes = [pd.read_parquet(file) for file in parquet_files]

        # Concatenate all DataFrames into a single DataFrame
        if not dataframes:
            return pd.DataFrame()
        return pd.concat(dataframes, ignore_index=True)

    @classmethod
    def load_by_time_range(
        cls, base_path: str, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Loads parquet files that fall within a specified time range based on the directory structure.

        The directory structure is expected to be in the format YYYY/MM/DD/HH.

        Args:
            base_path (str): The base directory where parquet files are stored.
            start_time (pd.Timestamp): The start timestamp.
            end_time (pd.Timestamp): The end timestamp.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files within the time range.
        """
        # Convert base path to a Path object
        base_path = Path(base_path)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(base_path)
        valid_files = []

        for file in parquet_files:
            try:
                # Extract the timestamp from the file's relative path
                folder_parts = file.relative_to(base_path).parts[
                    :4
                ]  # Extract YYYY/MM/DD/HH parts
                folder_time_str = "/".join(folder_parts)
                file_time = pd.to_datetime(folder_time_str, format="%Y/%m/%d/%H")

                # Check if the file's timestamp falls within the specified time range
                if start_time <= file_time <= end_time:
                    valid_files.append(file)
            except ValueError:
                # Skip files that do not follow the expected folder structure
                continue

        # Load all valid files into pandas DataFrames
        dataframes = [pd.read_parquet(file) for file in valid_files]
        if not dataframes:
            return pd.DataFrame()
        return pd.concat(dataframes, ignore_index=True)

    @classmethod
    def load_by_uuid_list(cls, base_path: str, uuid_list: list) -> pd.DataFrame:
        """
        Loads parquet files that match any UUID in the specified list.

        The UUIDs are expected to be part of the file names.

        Args:
            base_path (str): The base directory where parquet files are stored.
            uuid_list (list): A list of UUIDs to filter the files.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files with matching UUIDs.
        """
        # Convert base path to a Path object
        base_path = Path(base_path)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(base_path)
        valid_files = []

        for file in parquet_files:
            # Extract the file name without extension
            file_name = file.stem
            # Check if the file name contains any of the UUIDs in the list
            for uuid in uuid_list:
                if uuid in file_name:
                    valid_files.append(file)
                    break  # Stop checking other UUIDs for this file

        # Load all valid files into pandas DataFrames
        dataframes = [pd.read_parquet(file) for file in valid_files]
        if not dataframes:
            return pd.DataFrame()
        return pd.concat(dataframes, ignore_index=True)

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

        The directory structure is expected to be in the format YYYY/MM/DD/HH, and UUIDs are part of the file names.

        Args:
            base_path (str): The base directory where parquet files are stored.
            start_time (pd.Timestamp): The start timestamp.
            end_time (pd.Timestamp): The end timestamp.
            uuid_list (list): A list of UUIDs to filter the files.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files that meet both criteria.
        """
        # Convert base path to a Path object
        base_path = Path(base_path)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(base_path)
        valid_files = []

        for file in parquet_files:
            try:
                # Extract the timestamp from the file's relative path
                folder_parts = file.relative_to(base_path).parts[
                    :4
                ]  # Extract YYYY/MM/DD/HH parts
                folder_time_str = "/".join(folder_parts)
                file_time = pd.to_datetime(folder_time_str, format="%Y/%m/%d/%H")

                # Check if the file's timestamp falls within the specified time range
                if start_time <= file_time <= end_time:
                    # Extract the file name without extension
                    file_name = file.stem
                    # Check if the file name contains any of the UUIDs in the list
                    for uuid in uuid_list:
                        if uuid in file_name:
                            valid_files.append(file)
                            break  # Stop checking other UUIDs for this file
            except ValueError:
                # Skip files that do not follow the expected folder structure
                continue

        # Load all valid files into pandas DataFrames
        dataframes = [pd.read_parquet(file) for file in valid_files]
        if not dataframes:
            return pd.DataFrame()
        return pd.concat(dataframes, ignore_index=True)
