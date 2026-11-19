import io
import logging
import warnings
import zipfile
from collections.abc import Iterator
from functools import partial
from pathlib import Path, PurePosixPath

import pandas as pd

from ts_shape.errors import LoaderConfigWarning
from ts_shape.loader._utils import (
    retry_call,
    same_tz_bounds,
    trim_to_time_range,
    utc_naive,
    validate_local_path,
)
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs

logger = logging.getLogger(__name__)


def _folder_bounds(
    start_time: str | pd.Timestamp, end_time: str | pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Naive UTC bounds to compare ``YYYY/MM/DD/HH`` folder times with.

    The start is floored to the hour, so a mid-hour start still reads the
    folder that contains it. An end before the start raises ``ValueError``. tz-aware bounds are converted to UTC; naive
    bounds are taken as UTC (the folder layout's timezone).
    """
    start, end = same_tz_bounds(start_time, end_time)
    if end < start:
        raise ValueError(
            f"ParquetLoader: end ({end}) is before start ({start}); are they swapped?"
        )
    return utc_naive(start).floor("h"), utc_naive(end)


class _ZipMember:
    """A parquet file inside a zip export (path-like for the loader)."""

    def __init__(self, archive: Path, name: str) -> None:
        self.archive = archive
        self.name = name
        parts = PurePosixPath(name).parts
        # The YYYY/MM/DD/HH layout may sit below a top folder in the zip.
        start = next((i for i, p in enumerate(parts) if len(p) == 4 and p.isdigit()), 0)
        self._rel = PurePosixPath(*parts[start:])
        self.stem = PurePosixPath(name).stem

    def relative_to(self, _root: object) -> PurePosixPath:
        return self._rel

    def read(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.archive) as archive:
            return pd.read_parquet(io.BytesIO(archive.read(self.name)))

    def __repr__(self) -> str:
        return f"{self.archive}!{self.name}"


class _ZipArchive:
    """A zip export of ``YYYY/MM/DD/HH/<uuid>.parquet`` files."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def members(self) -> list[_ZipMember]:
        with zipfile.ZipFile(self.path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".parquet")]
        return [_ZipMember(self.path, n) for n in sorted(names)]

    def __str__(self) -> str:
        return str(self.path)


def _root(base_path: str | Path) -> Path | _ZipArchive:
    """A folder of hour folders, or a ``.zip`` export of one."""
    path = Path(base_path)
    if path.is_file() and zipfile.is_zipfile(path):
        return _ZipArchive(path)
    return validate_local_path(base_path, must_be_dir=True)


def _read_one(file: Path | _ZipMember) -> pd.DataFrame:
    return file.read() if isinstance(file, _ZipMember) else pd.read_parquet(file)


class ParquetLoader:
    """
    This class provides class methods to load parquet files from a specified directory structure.

    ``base_path`` is a folder of ``YYYY/MM/DD/HH/<uuid>.parquet`` files, or a
    ``.zip`` export of such a folder (read in place, no unpacking).

    Example:
        Files are named ``<uuid>.parquet`` in ``YYYY/MM/DD/HH`` folders:

        >>> import tempfile
        >>> from pathlib import Path
        >>> import ts_shape
        >>> from ts_shape import ParquetLoader
        >>> tmp = tempfile.TemporaryDirectory()
        >>> folder = Path(tmp.name, "2025", "01", "01", "00")
        >>> folder.mkdir(parents=True)
        >>> ts_shape.make_timeseries(["temp"], n_points=3, freq="1min", noise=0).to_parquet(
        ...     folder / "temp.parquet")
        >>> ParquetLoader.load_by_uuid_list(tmp.name, ["temp"])[
        ...     ["systime", "uuid", "value_double"]]
                      systime  uuid  value_double
        0 2025-01-01 00:00:00  temp         100.0
        1 2025-01-01 00:01:00  temp         100.0
        2 2025-01-01 00:02:00  temp         100.0
        >>> tmp.cleanup()
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
    def _get_parquet_files(cls, base_path: "Path | _ZipArchive") -> list:
        """
        Recursively finds all parquet files in the directory structure.

        Args:
            base_path (Path): The base directory path.

        Returns:
            list: A list of paths to all found parquet files.
        """
        if isinstance(base_path, _ZipArchive):
            return base_path.members()
        # Use rglob to recursively find all .parquet files in the directory
        return list(base_path.rglob("*.parquet"))

    @classmethod
    def _read_concat(cls, files: list, base_path: "Path | _ZipArchive") -> pd.DataFrame:
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
                partial(_read_one, file),
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
        root = _root(base_path)
        # Get all parquet files in the directory
        parquet_files = cls._get_parquet_files(root)
        return cls._read_concat(parquet_files, root)

    @classmethod
    @renamed_kwargs(start_time="start", end_time="end")
    def load_by_time_range(
        cls,
        base_path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        trim_to_range: bool = True,
    ) -> pd.DataFrame:
        """
        Loads parquet files that fall within a specified time range based on the directory structure.

        The directory structure is expected to be in the format YYYY/MM/DD/HH
        (UTC). The hour folder that contains ``start`` is included;
        tz-aware bounds are converted to UTC, naive ones are taken as UTC.

        Args:
            base_path (str): The base directory where parquet files are stored.
            start (pd.Timestamp): The start timestamp.
            end (pd.Timestamp): The end timestamp.
            trim_to_range: If True (default), drop rows whose ``systime``
                falls outside ``[start, end]`` (both bounds inclusive; naive
                bounds are UTC). Frames without a ``systime`` column are left
                as read. False keeps whole hour folders.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files within the time range.
        """
        root = _root(base_path)
        first_hour, last = _folder_bounds(start, end)
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

        out = cls._read_concat(valid_files, root)
        return trim_to_time_range(out, start, end) if trim_to_range else out

    @classmethod
    def stream_by_time_range(
        cls,
        base_path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        chunk: str = "1h",
        signal_uuids: list | None = None,
        compact: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Load ``[start, end)`` one slice at a time (for month-scale data).

        A month of a large line does not fit in memory at once; this yields
        one frame per ``chunk`` (via :func:`ts_shape.stream.time_range_source`),
        ready for :func:`ts_shape.run_chunked`::

            for frame in ParquetLoader.stream_by_time_range(path, t0, t1, chunk="6h"):
                ...

        Args:
            base_path: The folder of hour folders, or a ``.zip`` export.
            start: First instant.
            end: End of the range, exclusive.
            chunk: Slice length (``"1h"``, ``"6h"``, ``"1D"``).
            signal_uuids: Load only these signals (default: all).
            compact: Shrink each slice with
                :func:`ts_shape.utils.frames.compact_frame`.

        Yields:
            One DataFrame per slice (slices without data are skipped).
        """
        from ts_shape.stream import time_range_source
        from ts_shape.utils.frames import compact_frame

        def load(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LoaderConfigWarning)
                if signal_uuids is None:
                    frame = cls.load_by_time_range(base_path, lo, hi)
                else:
                    frame = cls.load_files_by_time_range_and_uuids(
                        base_path, lo, hi, signal_uuids
                    )
            return compact_frame(frame) if compact and len(frame) else frame

        for frame in time_range_source(load, start, end, chunk):
            if len(frame):
                yield frame

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
        root = _root(base_path)
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
    @renamed_kwargs(start_time="start", end_time="end")
    def load_files_by_time_range_and_uuids(
        cls,
        base_path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        uuid_list: list,
        *,
        trim_to_range: bool = True,
    ) -> pd.DataFrame:
        """
        Loads parquet files that fall within a specified time range and match any UUID in the list.

        The directory structure is expected to be in the format YYYY/MM/DD/HH
        (UTC; the hour folder that contains ``start`` is included), and a
        file matches when its name without ``.parquet`` equals a UUID.

        Args:
            base_path (str): The base directory where parquet files are stored.
            start (pd.Timestamp): The start timestamp.
            end (pd.Timestamp): The end timestamp.
            uuid_list (list): A list of UUIDs to filter the files.
            trim_to_range: If True (default), drop rows whose ``systime``
                falls outside ``[start, end]`` (both bounds inclusive; naive
                bounds are UTC). Frames without a ``systime`` column are left
                as read. False keeps whole hour folders.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the parquet files that meet both criteria.
        """
        root = _root(base_path)
        first_hour, last = _folder_bounds(start, end)
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

        out = cls._read_concat(valid_files, root)
        return trim_to_time_range(out, start, end) if trim_to_range else out
