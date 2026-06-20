from collections.abc import Iterable, Iterator
from pathlib import Path
import logging
import os
import warnings

import pandas as pd  # type: ignore

from ts_shape.errors import LoaderConfigWarning

logger = logging.getLogger(__name__)


class DatabricksUnityCatalogParquetLoader:
    """
    Load canonical parquet files governed by **Databricks Unity Catalog**.

    Unity Catalog exposes the *same* parquet files that already live in your
    cloud storage (an external UC Volume over, e.g., an Azure blob container).
    The on-disk layout is unchanged -- ``<prefix>/YYYY/MM/DD/HH/<uuid>.parquet``
    with the canonical columns ``systime, uuid, value_double, value_integer,
    value_string, value_bool, is_delta`` -- so the frames returned here flow
    straight into every ts-shape transformation, event detector, ``Pipeline``
    and ``DataIntegratorHybrid`` without any change.

    **Designed for use inside Databricks notebooks / pipelines.** A UC Volume is
    FUSE-mounted at ``/Volumes/<catalog>/<schema>/<volume>/...``, so this loader
    reads parquet *directly from that mounted path* with ``pandas.read_parquet``.
    It deliberately keeps the resource footprint low:

    - **No download step and no ``databricks-sdk`` / network client** -- the
      mounted Volume is read like a local directory.
    - **Navigates only the hour folders in range** (``YYYY/MM/DD/HH``) instead of
      scanning the whole tree, so a one-hour query never lists the whole Volume.
    - **Column projection and row-predicate pushdown** via ``columns`` / ``filters``
      forwarded to ``pandas.read_parquet`` (pyarrow), so only needed bytes are read.
    - **Streaming generators** (:meth:`stream_by_time_range`,
      :meth:`stream_files_by_time_range_and_uuids`) yield one frame at a time so
      the driver never has to hold the full dataset in memory.

    Reads are sequential by design (no thread pool) to avoid adding CPU/memory
    pressure on a shared cluster driver; pushdown + streaming keep them cheap.

    Off-cluster (outside Databricks) the same code works against any mounted or
    synced copy of the Volume; if you cannot mount it, use
    :class:`AzureBlobParquetLoader` against the underlying storage instead.
    """

    def __init__(
        self,
        volume_path: str | None = None,
        *,
        catalog: str | None = None,
        schema: str | None = None,
        volume: str | None = None,
        prefix: str = "",
        base_path: str = "/Volumes",
        hour_pattern: str = "{Y}/{m}/{d}/{H}/",
        validate: bool = True,
    ) -> None:
        """
        Resolve the mounted Unity Catalog Volume root to read from.

        Provide *either* an explicit ``volume_path`` *or* the
        ``catalog`` / ``schema`` / ``volume`` triple (joined under ``base_path``)::

            # explicit mounted path
            DatabricksUnityCatalogParquetLoader(
                volume_path="/Volumes/main/plant/timeseries", prefix="parquet",
            )

            # from catalog/schema/volume parts
            DatabricksUnityCatalogParquetLoader(
                catalog="main", schema="plant", volume="timeseries",
                prefix="parquet",
            )

        Args:
            volume_path: Full mounted Volume root, e.g.
                ``/Volumes/<catalog>/<schema>/<volume>``.
            catalog: UC catalog name (used when ``volume_path`` is omitted).
            schema: UC schema name (used when ``volume_path`` is omitted).
            volume: UC volume name (used when ``volume_path`` is omitted).
            prefix: Optional sub-path beneath the Volume root (e.g. ``"parquet"``).
            base_path: Mount root for Volumes; ``/Volumes`` inside Databricks.
            hour_pattern: Pattern for the hour-level subpath; tokens
                ``{Y} {m} {d} {H}``. Default ``YYYY/MM/DD/HH/``.
            validate: Warn (``LoaderConfigWarning``) if the resolved root does
                not exist, so a missing/mistyped mount fails loudly but cheaply.
        """
        if volume_path:
            root = Path(volume_path)
        elif catalog and schema and volume:
            root = Path(base_path) / catalog / schema / volume
        else:
            raise ValueError(
                "Provide either volume_path or all of catalog, schema and volume "
                "(e.g. volume_path='/Volumes/<catalog>/<schema>/<volume>')."
            )

        self.volume_root = root
        self.prefix = prefix.strip("/")
        # Directory under which hour folders live: <volume_root>/<prefix>
        self.base_path = root / self.prefix if self.prefix else root
        self.hour_pattern = hour_pattern

        if validate and not self.base_path.exists():
            warnings.warn(
                "DatabricksUnityCatalogParquetLoader path does not exist: "
                f"{self.base_path}. Inside Databricks ensure the Unity Catalog "
                "Volume is mounted (default '/Volumes/<catalog>/<schema>/<volume>') "
                "and that catalog/schema/volume/prefix are correct.",
                LoaderConfigWarning,
                stacklevel=2,
            )

    # ---- Hourly-layout helpers (mirror AzureBlobParquetLoader) ----
    @staticmethod
    def _hourly_slots(
        start_timestamp: str | pd.Timestamp, end_timestamp: str | pd.Timestamp
    ) -> Iterable[pd.Timestamp]:
        start = pd.to_datetime(start_timestamp)
        end = pd.to_datetime(end_timestamp)
        # Inclusive hourly range
        return pd.date_range(start=start, end=end, freq="h")

    def _hour_dir(self, ts: pd.Timestamp) -> Path:
        """Return the directory for a given hour, e.g. <base>/2024/01/31/09."""
        y = str(ts.year)
        m = str(ts.month).zfill(2)
        d = str(ts.day).zfill(2)
        h = str(ts.hour).zfill(2)
        sub = (
            self.hour_pattern.replace("{Y}", y)
            .replace("{m}", m)
            .replace("{d}", d)
            .replace("{H}", h)
        )
        return self.base_path / sub.strip("/")

    # ---- Filesystem listing (cheap; no file contents read) ----
    def _list_parquet_in(self, directory: Path) -> list[Path]:
        """List ``*.parquet`` files in a single hour directory (no recursion)."""
        try:
            with os.scandir(directory) as it:
                return [
                    Path(entry.path)
                    for entry in it
                    if entry.is_file() and entry.name.endswith(".parquet")
                ]
        except (FileNotFoundError, NotADirectoryError):
            return []
        except OSError as exc:
            logger.debug("Failed to list directory '%s': %s", directory, exc)
            return []

    def _walk_parquet(self, root: Path) -> list[Path]:
        """Recursively find all ``*.parquet`` files under ``root``.

        Only used by the explicit "load everything" entry points
        (:meth:`load_all_files`, :meth:`list_structure`).
        """
        if not root.exists():
            return []
        return sorted(root.rglob("*.parquet"))

    def _read_parquet(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame | None:
        """Read one mounted parquet file with column/predicate pushdown.

        Returns ``None`` on failure (missing, unauthorized, or corrupt) so the
        caller can count and report it rather than aborting the whole load.
        """
        try:
            return pd.read_parquet(path, columns=columns, filters=filters)
        except Exception as exc:
            logger.debug("Failed to read parquet '%s': %s", path, exc)
            return None

    def _read_many(
        self,
        paths: list[Path],
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> tuple[list[pd.DataFrame], int, int]:
        """Read a set of parquet files sequentially.

        Returns ``(frames, n_failed, n_empty)`` where ``frames`` are the
        non-empty DataFrames, ``n_failed`` counts files that could not be read
        or parsed, and ``n_empty`` counts files that parsed but had no rows.
        """
        frames: list[pd.DataFrame] = []
        n_failed = 0
        n_empty = 0
        for path in paths:
            df = self._read_parquet(path, columns, filters)
            if df is None:
                n_failed += 1
            elif df.empty:
                n_empty += 1
            else:
                frames.append(df)
        return frames, n_failed, n_empty

    def _warn_empty(
        self, method: str, detail: str, filters: list | None = None
    ) -> None:
        """Emit a ``LoaderConfigWarning`` explaining why a load returned no data."""
        if filters is not None:
            detail = f"{detail} The filters argument may be too strict."
        warnings.warn(
            f"DatabricksUnityCatalogParquetLoader.{method} returned an empty "
            f"DataFrame. {detail}",
            LoaderConfigWarning,
            stacklevel=3,
        )

    def _time_search_detail(self, slots: list[pd.Timestamp]) -> str:
        """Describe the path/pattern/hour-range probed by a time-range load."""
        base_repr = repr(str(self.base_path))
        pattern_repr = repr(self.hour_pattern)
        if not slots:
            return (
                f"base_path={base_repr}, hour_pattern={pattern_repr}, "
                "0 hour-folders (the time range is empty)"
            )
        first = slots[0].strftime("%Y-%m-%d %H:00")
        last = slots[-1].strftime("%Y-%m-%d %H:00")
        return (
            f"base_path={base_repr}, hour_pattern={pattern_repr}, "
            f"{len(slots)} hour-folder(s) {first} .. {last}"
        )

    # ---- Public API (mirrors AzureBlobParquetLoader) ----
    def load_all_files(
        self,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        """
        Load every parquet file under the Volume root (optionally below ``prefix``).

        This walks the whole tree; for large Volumes prefer
        :meth:`load_by_time_range` or :meth:`stream_by_time_range`.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        paths = self._walk_parquet(self.base_path)
        base_repr = repr(str(self.base_path))
        if not paths:
            self._warn_empty(
                "load_all_files",
                f"No .parquet files found under base_path={base_repr}. Check that "
                "the Unity Catalog Volume is mounted and that the prefix matches "
                "the stored layout.",
            )
            return pd.DataFrame()

        frames, n_failed, n_empty = self._read_many(paths, columns, filters)
        if not frames:
            self._warn_empty(
                "load_all_files",
                f"Found {len(paths)} .parquet file(s) under base_path={base_repr} "
                f"but none yielded rows ({n_failed} failed to read/parse, "
                f"{n_empty} had no rows). Check read permission on the Volume and "
                "that the files are valid parquet.",
                filters=filters,
            )
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_by_time_range(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        """
        Load parquet files under the hourly folders within ``[start, end]``.

        Visits only the ``YYYY/MM/DD/HH`` folders in range, so the rest of the
        Volume is never scanned.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        paths: list[Path] = []
        for ts in slots:
            paths.extend(self._list_parquet_in(self._hour_dir(ts)))

        search = self._time_search_detail(slots)
        if not paths:
            self._warn_empty(
                "load_by_time_range",
                f"No .parquet files found. {search}. Check that hour_pattern "
                "matches the real folder layout and that the time range overlaps "
                "the stored data.",
            )
            return pd.DataFrame()

        frames, n_failed, n_empty = self._read_many(paths, columns, filters)
        if not frames:
            self._warn_empty(
                "load_by_time_range",
                f"Found {len(paths)} .parquet file(s) but none yielded rows "
                f"({n_failed} failed to read/parse, {n_empty} had no rows). "
                f"{search}. Check read permission and file validity.",
                filters=filters,
            )
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def stream_by_time_range(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """
        Stream parquet frames under the hourly folders within ``[start, end]``.

        Yields ``(file_path, DataFrame)`` one at a time so the full dataset is
        never held in memory -- the recommended low-memory entry point for
        Databricks pipelines.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        yielded = 0
        for ts in slots:
            for path in self._list_parquet_in(self._hour_dir(ts)):
                df = self._read_parquet(path, columns, filters)
                if df is not None and not df.empty:
                    yielded += 1
                    yield (str(path), df)

        if yielded == 0:
            self._warn_empty(
                "stream_by_time_range",
                f"Yielded no DataFrames. {self._time_search_detail(slots)}. Check "
                "that hour_pattern matches the real folder layout and that the "
                "time range overlaps the stored data.",
                filters=filters,
            )

    def load_files_by_time_range_and_uuids(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        uuid_list: list[str],
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        """
        Load parquet files for the given UUIDs within ``[start, end]`` hours.

        Each hour folder is scanned once and files whose basename matches a
        requested UUID (``<uuid>.parquet``, case-insensitive) are read.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        if not uuid_list:
            return pd.DataFrame()

        variants = self._uuid_basenames(uuid_list)
        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        paths: list[Path] = []
        for ts in slots:
            for path in self._list_parquet_in(self._hour_dir(ts)):
                if path.name.lower() in variants:
                    paths.append(path)

        search = self._time_search_detail(slots)
        uuid_note = f"{len(variants)} UUID basename(s) searched"
        if not paths:
            self._warn_empty(
                "load_files_by_time_range_and_uuids",
                f"No matching .parquet files found. {search}, {uuid_note}. Check "
                "that the UUIDs, hour_pattern and time range match the stored data.",
            )
            return pd.DataFrame()

        frames, n_failed, n_empty = self._read_many(paths, columns, filters)
        if not frames:
            self._warn_empty(
                "load_files_by_time_range_and_uuids",
                f"Resolved {len(paths)} file(s) but none yielded rows "
                f"({n_failed} failed to read/parse, {n_empty} had no rows). "
                f"{search}, {uuid_note}. Check read permission and file validity.",
                filters=filters,
            )
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def stream_files_by_time_range_and_uuids(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        uuid_list: list[str],
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """
        Stream parquet frames for the given UUIDs within ``[start, end]`` hours.

        Yields ``(file_path, DataFrame)`` as they are read, keeping memory flat.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        if not uuid_list:
            return

        variants = self._uuid_basenames(uuid_list)
        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        yielded = 0
        for ts in slots:
            for path in self._list_parquet_in(self._hour_dir(ts)):
                if path.name.lower() not in variants:
                    continue
                df = self._read_parquet(path, columns, filters)
                if df is not None and not df.empty:
                    yielded += 1
                    yield (str(path), df)

        if yielded == 0:
            self._warn_empty(
                "stream_files_by_time_range_and_uuids",
                f"Yielded no DataFrames. {self._time_search_detail(slots)}, "
                f"{len(variants)} UUID basename(s) searched. Check that the "
                "UUIDs, hour_pattern and time range match the stored data.",
                filters=filters,
            )

    @staticmethod
    def _uuid_basenames(uuid_list: list[str]) -> set[str]:
        """Build the set of lowercase ``<uuid>.parquet`` basenames to match."""
        variants: set[str] = set()
        for u in uuid_list:
            s = str(u).strip().strip("{}").strip().lower()
            if s:
                variants.add(f"{s}.parquet")
        return variants

    def list_structure(
        self, parquet_only: bool = True, limit: int | None = None
    ) -> dict[str, list[str]]:
        """
        List the folders (hours) and files under the configured Volume path.

        Args:
            parquet_only: If True, only include files ending with ``.parquet``.
            limit: Optional cap on number of files collected for quick inspection.

        Returns:
            A dict with ``folders`` (sorted unique parent folders) and ``files``
            (sorted file paths) as strings relative-free full mounted paths.
        """
        folders: set[str] = set()
        files: list[str] = []
        collected = 0
        if not self.base_path.exists():
            return {"folders": [], "files": []}

        pattern = "*.parquet" if parquet_only else "*"
        for path in sorted(self.base_path.rglob(pattern)):
            if not path.is_file():
                continue
            files.append(str(path))
            folders.add(str(path.parent) + "/")
            collected += 1
            if limit is not None and collected >= limit:
                break

        return {"folders": sorted(folders), "files": sorted(files)}

    def fetch_data_as_dataframe(
        self,
        start_timestamp: str | pd.Timestamp | None = None,
        end_timestamp: str | pd.Timestamp | None = None,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        """
        Return a combined DataFrame, for ``Pipeline`` / ``DataIntegratorHybrid``.

        With a ``start``/``end`` pair this delegates to :meth:`load_by_time_range`
        (visiting only the in-range hour folders); with no bounds it falls back to
        :meth:`load_all_files`.
        """
        if start_timestamp is not None and end_timestamp is not None:
            return self.load_by_time_range(
                start_timestamp, end_timestamp, columns=columns, filters=filters
            )
        return self.load_all_files(columns=columns, filters=filters)
