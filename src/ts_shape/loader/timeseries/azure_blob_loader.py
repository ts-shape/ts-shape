import logging
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any

import pandas as pd  # type: ignore

from ts_shape.errors import LoaderConfigWarning

logger = logging.getLogger(__name__)


class AzureBlobParquetLoader:
    """
    Load parquet files from an Azure Blob Storage container filtered by a list of UUIDs.

    Optimized for speed by:
    - Using server-side prefix filtering when provided
    - Streaming blob listings and filtering client-side by UUID containment
    - Downloading and parsing parquet files concurrently
    """

    def __init__(
        self,
        container_name: str | None = None,
        *,
        connection_string: str | None = None,
        account_url: str | None = None,
        credential: Any | None = None,
        sas_url: str | None = None,
        prefix: str = "",
        max_workers: int = 8,
        hour_pattern: str = "{Y}/{m}/{d}/{H}/",
    ) -> None:
        """
        Initialize the loader with Azure connection details.

        Supports three authentication methods:

        1. **SAS URL** (simplest)::

            loader = AzureBlobParquetLoader(
                sas_url="https://account.blob.core.windows.net/container?sv=...&sig=..."
            )

           A SAS URL is also auto-detected when passed as ``connection_string``.

        2. **Connection string**::

            loader = AzureBlobParquetLoader(
                connection_string="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...",
                container_name="mycontainer",
            )

        3. **AAD credential** (account_url + credential)::

            loader = AzureBlobParquetLoader(
                account_url="https://account.blob.core.windows.net",
                container_name="mycontainer",
                credential=DefaultAzureCredential(),
            )

        Args:
            container_name: Target container name (not needed when using sas_url).
            connection_string: Azure Storage connection string.
            account_url: Full account URL for AAD auth.
            credential: Azure credential object for AAD auth.
            sas_url: Full Blob SAS URL including container and SAS token.
            prefix: Optional path prefix to narrow listing (e.g. "year/month/").
            max_workers: Max concurrent downloads/reads.
            hour_pattern: Pattern for hour-level subpath; tokens: {Y} {m} {d} {H}.
        """
        try:
            from azure.storage.blob import ContainerClient  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "azure-storage-blob is required for AzureBlobParquetLoader. "
                "Install with `pip install azure-storage-blob`."
            ) from exc

        self._ContainerClient = ContainerClient

        # Auto-detect: if connection_string looks like a URL, treat it as sas_url
        if connection_string and connection_string.strip().startswith("http"):
            sas_url = connection_string
            connection_string = None

        if sas_url:
            # SAS URL: https://account.blob.core.windows.net/container?sig=...
            self.container_client = ContainerClient.from_container_url(sas_url)
        elif account_url or (credential is not None and not connection_string):
            if not account_url:
                raise ValueError(
                    "account_url must be provided when using AAD credential auth"
                )
            if credential is None:
                raise ValueError(
                    "credential must be provided when using AAD credential auth"
                )
            if not container_name:
                raise ValueError(
                    "container_name is required when using account_url + credential"
                )
            self.container_client = ContainerClient(
                account_url=account_url,
                container_name=container_name,
                credential=credential,
            )
        else:
            if not connection_string:
                raise ValueError(
                    "Provide one of: sas_url, connection_string, or (account_url + credential)"
                )
            if not container_name:
                raise ValueError(
                    "container_name is required when using connection_string"
                )
            try:
                self.container_client = ContainerClient.from_connection_string(
                    conn_str=connection_string, container_name=container_name
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "Invalid connection string. Ensure it contains "
                    "'AccountName=...' and 'AccountKey=...' (or use "
                    "sas_url for SAS token auth, or account_url + credential for AAD). "
                    "Find the full string in Azure Portal → Storage Account → Access keys."
                ) from exc
        self.prefix = prefix
        self.max_workers = max_workers if max_workers > 0 else 1
        # Pattern for hour-level subpath; tokens: {Y} {m} {d} {H}
        # Default matches many data lake layouts: YYYY/MM/DD/HH/
        self.hour_pattern = hour_pattern

    @classmethod
    def from_account_name(
        cls,
        account_name: str,
        container_name: str,
        *,
        credential: Any | None = None,
        endpoint_suffix: str = "blob.core.windows.net",
        prefix: str = "",
        max_workers: int = 8,
    ) -> "AzureBlobParquetLoader":
        """
        Construct a loader using AAD credentials with an account name.

        Args:
            account_name: Storage account name.
            container_name: Target container.
            credential: Optional Azure credential (DefaultAzureCredential if None).
            endpoint_suffix: DNS suffix for the blob endpoint (e.g., for sovereign clouds).
            prefix: Optional listing prefix (e.g., "parquet/").
            max_workers: Concurrency for downloads.
        """
        account_url = f"https://{account_name}.{endpoint_suffix}"
        if credential is None:
            raise ValueError(
                "credential must be provided when using AAD credential auth"
            )
        return cls(
            container_name=container_name,
            account_url=account_url,
            credential=credential,
            prefix=prefix,
            max_workers=max_workers,
        )

    def _iter_matching_blob_names(self, uuids: set[str]) -> Iterable[str]:
        """
        Iterate over blob names that end with .parquet and contain any of the given UUIDs.

        Uses server-side prefix filtering when `self.prefix` is provided to reduce listing.
        """
        # Stream listing to handle large containers efficiently
        blob_iter = self.container_client.list_blobs(
            name_starts_with=self.prefix or None
        )
        for blob in blob_iter:  # type: ignore[attr-defined]
            name: str = blob.name  # type: ignore[attr-defined]
            if not name.endswith(".parquet"):
                continue
            # Fast path: check containment against UUID set
            # Assumes filenames or paths contain the UUID as substring
            if any(u in name for u in uuids):
                yield name

    def _download_parquet(
        self,
        blob_name: str,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame | None:
        """
        Download a parquet blob and return a DataFrame. Returns None if not found.

        ``columns`` and ``filters`` are pushed into ``pd.read_parquet`` so only
        the requested columns / rows are parsed. Both default to ``None``,
        which reads the full blob.
        """
        try:
            downloader = self.container_client.download_blob(blob_name)
            data = downloader.readall()
            return pd.read_parquet(BytesIO(data), columns=columns, filters=filters)
        except Exception as exc:
            logger.debug("Failed to download blob '%s': %s", blob_name, exc)
            return None

    def _download_many(
        self,
        blob_names: list[str],
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> tuple[list[pd.DataFrame], int, int]:
        """
        Download a set of parquet blobs concurrently.

        ``columns``/``filters`` are forwarded to :meth:`_download_parquet` for
        column projection and row predicate pushdown. If ``columns`` names a
        field absent from a blob the parquet engine raises, so that blob is
        counted in ``n_failed``.

        Returns:
            A tuple ``(frames, n_failed, n_empty)`` where ``frames`` are the
            non-empty DataFrames, ``n_failed`` counts blobs that could not be
            downloaded or parsed (missing, unauthorized, or corrupt), and
            ``n_empty`` counts blobs that parsed successfully but had no rows.
        """
        frames: list[pd.DataFrame] = []
        n_failed = 0
        n_empty = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(self._download_parquet, name, columns, filters): name
                for name in blob_names
            }
            for future in as_completed(future_to_name):
                df = future.result()
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
        """
        Emit a ``LoaderConfigWarning`` explaining why a load returned no data.
        """
        if filters is not None:
            detail = f"{detail} The filters argument may be too strict."
        warnings.warn(
            f"AzureBlobParquetLoader.{method} returned an empty DataFrame. {detail}",
            LoaderConfigWarning,
            stacklevel=3,
        )

    def _time_search_detail(self, slots: list[pd.Timestamp]) -> str:
        """Describe the prefix/pattern/hour-range probed by a time-range load."""
        prefix_repr = repr(self.prefix or "")
        pattern_repr = repr(self.hour_pattern)
        if not slots:
            return (
                f"prefix={prefix_repr}, hour_pattern={pattern_repr}, "
                "0 hour-prefixes (the time range is empty)"
            )
        first = slots[0].strftime("%Y-%m-%d %H:00")
        last = slots[-1].strftime("%Y-%m-%d %H:00")
        return (
            f"prefix={prefix_repr}, hour_pattern={pattern_repr}, "
            f"{len(slots)} hour-prefix(es) {first} .. {last}"
        )

    # ---- Helpers for time-structured containers parquet/YYYY/MM/DD/HH ----
    @staticmethod
    def _hourly_slots(
        start_timestamp: str | pd.Timestamp, end_timestamp: str | pd.Timestamp
    ) -> Iterable[pd.Timestamp]:
        start = pd.to_datetime(start_timestamp)
        end = pd.to_datetime(end_timestamp)
        # Ensure inclusive range per hour
        return pd.date_range(start=start, end=end, freq="h")

    def _hour_prefix(self, ts: pd.Timestamp) -> str:
        # Builds e.g. "parquet/2024/01/31/09/" if prefix="parquet/"
        y = str(ts.year)
        m = str(ts.month).zfill(2)
        d = str(ts.day).zfill(2)
        h = str(ts.hour).zfill(2)
        base = self.prefix or ""
        if base and not base.endswith("/"):
            base += "/"
        sub = (
            self.hour_pattern.replace("{Y}", y)
            .replace("{m}", m)
            .replace("{d}", d)
            .replace("{H}", h)
        )
        return f"{base}{sub}"

    def load_all_files(
        self,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        """
        Load all parquet blobs in the container (optionally under `prefix`).

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.

        Returns:
            A concatenated DataFrame of all parquet blobs. Returns an empty DataFrame
            if none are found.
        """
        # List all parquet blob names using optional prefix for server-side filtering
        blob_iter = self.container_client.list_blobs(
            name_starts_with=self.prefix or None
        )
        blob_names = [b.name for b in blob_iter if str(b.name).endswith(".parquet")]  # type: ignore[attr-defined]
        prefix_repr = repr(self.prefix or "")
        if not blob_names:
            self._warn_empty(
                "load_all_files",
                f"No .parquet blobs found under prefix={prefix_repr}. "
                "Check the container name and prefix; the container may be "
                "empty or the prefix may not match the stored layout.",
            )
            return pd.DataFrame()

        frames, n_failed, n_empty = self._download_many(blob_names, columns, filters)
        if not frames:
            self._warn_empty(
                "load_all_files",
                f"Listed {len(blob_names)} .parquet blob(s) under prefix="
                f"{prefix_repr} but none yielded rows ({n_failed} failed to "
                f"download/parse, {n_empty} had no rows). Check that the "
                "credentials have read permission and the files are valid parquet.",
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
        Load all parquet blobs under hourly folders within [start, end].

        Assumes container structure: prefix/year/month/day/hour/{file}.parquet
        Listing is constrained per-hour for speed.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]
        blob_names: list[str] = []
        for pfx in hour_prefixes:
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            blob_names.extend([b.name for b in blob_iter if str(b.name).endswith(".parquet")])  # type: ignore[attr-defined]

        search = self._time_search_detail(slots)
        if not blob_names:
            self._warn_empty(
                "load_by_time_range",
                f"No .parquet blobs found. {search}. Check that hour_pattern "
                "matches the real folder layout and that the time range "
                "overlaps the stored data.",
            )
            return pd.DataFrame()

        frames, n_failed, n_empty = self._download_many(blob_names, columns, filters)
        if not frames:
            self._warn_empty(
                "load_by_time_range",
                f"Listed {len(blob_names)} .parquet blob(s) but none yielded "
                f"rows ({n_failed} failed to download/parse, {n_empty} had no "
                f"rows). {search}. Check read permission and file validity.",
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
        Stream parquet DataFrames under hourly folders within [start, end].

        Yields (blob_name, DataFrame) one by one to avoid holding everything in memory.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]

        def _names_iter() -> Iterator[str]:
            for pfx in hour_prefixes:
                blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
                for b in blob_iter:  # type: ignore[attr-defined]
                    name = str(b.name)
                    if name.endswith(".parquet"):
                        yield name

        names_iter = _names_iter()
        yielded = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[Any, str] = {}
            # initial fill
            try:
                while len(futures) < self.max_workers:
                    n = next(names_iter)
                    futures[
                        executor.submit(self._download_parquet, n, columns, filters)
                    ] = n
            except StopIteration:
                pass

            while futures:
                # Drain current batch
                for fut in as_completed(list(futures.keys())):
                    name = futures.pop(fut)
                    try:
                        df = fut.result()
                    except Exception:
                        df = None
                    if df is not None and not df.empty:
                        yielded += 1
                        yield (name, df)

                # Refill
                try:
                    while len(futures) < self.max_workers:
                        n = next(names_iter)
                        futures[
                            executor.submit(self._download_parquet, n, columns, filters)
                        ] = n
                except StopIteration:
                    pass

        if yielded == 0:
            self._warn_empty(
                "stream_by_time_range",
                f"Yielded no DataFrames. {self._time_search_detail(slots)}. "
                "Check that hour_pattern matches the real folder layout and "
                "that the time range overlaps the stored data.",
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
        Load parquet blobs for given UUIDs within [start, end] hours.

        Strategy:
        1) List each hour prefix once and keep blobs whose basename matches a
           requested UUID variant. When listing succeeds it is authoritative,
           so only blobs that actually exist are downloaded.
        2) Only when listing fails entirely (e.g. a SAS token without list
           permission) fall back to constructing direct blob paths assuming
           the pattern prefix/YYYY/MM/DD/HH/{uuid}.parquet and downloading
           those blindly.

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        if not uuid_list:
            return pd.DataFrame()

        # Sanitize and deduplicate UUIDs while preserving order
        def _clean_uuid(u: object) -> str:
            s = str(u).strip().strip("{}").strip()
            return s

        raw = [_clean_uuid(u) for u in uuid_list]
        # Include lowercase variants to be tolerant of case differences in filenames
        variants_ordered: list[str] = []
        seen: set[str] = set()
        for u in raw:
            for v in (u, u.lower()):
                if v and v not in seen:
                    seen.add(v)
                    variants_ordered.append(v)

        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]

        # 1) Authoritative path: list each hour prefix, match by basename.
        basenames = {f"{u}.parquet" for u in variants_ordered}
        listed_names: list[str] = []
        listing_failed = False
        try:
            for pfx in hour_prefixes:
                blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
                for b in blob_iter:  # type: ignore[attr-defined]
                    name = str(b.name)
                    if not name.endswith(".parquet"):
                        continue
                    base = name.rsplit("/", 1)[-1]
                    if base in basenames:
                        listed_names.append(name)
        except Exception:
            # Listing not permitted/available: fall back to blind direct names.
            listing_failed = True

        if listing_failed:
            # 2) Fallback only: construct direct blob names and download blindly.
            all_blob_names = list(
                dict.fromkeys(
                    f"{pfx}{u}.parquet"
                    for pfx in hour_prefixes
                    for u in variants_ordered
                )
            )
            fallback_note = (
                " Listing was not available, so direct blob names were "
                "downloaded blindly; verify the credentials' list permission."
            )
        else:
            all_blob_names = list(dict.fromkeys(listed_names))
            fallback_note = ""

        search = self._time_search_detail(slots)
        uuid_note = f"{len(variants_ordered)} UUID variant(s) searched"
        if not all_blob_names:
            self._warn_empty(
                "load_files_by_time_range_and_uuids",
                f"No matching .parquet blobs found. {search}, {uuid_note}. "
                "Check that the UUIDs, hour_pattern and time range match the "
                f"stored data.{fallback_note}",
            )
            return pd.DataFrame()

        frames, n_failed, n_empty = self._download_many(
            all_blob_names, columns, filters
        )
        if not frames:
            self._warn_empty(
                "load_files_by_time_range_and_uuids",
                f"Resolved {len(all_blob_names)} blob name(s) but none yielded "
                f"rows ({n_failed} failed to download/parse, {n_empty} had no "
                f"rows). {search}, {uuid_note}. Check read permission and file "
                f"validity.{fallback_note}",
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
        Stream parquet DataFrames for given UUIDs within [start, end] hours.

        Yields (blob_name, DataFrame) as they arrive. Listing is authoritative
        when available; direct blob names are used only as a fallback when
        listing fails (see :meth:`load_files_by_time_range_and_uuids`).

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
        """
        if not uuid_list:
            return

        def _clean_uuid(u: object) -> str:
            return str(u).strip().strip("{}").strip()

        raw = [_clean_uuid(u) for u in uuid_list]
        variants_ordered: list[str] = []
        seen: set[str] = set()
        for u in raw:
            for v in (u, u.lower()):
                if v and v not in seen:
                    seen.add(v)
                    variants_ordered.append(v)

        slots = list(self._hourly_slots(start_timestamp, end_timestamp))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]
        basenames = {f"{u}.parquet" for u in variants_ordered}

        listed_names: list[str] = []
        listing_failed = False
        try:
            for pfx in hour_prefixes:
                blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
                for b in blob_iter:  # type: ignore[attr-defined]
                    name = str(b.name)
                    if not name.endswith(".parquet"):
                        continue
                    base = name.rsplit("/", 1)[-1]
                    if base in basenames:
                        listed_names.append(name)
        except Exception:
            listing_failed = True

        if listing_failed:
            all_blob_names = list(
                dict.fromkeys(
                    f"{pfx}{u}.parquet"
                    for pfx in hour_prefixes
                    for u in variants_ordered
                )
            )
            fallback_note = (
                " Listing was not available, so direct blob names were "
                "downloaded blindly; verify the credentials' list permission."
            )
        else:
            all_blob_names = list(dict.fromkeys(listed_names))
            fallback_note = ""

        names_iter = iter(all_blob_names)
        yielded = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[Any, str] = {}
            try:
                while len(futures) < self.max_workers:
                    n = next(names_iter)
                    futures[
                        executor.submit(self._download_parquet, n, columns, filters)
                    ] = n
            except StopIteration:
                pass

            while futures:
                for fut in as_completed(list(futures.keys())):
                    name = futures.pop(fut)
                    try:
                        df = fut.result()
                    except Exception:
                        df = None
                    if df is not None and not df.empty:
                        yielded += 1
                        yield (name, df)

                try:
                    while len(futures) < self.max_workers:
                        n = next(names_iter)
                        futures[
                            executor.submit(self._download_parquet, n, columns, filters)
                        ] = n
                except StopIteration:
                    pass

        if yielded == 0:
            self._warn_empty(
                "stream_files_by_time_range_and_uuids",
                f"Yielded no DataFrames. {self._time_search_detail(slots)}, "
                f"{len(variants_ordered)} UUID variant(s) searched. Check that "
                "the UUIDs, hour_pattern and time range match the stored "
                f"data.{fallback_note}",
                filters=filters,
            )

    def list_structure(
        self, parquet_only: bool = True, limit: int | None = None
    ) -> dict[str, list[str]]:
        """
        List folder prefixes (hours) and blob names under the configured `prefix`.

        Args:
            parquet_only: If True, only include blobs ending with .parquet.
            limit: Optional cap on number of files collected for quick inspection.

        Returns:
            A dict with:
            - folders: Sorted unique hour-level prefixes like 'parquet/YYYY/MM/DD/HH/'
            - files: Sorted blob names (full paths) matching the filter
        """
        folders: set[str] = set()
        files: list[str] = []
        collected = 0

        blob_iter = self.container_client.list_blobs(
            name_starts_with=self.prefix or None
        )
        for b in blob_iter:
            name = str(b.name)
            if parquet_only and not name.endswith(".parquet"):
                continue
            files.append(name)
            # Derive hour-level folder prefix
            if "/" in name:
                folders.add(name.rsplit("/", 1)[0].rstrip("/") + "/")
            collected += 1
            if limit is not None and collected >= limit:
                break

        return {
            "folders": sorted(folders),
            "files": sorted(files),
        }


class AzureBlobFlexibleFileLoader:
    """
    Load arbitrary file types from Azure Blob Storage under time-structured folders.

    Designed for containers with paths like: prefix/YYYY/MM/DD/HH/<unknown and non-static suffix>/file.ext
    This class lists by per-hour prefix and can filter by extensions and/or basenames,
    then downloads files concurrently as raw bytes.
    """

    # Parser registry: maps lowercase extensions (with dot) -> callable(content, name) -> Any
    _parsers: dict[str, Callable[[bytes, str], Any]] = {}
    _parsers_initialized: bool = False

    def __init__(
        self,
        container_name: str | None = None,
        *,
        connection_string: str | None = None,
        account_url: str | None = None,
        credential: Any | None = None,
        sas_url: str | None = None,
        prefix: str = "",
        max_workers: int = 8,
        hour_pattern: str = "{Y}/{m}/{d}/{H}/",
    ) -> None:
        """
        Initialize the loader with Azure connection details.

        Supports SAS URL, connection string, and AAD credential auth.
        See :class:`AzureBlobParquetLoader` for full parameter docs.

        A SAS URL is auto-detected when passed as ``connection_string``.
        """
        try:
            from azure.storage.blob import ContainerClient  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "azure-storage-blob is required for AzureBlobFlexibleFileLoader. "
                "Install with `pip install azure-storage-blob`."
            ) from exc

        # Auto-detect: if connection_string looks like a URL, treat it as sas_url
        if connection_string and connection_string.strip().startswith("http"):
            sas_url = connection_string
            connection_string = None

        if sas_url:
            self.container_client = ContainerClient.from_container_url(sas_url)
        elif account_url or (credential is not None and not connection_string):
            if not account_url:
                raise ValueError(
                    "account_url must be provided when using AAD credential auth"
                )
            if credential is None:
                raise ValueError(
                    "credential must be provided when using AAD credential auth"
                )
            if not container_name:
                raise ValueError(
                    "container_name is required when using account_url + credential"
                )
            self.container_client = ContainerClient(
                account_url=account_url,
                container_name=container_name,
                credential=credential,
            )
        else:
            if not connection_string:
                raise ValueError(
                    "Provide one of: sas_url, connection_string, or (account_url + credential)"
                )
            if not container_name:
                raise ValueError(
                    "container_name is required when using connection_string"
                )
            try:
                self.container_client = ContainerClient.from_connection_string(
                    conn_str=connection_string, container_name=container_name
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "Invalid connection string. Ensure it contains "
                    "'AccountName=...' and 'AccountKey=...' (or use "
                    "sas_url for SAS token auth, or account_url + credential for AAD). "
                    "Find the full string in Azure Portal → Storage Account → Access keys."
                ) from exc
        self.prefix = prefix
        self.max_workers = max_workers if max_workers > 0 else 1
        # Pattern for hour-level subpath; tokens: {Y} {m} {d} {H}
        self.hour_pattern = hour_pattern
        # Initialize parsers lazily once per process
        if not AzureBlobFlexibleFileLoader._parsers_initialized:
            self._enable_builtin_parsers()
            AzureBlobFlexibleFileLoader._parsers_initialized = True

    # ---- Shared helpers with Parquet loader ----
    @staticmethod
    def _hourly_slots(
        start_timestamp: str | pd.Timestamp, end_timestamp: str | pd.Timestamp
    ) -> Iterable[pd.Timestamp]:
        start = pd.to_datetime(start_timestamp)
        end = pd.to_datetime(end_timestamp)
        return pd.date_range(start=start, end=end, freq="h")

    def _hour_prefix(self, ts: pd.Timestamp) -> str:
        y = str(ts.year)
        m = str(ts.month).zfill(2)
        d = str(ts.day).zfill(2)
        h = str(ts.hour).zfill(2)
        base = self.prefix or ""
        if base and not base.endswith("/"):
            base += "/"
        sub = (
            self.hour_pattern.replace("{Y}", y)
            .replace("{m}", m)
            .replace("{d}", d)
            .replace("{H}", h)
        )
        return f"{base}{sub}"

    # ---- Core operations ----
    def _download_bytes(self, blob_name: str) -> bytes | None:
        try:
            downloader = self.container_client.download_blob(blob_name)
            return downloader.readall()
        except Exception:
            return None

    @staticmethod
    def _normalize_exts(exts: Iterable[str] | None) -> set[str] | None:
        if exts is None:
            return None
        norm: set[str] = set()
        for e in exts:
            s = str(e).strip().lower()
            if not s:
                continue
            if not s.startswith("."):
                s = "." + s
            norm.add(s)
        return norm or None

    # ---- Parser registry ----
    @classmethod
    def register_parser(cls, extension: str, func: Callable[[bytes, str], Any]) -> None:
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        cls._parsers[ext] = func

    @classmethod
    def unregister_parser(cls, extension: str) -> None:
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        cls._parsers.pop(ext, None)

    @classmethod
    def available_parsers(cls) -> set[str]:
        return set(cls._parsers.keys())

    @classmethod
    def _enable_builtin_parsers(cls) -> None:
        # Always register JSON (stdlib)
        import json

        def parse_json(content: bytes, name: str) -> Any:
            return json.loads(content.decode("utf-8"))

        cls._parsers.setdefault(".json", parse_json)

        # Parquet via pandas (already imported at module level)
        def parse_parquet(content: bytes, name: str) -> Any:
            from io import BytesIO as _BytesIO

            return pd.read_parquet(_BytesIO(content))

        cls._parsers.setdefault(".parquet", parse_parquet)

        # Optional: NumPy npy/npz
        try:
            import numpy as _np  # type: ignore

            def parse_npy(content: bytes, name: str) -> Any:
                from io import BytesIO as _BytesIO

                return _np.load(_BytesIO(content), allow_pickle=False)

            def parse_npz(content: bytes, name: str) -> Any:
                from io import BytesIO as _BytesIO

                return _np.load(_BytesIO(content), allow_pickle=False)

            cls._parsers.setdefault(".npy", parse_npy)
            cls._parsers.setdefault(".npz", parse_npz)
        except Exception:
            pass

        # Optional: images via Pillow
        try:
            from PIL import Image as _Image  # type: ignore

            def parse_image(content: bytes, name: str) -> Any:
                from io import BytesIO as _BytesIO

                return _Image.open(_BytesIO(content))

            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"):
                cls._parsers.setdefault(ext, parse_image)
        except Exception:
            pass

    @classmethod
    def _parse_bytes(cls, blob_name: str, content: bytes) -> Any:
        ext = "." + blob_name.lower().rsplit(".", 1)[-1] if "." in blob_name else ""
        parser = cls._parsers.get(ext)
        if parser is None:
            return content
        try:
            return parser(content, blob_name)
        except Exception:
            # Fall back to raw bytes on parse errors
            return content

    def list_files_by_time_range(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """
        List blob names under each hourly prefix within [start, end].

        Args:
            extensions: Optional set/list of file extensions (e.g., {"json", ".bmp"}). Case-insensitive.
            limit: Optional cap on number of files collected.
        """
        allowed_exts = self._normalize_exts(extensions)
        names: list[str] = []
        collected = 0
        for pfx in (
            self._hour_prefix(ts)
            for ts in self._hourly_slots(start_timestamp, end_timestamp)
        ):
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            for b in blob_iter:  # type: ignore[attr-defined]
                name = str(b.name)
                if allowed_exts is not None:
                    lower_name = name.lower()
                    if not any(lower_name.endswith(ext) for ext in allowed_exts):
                        continue
                names.append(name)
                collected += 1
                if limit is not None and collected >= limit:
                    return names
        return names

    def iter_file_names_by_time_range(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
    ) -> Iterator[str]:
        """
        Yield blob names under each hourly prefix within [start, end].
        Uses server-side prefix listing and client-side extension filtering.
        """
        allowed_exts = self._normalize_exts(extensions)
        for pfx in (
            self._hour_prefix(ts)
            for ts in self._hourly_slots(start_timestamp, end_timestamp)
        ):
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            for b in blob_iter:  # type: ignore[attr-defined]
                name = str(b.name)
                if allowed_exts is not None:
                    if not any(name.lower().endswith(ext) for ext in allowed_exts):
                        continue
                yield name

    def fetch_files_by_time_range(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
    ) -> dict[str, Any]:
        """
        Download files that match extensions within [start, end] hour prefixes.
        Returns a dict mapping blob_name -> parsed object (if parse=True and a parser exists),
        otherwise raw bytes.
        """
        blob_names = self.list_files_by_time_range(
            start_timestamp, end_timestamp, extensions=extensions
        )
        if not blob_names:
            return {}
        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(self._download_bytes, n): n for n in blob_names
            }
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                content = fut.result()
                if content is not None:
                    results[name] = (
                        self._parse_bytes(name, content) if parse else content
                    )
        return results

    def stream_files_by_time_range(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
    ) -> Iterator[tuple[str, Any]]:
        """
        Stream matching files as (blob_name, bytes-or-parsed) within [start, end].
        Maintains up to `max_workers` concurrent downloads while yielding incrementally.
        """
        names_iter = self.iter_file_names_by_time_range(
            start_timestamp, end_timestamp, extensions=extensions
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name: dict[Any, str] = {}
            # initial fill
            try:
                while len(future_to_name) < self.max_workers:
                    n = next(names_iter)
                    future_to_name[executor.submit(self._download_bytes, n)] = n
            except StopIteration:
                pass

            while future_to_name:
                # Drain
                for fut in as_completed(list(future_to_name.keys())):
                    name = future_to_name.pop(fut)
                    try:
                        content = fut.result()
                    except Exception:
                        content = None
                    if content is not None:
                        yield (
                            name,
                            self._parse_bytes(name, content) if parse else content,
                        )

                # Refill
                try:
                    while len(future_to_name) < self.max_workers:
                        n = next(names_iter)
                        future_to_name[executor.submit(self._download_bytes, n)] = n
                except StopIteration:
                    pass

    def fetch_files_by_time_range_and_basenames(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        basenames: Iterable[str],
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
    ) -> dict[str, Any]:
        """
        Download files whose basename (final path segment) is in `basenames`,
        optionally filtered by extensions, within [start, end] hour prefixes.
        Returns blob_name -> parsed object (if parse=True and a parser exists), otherwise raw bytes.
        """
        base_set = {str(b).strip() for b in basenames if str(b).strip()}
        allowed_exts = self._normalize_exts(extensions)
        candidates: list[str] = []
        for pfx in (
            self._hour_prefix(ts)
            for ts in self._hourly_slots(start_timestamp, end_timestamp)
        ):
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            for b in blob_iter:  # type: ignore[attr-defined]
                name = str(b.name)
                base = name.rsplit("/", 1)[-1]
                if base not in base_set:
                    continue
                if allowed_exts is not None and not any(
                    name.lower().endswith(ext) for ext in allowed_exts
                ):
                    continue
                candidates.append(name)

        if not candidates:
            return {}

        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(self._download_bytes, n): n for n in candidates
            }
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                content = fut.result()
                if content is not None:
                    results[name] = (
                        self._parse_bytes(name, content) if parse else content
                    )
        return results

    def stream_files_by_time_range_and_basenames(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        basenames: Iterable[str],
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
    ) -> Iterator[tuple[str, Any]]:
        """
        Stream files whose basename is in `basenames` within [start, end].
        Yields (blob_name, bytes-or-parsed) incrementally with bounded concurrency.
        """
        base_set = {str(b).strip() for b in basenames if str(b).strip()}
        allowed_exts = self._normalize_exts(extensions)

        def _names_iter() -> Iterator[str]:
            for pfx in (
                self._hour_prefix(ts)
                for ts in self._hourly_slots(start_timestamp, end_timestamp)
            ):
                blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
                for b in blob_iter:  # type: ignore[attr-defined]
                    name = str(b.name)
                    base = name.rsplit("/", 1)[-1]
                    if base not in base_set:
                        continue
                    if allowed_exts is not None and not any(
                        name.lower().endswith(ext) for ext in allowed_exts
                    ):
                        continue
                    yield name

        names_iter = _names_iter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name: dict[Any, str] = {}
            # initial fill
            try:
                while len(future_to_name) < self.max_workers:
                    n = next(names_iter)
                    future_to_name[executor.submit(self._download_bytes, n)] = n
            except StopIteration:
                pass

            while future_to_name:
                for fut in as_completed(list(future_to_name.keys())):
                    name = future_to_name.pop(fut)
                    try:
                        content = fut.result()
                    except Exception:
                        content = None
                    if content is not None:
                        yield (
                            name,
                            self._parse_bytes(name, content) if parse else content,
                        )

                try:
                    while len(future_to_name) < self.max_workers:
                        n = next(names_iter)
                        future_to_name[executor.submit(self._download_bytes, n)] = n
                except StopIteration:
                    pass
