import logging
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Self, TypeVar

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning, LoaderConfigWarning, LoaderError
from ts_shape.loader._utils import (
    ErrorsMode,
    check_errors_mode,
    describe_failures,
    is_not_found,
    missing_optional,
    read_errors,
    report_failures,
    trim_to_time_range,
    utc_folder_hours,
)
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _download_all(
    fetch: Callable[[str], T],
    names: list[str],
    max_workers: int,
    *,
    missing_ok: bool = False,
) -> tuple[list[tuple[str, T]], list[tuple[str, Exception]]]:
    """Run ``fetch`` on every name concurrently.

    Returns ``(results, failures)``: ``(name, value)`` for every successful
    read and ``(name, error)`` for every read that raised one of
    :func:`~ts_shape.loader._utils.read_errors`, both sorted by name. With
    ``missing_ok`` an object that does not exist is left out of both.
    """
    results: list[tuple[str, T]] = []
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(fetch, name): name for name in names}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results.append((name, future.result()))
            except read_errors() as exc:
                if not (missing_ok and is_not_found(exc)):
                    failures.append((name, exc))
    results.sort(key=lambda item: item[0])
    failures.sort(key=lambda item: item[0])
    return results, failures


def _stream_downloads(
    fetch: Callable[[str], T],
    names: Iterator[str],
    max_workers: int,
    *,
    failures: list[tuple[str, Exception]],
    errors: ErrorsMode,
    what: str,
    missing_ok: bool = False,
) -> Iterator[tuple[str, T]]:
    """Run ``fetch`` over ``names`` with up to ``max_workers`` reads in flight.

    Yields ``(name, value)`` as reads complete. A read that raises one of
    :func:`~ts_shape.loader._utils.read_errors` is appended to ``failures``
    (``errors="warn"``) or raises :class:`~ts_shape.errors.LoaderError` at
    once (``errors="raise"``); with ``missing_ok`` an object that does not
    exist is skipped.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[T], str] = {}

        def _fill() -> None:
            while len(futures) < max_workers:
                name = next(names, None)
                if name is None:
                    return
                futures[executor.submit(fetch, name)] = name

        _fill()
        while futures:
            for fut in as_completed(list(futures)):
                name = futures.pop(fut)
                try:
                    value = fut.result()
                except read_errors() as exc:
                    if missing_ok and is_not_found(exc):
                        continue
                    if errors == "raise":
                        raise LoaderError(
                            describe_failures([(name, exc)], what)
                        ) from exc
                    failures.append((name, exc))
                    continue
                yield (name, value)
            _fill()


class AzureBlobParquetLoader:
    """
    Load parquet files from an Azure Blob Storage container filtered by a list of UUIDs.

    Optimized for speed by:
    - Using server-side prefix filtering when provided
    - Streaming blob listings and filtering client-side by UUID containment
    - Downloading and parsing parquet files concurrently

    Hour folders are UTC: a tz-aware time bound is converted to UTC before
    the folders to visit are chosen, and a naive bound is read as UTC.
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
            from azure.storage.blob import ContainerClient
        except ImportError as exc:  # pragma: no cover - import guard
            raise missing_optional("azure-storage-blob", "azure") from exc

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
    ) -> Self:
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

    def _download_parquet(
        self,
        blob_name: str,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        """
        Download a parquet blob and return a DataFrame.

        ``columns`` and ``filters`` are pushed into ``pd.read_parquet`` so only
        the requested columns / rows are parsed. Both default to ``None``,
        which reads the full blob.

        Raises:
            One of :func:`~ts_shape.loader._utils.read_errors` when the blob is
            missing, not readable with the credentials, or not valid parquet.
        """
        downloader = self.container_client.download_blob(blob_name)
        data = downloader.readall()
        return pd.read_parquet(BytesIO(data), columns=columns, filters=filters)

    def _download_many(
        self,
        blob_names: list[str],
        columns: list[str] | None = None,
        filters: list | None = None,
        *,
        missing_ok: bool = False,
    ) -> tuple[list[pd.DataFrame], list[tuple[str, Exception]], int]:
        """
        Download a set of parquet blobs concurrently.

        ``columns``/``filters`` are forwarded to :meth:`_download_parquet` for
        column projection and row predicate pushdown. If ``columns`` names a
        field absent from a blob the parquet engine raises, so that blob is
        reported as a failure.

        Args:
            blob_names: Blobs to download.
            columns: Column projection.
            filters: Row predicate pushdown.
            missing_ok: Treat a blob that does not exist as absent rather than
                as a failure (used when blob names are guessed).

        Returns:
            A tuple ``(frames, failures, n_empty)`` where ``frames`` are the
            non-empty DataFrames, ``failures`` lists ``(blob name, error)`` for
            every blob that could not be downloaded or parsed (missing,
            unauthorized, or corrupt), and ``n_empty`` counts blobs that parsed
            successfully but had no rows.
        """
        results, failures = _download_all(
            lambda name: self._download_parquet(name, columns, filters),
            blob_names,
            self.max_workers,
            missing_ok=missing_ok,
        )
        frames = [df for _, df in results if not df.empty]
        n_empty = len(results) - len(frames)
        return frames, failures, n_empty

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
            stacklevel=caller_stacklevel(),
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
        *,
        errors: ErrorsMode = "warn",
    ) -> pd.DataFrame:
        """
        Load all parquet blobs in the container (optionally under `prefix`).

        Args:
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
            errors: ``"warn"`` (default) skips a blob that cannot be downloaded
                or parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.

        Returns:
            A concatenated DataFrame of all parquet blobs. Returns an empty DataFrame
            if none are found.
        """
        check_errors_mode(errors)
        # List all parquet blob names using optional prefix for server-side filtering
        blob_iter = self.container_client.list_blobs(
            name_starts_with=self.prefix or None
        )
        blob_names = [b.name for b in blob_iter if str(b.name).endswith(".parquet")]
        prefix_repr = repr(self.prefix or "")
        if not blob_names:
            self._warn_empty(
                "load_all_files",
                f"No .parquet blobs found under prefix={prefix_repr}. "
                "Check the container name and prefix; the container may be "
                "empty or the prefix may not match the stored layout.",
            )
            return pd.DataFrame()

        frames, failures, n_empty = self._download_many(blob_names, columns, filters)
        report_failures(failures, what="parquet blob(s)", errors=errors)
        if not frames:
            self._warn_empty(
                "load_all_files",
                f"Listed {len(blob_names)} .parquet blob(s) under prefix="
                f"{prefix_repr} but none yielded rows ({len(failures)} failed to "
                f"download/parse, {n_empty} had no rows). Check that the "
                "credentials have read permission and the files are valid parquet.",
                filters=filters,
            )
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def load_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        columns: list[str] | None = None,
        filters: list | None = None,
        *,
        trim_to_range: bool = True,
        errors: ErrorsMode = "warn",
    ) -> pd.DataFrame:
        """
        Load all parquet blobs under hourly folders within [start, end].

        Assumes container structure: prefix/year/month/day/hour/{file}.parquet
        Listing is constrained per-hour for speed.

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts).
                Every hour folder that overlaps ``[start, end]`` is read in
                full (the start is floored to the hour); see ``trim_to_range``.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
            trim_to_range: If True (default), drop rows whose ``systime``
                falls outside ``[start, end]`` (both bounds inclusive). Frames
                without a ``systime`` column are left as read. False keeps
                whole hour folders.
            errors: ``"warn"`` (default) skips a blob that cannot be downloaded
                or parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        slots = list(utc_folder_hours(start, end))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]
        blob_names: list[str] = []
        for pfx in hour_prefixes:
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            blob_names.extend(
                [b.name for b in blob_iter if str(b.name).endswith(".parquet")]
            )

        search = self._time_search_detail(slots)
        if not blob_names:
            self._warn_empty(
                "load_by_time_range",
                f"No .parquet blobs found. {search}. Check that hour_pattern "
                "matches the real folder layout and that the time range "
                "overlaps the stored data.",
            )
            return pd.DataFrame()

        frames, failures, n_empty = self._download_many(blob_names, columns, filters)
        report_failures(failures, what="parquet blob(s)", errors=errors)
        if not frames:
            self._warn_empty(
                "load_by_time_range",
                f"Listed {len(blob_names)} .parquet blob(s) but none yielded "
                f"rows ({len(failures)} failed to download/parse, {n_empty} had no "
                f"rows). {search}. Check read permission and file validity.",
                filters=filters,
            )
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if trim_to_range:
            out = trim_to_time_range(out, start, end)
        return out

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def stream_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        columns: list[str] | None = None,
        filters: list | None = None,
        *,
        trim_to_range: bool = True,
        errors: ErrorsMode = "warn",
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """
        Stream parquet DataFrames under hourly folders within [start, end].

        Yields (blob_name, DataFrame) one by one to avoid holding everything in memory.

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts).
                Every hour folder that overlaps ``[start, end]`` is read in
                full (the start is floored to the hour); see ``trim_to_range``.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
            trim_to_range: If True (default), drop rows whose ``systime``
                falls outside ``[start, end]`` (both bounds inclusive). Frames
                without a ``systime`` column are left as read. False keeps
                whole hour folders.
            errors: ``"warn"`` (default) skips a blob that cannot be downloaded
                or parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        slots = list(utc_folder_hours(start, end))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]

        def _names_iter() -> Iterator[str]:
            for pfx in hour_prefixes:
                blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
                for b in blob_iter:
                    name = str(b.name)
                    if name.endswith(".parquet"):
                        yield name

        failures: list[tuple[str, Exception]] = []
        yielded = 0
        for name, df in self._stream_parquet(
            _names_iter(),
            columns,
            filters,
            failures=failures,
            errors=errors,
        ):
            if trim_to_range:
                df = trim_to_time_range(df, start, end)
            if not df.empty:
                yielded += 1
                yield (name, df)
        report_failures(failures, what="parquet blob(s)", errors=errors)

        if yielded == 0:
            self._warn_empty(
                "stream_by_time_range",
                f"Yielded no DataFrames. {self._time_search_detail(slots)}. "
                "Check that hour_pattern matches the real folder layout and "
                "that the time range overlaps the stored data.",
                filters=filters,
            )

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def load_files_by_time_range_and_uuids(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        uuid_list: list[str],
        columns: list[str] | None = None,
        filters: list | None = None,
        *,
        trim_to_range: bool = True,
        errors: ErrorsMode = "warn",
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
            start: Start of the range (anything ``pd.to_datetime`` accepts).
                Every hour folder that overlaps ``[start, end]`` is read in
                full (the start is floored to the hour); see ``trim_to_range``.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            uuid_list: Signal UUIDs to load; each matches a ``<uuid>.parquet`` file.
                Surrounding whitespace and braces are stripped and a lowercase variant
                is also tried.
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
            trim_to_range: If True (default), drop rows whose ``systime``
                falls outside ``[start, end]`` (both bounds inclusive). Frames
                without a ``systime`` column are left as read. False keeps
                whole hour folders.
            errors: ``"warn"`` (default) skips a blob that cannot be downloaded
                or parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        if not uuid_list:
            return pd.DataFrame()

        variants_ordered = self._uuid_variants(uuid_list)
        slots = list(utc_folder_hours(start, end))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]
        all_blob_names, guessed, fallback_note = self._resolve_uuid_blobs(
            "load_files_by_time_range_and_uuids", hour_prefixes, variants_ordered
        )

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

        frames, failures, n_empty = self._download_many(
            all_blob_names, columns, filters, missing_ok=guessed
        )
        report_failures(failures, what="parquet blob(s)", errors=errors)
        if not frames:
            self._warn_empty(
                "load_files_by_time_range_and_uuids",
                f"Resolved {len(all_blob_names)} blob name(s) but none yielded "
                f"rows ({len(failures)} failed to download/parse, {n_empty} had "
                f"no rows). {search}, {uuid_note}. Check read permission and "
                f"file validity.{fallback_note}",
                filters=filters,
            )
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if trim_to_range:
            out = trim_to_time_range(out, start, end)
        return out

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def stream_files_by_time_range_and_uuids(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        uuid_list: list[str],
        columns: list[str] | None = None,
        filters: list | None = None,
        *,
        trim_to_range: bool = True,
        errors: ErrorsMode = "warn",
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """
        Stream parquet DataFrames for given UUIDs within [start, end] hours.

        Yields (blob_name, DataFrame) as they arrive. Listing is authoritative
        when available; direct blob names are used only as a fallback when
        listing fails (see :meth:`load_files_by_time_range_and_uuids`).

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts).
                Every hour folder that overlaps ``[start, end]`` is read in
                full (the start is floored to the hour); see ``trim_to_range``.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            uuid_list: Signal UUIDs to load; each matches a ``<uuid>.parquet`` file.
                Surrounding whitespace and braces are stripped and a lowercase variant
                is also tried.
            columns: Subset of parquet columns to read; None reads all columns.
            filters: pyarrow-style predicate pushdown in DNF form, e.g.
                ``[("value_double", ">", 0.0)]``; None reads all rows.
            trim_to_range: If True (default), drop rows whose ``systime``
                falls outside ``[start, end]`` (both bounds inclusive). Frames
                without a ``systime`` column are left as read. False keeps
                whole hour folders.
            errors: ``"warn"`` (default) skips a blob that cannot be downloaded
                or parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        if not uuid_list:
            return

        variants_ordered = self._uuid_variants(uuid_list)
        slots = list(utc_folder_hours(start, end))
        hour_prefixes = [self._hour_prefix(ts) for ts in slots]
        all_blob_names, guessed, fallback_note = self._resolve_uuid_blobs(
            "stream_files_by_time_range_and_uuids", hour_prefixes, variants_ordered
        )

        failures: list[tuple[str, Exception]] = []
        yielded = 0
        for name, df in self._stream_parquet(
            iter(all_blob_names),
            columns,
            filters,
            failures=failures,
            errors=errors,
            missing_ok=guessed,
        ):
            if trim_to_range:
                df = trim_to_time_range(df, start, end)
            if not df.empty:
                yielded += 1
                yield (name, df)
        report_failures(failures, what="parquet blob(s)", errors=errors)

        if yielded == 0:
            self._warn_empty(
                "stream_files_by_time_range_and_uuids",
                f"Yielded no DataFrames. {self._time_search_detail(slots)}, "
                f"{len(variants_ordered)} UUID variant(s) searched. Check that "
                "the UUIDs, hour_pattern and time range match the stored "
                f"data.{fallback_note}",
                filters=filters,
            )

    @staticmethod
    def _uuid_variants(uuid_list: list[str]) -> list[str]:
        """Cleaned UUIDs plus their lowercase variants, deduplicated in order.

        Surrounding whitespace and braces are stripped; the lowercase variant
        tolerates case differences in file names.
        """
        variants: list[str] = []
        seen: set[str] = set()
        for u in uuid_list:
            cleaned = str(u).strip().strip("{}").strip()
            for v in (cleaned, cleaned.lower()):
                if v and v not in seen:
                    seen.add(v)
                    variants.append(v)
        return variants

    def _resolve_uuid_blobs(
        self, method: str, hour_prefixes: list[str], variants: list[str]
    ) -> tuple[list[str], bool, str]:
        """Blob names for ``variants`` under ``hour_prefixes``.

        Listing each hour prefix is authoritative. Only when listing fails
        (e.g. a SAS token without list permission) are the names guessed as
        ``<hour prefix><uuid>.parquet``; a :class:`LoaderConfigWarning` says so.

        Returns:
            ``(names, guessed, note)``: the blob names, whether they were
            guessed (so a missing blob is expected), and a note for the
            empty-result warning.
        """
        basenames = {f"{u}.parquet" for u in variants}
        listed: list[str] = []
        try:
            for pfx in hour_prefixes:
                for b in self.container_client.list_blobs(name_starts_with=pfx):
                    name = str(b.name)
                    if (
                        name.endswith(".parquet")
                        and name.rsplit("/", 1)[-1] in basenames
                    ):
                        listed.append(name)
        except read_errors() as exc:
            warnings.warn(
                f"AzureBlobParquetLoader.{method}: listing the hour folders "
                f"failed ({type(exc).__name__}: {exc}); downloading the "
                "expected blob names <hour prefix>/<uuid>.parquet directly. "
                "Verify the credentials' list permission.",
                LoaderConfigWarning,
                stacklevel=caller_stacklevel(),
            )
            guessed = list(
                dict.fromkeys(
                    f"{pfx}{u}.parquet" for pfx in hour_prefixes for u in variants
                )
            )
            note = (
                " Listing was not available, so direct blob names were "
                "downloaded blindly; verify the credentials' list permission."
            )
            return guessed, True, note
        return list(dict.fromkeys(listed)), False, ""

    def _stream_parquet(
        self,
        names: Iterator[str],
        columns: list[str] | None,
        filters: list | None,
        *,
        failures: list[tuple[str, Exception]],
        errors: ErrorsMode,
        missing_ok: bool = False,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Download ``names`` with up to ``max_workers`` reads in flight.

        Yields ``(blob name, frame)`` as downloads complete. A blob that
        cannot be read is appended to ``failures`` (``errors="warn"``) or
        raises :class:`~ts_shape.errors.LoaderError` at once
        (``errors="raise"``); with ``missing_ok`` a blob that does not exist
        is skipped quietly.
        """
        return _stream_downloads(
            lambda name: self._download_parquet(name, columns, filters),
            names,
            self.max_workers,
            failures=failures,
            errors=errors,
            what="parquet blob(s)",
            missing_ok=missing_ok,
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

    Hour folders are UTC: a tz-aware time bound is converted to UTC before
    the folders to visit are chosen, and a naive bound is read as UTC.
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

        Args:
            container_name: Target container name (not needed when using ``sas_url``).
            connection_string: Azure Storage connection string. A value starting with
                ``http`` is treated as ``sas_url``.
            account_url: Full account URL for AAD auth (used with ``credential``).
            credential: Azure credential object for AAD auth.
            sas_url: Full Blob SAS URL including container and SAS token.
            prefix: Path prefix prepended to every hour folder (e.g. ``"raw/"``).
                Default ``""``.
            max_workers: Maximum concurrent downloads; values below 1 are treated as 1.
                Default 8.
            hour_pattern: Pattern for the hour-level subpath, with tokens ``{Y}``
                ``{m}`` ``{d}`` ``{H}``. Default ``"{Y}/{m}/{d}/{H}/"``.
        """
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:  # pragma: no cover - import guard
            raise missing_optional("azure-storage-blob", "azure") from exc

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
    def _download_bytes(self, blob_name: str) -> bytes:
        """Download one blob's content.

        Raises:
            One of :func:`~ts_shape.loader._utils.read_errors` when the blob is
            missing or cannot be read with the credentials.
        """
        downloader = self.container_client.download_blob(blob_name)
        return bytes(downloader.readall())

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
        """Register a parser for a file extension, replacing any existing one.

        Args:
            extension: File extension, with or without the leading dot
                (case-insensitive).
            func: Callable receiving the blob content and blob name and
                returning the parsed object.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        cls._parsers[ext] = func

    @classmethod
    def unregister_parser(cls, extension: str) -> None:
        """Remove the parser for a file extension, if one is registered.

        Args:
            extension: File extension, with or without the leading dot
                (case-insensitive).
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        cls._parsers.pop(ext, None)

    @classmethod
    def available_parsers(cls) -> set[str]:
        """Return the set of file extensions that have a registered parser."""
        return set(cls._parsers.keys())

    @classmethod
    def _enable_builtin_parsers(cls) -> None:
        # Always register JSON (stdlib)
        import json

        def parse_json(content: bytes, name: str) -> Any:
            return json.loads(content.decode("utf-8"))

        cls._parsers.setdefault(".json", parse_json)

        # Parquet via pandas
        def parse_parquet(content: bytes, name: str) -> Any:
            return pd.read_parquet(BytesIO(content))

        cls._parsers.setdefault(".parquet", parse_parquet)

        # NumPy npy/npz (numpy is a core dependency)
        def parse_npy(content: bytes, name: str) -> Any:
            return np.load(BytesIO(content), allow_pickle=False)

        cls._parsers.setdefault(".npy", parse_npy)
        cls._parsers.setdefault(".npz", parse_npy)

        # Images via Pillow, when it is installed; without it image blobs are
        # returned as raw bytes (no parser is registered for them).
        try:
            from PIL import Image as _Image
        except ImportError:
            return

        def parse_image(content: bytes, name: str) -> Any:
            return _Image.open(BytesIO(content))

        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"):
            cls._parsers.setdefault(ext, parse_image)

    @classmethod
    def _parse_bytes(cls, blob_name: str, content: bytes) -> Any:
        ext = "." + blob_name.lower().rsplit(".", 1)[-1] if "." in blob_name else ""
        parser = cls._parsers.get(ext)
        if parser is None:
            return content
        try:
            return parser(content, blob_name)
        # Parsers are user-registrable callables, so any exception is possible;
        # it is reported, and the raw bytes are returned instead.
        except Exception as exc:
            warnings.warn(
                f"Could not parse blob {blob_name!r} with the {ext!r} parser "
                f"({type(exc).__name__}: {exc}); returning its raw bytes.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return content

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def list_files_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """
        List blob names under each hourly prefix within [start, end].

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts);
                hour folders from this hour on are listed.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            extensions: Optional set/list of file extensions (e.g., {"json", ".bmp"}). Case-insensitive.
            limit: Optional cap on number of files collected.
        """
        allowed_exts = self._normalize_exts(extensions)
        names: list[str] = []
        collected = 0
        for pfx in (self._hour_prefix(ts) for ts in utc_folder_hours(start, end)):
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            for b in blob_iter:
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

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def iter_file_names_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
    ) -> Iterator[str]:
        """
        Yield blob names under each hourly prefix within [start, end].
        Uses server-side prefix listing and client-side extension filtering.

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts);
                hour folders from this hour on are listed.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            extensions: File extensions to keep, with or without the dot (e.g.
                ``{"json", ".bmp"}``), case-insensitive. None keeps every file.
        """
        allowed_exts = self._normalize_exts(extensions)
        for pfx in (self._hour_prefix(ts) for ts in utc_folder_hours(start, end)):
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            for b in blob_iter:
                name = str(b.name)
                if allowed_exts is not None:
                    if not any(name.lower().endswith(ext) for ext in allowed_exts):
                        continue
                yield name

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def fetch_files_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
        errors: ErrorsMode = "warn",
    ) -> dict[str, Any]:
        """
        Download files that match extensions within [start, end] hour prefixes.
        Returns a dict mapping blob_name -> parsed object (if parse=True and a parser exists),
        otherwise raw bytes.

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts);
                hour folders from this hour on are listed.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            extensions: File extensions to keep, with or without the dot (e.g.
                ``{"json", ".bmp"}``), case-insensitive. None keeps every file.
            parse: If True, parse each file with the parser registered for its extension
                (see `register_parser`), falling back to raw bytes when none exists or
                parsing fails (with a :class:`~ts_shape.errors.DataQualityWarning`).
                Default False returns raw bytes.
            errors: ``"warn"`` (default) skips a blob that cannot be
                downloaded and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        blob_names = self.list_files_by_time_range(start, end, extensions=extensions)
        return self._fetch_many(blob_names, parse=parse, errors=errors)

    def _fetch_many(
        self, blob_names: list[str], *, parse: bool, errors: ErrorsMode
    ) -> dict[str, Any]:
        """Download ``blob_names`` concurrently; report the ones that fail."""
        if not blob_names:
            return {}
        downloaded, failures = _download_all(
            self._download_bytes, blob_names, self.max_workers
        )
        report_failures(failures, what="blob(s)", errors=errors)
        return {
            name: self._parse_bytes(name, content) if parse else content
            for name, content in downloaded
        }

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def stream_files_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
        errors: ErrorsMode = "warn",
    ) -> Iterator[tuple[str, Any]]:
        """
        Stream matching files as (blob_name, bytes-or-parsed) within [start, end].
        Maintains up to `max_workers` concurrent downloads while yielding incrementally.

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts);
                hour folders from this hour on are listed.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            extensions: File extensions to keep, with or without the dot (e.g.
                ``{"json", ".bmp"}``), case-insensitive. None keeps every file.
            parse: If True, parse each file with the parser registered for its extension
                (see `register_parser`), falling back to raw bytes when none exists or
                parsing fails (with a :class:`~ts_shape.errors.DataQualityWarning`).
                Default False returns raw bytes.
            errors: ``"warn"`` (default) skips a blob that cannot be
                downloaded and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        names_iter = self.iter_file_names_by_time_range(
            start, end, extensions=extensions
        )
        yield from self._stream_many(names_iter, parse=parse, errors=errors)

    def _stream_many(
        self, names: Iterator[str], *, parse: bool, errors: ErrorsMode
    ) -> Iterator[tuple[str, Any]]:
        """Stream ``names`` with bounded concurrency; report the ones that fail."""
        failures: list[tuple[str, Exception]] = []
        for name, content in _stream_downloads(
            self._download_bytes,
            names,
            self.max_workers,
            failures=failures,
            errors=errors,
            what="blob(s)",
        ):
            yield (name, self._parse_bytes(name, content) if parse else content)
        report_failures(failures, what="blob(s)", errors=errors)

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def fetch_files_by_time_range_and_basenames(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        basenames: Iterable[str],
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
        errors: ErrorsMode = "warn",
    ) -> dict[str, Any]:
        """
        Download files whose basename (final path segment) is in `basenames`,
        optionally filtered by extensions, within [start, end] hour prefixes.
        Returns blob_name -> parsed object (if parse=True and a parser exists), otherwise raw bytes.

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts);
                hour folders from this hour on are listed.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            basenames: File names (final path segment, extension included) to keep;
                matched exactly, after stripping whitespace.
            extensions: File extensions to keep, with or without the dot (e.g.
                ``{"json", ".bmp"}``), case-insensitive. None keeps every file.
            parse: If True, parse each file with the parser registered for its extension
                (see `register_parser`), falling back to raw bytes when none exists or
                parsing fails (with a :class:`~ts_shape.errors.DataQualityWarning`).
                Default False returns raw bytes.
            errors: ``"warn"`` (default) skips a blob that cannot be
                downloaded and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        base_set = {str(b).strip() for b in basenames if str(b).strip()}
        allowed_exts = self._normalize_exts(extensions)
        candidates: list[str] = []
        for pfx in (self._hour_prefix(ts) for ts in utc_folder_hours(start, end)):
            blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
            for b in blob_iter:
                name = str(b.name)
                base = name.rsplit("/", 1)[-1]
                if base not in base_set:
                    continue
                if allowed_exts is not None and not any(
                    name.lower().endswith(ext) for ext in allowed_exts
                ):
                    continue
                candidates.append(name)

        return self._fetch_many(candidates, parse=parse, errors=errors)

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def stream_files_by_time_range_and_basenames(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        basenames: Iterable[str],
        *,
        extensions: Iterable[str] | None = None,
        parse: bool = False,
        errors: ErrorsMode = "warn",
    ) -> Iterator[tuple[str, Any]]:
        """
        Stream files whose basename is in `basenames` within [start, end].
        Yields (blob_name, bytes-or-parsed) incrementally with bounded concurrency.

        Args:
            start: Start of the range (anything ``pd.to_datetime`` accepts);
                hour folders from this hour on are listed.
            end: End of the range, inclusive (anything ``pd.to_datetime``
                accepts).
            basenames: File names (final path segment, extension included) to keep;
                matched exactly, after stripping whitespace.
            extensions: File extensions to keep, with or without the dot (e.g.
                ``{"json", ".bmp"}``), case-insensitive. None keeps every file.
            parse: If True, parse each file with the parser registered for its extension
                (see `register_parser`), falling back to raw bytes when none exists or
                parsing fails (with a :class:`~ts_shape.errors.DataQualityWarning`).
                Default False returns raw bytes.
            errors: ``"warn"`` (default) skips a blob that cannot be
                downloaded and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped blob; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.
        """
        check_errors_mode(errors)
        base_set = {str(b).strip() for b in basenames if str(b).strip()}
        allowed_exts = self._normalize_exts(extensions)

        def _names_iter() -> Iterator[str]:
            for pfx in (self._hour_prefix(ts) for ts in utc_folder_hours(start, end)):
                blob_iter = self.container_client.list_blobs(name_starts_with=pfx)
                for b in blob_iter:
                    name = str(b.name)
                    base = name.rsplit("/", 1)[-1]
                    if base not in base_set:
                        continue
                    if allowed_exts is not None and not any(
                        name.lower().endswith(ext) for ext in allowed_exts
                    ):
                        continue
                    yield name

        yield from self._stream_many(_names_iter(), parse=parse, errors=errors)
