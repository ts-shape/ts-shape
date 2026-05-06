from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import PurePosixPath
from typing import Dict, Iterator, List, Optional, Tuple, Union
import logging

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

_METADATA_PATH = ".meta/series.csv"
_CSV_PREFIX = "csv/"
_EMPTY_SCHEMA = ["systime", "uuid", "value_double", "is_delta"]
_META_COLUMNS = [
    "id",
    "label_lvl1", "label_lvl2", "label_lvl3", "label_lvl4",
    "description", "unit",
    "hierarchy_lvl1", "hierarchy_lvl2", "hierarchy_lvl3",
    "hierarchy_lvl4", "hierarchy_lvl5", "hierarchy_lvl6",
]


class AzureBlobEnergyLoader:
    """
    Load CSV energy timeseries and series metadata from Azure Blob Storage.

    Bucket structure::

        <system>-<region>-<location>-energy-data   (container)
          .meta/
            series.csv           ← series metadata
          csv/
            YYYY/MM/DD/
              <series_id>.csv    ← interval readings: time, value columns

    Authentication supports three methods identical to AzureBlobParquetLoader:

    1. **SAS URL**::

        loader = AzureBlobEnergyLoader(
            sas_url="https://account.blob.core.windows.net/container?sv=...&sig=..."
        )

    2. **Connection string**::

        loader = AzureBlobEnergyLoader(
            connection_string="DefaultEndpointsProtocol=https;AccountName=...;",
            container_name="prod-west-plantA-energy-data",
        )

    3. **AAD credential**::

        loader = AzureBlobEnergyLoader(
            account_url="https://account.blob.core.windows.net",
            container_name="prod-west-plantA-energy-data",
            credential=DefaultAzureCredential(),
        )

    All load methods return a DataFrame with the standard ts-shape schema::

        systime | uuid | value_double | is_delta

    where ``uuid`` is the ``series_id`` (stem of the CSV filename).
    """

    def __init__(
        self,
        container_name: Optional[str] = None,
        *,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        credential: Optional[object] = None,
        sas_url: Optional[str] = None,
        prefix: str = "",
        max_workers: int = 8,
        thousands: Optional[str] = None,
        decimal: str = ".",
    ) -> None:
        """
        Args:
            container_name: Target container (not needed with sas_url).
            connection_string: Azure Storage connection string.
            account_url: Full account URL for AAD auth.
            credential: Azure credential object for AAD auth.
            sas_url: Full Blob SAS URL including container and SAS token.
            prefix: Optional blob prefix to narrow all listings.
            max_workers: Max concurrent downloads.
            thousands: Thousands separator for pd.read_csv (default None — standard floats).
            decimal: Decimal separator for pd.read_csv (default ".").
        """
        try:
            from azure.storage.blob import ContainerClient  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "azure-storage-blob is required for AzureBlobEnergyLoader. "
                "Install with `pip install azure-storage-blob`."
            ) from exc

        # Auto-detect SAS URL passed as connection_string
        if connection_string and connection_string.strip().startswith("http"):
            sas_url = connection_string
            connection_string = None

        if sas_url:
            self.container_client = ContainerClient.from_container_url(sas_url)
        elif account_url or (credential is not None and not connection_string):
            if not account_url:
                raise ValueError("account_url must be provided when using AAD credential auth")
            if credential is None:
                raise ValueError("credential must be provided when using AAD credential auth")
            if not container_name:
                raise ValueError("container_name is required when using account_url + credential")
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
                raise ValueError("container_name is required when using connection_string")
            try:
                self.container_client = ContainerClient.from_connection_string(
                    conn_str=connection_string, container_name=container_name
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "Invalid connection string. Ensure it contains "
                    "'AccountName=...' and 'AccountKey=...' (or use "
                    "sas_url for SAS token auth, or account_url + credential for AAD)."
                ) from exc

        self.prefix = prefix
        self.max_workers = max(1, max_workers)
        self.thousands = thousands
        self.decimal = decimal

    @classmethod
    def from_account_name(
        cls,
        account_name: str,
        container_name: str,
        *,
        credential: Optional[object] = None,
        endpoint_suffix: str = "blob.core.windows.net",
        prefix: str = "",
        max_workers: int = 8,
        thousands: Optional[str] = None,
        decimal: str = ".",
    ) -> "AzureBlobEnergyLoader":
        """
        Construct a loader using AAD credentials with an account name.

        Args:
            account_name: Storage account name.
            container_name: Target container.
            credential: Azure credential (required).
            endpoint_suffix: DNS suffix for sovereign clouds.
            prefix: Optional listing prefix.
            max_workers: Concurrency for downloads.
            thousands: Thousands separator passed to pd.read_csv.
            decimal: Decimal separator passed to pd.read_csv.
        """
        if credential is None:
            raise ValueError("credential must be provided when using AAD credential auth")
        account_url = f"https://{account_name}.{endpoint_suffix}"
        return cls(
            container_name=container_name,
            account_url=account_url,
            credential=credential,
            prefix=prefix,
            max_workers=max_workers,
            thousands=thousands,
            decimal=decimal,
        )

    # ── Metadata ────────────────────────────────────────────────────────────

    def load_series_metadata(self) -> pd.DataFrame:
        """
        Download ``.meta/series.csv`` and return as a DataFrame.

        Columns: id, label_lvl1, label_lvl2, label_lvl3, label_lvl4,
                 description, unit, hierarchy_lvl1 … hierarchy_lvl6

        Returns an empty DataFrame with the expected columns when the blob
        does not exist.
        """
        blob_name = self._full_path(_METADATA_PATH)
        try:
            downloader = self.container_client.download_blob(blob_name)
            data = downloader.readall()
            df = pd.read_csv(BytesIO(data), sep="\t", dtype=str)
            # Ensure all expected columns are present, fill missing with NA
            for col in _META_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            return df[_META_COLUMNS].reset_index(drop=True)
        except Exception as exc:
            logger.debug("Could not load series metadata from '%s': %s", blob_name, exc)
            return pd.DataFrame(columns=_META_COLUMNS)

    # ── Timeseries ───────────────────────────────────────────────────────────

    def load_by_time_range(
        self,
        start: Union[str, "pd.Timestamp"],
        end: Union[str, "pd.Timestamp"],
        series_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load all CSV files in ``csv/YYYY/MM/DD/`` for each date in [start, end].

        Args:
            start: Start date/datetime (inclusive).
            end: End date/datetime (inclusive).
            series_ids: Optional list of series IDs to load. Loads all if None.

        Returns:
            Standard schema DataFrame: systime | uuid | value_double | is_delta
        """
        ids_set = set(series_ids) if series_ids else None
        blob_names = self._list_blobs_by_date_range(start, end, ids_set)
        return self._download_and_concat(blob_names)

    def load_by_series_ids(
        self,
        series_ids: List[str],
        start: Optional[Union[str, "pd.Timestamp"]] = None,
        end: Optional[Union[str, "pd.Timestamp"]] = None,
    ) -> pd.DataFrame:
        """
        Load specific series by ID.

        With start/end: constructs direct paths for each (date, series_id) pair.
        Without dates: lists all blobs under ``csv/`` and filters by stem.

        Args:
            series_ids: Series IDs to load.
            start: Optional start date filter.
            end: Optional end date filter.

        Returns:
            Standard schema DataFrame: systime | uuid | value_double | is_delta
        """
        if start is not None and end is not None:
            blob_names = self._list_blobs_by_date_range(start, end, set(series_ids))
        else:
            ids_set = set(series_ids)
            prefix = self._full_path(_CSV_PREFIX)
            all_blobs = self.container_client.list_blobs(name_starts_with=prefix or None)
            blob_names = [
                b.name for b in all_blobs
                if str(b.name).endswith(".csv")
                and self._series_id_from_blob_name(b.name) in ids_set
            ]
        return self._download_and_concat(blob_names)

    def stream_by_time_range(
        self,
        start: Union[str, "pd.Timestamp"],
        end: Union[str, "pd.Timestamp"],
        series_ids: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, pd.DataFrame]]:
        """
        Stream CSV files one at a time as (series_id, DataFrame) tuples.

        Memory-efficient alternative to load_by_time_range for large date ranges.

        Args:
            start: Start date/datetime (inclusive).
            end: End date/datetime (inclusive).
            series_ids: Optional series filter.

        Yields:
            (series_id, DataFrame) where DataFrame has the standard schema.
        """
        ids_set = set(series_ids) if series_ids else None
        blob_names = self._list_blobs_by_date_range(start, end, ids_set)
        for blob_name in blob_names:
            df = self._download_csv(blob_name)
            if df is not None and not df.empty:
                yield self._series_id_from_blob_name(blob_name), df

    def list_series(self) -> List[str]:
        """
        List all series IDs present in the blob store by scanning ``csv/``.

        Returns:
            Sorted list of unique series ID strings.
        """
        prefix = self._full_path(_CSV_PREFIX)
        blob_iter = self.container_client.list_blobs(name_starts_with=prefix or None)
        seen = set()
        for blob in blob_iter:
            name: str = blob.name
            if name.endswith(".csv"):
                seen.add(self._series_id_from_blob_name(name))
        return sorted(seen)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _full_path(self, relative: str) -> str:
        """Prepend self.prefix to a relative blob path."""
        if not self.prefix:
            return relative
        base = self.prefix.rstrip("/")
        return f"{base}/{relative}"

    def _build_date_paths(
        self,
        start: Union[str, "pd.Timestamp"],
        end: Union[str, "pd.Timestamp"],
    ) -> List[str]:
        """Return list of day-level blob prefixes: csv/YYYY/MM/DD/"""
        dates = pd.date_range(
            start=pd.to_datetime(start).normalize(),
            end=pd.to_datetime(end).normalize(),
            freq="D",
        )
        return [
            self._full_path(
                f"{_CSV_PREFIX}{ts.year}/{str(ts.month).zfill(2)}/{str(ts.day).zfill(2)}/"
            )
            for ts in dates
        ]

    def _list_blobs_by_date_range(
        self,
        start: Union[str, "pd.Timestamp"],
        end: Union[str, "pd.Timestamp"],
        ids_set: Optional[set],
    ) -> List[str]:
        """List CSV blob names across the date range, optionally filtered by ids_set."""
        day_prefixes = self._build_date_paths(start, end)
        blob_names: List[str] = []
        for pfx in day_prefixes:
            for blob in self.container_client.list_blobs(name_starts_with=pfx):
                name: str = blob.name
                if not name.endswith(".csv"):
                    continue
                if ids_set is not None and self._series_id_from_blob_name(name) not in ids_set:
                    continue
                blob_names.append(name)
        return blob_names

    @staticmethod
    def _series_id_from_blob_name(blob_name: str) -> str:
        """Extract series_id from blob path: csv/2026/01/13/sensor_001.csv → sensor_001"""
        return PurePosixPath(blob_name).stem

    def _download_csv(self, blob_name: str) -> Optional[pd.DataFrame]:
        """Download a single CSV blob and normalize to standard schema. Returns None on error."""
        try:
            downloader = self.container_client.download_blob(blob_name)
            data = downloader.readall()
            raw = pd.read_csv(
                BytesIO(data),
                thousands=self.thousands,
                decimal=self.decimal,
            )
            series_id = self._series_id_from_blob_name(blob_name)
            return self._normalize_df(raw, series_id)
        except Exception as exc:
            logger.debug("Failed to download blob '%s': %s", blob_name, exc)
            return None

    @staticmethod
    def _normalize_df(raw: pd.DataFrame, series_id: str) -> pd.DataFrame:
        """
        Convert raw two-column CSV DataFrame to standard ts-shape schema.

        Input columns: time, value
        Output columns: systime, uuid, value_double, is_delta
        """
        if raw.empty or "time" not in raw.columns or "value" not in raw.columns:
            return pd.DataFrame(columns=_EMPTY_SCHEMA)

        out = pd.DataFrame()
        out["systime"] = pd.to_datetime(raw["time"], utc=True)
        out["uuid"] = series_id
        out["value_double"] = pd.to_numeric(raw["value"], errors="coerce")
        out["is_delta"] = True
        return out.sort_values("systime").reset_index(drop=True)

    def _download_and_concat(self, blob_names: List[str]) -> pd.DataFrame:
        """Download a list of CSV blobs concurrently and concatenate results."""
        if not blob_names:
            return pd.DataFrame(columns=_EMPTY_SCHEMA)

        frames: List[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._download_csv, name): name for name in blob_names}
            for future in as_completed(futures):
                df = future.result()
                if df is not None and not df.empty:
                    frames.append(df)

        if not frames:
            return pd.DataFrame(columns=_EMPTY_SCHEMA)

        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("systime").reset_index(drop=True)
