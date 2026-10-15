import logging
import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ts_shape.errors import DataQualityWarning, LoaderConfigWarning, LoaderError
from ts_shape.loader._utils import (
    ErrorsMode,
    check_errors_mode,
    describe_failures,
    read_errors,
    report_failures,
)
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import caller_stacklevel

logger = logging.getLogger(__name__)

_METADATA_PATH = ".meta/series.csv"
_CSV_PREFIX = "csv"
_EMPTY_SCHEMA = ["systime", "uuid", "value_double", "is_delta"]
_META_COLUMNS = [
    "id",
    "label_lvl1",
    "label_lvl2",
    "label_lvl3",
    "label_lvl4",
    "description",
    "unit",
    "hierarchy_lvl1",
    "hierarchy_lvl2",
    "hierarchy_lvl3",
    "hierarchy_lvl4",
    "hierarchy_lvl5",
    "hierarchy_lvl6",
]


class DatabricksUnityEnergyLoader:
    """
    Load CSV energy timeseries and series metadata governed by **Unity Catalog**.

    Unity Catalog exposes the *same* CSV energy files that already live in your
    cloud storage (an external UC Volume over, e.g., an Azure blob container).
    The on-disk layout is unchanged::

        <volume_root>/<prefix>/
          .meta/
            series.csv           ← series metadata (tab-separated)
          csv/
            YYYY/MM/DD/
              <series_id>.csv    ← interval readings: time, value columns

    All load methods return the standard ts-shape schema::

        systime | uuid | value_double | is_delta

    where ``uuid`` is the ``series_id`` (stem of the CSV filename), so the frames
    flow straight into every ts-shape transformation, event detector, ``Pipeline``
    and ``DataIntegratorHybrid`` without any change.

    **Designed for use inside Databricks notebooks / pipelines.** A UC Volume is
    FUSE-mounted at ``/Volumes/<catalog>/<schema>/<volume>/...``, so this loader
    reads CSV *directly from that mounted path* with ``pandas.read_csv`` and keeps
    the resource footprint low:

    - **No download step and no ``databricks-sdk`` / network client** -- the
      mounted Volume is read like a local directory.
    - **Navigates only the day folders in range** (``csv/YYYY/MM/DD``) instead of
      scanning the whole tree, so a one-day query never lists the whole Volume.
    - **Streaming generator** (:meth:`stream_by_time_range`) yields one series at a
      time so the driver never has to hold the full dataset in memory.

    Reads are sequential by design (no thread pool) to avoid adding CPU/memory
    pressure on a shared cluster driver.

    Off-cluster (outside Databricks) the same code works against any mounted or
    synced copy of the Volume; if you cannot mount it, use
    :class:`AzureBlobEnergyLoader` against the underlying storage instead.
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
        thousands: str | None = None,
        decimal: str = ".",
        validate: bool = True,
    ) -> None:
        """
        Resolve the mounted Unity Catalog Volume root to read from.

        Provide *either* an explicit ``volume_path`` *or* the
        ``catalog`` / ``schema`` / ``volume`` triple (joined under ``base_path``)::

            # explicit mounted path
            DatabricksUnityEnergyLoader(
                volume_path="/Volumes/main/plant/energy",
            )

            # from catalog/schema/volume parts
            DatabricksUnityEnergyLoader(
                catalog="main", schema="plant", volume="energy",
            )

        Args:
            volume_path: Full mounted Volume root, e.g.
                ``/Volumes/<catalog>/<schema>/<volume>``.
            catalog: UC catalog name (used when ``volume_path`` is omitted).
            schema: UC schema name (used when ``volume_path`` is omitted).
            volume: UC volume name (used when ``volume_path`` is omitted).
            prefix: Optional sub-path beneath the Volume root holding the
                ``.meta/`` and ``csv/`` folders.
            base_path: Mount root for Volumes; ``/Volumes`` inside Databricks.
            thousands: Thousands separator for ``pd.read_csv`` (default None).
            decimal: Decimal separator for ``pd.read_csv`` (default ".").
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
        # Directory under which .meta/ and csv/ live: <volume_root>/<prefix>
        self.base_path = root / self.prefix if self.prefix else root
        self.thousands = thousands
        self.decimal = decimal

        if validate and not self.base_path.exists():
            warnings.warn(
                "DatabricksUnityEnergyLoader path does not exist: "
                f"{self.base_path}. Inside Databricks ensure the Unity Catalog "
                "Volume is mounted (default '/Volumes/<catalog>/<schema>/<volume>') "
                "and that catalog/schema/volume/prefix are correct.",
                LoaderConfigWarning,
                stacklevel=caller_stacklevel(),
            )

    # ── Metadata ────────────────────────────────────────────────────────────

    def load_series_metadata(self) -> pd.DataFrame:
        """
        Read ``.meta/series.csv`` and return as a DataFrame.

        Columns: id, label_lvl1, label_lvl2, label_lvl3, label_lvl4,
                 description, unit, hierarchy_lvl1 … hierarchy_lvl6

        Returns an empty DataFrame with the expected columns, and emits a
        :class:`~ts_shape.errors.LoaderConfigWarning`, when the file does not
        exist.

        Raises:
            LoaderError: When the file exists but cannot be read or parsed.
        """
        path = self.base_path / _METADATA_PATH
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
        except FileNotFoundError:
            warnings.warn(
                f"Series metadata file {str(path)!r} does not exist; returning "
                "an empty metadata table.",
                LoaderConfigWarning,
                stacklevel=caller_stacklevel(),
            )
            return pd.DataFrame(columns=_META_COLUMNS)
        except read_errors() as exc:
            raise LoaderError(
                describe_failures([(str(path), exc)], "metadata file(s)")
            ) from exc
        for col in _META_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[_META_COLUMNS].reset_index(drop=True)

    # ── Timeseries ───────────────────────────────────────────────────────────

    def load_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        series_ids: list[str] | None = None,
        *,
        errors: ErrorsMode = "warn",
    ) -> pd.DataFrame:
        """
        Load all CSV files in ``csv/YYYY/MM/DD/`` for each date in [start, end].

        Args:
            start: Start date/datetime (inclusive).
            end: End date/datetime (inclusive).
            series_ids: Optional list of series IDs to load. Loads all if None.
            errors: ``"warn"`` (default) skips a CSV that cannot be read or
                parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped file; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.

        Returns:
            Standard schema DataFrame: systime | uuid | value_double | is_delta
        """
        check_errors_mode(errors)
        ids_set = set(series_ids) if series_ids else None
        paths = self._list_csv_by_date_range(start, end, ids_set)
        return self._read_and_concat(paths, errors=errors)

    def load_by_series_ids(
        self,
        series_ids: list[str],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        *,
        errors: ErrorsMode = "warn",
    ) -> pd.DataFrame:
        """
        Load specific series by ID.

        With start/end: visits only the day folders in range and keeps files
        whose stem matches a requested series_id. Without dates: walks ``csv/``
        and filters by stem. With only one bound the range is open on the
        other side (day folders on or after ``start``, or on or before ``end``).

        Args:
            series_ids: Series IDs to load.
            start: Optional start date filter (inclusive, whole days).
            end: Optional end date filter (inclusive, whole days).
            errors: ``"warn"`` (default) skips a CSV that cannot be read or
                parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped file; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.

        Returns:
            Standard schema DataFrame: systime | uuid | value_double | is_delta
        """
        check_errors_mode(errors)
        ids_set = set(series_ids)
        if start is not None and end is not None:
            paths = self._list_csv_by_date_range(start, end, ids_set)
        else:
            paths = self._list_csv_open_range(start, end, ids_set)
        return self._read_and_concat(paths, errors=errors)

    def stream_by_time_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        series_ids: list[str] | None = None,
        *,
        errors: ErrorsMode = "warn",
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """
        Stream CSV files one at a time as (series_id, DataFrame) tuples.

        Memory-efficient alternative to :meth:`load_by_time_range` for large
        date ranges -- the recommended low-memory entry point in pipelines.

        Args:
            start: Start date/datetime (inclusive).
            end: End date/datetime (inclusive).
            series_ids: Optional series filter.
            errors: ``"warn"`` (default) skips a CSV that cannot be read or
                parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped file; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.

        Yields:
            (series_id, DataFrame) where DataFrame has the standard schema.
        """
        check_errors_mode(errors)
        ids_set = set(series_ids) if series_ids else None
        failures: list[tuple[str, Exception]] = []
        for path in self._list_csv_by_date_range(start, end, ids_set):
            try:
                df = self._read_csv(path)
            except read_errors() as exc:
                if errors == "raise":
                    raise LoaderError(
                        describe_failures([(str(path), exc)], "CSV file(s)")
                    ) from exc
                failures.append((str(path), exc))
                continue
            if not df.empty:
                yield self._series_id_from_path(path), df
        report_failures(failures, what="CSV file(s)", errors=errors)

    def list_series(self) -> list[str]:
        """
        List all series IDs present in the Volume by scanning ``csv/``.

        Returns:
            Sorted list of unique series ID strings.
        """
        seen: set[str] = set()
        for path in self._walk_csv(self.base_path / _CSV_PREFIX):
            seen.add(self._series_id_from_path(path))
        return sorted(seen)

    def fetch_data_as_dataframe(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        series_ids: list[str] | None = None,
        *,
        errors: ErrorsMode = "warn",
    ) -> pd.DataFrame:
        """
        Return a combined DataFrame, for ``Pipeline`` / ``DataIntegratorHybrid``.

        With a ``start``/``end`` pair this delegates to :meth:`load_by_time_range`
        (visiting only the in-range day folders). With only one bound the range
        is open on the other side: every day folder on or after ``start`` (or on
        or before ``end``) is read. With no bounds it loads every series under
        ``csv/``.

        Args:
            start: Start date/datetime (inclusive, whole days); None leaves the
                start open.
            end: End date/datetime (inclusive, whole days); None leaves the end
                open.
            series_ids: Series IDs to load. None loads every series.
            errors: ``"warn"`` (default) skips a CSV that cannot be read or
                parsed and emits one
                :class:`~ts_shape.errors.DataQualityWarning` naming every
                skipped file; ``"raise"`` raises
                :class:`~ts_shape.errors.LoaderError` instead.

        Returns:
            Standard schema DataFrame: systime | uuid | value_double | is_delta.
        """
        if start is not None and end is not None:
            return self.load_by_time_range(
                start, end, series_ids=series_ids, errors=errors
            )
        check_errors_mode(errors)
        ids_set = set(series_ids) if series_ids else None
        return self._read_and_concat(
            self._list_csv_open_range(start, end, ids_set), errors=errors
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _day_dirs(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> list[Path]:
        """Return the day-level directories: <base>/csv/YYYY/MM/DD."""
        dates = pd.date_range(
            start=pd.to_datetime(start).normalize(),
            end=pd.to_datetime(end).normalize(),
            freq="D",
        )
        csv_root = self.base_path / _CSV_PREFIX
        return [
            csv_root / str(ts.year) / str(ts.month).zfill(2) / str(ts.day).zfill(2)
            for ts in dates
        ]

    def _list_csv_in(self, directory: Path) -> list[Path]:
        """List ``*.csv`` files in a single day directory (no recursion)."""
        try:
            with os.scandir(directory) as it:
                return [
                    Path(entry.path)
                    for entry in it
                    if entry.is_file() and entry.name.endswith(".csv")
                ]
        except (FileNotFoundError, NotADirectoryError):
            return []  # no folder for this day: no data, not an error
        except OSError as exc:
            warnings.warn(
                f"Could not list {str(directory)!r} ({type(exc).__name__}: {exc}); "
                "its CSV files are skipped.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return []

    def _walk_csv(self, root: Path) -> list[Path]:
        """Recursively find all ``*.csv`` files under ``root`` (sorted)."""
        if not root.exists():
            return []
        return sorted(root.rglob("*.csv"))

    def _list_csv_by_date_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        ids_set: set | None,
    ) -> list[Path]:
        """List CSV files across the date range, optionally filtered by ids_set."""
        paths: list[Path] = []
        for directory in self._day_dirs(start, end):
            for path in self._list_csv_in(directory):
                if (
                    ids_set is not None
                    and self._series_id_from_path(path) not in ids_set
                ):
                    continue
                paths.append(path)
        return paths

    def _list_csv_open_range(
        self,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
        ids_set: set | None,
    ) -> list[Path]:
        """List CSV files in day folders within an optionally open date range."""
        lo = None if start is None else pd.to_datetime(start).tz_localize(None)
        hi = None if end is None else pd.to_datetime(end).tz_localize(None)
        lo = None if lo is None else lo.normalize()
        hi = None if hi is None else hi.normalize()
        csv_root = self.base_path / _CSV_PREFIX
        paths: list[Path] = []
        for path in self._walk_csv(csv_root):
            if ids_set is not None and self._series_id_from_path(path) not in ids_set:
                continue
            if lo is not None or hi is not None:
                day = self._day_of_path(path, csv_root)
                if day is None:
                    continue
                if (lo is not None and day < lo) or (hi is not None and day > hi):
                    continue
            paths.append(path)
        return paths

    @staticmethod
    def _day_of_path(path: Path, csv_root: Path) -> pd.Timestamp | None:
        """Return the ``YYYY/MM/DD`` day folder of ``path``, or None if off-layout."""
        try:
            parts = path.relative_to(csv_root).parts
        except ValueError:
            return None
        if len(parts) != 4:
            return None
        try:
            return pd.Timestamp(
                year=int(parts[0]), month=int(parts[1]), day=int(parts[2])
            )
        except ValueError:
            return None

    @staticmethod
    def _series_id_from_path(path: Path) -> str:
        """Extract series_id from path: csv/2026/01/13/sensor_001.csv → sensor_001."""
        return path.stem

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read a single CSV file and normalize it to the standard schema.

        Raises:
            One of :func:`~ts_shape.loader._utils.read_errors` when the file
            cannot be read or parsed, or lacks the ``time``/``value`` columns.
        """
        raw = pd.read_csv(path, thousands=self.thousands, decimal=self.decimal)
        return self._normalize_df(raw, self._series_id_from_path(path))

    @staticmethod
    def _normalize_df(raw: pd.DataFrame, series_id: str) -> pd.DataFrame:
        """
        Convert raw two-column CSV DataFrame to standard ts-shape schema.

        Input columns: time, value
        Output columns: systime, uuid, value_double, is_delta

        Raises:
            ValueError: When a non-empty CSV lacks the ``time`` or ``value``
                column.
        """
        if raw.empty:
            return pd.DataFrame(columns=_EMPTY_SCHEMA)
        missing = [c for c in ("time", "value") if c not in raw.columns]
        if missing:
            raise ValueError(
                f"series {series_id!r}: CSV has no {missing} column(s); "
                f"columns are {list(raw.columns)}"
            )

        out = pd.DataFrame()
        out["systime"] = pd.to_datetime(raw["time"], utc=True)
        out["uuid"] = series_id
        out["value_double"] = to_numeric_checked(
            raw["value"], what=f"series {series_id!r}"
        )
        out["is_delta"] = True
        return out.sort_values("systime").reset_index(drop=True)

    def _read_and_concat(
        self, paths: list[Path], *, errors: ErrorsMode = "warn"
    ) -> pd.DataFrame:
        """Read CSV files sequentially and concatenate the results.

        A file that cannot be read is reported according to ``errors`` (see
        :func:`~ts_shape.loader._utils.report_failures`).
        """
        if not paths:
            return pd.DataFrame(columns=_EMPTY_SCHEMA)

        frames: list[pd.DataFrame] = []
        failures: list[tuple[str, Exception]] = []
        for path in paths:
            try:
                df = self._read_csv(path)
            except read_errors() as exc:
                failures.append((str(path), exc))
                continue
            if not df.empty:
                frames.append(df)
        report_failures(failures, what="CSV file(s)", errors=errors)

        if not frames:
            return pd.DataFrame(columns=_EMPTY_SCHEMA)

        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("systime").reset_index(drop=True)
