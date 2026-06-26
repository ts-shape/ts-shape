"""Spark-native Databricks loader for the canonical hourly parquet layout.

Unlike :class:`DatabricksUnityParquetLoader` -- which reads a FUSE-mounted Unity
Catalog Volume directly with ``pandas.read_parquet`` -- this loader uses the
cluster's **Spark** session to read the same ``<base>/YYYY/MM/DD/HH/<uuid>.parquet``
layout. Use it when you want Spark to do the scan (predicate/column pushdown,
distributed read) and only collect to pandas at the end, e.g. inside a Databricks
notebook or job.

It generalizes the common notebook idiom::

    hours = pd.date_range(start, end, freq="h")
    paths = [f"{base}/{h:%Y/%m/%d/%H}/{uuid}.parquet" for h in hours]
    sdf = spark.read.option("basePath", base).parquet(*paths)
    df = sdf.select("systime", "uuid", "value_integer").toPandas()

into a reusable, parameterized loader: many UUIDs, configurable hour layout,
column projection, a Spark filter expression, optional missing-path tolerance,
and a choice of Spark or pandas output. The path-building step is a pure,
Spark-free method so it can be inspected and unit-tested on its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd  # type: ignore

from ts_shape.errors import LoaderError

logger = logging.getLogger(__name__)


def _active_spark() -> Any | None:
    """Return the active SparkSession if one exists, else ``None``.

    Imported lazily so the module loads fine off-cluster where ``pyspark`` is
    not installed.
    """
    try:
        from pyspark.sql import SparkSession  # type: ignore
    except ImportError:
        return None
    return SparkSession.getActiveSession()


class DatabricksSparkParquetLoader:
    """Read the canonical hourly parquet layout through a Spark session.

    Args:
        base_path: Root under which the hour folders live, e.g.
            ``/Volumes/main/plant/timeseries`` or ``abfss://.../timeseries``.
        spark: The SparkSession to use. When omitted, the active session is
            used (``SparkSession.getActiveSession()``); a clear
            :class:`~ts_shape.errors.LoaderError` is raised at read time if none
            is available.
        hour_pattern: ``strftime`` pattern for the hour subfolder. Default
            ``"%Y/%m/%d/%H"`` matches the rest of ts-shape.
        file_template: Per-file name template; ``{uuid}`` is substituted.
            Default ``"{uuid}.parquet"``.
    """

    def __init__(
        self,
        base_path: str,
        spark: Any | None = None,
        *,
        hour_pattern: str = "%Y/%m/%d/%H",
        file_template: str = "{uuid}.parquet",
    ) -> None:
        if not base_path:
            raise LoaderError("base_path must be a non-empty string.")
        self.base_path = base_path.rstrip("/")
        self.spark = spark if spark is not None else _active_spark()
        self.hour_pattern = hour_pattern
        self.file_template = file_template

    # ---- Pure path building (no Spark required) ----
    @staticmethod
    def _as_uuid_list(uuids: str | Sequence[str]) -> list[str]:
        """Accept a single UUID or a sequence and normalize to a list."""
        if isinstance(uuids, str):
            return [uuids]
        return list(uuids)

    def build_paths(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        uuids: str | Sequence[str],
        *,
        freq: str = "h",
    ) -> list[str]:
        """Build the explicit parquet paths covering ``[start, end]`` per UUID.

        This is the reusable, testable core: it mirrors what Spark will read,
        with one path per (hour, uuid) pair. Hours are inclusive of both ends.

        Args:
            start_timestamp: Window start (parsed by ``pandas.to_datetime``).
            end_timestamp: Window end (inclusive).
            uuids: A single UUID or a sequence of UUIDs.
            freq: Folder granularity; defaults to hourly (``"h"``).

        Returns:
            A list of fully-qualified parquet paths.

        Raises:
            LoaderError: If no UUIDs are supplied.
        """
        uuid_list = self._as_uuid_list(uuids)
        if not uuid_list:
            raise LoaderError("At least one uuid is required.")
        slots = pd.date_range(
            start=pd.to_datetime(start_timestamp),
            end=pd.to_datetime(end_timestamp),
            freq=freq,
        )
        paths: list[str] = []
        for ts in slots:
            hour_dir = ts.strftime(self.hour_pattern)
            for uuid in uuid_list:
                fname = self.file_template.format(uuid=uuid)
                paths.append(f"{self.base_path}/{hour_dir}/{fname}")
        return paths

    # ---- Spark read ----
    def _require_spark(self) -> Any:
        if self.spark is None:
            raise LoaderError(
                "No SparkSession available. Pass spark=... explicitly, or run "
                "inside a Databricks/Spark environment where an active session "
                "exists."
            )
        return self.spark

    def load_by_time_range_and_uuids(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        uuids: str | Sequence[str],
        *,
        columns: Sequence[str] | None = None,
        filter_expr: str | None = None,
        freq: str = "h",
        ignore_missing: bool = True,
        path_exists: Callable[[str], bool] | None = None,
        as_pandas: bool = True,
    ) -> Any:
        """Read the hourly parquet files for ``uuids`` within ``[start, end]``.

        Args:
            start_timestamp: Window start.
            end_timestamp: Window end (inclusive).
            uuids: A single UUID or a sequence of UUIDs.
            columns: Optional columns to ``select`` (projection pushdown).
            filter_expr: Optional Spark SQL predicate applied with ``.where``,
                e.g. ``"value_integer > 0"``.
            freq: Folder granularity; defaults to hourly.
            ignore_missing: When True, set ``spark.sql.files.ignoreMissingFiles``
                so hours with no file for a UUID are skipped instead of failing.
            path_exists: Optional callable used to pre-filter paths to those that
                exist (e.g. a wrapper over ``dbutils.fs``). Useful when
                ``ignore_missing`` is not sufficient for your storage layer.
            as_pandas: When True (default) return a pandas DataFrame via
                ``toPandas()``; when False return the Spark DataFrame.

        Returns:
            A pandas DataFrame, or the Spark DataFrame when ``as_pandas`` is
            False. An empty pandas DataFrame is returned if no candidate paths
            remain after ``path_exists`` filtering.

        Raises:
            LoaderError: If no SparkSession is available.
        """
        spark = self._require_spark()
        paths = self.build_paths(start_timestamp, end_timestamp, uuids, freq=freq)
        if path_exists is not None:
            paths = [p for p in paths if path_exists(p)]
        if not paths:
            logger.debug("No candidate parquet paths to read.")
            return pd.DataFrame() if as_pandas else spark.createDataFrame([], "")

        if ignore_missing:
            spark.conf.set("spark.sql.files.ignoreMissingFiles", "true")

        sdf = spark.read.option("basePath", self.base_path).parquet(*paths)
        if columns:
            sdf = sdf.select(*columns)
        if filter_expr:
            sdf = sdf.where(filter_expr)
        return sdf.toPandas() if as_pandas else sdf

    def fetch_data_as_dataframe(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        uuids: str | Sequence[str],
        *,
        columns: Sequence[str] | None = None,
        filter_expr: str | None = None,
    ) -> pd.DataFrame:
        """Pandas-returning convenience for ``Pipeline`` / ``DataIntegratorHybrid``."""
        return self.load_by_time_range_and_uuids(
            start_timestamp,
            end_timestamp,
            uuids,
            columns=columns,
            filter_expr=filter_expr,
            as_pandas=True,
        )
