"""Tests for the Spark-native Databricks loader.

pyspark is not a dependency, so the Spark read is exercised with a small fake
session that records what was requested and returns a stub DataFrame.
"""

import pandas as pd  # type: ignore
import pytest

from ts_shape.errors import LoaderError
from ts_shape.loader.timeseries.databricks_spark_parquet_loader import (
    DatabricksSparkParquetLoader,
)

# ---------------------------------------------------------------------------
# Fake Spark session
# ---------------------------------------------------------------------------


class FakeSparkDataFrame:
    def __init__(self, pdf, log):
        self._pdf = pdf
        self._log = log

    def select(self, *cols):
        self._log["select"] = list(cols)
        return FakeSparkDataFrame(self._pdf[list(cols)], self._log)

    def where(self, expr):
        self._log["where"] = expr
        return self

    def toPandas(self):
        return self._pdf


class FakeReader:
    def __init__(self, pdf, log):
        self._pdf = pdf
        self._log = log

    def option(self, key, value):
        self._log.setdefault("options", {})[key] = value
        return self

    def parquet(self, *paths):
        self._log["paths"] = list(paths)
        return FakeSparkDataFrame(self._pdf, self._log)


class FakeConf:
    def __init__(self, log):
        self._log = log

    def set(self, key, value):
        self._log.setdefault("conf", {})[key] = value


class FakeSpark:
    def __init__(self, pdf):
        self.log: dict = {}
        self._pdf = pdf

    @property
    def read(self):
        return FakeReader(self._pdf, self.log)

    @property
    def conf(self):
        return FakeConf(self.log)


@pytest.fixture()
def sample_pdf():
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(["2026-06-19 06:00", "2026-06-19 07:00"]),
            "uuid": ["ABC", "ABC"],
            "value_integer": [1, 2],
        }
    )


# ---------------------------------------------------------------------------
# build_paths (pure, no Spark)
# ---------------------------------------------------------------------------


def test_build_paths_single_uuid():
    ldr = DatabricksSparkParquetLoader("/base", spark=object())
    paths = ldr.build_paths("2026-06-19 06:00", "2026-06-19 08:00", "ABC")
    assert paths == [
        "/base/2026/06/19/06/ABC.parquet",
        "/base/2026/06/19/07/ABC.parquet",
        "/base/2026/06/19/08/ABC.parquet",
    ]


def test_build_paths_multi_uuid_pairs_every_hour():
    ldr = DatabricksSparkParquetLoader("/base/", spark=object())  # trailing slash
    paths = ldr.build_paths("2026-06-19 06:00", "2026-06-19 07:00", ["A", "B"])
    assert paths == [
        "/base/2026/06/19/06/A.parquet",
        "/base/2026/06/19/06/B.parquet",
        "/base/2026/06/19/07/A.parquet",
        "/base/2026/06/19/07/B.parquet",
    ]


def test_build_paths_requires_uuids():
    ldr = DatabricksSparkParquetLoader("/base", spark=object())
    with pytest.raises(LoaderError, match="at least one uuid|At least one uuid"):
        ldr.build_paths("2026-06-19 06:00", "2026-06-19 07:00", [])


def test_empty_base_path_rejected():
    with pytest.raises(LoaderError, match="base_path"):
        DatabricksSparkParquetLoader("", spark=object())


# ---------------------------------------------------------------------------
# load_by_time_range_and_uuids (fake Spark)
# ---------------------------------------------------------------------------


def test_load_reads_paths_and_returns_pandas(sample_pdf):
    spark = FakeSpark(sample_pdf)
    ldr = DatabricksSparkParquetLoader("/base", spark=spark)

    out = ldr.load_by_time_range_and_uuids(
        "2026-06-19 06:00",
        "2026-06-19 07:00",
        "ABC",
        columns=["systime", "uuid", "value_integer"],
        filter_expr="value_integer > 0",
    )

    assert isinstance(out, pd.DataFrame)
    # basePath option set, missing-files tolerance on, both hour paths read.
    assert spark.log["options"]["basePath"] == "/base"
    assert spark.log["conf"]["spark.sql.files.ignoreMissingFiles"] == "true"
    assert spark.log["paths"] == [
        "/base/2026/06/19/06/ABC.parquet",
        "/base/2026/06/19/07/ABC.parquet",
    ]
    assert spark.log["select"] == ["systime", "uuid", "value_integer"]
    assert spark.log["where"] == "value_integer > 0"


def test_load_can_return_spark_dataframe(sample_pdf):
    spark = FakeSpark(sample_pdf)
    ldr = DatabricksSparkParquetLoader("/base", spark=spark)
    out = ldr.load_by_time_range_and_uuids(
        "2026-06-19 06:00", "2026-06-19 06:00", "ABC", as_pandas=False
    )
    assert isinstance(out, FakeSparkDataFrame)


def test_path_exists_prefilter_short_circuits_to_empty(sample_pdf):
    spark = FakeSpark(sample_pdf)
    ldr = DatabricksSparkParquetLoader("/base", spark=spark)
    out = ldr.load_by_time_range_and_uuids(
        "2026-06-19 06:00",
        "2026-06-19 07:00",
        "ABC",
        path_exists=lambda _p: False,  # nothing exists
    )
    assert isinstance(out, pd.DataFrame) and out.empty
    # No Spark read attempted.
    assert "paths" not in spark.log


def test_missing_spark_session_raises():
    ldr = DatabricksSparkParquetLoader("/base", spark=None)
    with pytest.raises(LoaderError, match="No SparkSession"):
        ldr.load_by_time_range_and_uuids("2026-06-19 06:00", "2026-06-19 07:00", "ABC")
