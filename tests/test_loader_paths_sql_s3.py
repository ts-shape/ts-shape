"""TimescaleDB, S3 proxy and metadata-DB loaders against in-process fakes.

The database engine and the S3 filesystem are replaced, so the query,
parquet-export and retry paths run without a server.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")


# --------------------------------------------------------------------------
# TimescaleDB
# --------------------------------------------------------------------------

sqlalchemy = pytest.importorskip("sqlalchemy")

from ts_shape.loader.timeseries import timescale_loader  # noqa: E402
from ts_shape.loader.timeseries.timescale_loader import (  # noqa: E402
    TimescaleDBDataAccess,
)

DB = {"db_user": "u", "db_pass": "p", "db_host": "h", "db_name": "plant"}


@pytest.fixture
def engine_urls(monkeypatch):
    """Record the URL every engine is created with; no driver is loaded."""
    urls: list[str] = []

    def fake_engine(url, *args, **kwargs):
        urls.append(url)
        return f"engine:{url}"

    # the loader may import create_engine at module level or lazily
    monkeypatch.setattr(sqlalchemy, "create_engine", fake_engine)
    monkeypatch.setattr(timescale_loader, "create_engine", fake_engine, raising=False)
    return urls


def _rows(uuid: str, start: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": uuid,
            "systime": pd.date_range(start, periods=n, freq="20min"),
            "value_double": range(n),
        }
    )


@pytest.fixture
def read_sql_calls(monkeypatch):
    """Serve two chunks per uuid from ``pd.read_sql`` and record the calls."""
    calls: list[dict] = []

    def fake_read_sql(query, engine, params=None, chunksize=None):
        calls.append(
            {
                "sql": str(query),
                "engine": engine,
                "params": params,
                "chunksize": chunksize,
            }
        )
        uuid = params["uuid"]
        if uuid == "empty":
            return iter([pd.DataFrame(columns=["uuid", "systime"])])
        return iter(
            [_rows(uuid, "2024-01-01 00:00", 3), _rows(uuid, "2024-01-01 01:00", 3)]
        )

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    return calls


def _timescale(uuids):
    return TimescaleDBDataAccess(
        start="2024-01-01 00:00:00",
        end="2024-01-01 02:00:00",
        uuids=uuids,
        db_config=DB,
    )


def test_timescale_engine_url(engine_urls):
    _timescale(["a"])
    assert engine_urls == ["postgresql://u:p@h/plant"]


def test_timescale_query_binds_parameters(engine_urls, read_sql_calls):
    loader = _timescale(["temp'1", "pressure"])
    df = loader.fetch_data_as_dataframe()
    assert len(df) == 12
    first = read_sql_calls[0]
    assert ":uuid" in first["sql"] and "temp'1" not in first["sql"]
    assert first["params"] == {
        "uuid": "temp'1",
        "start_ts": "2024-01-01 00:00:00",
        "end_ts": "2024-01-01 02:00:00",
    }
    assert first["chunksize"] == 10000
    assert first["engine"] == "engine:postgresql://u:p@h/plant"


def test_timescale_no_uuids_is_an_empty_frame(engine_urls, read_sql_calls):
    assert _timescale([]).fetch_data_as_dataframe().empty
    assert read_sql_calls == []


def test_timescale_parquet_export_one_file_per_uuid_and_hour(
    engine_urls, read_sql_calls, tmp_path
):
    _timescale(["temp", "empty"]).fetch_data_as_parquet(str(tmp_path))
    files = sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet")
    )
    assert files == ["2024/01/01/00/temp.parquet", "2024/01/01/01/temp.parquet"]
    hour0 = pd.read_parquet(tmp_path / "2024/01/01/00/temp.parquet")
    assert len(hour0) == 3


def test_timescale_parquet_export_merges_chunks_of_one_hour(
    engine_urls, monkeypatch, tmp_path
):
    def same_hour_chunks(query, engine, params=None, chunksize=None):
        return iter(
            [_rows("temp", "2024-01-01 05:00", 2), _rows("temp", "2024-01-01 05:40", 1)]
        )

    monkeypatch.setattr(pd, "read_sql", same_hour_chunks)
    _timescale(["temp"]).fetch_data_as_parquet(str(tmp_path))
    out = pd.read_parquet(tmp_path / "2024/01/01/05/temp.parquet")
    assert len(out) == 3


# --------------------------------------------------------------------------
# S3 proxy
# --------------------------------------------------------------------------

s3fs = pytest.importorskip("s3fs")

from ts_shape.errors import LoaderError  # noqa: E402
from ts_shape.loader.timeseries.s3proxy_parquet_loader import (  # noqa: E402
    S3ProxyDataAccess,
)

S3_CONFIG = {
    "endpoint_url": "http://localhost:9000",
    "key": "k",
    "secret": "s",
    "use_ssl": False,
    "version_aware": False,
    "s3_path_base": "lake/",
}


def _parquet_bytes(uuid: str, hour: str) -> bytes:
    buf = io.BytesIO()
    _rows(uuid, hour, 3).to_parquet(buf, index=False)
    return buf.getvalue()


class FakeS3:
    """An S3 filesystem holding a few hourly parquet files."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.files = {
            "lake/2024/01/01/00/temp.parquet": _parquet_bytes(
                "temp", "2024-01-01 00:00"
            ),
            "lake/2024/01/01/01/temp.parquet": _parquet_bytes(
                "temp", "2024-01-01 01:00"
            ),
            "lake/2024/01/01/01/flow.parquet": _parquet_bytes(
                "flow", "2024-01-01 01:00"
            ),
        }
        self.flaky: dict[str, int] = {}
        self.opened: list[str] = []

    def open(self, path, mode):
        self.opened.append(path)
        if self.flaky.get(path, 0) > 0:
            self.flaky[path] -= 1
            raise ConnectionResetError("connection reset by peer")
        if path not in self.files:
            raise FileNotFoundError(path)
        return io.BytesIO(self.files[path])


@pytest.fixture
def fake_s3(monkeypatch):
    monkeypatch.setattr(s3fs, "S3FileSystem", FakeS3)


def _s3(uuids, end="2024-01-01 01:00:00"):
    return S3ProxyDataAccess(
        start="2024-01-01 00:00:00", end=end, uuids=uuids, s3_config=S3_CONFIG
    )


def test_s3_connection_settings(fake_s3):
    loader = _s3(["temp"])
    assert loader.s3.kwargs == {
        "endpoint_url": "http://localhost:9000",
        "key": "k",
        "secret": "s",
        "use_ssl": False,
        "version_aware": False,
    }
    assert loader.s3_path_base == "lake/"


def test_s3_missing_config_key_is_named(fake_s3):
    config = {k: v for k, v in S3_CONFIG.items() if k != "secret"}
    with pytest.raises(LoaderError, match="secret"):
        S3ProxyDataAccess(start="2024", end="2024", uuids=[], s3_config=config)


def test_s3_dataframe_reads_every_hour_and_skips_missing(fake_s3):
    loader = _s3(["temp", "flow", "temp"])
    df = loader.fetch_data_as_dataframe()
    assert sorted(df["uuid"].unique()) == ["flow", "temp"]
    assert len(df) == 9  # temp 00, temp 01, flow 01; flow 00 does not exist
    paths = [Path(p).as_posix() for p in loader.s3.opened]
    assert "lake/2024/01/01/00/flow.parquet" in paths


def test_s3_retries_a_transient_error(fake_s3):
    loader = _s3(["temp"], end="2024-01-01 00:00:00")
    loader.s3.flaky["lake/2024/01/01/00/temp.parquet"] = 1
    df = loader.fetch_data_as_dataframe()
    assert len(df) == 3
    assert loader.s3.opened.count("lake/2024/01/01/00/temp.parquet") == 2


def test_s3_parquet_export(fake_s3, tmp_path):
    _s3(["temp", "flow"]).fetch_data_as_parquet(str(tmp_path))
    files = sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet")
    )
    assert files == [
        "2024/01/01/00/temp.parquet",
        "2024/01/01/01/flow.parquet",
        "2024/01/01/01/temp.parquet",
    ]


def test_s3_timeslot_paths_are_hourly(fake_s3):
    slots = [p.as_posix() for p in _s3(["temp"])._generate_timeslot_paths()]
    assert slots == ["2024/01/01/00", "2024/01/01/01"]


# --------------------------------------------------------------------------
# metadata database
# --------------------------------------------------------------------------

from ts_shape.loader.metadata import metadata_db_loader  # noqa: E402
from ts_shape.loader.metadata.metadata_db_loader import DatapointDB  # noqa: E402


class _Conn:
    def __init__(self, rows_by_device, queries):
        self.rows_by_device = rows_by_device
        self.queries = queries

    def execute(self, query, params):
        self.queries.append((str(query), params))
        rows = self.rows_by_device[params["device_name"]]
        return type("Result", (), {"fetchall": lambda self: rows})()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def metadata_db(monkeypatch):
    queries: list = []
    urls: list = []
    rows = {
        "press 1": [
            ("u1", "temp", {"unit": "degC"}),
            ("u2", "pressure", {"unit": "bar"}),
        ],
        "press 2": [("u3", "flow", {"unit": "l/min"})],
    }

    def fake_engine(url, *args, **kwargs):
        urls.append(url)
        return type("Engine", (), {"connect": lambda self: _Conn(rows, queries)})()

    monkeypatch.setattr(sqlalchemy, "create_engine", fake_engine)
    monkeypatch.setattr(metadata_db_loader, "create_engine", fake_engine, raising=False)
    return queries, urls


def test_metadata_db_filters_and_exports(metadata_db, tmp_path, caplog):
    queries, urls = metadata_db
    db = DatapointDB(
        ["press 1", "press 2"],
        "user",
        "pw",
        "dbhost",
        output_path=str(tmp_path),
        required_uuid_list=["u1", "u3"],
    )
    assert urls == ["postgresql://user:pw@dbhost:5432/config_repository"]
    assert all("dp.enabled = true" in q for q, _ in queries)
    assert db.get_all_uuids() == {"press 1": ["u1"], "press 2": ["u3"]}
    assert db.get_all_metadata()["press 2"] == [
        {"uuid": "u3", "label": "flow", "config": {"unit": "l/min"}}
    ]
    exported = json.loads((tmp_path / "press_1_data_points.json").read_text())
    assert exported == [{"uuid": "u1", "label": "temp", "config": {"unit": "degC"}}]

    caplog.set_level("INFO", logger=metadata_db_loader.__name__)
    db.display_dataframe(aggregate=True)
    db.display_dataframe("press 1")
    db.display_dataframe("unknown")
    db.display_dataframe()
    text = caplog.text
    assert "Aggregated metadata" in text
    assert "No metadata found for device: unknown" in text
    assert "Metadata for device: press 2" in text


def test_metadata_db_without_filter(metadata_db, tmp_path):
    queries, _ = metadata_db
    db = DatapointDB(
        ["press 1"],
        "user",
        "pw",
        "dbhost",
        output_path=str(tmp_path),
        filter_enabled=False,
    )
    assert "enabled" not in queries[0][0]
    assert db.get_all_uuids() == {"press 1": ["u1", "u2"]}
