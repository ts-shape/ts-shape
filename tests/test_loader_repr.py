"""Loader ``repr`` never shows a secret, and the PostgreSQL URL is built safely.

Everything here runs offline: database engines are replaced, Azure clients
are only constructed (which makes no request), and no fetch is made.
"""

from __future__ import annotations

import importlib.util

import pytest

from ts_shape.loader._utils import (
    create_postgres_engine,
    redact_mapping,
    redact_url,
)

PASSWORD = "p@ss:w/rd%?#&"
SECRET_PARTS = ("p@ss", "w/rd", "SECRETKEY", "sig=abc", "TOKEN123", "s3cr3t")


def _assert_no_secret(text: str) -> None:
    for part in SECRET_PARTS:
        assert part not in text, (part, text)


# ---- redaction helpers --------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com/v1", "https://api.example.com/v1"),
        (
            "https://acct.blob.core.windows.net/c?sv=2022&sig=abc",
            "https://acct.blob.core.windows.net/c?sv=***&sig=***",
        ),
        ("https://u:s3cr3t@host:8080/p", "https://u:***@host:8080/p"),
        ("https://host/p?flag", "https://host/p?flag"),
    ],
)
def test_redact_url(url, expected):
    assert redact_url(url) == expected


def test_redact_mapping_keeps_keys_only():
    assert redact_mapping({"Authorization": "Bearer TOKEN123"}) == {
        "Authorization": "***"
    }
    assert redact_mapping(None) == {}


# ---- PostgreSQL URL -------------------------------------------------------------


def _sqlalchemy():
    return pytest.importorskip("sqlalchemy")


def test_postgres_url_escapes_password_and_sets_driver_and_port():
    _sqlalchemy()
    from sqlalchemy.engine import make_url

    from ts_shape.loader._utils import postgres_url

    url = postgres_url(
        username="reader",
        password=PASSWORD,
        host="db.example.com",
        port=6543,
        database="telemetry",
    )
    assert url.drivername == "postgresql+psycopg2"
    assert url.password == PASSWORD
    assert url.port == 6543
    assert url.host == "db.example.com"
    assert url.database == "telemetry"
    # The rendered string round-trips: the password was escaped.
    rendered = url.render_as_string(hide_password=False)
    assert make_url(rendered).password == PASSWORD
    assert make_url(rendered).host == "db.example.com"
    # str() masks it.
    assert str(url) == "postgresql+psycopg2://reader:***@db.example.com:6543/telemetry"


def test_postgres_url_port_default_and_string_port():
    _sqlalchemy()
    from ts_shape.loader._utils import postgres_url

    common = {"username": "u", "password": "p", "host": "h", "database": "d"}
    assert postgres_url(**common).port == 5432
    assert postgres_url(**common, port="5433").port == 5433
    assert postgres_url(**common, driver="psycopg").drivername == ("postgresql+psycopg")


def test_missing_psycopg2_names_the_postgres_extra():
    _sqlalchemy()
    from ts_shape.loader._utils import postgres_url

    def no_driver(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'psycopg2'")

    url = postgres_url(username="u", password="p", host="h", database="d")
    with pytest.raises(ImportError, match=r"psycopg2-binary.*ts-shape\[postgres\]"):
        create_postgres_engine(no_driver, url)


@pytest.mark.skipif(
    importlib.util.find_spec("psycopg2") is not None, reason="psycopg2 installed"
)
def test_real_create_engine_without_psycopg2_names_the_extra():
    _sqlalchemy()
    from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess

    with pytest.raises(ImportError, match=r"ts-shape\[postgres\]"):
        TimescaleDBDataAccess(
            start="2024-01-01",
            end="2024-01-02",
            uuids=["a"],
            db_config={
                "db_user": "u",
                "db_pass": "p",
                "db_host": "h",
                "db_name": "d",
            },
        )


def test_other_driver_import_error_is_not_rewritten():
    _sqlalchemy()
    from ts_shape.loader._utils import postgres_url

    def no_driver(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'pg8000'")

    url = postgres_url(
        username="u", password="p", host="h", database="d", driver="pg8000"
    )
    with pytest.raises(ModuleNotFoundError, match="pg8000"):
        create_postgres_engine(no_driver, url)


# ---- TimescaleDB ---------------------------------------------------------------


def _capture_engine(monkeypatch, module: str) -> list:
    seen: list = []
    monkeypatch.setattr(
        f"{module}.create_engine", lambda url, **kwargs: seen.append(url) or object()
    )
    return seen


def test_timescale_url_and_repr(monkeypatch):
    _sqlalchemy()
    from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess

    seen = _capture_engine(monkeypatch, "ts_shape.loader.timeseries.timescale_loader")
    loader = TimescaleDBDataAccess(
        start="2024-01-01",
        end="2024-01-02",
        uuids=["a", "b"],
        db_config={
            "db_user": "reader",
            "db_pass": PASSWORD,
            "db_host": "db.example.com",
            "db_port": 6543,
            "db_name": "telemetry",
        },
    )
    (url,) = seen
    assert url.password == PASSWORD
    assert url.port == 6543
    assert url.drivername == "postgresql+psycopg2"
    text = repr(loader)
    _assert_no_secret(text)
    assert text == (
        "TimescaleDBDataAccess(start='2024-01-01', end='2024-01-02', uuid_count=2, "
        "url='postgresql+psycopg2://reader:***@db.example.com:6543/telemetry')"
    )


def test_timescale_default_port(monkeypatch):
    _sqlalchemy()
    from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess

    seen = _capture_engine(monkeypatch, "ts_shape.loader.timeseries.timescale_loader")
    TimescaleDBDataAccess(
        start="2024-01-01",
        end="2024-01-02",
        uuids=["a"],
        db_config={"db_user": "u", "db_pass": "p", "db_host": "h", "db_name": "n"},
    )
    assert seen[0].port == 5432
    assert seen[0].database == "n"


# ---- DatapointDB ---------------------------------------------------------------


def test_datapoint_db_url_and_repr(monkeypatch, tmp_path):
    _sqlalchemy()
    from ts_shape.loader.metadata.metadata_db_loader import DatapointDB

    monkeypatch.setattr(DatapointDB, "_db_access", lambda self: None)
    db = DatapointDB(
        ["Press 1"],
        "reader",
        PASSWORD,
        "db.example.com",
        output_path=str(tmp_path),
        db_port=6543,
        db_name="configs",
    )
    assert db.url.password == PASSWORD
    assert db.url.port == 6543
    assert db.url.database == "configs"
    text = repr(db)
    _assert_no_secret(text)
    assert text == (
        "DatapointDB(device_names=['Press 1'], "
        "url='postgresql+psycopg2://reader:***@db.example.com:6543/configs')"
    )


def test_datapoint_db_defaults(monkeypatch):
    _sqlalchemy()
    from ts_shape.loader.metadata.metadata_db_loader import DatapointDB

    monkeypatch.setattr(DatapointDB, "_db_access", lambda self: None)
    db = DatapointDB(["d"], "u", "p", "h")
    assert db.url.port == 5432
    assert db.url.database == "config_repository"


# ---- API loaders ---------------------------------------------------------------


def test_datapoint_api_repr(monkeypatch):
    pytest.importorskip("requests")
    from ts_shape.loader.metadata.metadata_api_loader import DatapointAPI

    monkeypatch.setattr(DatapointAPI, "_api_access", lambda self: None)
    api = DatapointAPI(
        ["Press 1"], "https://datadash.example.com/", api_token="TOKEN123"
    )
    text = repr(api)
    _assert_no_secret(text)
    assert text == (
        "DatapointAPI(device_names=['Press 1'], "
        "base_url='https://datadash.example.com', api_token='***')"
    )


def test_energy_api_repr():
    pytest.importorskip("requests")
    from ts_shape.loader.timeseries.energy_api_loader import EnergyAPILoader

    loader = EnergyAPILoader(
        "https://api.example.com/v1?api_key=SECRETKEY",
        headers={"Authorization": "Bearer TOKEN123"},
    )
    text = repr(loader)
    _assert_no_secret(text)
    assert text == (
        "EnergyAPILoader(base_url='https://api.example.com/v1?api_key=***', "
        "headers={'Authorization': '***'}, timeout=30)"
    )


# ---- S3 ------------------------------------------------------------------------


def test_s3proxy_repr(monkeypatch):
    s3fs = pytest.importorskip("s3fs")
    from ts_shape.loader.timeseries.s3proxy_parquet_loader import S3ProxyDataAccess

    monkeypatch.setattr(s3fs, "S3FileSystem", lambda **kwargs: object())
    loader = S3ProxyDataAccess(
        start="2024-01-01",
        end="2024-01-02",
        uuids=["a"],
        s3_config={
            "endpoint_url": "https://s3.example.com",
            "key": "SECRETKEY",
            "secret": "s3cr3t",
            "use_ssl": True,
            "version_aware": False,
            "s3_path_base": "bucket/telemetry",
        },
    )
    text = repr(loader)
    _assert_no_secret(text)
    assert text == (
        "S3ProxyDataAccess(start='2024-01-01', end='2024-01-02', uuid_count=1, "
        "endpoint_url='https://s3.example.com', s3_path_base='bucket/telemetry')"
    )


# ---- Azure ---------------------------------------------------------------------

_CONN = (
    "DefaultEndpointsProtocol=https;AccountName=acct;"
    "AccountKey=U0VDUkVUS0VZ;EndpointSuffix=core.windows.net"
)
_SAS = "https://acct.blob.core.windows.net/data?sv=2022-11-02&sig=abc"


@pytest.mark.parametrize(
    "cls_path",
    [
        "ts_shape.loader.timeseries.azure_blob_loader.AzureBlobParquetLoader",
        "ts_shape.loader.timeseries.azure_blob_loader.AzureBlobFlexibleFileLoader",
        "ts_shape.loader.timeseries.azure_blob_energy_loader.AzureBlobEnergyLoader",
    ],
)
@pytest.mark.parametrize("auth", ["connection_string", "sas_url"])
def test_azure_repr(cls_path, auth):
    pytest.importorskip("azure.storage.blob")
    import importlib

    module, _, name = cls_path.rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    if auth == "connection_string":
        loader = cls("data", connection_string=_CONN, prefix="raw/")
    else:
        loader = cls(sas_url=_SAS, prefix="raw/")
    text = repr(loader)
    _assert_no_secret(text)
    assert "U0VDUkVUS0VZ" not in text
    assert text == f"{name}(account_name='acct', container_name='data', prefix='raw/')"
