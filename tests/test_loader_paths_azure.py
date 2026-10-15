"""Azure Blob loaders against an in-memory container.

The loaders are built through their public constructors with
``azure.storage.blob.ContainerClient`` replaced by a fake, so every
authentication branch runs; the fake serves real parquet / JSON / CSV bytes.
"""

from __future__ import annotations

import io
import json
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

blob = pytest.importorskip("azure.storage.blob")
pytest.importorskip("pyarrow")
from azure.core.exceptions import (  # noqa: E402
    HttpResponseError,
    ResourceNotFoundError,
)

from ts_shape.errors import DataQualityWarning, LoaderConfigWarning  # noqa: E402
from ts_shape.loader.timeseries.azure_blob_energy_loader import (  # noqa: E402
    AzureBlobEnergyLoader,
)
from ts_shape.loader.timeseries.azure_blob_loader import (  # noqa: E402
    AzureBlobFlexibleFileLoader,
    AzureBlobParquetLoader,
)


def _parquet(uuid: str, start: str, n: int = 4) -> bytes:
    frame = pd.DataFrame(
        {
            "systime": pd.date_range(start, periods=n, freq="15min"),
            "uuid": uuid,
            "value_double": [float(i) for i in range(n)],
        }
    )
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


BLOBS = {
    "pq/2024/01/01/08/temp.parquet": _parquet("temp", "2024-01-01 08:00"),
    "pq/2024/01/01/08/pressure.parquet": _parquet("pressure", "2024-01-01 08:00"),
    "pq/2024/01/01/09/temp.parquet": _parquet("temp", "2024-01-01 09:00"),
    "pq/2024/01/01/09/notes.json": json.dumps({"shift": "early"}).encode(),
    "pq/2024/01/01/09/broken.parquet": b"not parquet",
}


class FakeContainer:
    """Stands in for ``ContainerClient``; records how it was built."""

    built: list[tuple[str, dict]] = []

    def __init__(self, blobs=None, *, list_fails=False, **kwargs):
        self.blobs = dict(BLOBS if blobs is None else blobs)
        self.list_fails = list_fails
        FakeContainer.built.append(("init", kwargs))

    @classmethod
    def from_container_url(cls, url):
        inst = cls()
        cls.built.append(("sas", {"url": url}))
        return inst

    @classmethod
    def from_connection_string(cls, conn_str, container_name):
        if "AccountName" not in conn_str:
            raise ValueError("bad connection string")
        inst = cls()
        cls.built.append(("conn", {"container_name": container_name}))
        return inst

    def list_blobs(self, name_starts_with=None):
        if self.list_fails:
            raise HttpResponseError("This request is not authorized to list blobs")
        return [
            SimpleNamespace(name=n)
            for n in sorted(self.blobs)
            if not name_starts_with or n.startswith(name_starts_with)
        ]

    def download_blob(self, name):
        if name not in self.blobs:
            raise ResourceNotFoundError(f"{name} not found")
        content = self.blobs[name]
        return SimpleNamespace(readall=lambda: content)


@pytest.fixture(autouse=True)
def fake_container(monkeypatch):
    FakeContainer.built = []
    monkeypatch.setattr(blob, "ContainerClient", FakeContainer)
    return FakeContainer


# --------------------------------------------------------------------------
# authentication branches
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls", [AzureBlobParquetLoader, AzureBlobFlexibleFileLoader, AzureBlobEnergyLoader]
)
class TestAuthentication:
    def test_sas_url(self, cls):
        cls(sas_url="https://acct.blob.core.windows.net/c?sig=x")
        assert FakeContainer.built[-1][0] == "sas"

    def test_connection_string_that_is_a_url_is_a_sas_url(self, cls):
        cls(connection_string="  https://acct.blob.core.windows.net/c?sig=x")
        assert FakeContainer.built[-1] == (
            "sas",
            {"url": "  https://acct.blob.core.windows.net/c?sig=x"},
        )

    def test_connection_string(self, cls):
        loader = cls("c", connection_string="AccountName=a;AccountKey=k", prefix="pq")
        assert FakeContainer.built[-1] == ("conn", {"container_name": "c"})
        assert loader.prefix == "pq"

    def test_invalid_connection_string_explains_the_format(self, cls):
        with pytest.raises(ValueError, match="AccountName=...' and 'AccountKey"):
            cls("c", connection_string="garbage")

    def test_connection_string_needs_container(self, cls):
        with pytest.raises(ValueError, match="container_name is required"):
            cls(connection_string="AccountName=a;AccountKey=k")

    def test_nothing_given(self, cls):
        with pytest.raises(ValueError, match="Provide one of"):
            cls("c")

    def test_aad_credential(self, cls):
        token = object()
        loader = cls("c", account_url="https://a.blob", credential=token, max_workers=0)
        assert FakeContainer.built[-1] == (
            "init",
            {
                "account_url": "https://a.blob",
                "container_name": "c",
                "credential": token,
            },
        )
        assert loader.max_workers == 1  # below 1 is treated as 1

    def test_credential_without_account_url(self, cls):
        with pytest.raises(ValueError, match="account_url must be provided"):
            cls("c", credential=object())

    def test_account_url_without_credential(self, cls):
        with pytest.raises(ValueError, match="credential must be provided"):
            cls("c", account_url="https://a.blob")

    def test_account_url_without_container(self, cls):
        with pytest.raises(ValueError, match="container_name is required"):
            cls(account_url="https://a.blob", credential=object())


@pytest.mark.parametrize("cls", [AzureBlobParquetLoader, AzureBlobEnergyLoader])
def test_from_account_name_builds_the_account_url(cls):
    token = object()
    loader = cls.from_account_name(
        "plant", "c", credential=token, endpoint_suffix="blob.core.chinacloudapi.cn"
    )
    assert isinstance(loader, cls)
    assert FakeContainer.built[-1][1]["account_url"] == (
        "https://plant.blob.core.chinacloudapi.cn"
    )


@pytest.mark.parametrize("cls", [AzureBlobParquetLoader, AzureBlobEnergyLoader])
def test_from_account_name_requires_a_credential(cls):
    with pytest.raises(ValueError, match="credential must be provided"):
        cls.from_account_name("plant", "c")


# --------------------------------------------------------------------------
# parquet loader query paths
# --------------------------------------------------------------------------


def _parquet_loader(**container):
    loader = AzureBlobParquetLoader(
        sas_url="https://a/c?sig=x", prefix="pq", max_workers=2
    )
    loader.container_client = FakeContainer(**container)
    return loader


def test_load_files_by_uuid_trims_to_the_range():
    out = _parquet_loader().load_files_by_time_range_and_uuids(
        "2024-01-01 08:30", "2024-01-01 09:15", ["temp"]
    )
    assert set(out["uuid"]) == {"temp"}
    assert out["systime"].min() == pd.Timestamp("2024-01-01 08:30")
    assert out["systime"].max() == pd.Timestamp("2024-01-01 09:15")


def test_load_files_by_uuid_keeps_whole_hours_without_trim():
    out = _parquet_loader().load_files_by_time_range_and_uuids(
        "2024-01-01 08:30", "2024-01-01 08:31", ["{TEMP} "], trim_to_range=False
    )
    assert len(out) == 4  # the whole 08:00 file; "{TEMP} " is cleaned and lowercased


def test_load_files_by_uuid_without_uuids_is_empty():
    assert (
        _parquet_loader().load_files_by_time_range_and_uuids("2024", "2024", []).empty
    )


def test_load_files_by_uuid_none_found_warns():
    with pytest.warns(LoaderConfigWarning, match="No matching .parquet blobs"):
        out = _parquet_loader().load_files_by_time_range_and_uuids(
            "2024-01-01 08:00", "2024-01-01 09:00", ["flow"]
        )
    assert out.empty


def test_load_files_by_uuid_guesses_names_when_listing_fails():
    loader = _parquet_loader(list_fails=True)
    with pytest.warns(LoaderConfigWarning, match="listing the hour folders failed"):
        out = loader.load_files_by_time_range_and_uuids(
            "2024-01-01 08:00", "2024-01-01 10:00", ["pressure"]
        )
    # 09:00 and 10:00 have no pressure blob; guessed names that are missing
    # are skipped quietly.
    assert set(out["uuid"]) == {"pressure"}
    assert len(out) == 4


def test_load_files_by_uuid_all_unreadable_warns():
    blobs = {"pq/2024/01/01/08/temp.parquet": b"corrupt"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _parquet_loader(blobs=blobs).load_files_by_time_range_and_uuids(
            "2024-01-01 08:00", "2024-01-01 08:30", ["temp"]
        )
    assert out.empty
    kinds = {type(w.message) for w in caught}
    assert {DataQualityWarning, LoaderConfigWarning} <= kinds


def test_stream_files_by_uuid_yields_each_blob():
    loader = _parquet_loader()
    got = dict(
        loader.stream_files_by_time_range_and_uuids(
            "2024-01-01 08:00", "2024-01-01 09:59", ["temp", "pressure"]
        )
    )
    assert sorted(got) == [
        "pq/2024/01/01/08/pressure.parquet",
        "pq/2024/01/01/08/temp.parquet",
        "pq/2024/01/01/09/temp.parquet",
    ]
    assert all(not frame.empty for frame in got.values())


def test_stream_files_by_uuid_without_uuids_yields_nothing():
    assert (
        list(_parquet_loader().stream_files_by_time_range_and_uuids("2024", "2024", []))
        == []
    )


def test_stream_files_by_uuid_out_of_range_warns():
    with pytest.warns(LoaderConfigWarning, match="Yielded no DataFrames"):
        got = list(
            _parquet_loader().stream_files_by_time_range_and_uuids(
                "2024-02-01 08:00",
                "2024-02-01 09:00",
                ["temp"],
                filters=[("value_double", ">", 0)],
            )
        )
    assert got == []


def test_load_by_time_range_reports_broken_blob():
    with pytest.warns(DataQualityWarning, match="broken.parquet"):
        out = _parquet_loader().load_by_time_range(
            "2024-01-01 08:00", "2024-01-01 09:59"
        )
    assert set(out["uuid"]) == {"temp", "pressure"}


def test_load_by_time_range_all_broken_warns_empty():
    blobs = {"pq/2024/01/01/09/broken.parquet": b"x"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _parquet_loader(blobs=blobs).load_by_time_range(
            "2024-01-01 09:00", "2024-01-01 09:30", filters=[("value_double", ">", 0)]
        )
    assert out.empty
    assert any("filters argument may be too strict" in str(w.message) for w in caught)


def test_list_structure_and_limit():
    loader = _parquet_loader()
    full = loader.list_structure()
    assert full["folders"] == ["pq/2024/01/01/08/", "pq/2024/01/01/09/"]
    assert "pq/2024/01/01/09/notes.json" not in full["files"]
    every = loader.list_structure(parquet_only=False, limit=2)
    assert len(every["files"]) == 2


def test_time_search_detail_empty_range():
    detail = _parquet_loader()._time_search_detail([])
    assert "0 hour-prefixes" in detail


# --------------------------------------------------------------------------
# flexible loader
# --------------------------------------------------------------------------


def _flexible(**container):
    loader = AzureBlobFlexibleFileLoader(sas_url="https://a/c?sig=x", prefix="pq")
    loader.container_client = FakeContainer(**container)
    return loader


def test_flexible_lists_by_extension_and_limit():
    loader = _flexible()
    names = loader.list_files_by_time_range(
        "2024-01-01 08:00",
        "2024-01-01 09:00",
        extensions=["JSON", " ", "parquet"],
        limit=2,
    )
    assert len(names) == 2
    names = list(
        loader.iter_file_names_by_time_range(
            "2024-01-01 09:00", "2024-01-01 09:00", extensions=[".json"]
        )
    )
    assert names == ["pq/2024/01/01/09/notes.json"]


def test_flexible_fetch_parses_and_keeps_raw_bytes_on_parse_errors():
    loader = _flexible()
    with pytest.warns(DataQualityWarning, match="broken.parquet"):
        got = loader.fetch_files_by_time_range(
            "2024-01-01 09:00", "2024-01-01 09:00", parse=True
        )
    assert got["pq/2024/01/01/09/notes.json"] == {"shift": "early"}
    assert isinstance(got["pq/2024/01/01/09/temp.parquet"], pd.DataFrame)
    assert got["pq/2024/01/01/09/broken.parquet"] == b"not parquet"


def test_flexible_fetch_nothing_in_range_is_empty():
    assert _flexible().fetch_files_by_time_range("2025-01-01", "2025-01-01") == {}


def test_flexible_stream_by_basenames():
    loader = _flexible()
    got = dict(
        loader.stream_files_by_time_range_and_basenames(
            "2024-01-01 08:00",
            "2024-01-01 09:00",
            ["temp.parquet", "notes.json", " "],
            extensions=["parquet"],
        )
    )
    assert sorted(got) == [
        "pq/2024/01/01/08/temp.parquet",
        "pq/2024/01/01/09/temp.parquet",
    ]
    assert all(isinstance(v, bytes) for v in got.values())


def test_flexible_stream_by_time_range_parses():
    loader = _flexible()
    got = dict(
        loader.stream_files_by_time_range(
            "2024-01-01 09:00", "2024-01-01 09:00", extensions=["json"], parse=True
        )
    )
    assert got == {"pq/2024/01/01/09/notes.json": {"shift": "early"}}


def test_parser_registry_roundtrip():
    def parse_csv(content, name):
        return pd.read_csv(io.BytesIO(content))

    AzureBlobFlexibleFileLoader.register_parser("CSV", parse_csv)
    try:
        assert ".csv" in AzureBlobFlexibleFileLoader.available_parsers()
        parsed = AzureBlobFlexibleFileLoader._parse_bytes("a/b.csv", b"x,y\n1,2\n")
        assert parsed["y"].tolist() == [2]
    finally:
        AzureBlobFlexibleFileLoader.unregister_parser("csv")
    assert ".csv" not in AzureBlobFlexibleFileLoader.available_parsers()
    assert AzureBlobFlexibleFileLoader._parse_bytes("a/b.csv", b"x") == b"x"
    assert AzureBlobFlexibleFileLoader._parse_bytes("a/noext", b"x") == b"x"
    AzureBlobFlexibleFileLoader.unregister_parser(".never-registered")


def test_normalize_exts():
    norm = AzureBlobFlexibleFileLoader._normalize_exts
    assert norm(None) is None
    assert norm(["", " "]) is None
    assert norm(["CSV", ".json"]) == {".csv", ".json"}


def test_flexible_hour_prefix_adds_the_slash():
    loader = _flexible()
    assert loader._hour_prefix(pd.Timestamp("2024-03-05 07:00")) == "pq/2024/03/05/07/"
