import pytest
import pandas as pd  # type: ignore

s3fs = pytest.importorskip("s3fs")

from ts_shape.loader.timeseries.s3proxy_parquet_loader import S3ProxyDataAccess


class DummyS3:
    def __init__(self, *args, **kwargs):
        pass

    def open(self, path, mode):
        # not used in the test because we patch _fetch_parquet
        raise FileNotFoundError


def test_s3proxy_fetch_data_as_dataframe_monkeypatched(monkeypatch):
    # Patch S3FileSystem to avoid real connections
    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3)

    loader = S3ProxyDataAccess(
        start_timestamp="2024-01-01 00:00:00",
        end_timestamp="2024-01-01 02:00:00",
        uuids=["u1", "u2"],
        s3_config={
            "endpoint_url": "http://localhost",
            "key": "k",
            "secret": "s",
            "use_ssl": False,
            "version_aware": False,
            "s3_path_base": "bucket/",
        },
    )

    # Patch _fetch_parquet to return small frames
    calls = []

    def fake_fetch(uuid, ts_dir):
        calls.append((uuid, str(ts_dir)))
        return pd.DataFrame({"uuid": [uuid], "systime": [pd.Timestamp("2024-01-01")]})

    monkeypatch.setattr(loader, "_fetch_parquet", fake_fetch)

    df = loader.fetch_data_as_dataframe()
    # Non-empty and contains both uuids
    assert not df.empty
    assert set(df["uuid"].unique()) == {"u1", "u2"}
    assert len(calls) > 0
