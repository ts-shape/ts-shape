from pathlib import Path
import pandas as pd  # type: ignore
import pytest

from ts_shape.errors import LoaderConfigWarning, LoaderError
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader


def test_parquet_loader_load_all_and_filters(monkeypatch, tmp_path):
    # Create fake file list
    base = tmp_path
    files = [
        base / "2024/01/01/00" / "u1.parquet",
        base / "2024/01/01/01" / "u2.parquet",
        base / "2024/01/02/00" / "u3.parquet",
    ]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()

    # Patch readers
    def fake_get_parquet_files(_):
        return [Path(str(p)) for p in files]

    def fake_read_parquet(p):  # p is a Path
        name = Path(p).stem
        return pd.DataFrame({"uuid": [name]})

    monkeypatch.setattr(
        ParquetLoader,
        "_get_parquet_files",
        classmethod(lambda cls, p: fake_get_parquet_files(p)),
    )
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    # load all
    all_df = ParquetLoader.load_all_files(str(base))
    assert set(all_df["uuid"]) == {"u1", "u2", "u3"}

    # time range: include only first two
    start = pd.Timestamp("2024-01-01 00:00:00")
    end = pd.Timestamp("2024-01-01 23:59:59")
    rng_df = ParquetLoader.load_by_time_range(str(base), start, end)
    assert set(rng_df["uuid"]) == {"u1", "u2"}

    # by uuid list
    uu_df = ParquetLoader.load_by_uuid_list(str(base), ["u2"])
    assert set(uu_df["uuid"]) == {"u2"}

    both_df = ParquetLoader.load_files_by_time_range_and_uuids(
        str(base), start, end, ["u1", "u3"]
    )
    assert set(both_df["uuid"]) == {"u1"}


def test_load_all_files_raises_on_missing_path(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(LoaderError, match="does not exist"):
        ParquetLoader.load_all_files(str(missing))


def test_init_raises_on_missing_path(tmp_path):
    with pytest.raises(LoaderError):
        ParquetLoader(str(tmp_path / "nope"))


def test_load_all_files_warns_when_no_files_match(tmp_path):
    # Path exists but contains no parquet files.
    with pytest.warns(LoaderConfigWarning, match="No parquet files matched"):
        out = ParquetLoader.load_all_files(str(tmp_path))
    assert out.empty


def test_load_all_files_retries_transient_read(monkeypatch, tmp_path):
    f = tmp_path / "2024/01/01/00" / "u1.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch()

    monkeypatch.setattr(
        ParquetLoader,
        "_get_parquet_files",
        classmethod(lambda cls, p: [Path(str(f))]),
    )

    attempts = {"n": 0}

    def flaky_read(_path):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("transient mount glitch")
        return pd.DataFrame({"uuid": ["u1"]})

    monkeypatch.setattr(pd, "read_parquet", flaky_read)
    # Avoid real backoff sleeps in the test.
    monkeypatch.setattr("ts_shape.loader._utils.time.sleep", lambda _: None)

    out = ParquetLoader.load_all_files(str(tmp_path))
    assert set(out["uuid"]) == {"u1"}
    assert attempts["n"] == 2  # failed once, then succeeded
