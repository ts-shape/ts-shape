from pathlib import Path
import pandas as pd  # type: ignore
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
