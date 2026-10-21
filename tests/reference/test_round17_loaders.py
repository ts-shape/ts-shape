"""Round 17 (PR J, part c): loader time ranges, folders and record shapes.

Covers UTC hour folders and row trims of the S3 proxy loader, the empty
result when no file exists, hour patterns without ``{H}`` (one listing per
day, not 24), naive bounds read as UTC in ``trim_to_time_range``, the row
trim of ``ParquetLoader`` and both energy loaders, UTC day folders of the
Databricks energy loader, ``float64`` energy values, the Timescale parquet
export without a helper column, epoch units in ``parse_utc_times``, the
columnar ``MetadataJsonLoader`` input with uneven or mixed index keys, and
``DatapointAPI`` records without ``enabled`` / ``config``.

Each test states the rule it checks; the expected values follow from that
rule (hand-counted rows), not from the implementation. No network: object
stores are in-memory fakes, local layouts are temp dirs.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ts_shape.loader._utils import (
    SIGNAL_COLUMNS,
    epoch_unit,
    is_date_only,
    parse_utc_times,
    trim_to_day_range,
    trim_to_time_range,
)
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.timeseries.azure_blob_energy_loader import AzureBlobEnergyLoader
from ts_shape.loader.timeseries.azure_blob_loader import (
    AzureBlobFlexibleFileLoader,
    AzureBlobParquetLoader,
)
from ts_shape.loader.timeseries.databricks_spark_parquet_loader import (
    DatabricksSparkParquetLoader,
)
from ts_shape.loader.timeseries.databricks_unity_energy_loader import (
    DatabricksUnityEnergyLoader,
)
from ts_shape.loader.timeseries.databricks_unity_parquet_loader import (
    DatabricksUnityParquetLoader,
)
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader

pytest.importorskip("pyarrow")

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def _quarter_hours(hour: str, uuid: str = "t", n: int = 4) -> pd.DataFrame:
    """``n`` rows 15 minutes apart from ``hour`` (naive UTC)."""
    return pd.DataFrame(
        {
            "systime": pd.date_range(hour, periods=n, freq="15min"),
            "uuid": uuid,
            "value_double": [float(i) for i in range(n)],
        }
    )


class FakeContainer:
    """In-memory stand-in for ``azure.storage.blob.ContainerClient``."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs
        self.listed: list[str | None] = []

    def list_blobs(self, name_starts_with=None):
        self.listed.append(name_starts_with)
        return [
            SimpleNamespace(name=name)
            for name in sorted(self.blobs)
            if not name_starts_with or name.startswith(name_starts_with)
        ]

    def download_blob(self, name, **_kwargs):
        content = self.blobs[name]
        return SimpleNamespace(readall=lambda: content)


def _azure(cls, blobs: dict[str, bytes], hour_pattern: str):
    loader = object.__new__(cls)
    loader.container_client = FakeContainer(blobs)
    loader.prefix = ""
    loader.hour_pattern = hour_pattern
    loader.max_workers = 2
    return loader


#: One day of hourly rows, 00:00 .. 23:00 UTC.
DAY = pd.DataFrame(
    {
        "systime": pd.date_range("2024-01-01", periods=24, freq="h"),
        "uuid": "t",
        "value_double": 1.0,
    }
)

# ---------------------------------------------------------------------------
# 1, 2: S3 proxy
# ---------------------------------------------------------------------------

S3_CONFIG = {
    "endpoint_url": "http://localhost",
    "key": "k",
    "secret": "s",
    "use_ssl": False,
    "version_aware": False,
    "s3_path_base": "lake/",
}


class FakeS3:
    files: dict[str, bytes] = {}

    def __init__(self, *args, **kwargs):
        pass

    def open(self, path, mode):
        if path not in self.files:
            raise FileNotFoundError(path)
        return io.BytesIO(self.files[path])


@pytest.fixture
def s3(monkeypatch):
    s3fs = pytest.importorskip("s3fs")
    monkeypatch.setattr(s3fs, "S3FileSystem", FakeS3)
    FakeS3.files = {}
    from ts_shape.loader.timeseries.s3proxy_parquet_loader import S3ProxyDataAccess

    return S3ProxyDataAccess


def test_s3_reads_floored_utc_hours_and_trims_rows(s3):
    """09:30 .. 11:15 needs the folders 09, 10 and 11; the rows kept are the
    quarter hours inside the range: 09:30, 09:45, 10:00..10:45, 11:00,
    11:15 = 8 rows."""
    FakeS3.files = {
        f"lake/2024/01/01/{h}/t.parquet": _parquet_bytes(
            _quarter_hours(f"2024-01-01 {h}:00")
        )
        for h in ("09", "10", "11")
    }
    loader = s3("2024-01-01 09:30", "2024-01-01 11:15", ["t"], S3_CONFIG)
    slots = [p.as_posix() for p in loader._generate_timeslot_paths()]
    assert slots == ["2024/01/01/09", "2024/01/01/10", "2024/01/01/11"]
    df = loader.fetch_data_as_dataframe()
    assert len(df) == 8
    assert df["systime"].min() == pd.Timestamp("2024-01-01 09:30")
    assert df["systime"].max() == pd.Timestamp("2024-01-01 11:15")


def test_s3_aware_bounds_pick_utc_folders(s3):
    """10:30 .. 11:30 Berlin (UTC+1 in winter) is 09:30 .. 10:30 UTC."""
    berlin = "Europe/Berlin"
    loader = s3(
        pd.Timestamp("2024-01-01 10:30", tz=berlin),
        pd.Timestamp("2024-01-01 11:30", tz=berlin),
        ["t"],
        S3_CONFIG,
    )
    slots = [p.as_posix() for p in loader._generate_timeslot_paths()]
    assert slots == ["2024/01/01/09", "2024/01/01/10"]


def test_s3_parquet_export_trims_rows(s3, tmp_path):
    """The export keeps only the rows in [start, end]: 09:30 and 09:45."""
    FakeS3.files = {
        "lake/2024/01/01/09/t.parquet": _parquet_bytes(
            _quarter_hours("2024-01-01 09:00")
        )
    }
    s3("2024-01-01 09:30", "2024-01-01 09:50", ["t"], S3_CONFIG).fetch_data_as_parquet(
        str(tmp_path)
    )
    out = pd.read_parquet(tmp_path / "2024/01/01/09/t.parquet")
    assert out["systime"].dt.minute.tolist() == [30, 45]


def test_s3_no_files_gives_an_empty_signal_frame(s3):
    """No file in any folder: an empty long-format frame, not a concat error."""
    df = s3(
        "2024-01-01", "2024-01-01 02:00", ["t", "u"], S3_CONFIG
    ).fetch_data_as_dataframe()
    assert df.empty
    assert tuple(df.columns) == SIGNAL_COLUMNS


# ---------------------------------------------------------------------------
# 3: hour patterns without {H}
# ---------------------------------------------------------------------------


def test_azure_parquet_day_pattern_lists_each_day_once():
    """A day folder holds 24 rows; a 24-hour range must return them once."""
    loader = _azure(
        AzureBlobParquetLoader,
        {"2024/01/01/t.parquet": _parquet_bytes(DAY)},
        "{Y}/{m}/{d}/",
    )
    start, end = "2024-01-01 00:00", "2024-01-01 23:59"
    assert len(loader.load_by_time_range(start, end)) == 24
    assert sum(len(df) for _, df in loader.stream_by_time_range(start, end)) == 24
    assert len(loader.load_files_by_time_range_and_uuids(start, end, ["t"])) == 24
    streamed = loader.stream_files_by_time_range_and_uuids(start, end, ["t"])
    assert sum(len(df) for _, df in streamed) == 24
    # one listing of the day folder per call, not 24
    assert loader.container_client.listed == ["2024/01/01/"] * 4


def test_azure_flexible_day_pattern_lists_each_file_once():
    """One file per day folder, listed once per day, not once per hour."""
    loader = _azure(
        AzureBlobFlexibleFileLoader,
        {"2024/01/01/a.json": b"{}", "2024/01/02/b.json": b"{}"},
        "{Y}/{m}/{d}/",
    )
    start, end = "2024-01-01 00:00", "2024-01-02 23:00"
    assert loader.list_files_by_time_range(start, end) == [
        "2024/01/01/a.json",
        "2024/01/02/b.json",
    ]
    assert len(list(loader.iter_file_names_by_time_range(start, end))) == 2
    fetched = loader.fetch_files_by_time_range_and_basenames(start, end, ["a.json"])
    assert list(fetched) == ["2024/01/01/a.json"]
    streamed = loader.stream_files_by_time_range_and_basenames(start, end, ["a.json"])
    assert [name for name, _ in streamed] == ["2024/01/01/a.json"]


def test_databricks_parquet_day_pattern_reads_each_file_once(tmp_path):
    """The same 24-row day file through the Unity Volume loader."""
    folder = tmp_path / "2024/01/01"
    folder.mkdir(parents=True)
    DAY.to_parquet(folder / "t.parquet")
    loader = DatabricksUnityParquetLoader(
        volume_path=str(tmp_path), hour_pattern="{Y}/{m}/{d}/"
    )
    start, end = "2024-01-01 00:00", "2024-01-01 23:59"
    assert len(loader.load_by_time_range(start, end)) == 24
    assert sum(len(df) for _, df in loader.stream_by_time_range(start, end)) == 24
    assert len(loader.load_files_by_time_range_and_uuids(start, end, ["t"])) == 24
    streamed = loader.stream_files_by_time_range_and_uuids(start, end, ["t"])
    assert sum(len(df) for _, df in streamed) == 24


def test_databricks_spark_day_pattern_builds_each_path_once():
    """A ``%Y/%m/%d`` pattern over one day gives one path per uuid."""
    loader = DatabricksSparkParquetLoader(
        base_path="/mnt/lake", hour_pattern="%Y/%m/%d", spark=object()
    )
    paths = loader.build_paths("2024-01-01 00:00", "2024-01-01 23:00", ["t"])
    assert paths == ["/mnt/lake/2024/01/01/t.parquet"]


# ---------------------------------------------------------------------------
# 4: trim_to_time_range reads naive bounds as UTC
# ---------------------------------------------------------------------------


def test_berlin_column_naive_bounds_are_utc(tmp_path):
    """Rows every 30 min from 08:00 to 13:30 UTC, stored in Berlin time.
    The naive range 10:00 .. 12:00 is UTC (as the folder selection reads it):
    10:00, 10:30, 11:00, 11:30, 12:00 UTC = 5 rows."""
    times = pd.date_range(
        "2024-01-01 08:00", "2024-01-01 13:30", freq="30min", tz="UTC"
    )
    frame = pd.DataFrame(
        {"systime": times.tz_convert("Europe/Berlin"), "uuid": "t", "value_double": 1.0}
    )
    for hour, part in frame.groupby(times.floor("h")):
        folder = tmp_path / hour.strftime("%Y/%m/%d/%H")
        folder.mkdir(parents=True)
        part.to_parquet(folder / "t.parquet")
    loader = DatabricksUnityParquetLoader(volume_path=str(tmp_path))
    out = loader.load_by_time_range("2024-01-01 10:00", "2024-01-01 12:00")
    utc = out["systime"].dt.tz_convert("UTC")
    assert len(out) == 5
    assert utc.min() == pd.Timestamp("2024-01-01 10:00", tz="UTC")
    assert utc.max() == pd.Timestamp("2024-01-01 12:00", tz="UTC")
    assert str(out["systime"].dt.tz) == "Europe/Berlin"  # the column keeps its zone


def test_trim_text_times_with_mixed_offsets():
    """10:00+01:00 is 09:00 UTC, 10:00+02:00 is 08:00 UTC, 12:00Z is 12:00 UTC;
    08:00 .. 09:00 UTC keeps the first two."""
    df = pd.DataFrame(
        {
            "systime": [
                "2024-01-01T10:00:00+01:00",
                "2024-01-01T10:00:00+02:00",
                "2024-01-01T12:00:00+00:00",
            ],
            "v": [1, 2, 3],
        }
    )
    out = trim_to_time_range(df, "2024-01-01 08:00", "2024-01-01 09:00")
    assert out["v"].tolist() == [1, 2]


def test_trim_ambiguous_and_nonexistent_naive_bounds_on_dst_column():
    """02:30 on 27 Oct is ambiguous in Berlin, 02:30 on 31 Mar does not exist;
    as UTC bounds both are plain instants."""
    autumn = pd.DataFrame(
        {
            "systime": pd.date_range(
                "2024-10-27 00:00", periods=6, freq="h", tz="UTC"
            ).tz_convert("Europe/Berlin")
        }
    )
    # 00:00 .. 05:00 UTC; 02:30 .. 04:00 UTC keeps 03:00 and 04:00
    assert len(trim_to_time_range(autumn, "2024-10-27 02:30", "2024-10-27 04:00")) == 2
    spring = pd.DataFrame(
        {
            "systime": pd.date_range(
                "2024-03-31 00:00", periods=4, freq="h", tz="UTC"
            ).tz_convert("Europe/Berlin")
        }
    )
    # 00:00 .. 03:00 UTC; from 02:30 UTC keeps 03:00
    assert len(trim_to_time_range(spring, "2024-03-31 02:30", None)) == 1


def test_trim_naive_column_and_aware_bounds_compare_in_utc():
    """A naive column is UTC; 11:00 Berlin is 10:00 UTC."""
    df = pd.DataFrame(
        {"systime": pd.date_range("2024-01-01 09:00", periods=3, freq="h")}
    )
    out = trim_to_time_range(
        df, pd.Timestamp("2024-01-01 11:00", tz="Europe/Berlin"), None
    )
    assert out["systime"].tolist() == [
        pd.Timestamp("2024-01-01 10:00"),
        pd.Timestamp("2024-01-01 11:00"),
    ]


# ---------------------------------------------------------------------------
# 5: ParquetLoader trims rows
# ---------------------------------------------------------------------------


@pytest.fixture
def local_hour(tmp_path):
    folder = tmp_path / "2024/01/01/09"
    folder.mkdir(parents=True)
    _quarter_hours("2024-01-01 09:00").to_parquet(folder / "t.parquet")
    return tmp_path


def test_parquet_loader_trims_time_range_rows(local_hour):
    """The 09 file has 09:00, 09:15, 09:30, 09:45; 09:20 .. 09:40 keeps 09:30."""
    base = str(local_hour)
    start, end = "2024-01-01 09:20", "2024-01-01 09:40"
    assert len(ParquetLoader.load_by_time_range(base, start, end)) == 1
    out = ParquetLoader.load_files_by_time_range_and_uuids(base, start, end, ["t"])
    assert out["systime"].tolist() == [pd.Timestamp("2024-01-01 09:30")]


def test_parquet_loader_trim_to_range_false_keeps_whole_hours(local_hour):
    base = str(local_hour)
    start, end = "2024-01-01 09:20", "2024-01-01 09:40"
    assert (
        len(ParquetLoader.load_by_time_range(base, start, end, trim_to_range=False))
        == 4
    )
    out = ParquetLoader.load_files_by_time_range_and_uuids(
        base, start, end, ["t"], trim_to_range=False
    )
    assert len(out) == 4


# ---------------------------------------------------------------------------
# 6, 8: Azure energy loader
# ---------------------------------------------------------------------------


def _csv(day: str) -> bytes:
    """Three integer readings on ``day``: 00:00, 12:00, 23:00 UTC."""
    return (
        "time,value\n" f"{day}T00:00:00Z,1\n{day}T12:00:00Z,2\n{day}T23:00:00Z,3\n"
    ).encode()


def _energy() -> AzureBlobEnergyLoader:
    loader = object.__new__(AzureBlobEnergyLoader)
    loader.container_client = FakeContainer(
        {f"csv/2024/01/0{d}/s.csv": _csv(f"2024-01-0{d}") for d in (1, 2, 3)}
    )
    loader.prefix = ""
    loader.max_workers = 2
    loader.thousands = None
    loader.decimal = "."
    return loader


def test_azure_energy_time_range_trims_rows():
    """06:00 .. 18:00 on 1 Jan keeps only the 12:00 reading."""
    loader = _energy()
    df = loader.load_by_time_range("2024-01-01 06:00", "2024-01-01 18:00")
    assert df["value_double"].tolist() == [2.0]
    streamed = list(loader.stream_by_time_range("2024-01-01 06:00", "2024-01-01 18:00"))
    assert [len(df) for _, df in streamed] == [1]


def test_azure_energy_date_only_end_keeps_the_whole_day():
    """("2024-01-01", "2024-01-01") is the whole of 1 January: 3 rows."""
    assert len(_energy().load_by_time_range("2024-01-01", "2024-01-01")) == 3


def test_azure_energy_series_ids_open_ranges():
    """start 2 Jan: days 2 and 3 (6 rows); end 2 Jan: days 1 and 2 (6 rows);
    both: 2 Jan only (3 rows)."""
    loader = _energy()
    after = loader.load_by_series_ids(["s"], start="2024-01-02")
    before = loader.load_by_series_ids(["s"], end="2024-01-02")
    both = loader.load_by_series_ids(["s"], start="2024-01-02", end="2024-01-02")
    assert sorted(after["systime"].dt.day.unique()) == [2, 3]
    assert sorted(before["systime"].dt.day.unique()) == [1, 2]
    assert len(after) == len(before) == 6
    assert len(both) == 3


def test_azure_energy_integer_values_are_float64():
    df = _energy().load_by_time_range("2024-01-01", "2024-01-01")
    assert df["value_double"].dtype == "float64"


# ---------------------------------------------------------------------------
# 7, 8: Databricks energy loader
# ---------------------------------------------------------------------------


@pytest.fixture
def energy_volume(tmp_path):
    for day in ("01", "02"):
        folder = tmp_path / "csv/2024/01" / day
        folder.mkdir(parents=True)
        (folder / "s.csv").write_bytes(_csv(f"2024-01-{day}"))
    return DatabricksUnityEnergyLoader(volume_path=str(tmp_path))


def test_databricks_energy_day_folders_are_utc(energy_volume):
    """00:30 .. 00:45 Berlin on 2 Jan is 23:30 .. 23:45 UTC on 1 Jan."""
    berlin = "Europe/Berlin"
    days = energy_volume._day_dirs(
        pd.Timestamp("2024-01-02 00:30", tz=berlin),
        pd.Timestamp("2024-01-02 00:45", tz=berlin),
    )
    assert [p.as_posix()[-10:] for p in days] == ["2024/01/01"]
    df = energy_volume.load_by_time_range(
        pd.Timestamp("2024-01-01 23:00", tz="UTC"),
        pd.Timestamp("2024-01-02 01:00", tz=berlin),
    )
    # 23:00 UTC on 1 Jan .. 00:00 UTC on 2 Jan: the 23:00 and 00:00 readings
    assert df["value_double"].tolist() == [3.0, 1.0]


def test_databricks_energy_mixed_naive_and_aware_bounds(energy_volume):
    """A naive start next to an aware end is read in the end's zone."""
    df = energy_volume.load_by_time_range(
        "2024-01-01 06:00", pd.Timestamp("2024-01-02 06:00", tz="UTC")
    )
    assert df["value_double"].tolist() == [2.0, 3.0, 1.0]


def test_databricks_energy_open_range_uses_utc_days(energy_volume):
    """01:00 Berlin on 2 Jan is 00:00 UTC on 2 Jan: only the 2 Jan folder."""
    df = energy_volume.fetch_data_as_dataframe(
        start=pd.Timestamp("2024-01-02 01:00", tz="Europe/Berlin")
    )
    assert sorted(df["systime"].dt.day.unique()) == [2]


def test_databricks_energy_values_and_mixed_offsets(tmp_path):
    """Integers become float64; ``+01:00`` and ``Z`` times both parse to UTC."""
    folder = tmp_path / "csv/2024/01/01"
    folder.mkdir(parents=True)
    (folder / "s.csv").write_text(
        "time,value\n2024-01-01T01:00:00+01:00,1\n2024-01-01T01:00:00Z,2\n"
    )
    df = DatabricksUnityEnergyLoader(volume_path=str(tmp_path)).load_by_time_range(
        "2024-01-01", "2024-01-01"
    )
    assert df["value_double"].dtype == "float64"
    assert df["systime"].tolist() == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
    ]


def test_energy_api_integer_values_are_float64():
    pytest.importorskip("requests")
    from ts_shape.loader.timeseries.energy_api_loader import EnergyAPILoader

    loader = object.__new__(EnergyAPILoader)
    out = loader._normalize(
        [{"timestamp": 1704067200000, "value": 3}],
        time_key="timestamp",
        value_key="value",
        uuid_value="u",
        data_root=None,
    )
    assert out["value_double"].dtype == "float64"
    assert out["systime"].tolist() == [pd.Timestamp("2024-01-01", tz="UTC")]


# ---------------------------------------------------------------------------
# 9: Timescale parquet export
# ---------------------------------------------------------------------------


def test_timescale_parquet_export_has_no_hour_column(tmp_path):
    """Rows at 00:00, 00:40 and 01:20 go to the 00 and 01 files, with the
    queried columns only."""
    pytest.importorskip("sqlalchemy")
    from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess

    loader = object.__new__(TimescaleDBDataAccess)
    loader.uuids = ["u"]
    chunk = pd.DataFrame(
        {
            "uuid": "u",
            "systime": pd.date_range("2024-01-01", periods=3, freq="40min"),
            "value_double": [1.0, 2.0, 3.0],
        }
    )
    loader._fetch_data = lambda uuid: iter([chunk.copy()])
    loader.fetch_data_as_parquet(str(tmp_path))
    first = pd.read_parquet(tmp_path / "2024/01/01/00/u.parquet")
    second = pd.read_parquet(tmp_path / "2024/01/01/01/u.parquet")
    assert list(first.columns) == ["uuid", "systime", "value_double"]
    assert first["value_double"].tolist() == [1.0, 2.0]
    assert second["value_double"].tolist() == [3.0]


# ---------------------------------------------------------------------------
# 10: epoch units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        (1704067200, "s"),
        (1704067200.5, "s"),
        (1704067200000, "ms"),
        (1704067200000000, "us"),
        (1704067200000000000, "ns"),
    ],
)
def test_epoch_unit_is_inferred_from_magnitude(value, unit):
    """s below 1e11, ms below 1e14, us below 1e17, otherwise ns."""
    values = pd.Series([value])
    assert epoch_unit(values) == unit
    expected = pd.Timestamp("2024-01-01", tz="UTC")
    if value == 1704067200.5:
        expected += pd.Timedelta(milliseconds=500)
    assert parse_utc_times(values)[0] == expected


def test_epoch_unit_can_be_given():
    """An explicit unit wins: 1704067200000 seconds is out of range for ms
    inference, but read as ms it is 2024-01-01."""
    out = parse_utc_times(pd.Series([1704067200]), unit="s")
    assert out[0] == pd.Timestamp("2024-01-01", tz="UTC")
    out = parse_utc_times(pd.Series([86_400_000]), unit="ms")
    assert out[0] == pd.Timestamp("1970-01-02", tz="UTC")


def test_epoch_nulls_and_text_are_unchanged():
    out = parse_utc_times(pd.Series([1704067200.0, None]))
    assert out[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert pd.isna(out[1])
    out = parse_utc_times(pd.Series(["2024-01-01T01:00:00+01:00"]))
    assert out[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_trim_epoch_column():
    """Epoch-second rows at 00:00, 01:00, 02:00 UTC; from 01:00 keeps 2."""
    df = pd.DataFrame({"systime": [1704067200, 1704070800, 1704074400]})
    assert len(trim_to_time_range(df, "2024-01-01 01:00", None)) == 2


def test_date_only_detection():
    import datetime

    assert is_date_only("2024-01-31")
    assert is_date_only("2024/01/31")
    assert is_date_only(datetime.date(2024, 1, 31))
    assert not is_date_only("2024-01-31 00:00")
    assert not is_date_only(pd.Timestamp("2024-01-31"))
    df = pd.DataFrame({"systime": pd.date_range("2024-01-31", periods=3, freq="12h")})
    # 31 Jan 00:00 and 12:00; 1 Feb 00:00 is the next day
    assert len(trim_to_day_range(df, "2024-01-31", "2024-01-31")) == 2


# ---------------------------------------------------------------------------
# 11: MetadataJsonLoader columnar index maps
# ---------------------------------------------------------------------------

UNEVEN = {
    "uuid": {"0": "a", "1": "b"},
    "label": {"0": "A", "1": "B"},
    "config": {"0": {"unit": "C"}},
}


def test_json_uneven_index_maps_raise_in_strict_mode():
    """strict promises lined-up columns: a uuid key missing from config raises."""
    with pytest.raises(ValueError, match="config"):
        MetadataJsonLoader(UNEVEN)


def test_json_uneven_index_maps_fill_when_not_strict():
    """Not strict: "b" is kept with an empty config; a label-only key without
    uuid is dropped."""
    data = {**UNEVEN, "label": {"0": "A", "1": "B", "2": "C"}}
    df = MetadataJsonLoader(data, strict=False).df
    assert list(df.index) == ["a", "b"]
    assert df.loc["a", "unit"] == "C"
    assert pd.isna(df.loc["b", "unit"])
    data = {**UNEVEN, "label": {"0": "A"}}
    df = MetadataJsonLoader(data, strict=False).df
    assert pd.isna(df.loc["b", "label"])


def test_json_mixed_digit_and_text_keys_sort():
    """Digit keys in numeric order (2 before 10), then text keys."""
    keys = ["10", "x", "2"]
    data = {
        "uuid": {k: f"u{k}" for k in keys},
        "label": {k: k for k in keys},
        "config": {k: {} for k in keys},
    }
    assert list(MetadataJsonLoader(data).df.index) == ["u2", "u10", "ux"]


# ---------------------------------------------------------------------------
# 12: DatapointAPI
# ---------------------------------------------------------------------------


def _api(monkeypatch, tmp_path, datapoints, **kwargs):
    requests = pytest.importorskip("requests")
    from ts_shape.loader.metadata.metadata_api_loader import DatapointAPI

    def fake_get(url, headers=None, timeout=None):
        if "data_points" in url:
            data = datapoints
        elif url.endswith("devices"):
            data = [{"id": 10, "name": "Dev"}]
        else:
            data = [{"id": 1}]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    monkeypatch.setattr(requests, "get", fake_get)
    return DatapointAPI(
        ["Dev", "Unknown"], "http://api", "tok", output_path=str(tmp_path), **kwargs
    )


def test_datapoint_api_without_enabled_or_config(monkeypatch, tmp_path):
    """No ``enabled``: not enabled (filtered out); no ``config``: None."""
    points = [{"uuid": "u1", "label": "L1"}, {"uuid": "u2", "enabled": True}]
    api = _api(monkeypatch, tmp_path, points)
    assert api.get_all_uuids()["Dev"] == ["u2"]
    unfiltered = _api(monkeypatch, tmp_path, points, filter_enabled=False)
    records = unfiltered.get_all_metadata()["Dev"]
    assert [r["uuid"] for r in records] == ["u1", "u2"]
    assert records[0]["config"] is None
    assert records[1]["label"] is None


def test_datapoint_api_unknown_device_gets_an_empty_frame(monkeypatch, tmp_path):
    """As DatapointDB: an unknown device gets an empty uuid/label/config frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        api = _api(monkeypatch, tmp_path, [{"uuid": "u1", "enabled": True}])
    assert api.get_all_uuids()["Unknown"] == []
    frame = api.device_metadata["Unknown"]
    assert frame.empty
    assert list(frame.columns) == ["uuid", "label", "config"]
    assert Path(tmp_path, "Unknown_data_points.json").read_text() == "[]"
