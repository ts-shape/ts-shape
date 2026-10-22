"""Round 18 (PR K, part a): open-range day folders and ``Base`` signal rows.

K1. An open-range fetch from the Databricks Unity Volume loader floors its
bound to the folder granularity of ``hour_pattern``: a day layout
(``{Y}/{m}/{d}/``) dated every file at 00:00, so a mid-day ``start`` used
to skip that day's folder. The Azure parquet and flexible loaders list
folders from hour slots, which map to the day folder of a day layout; the
tests pin that their range and stream paths read the start day too.

K2. A detector that reads only its own signals keeps only their rows, but
warns exactly as the full constructor does (the warnings come from the
whole input).

Expected values are hand-counted from the fixtures (hourly rows).
"""

from __future__ import annotations

import io
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.loader.timeseries.azure_blob_loader import (
    AzureBlobFlexibleFileLoader,
    AzureBlobParquetLoader,
)
from ts_shape.loader.timeseries.databricks_unity_parquet_loader import (
    DatabricksUnityParquetLoader,
)
from ts_shape.utils.base import Base
from ts_shape.utils.signal_checks import signal_source

# ---------------------------------------------------------------------------
# fakes (as in test_round17_loaders)
# ---------------------------------------------------------------------------


def _day(day: str) -> pd.DataFrame:
    """24 hourly rows of ``day`` (naive UTC), values 0 .. 23."""
    return pd.DataFrame(
        {
            "systime": pd.date_range(day, periods=24, freq="h"),
            "uuid": "t",
            "value_double": np.arange(24, dtype=float),
        }
    )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


class FakeContainer:
    """In-memory stand-in for ``azure.storage.blob.ContainerClient``."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs

    def list_blobs(self, name_starts_with=None):
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


@pytest.fixture
def day_volume(tmp_path):
    """Two day folders (``{Y}/{m}/{d}/``), 24 hourly rows each."""
    pytest.importorskip("pyarrow")
    for day in ("2024-01-01", "2024-01-02"):
        folder = tmp_path / day.replace("-", "/")
        folder.mkdir(parents=True)
        _day(day).to_parquet(folder / "t.parquet")
    return DatabricksUnityParquetLoader(
        volume_path=str(tmp_path), hour_pattern="{Y}/{m}/{d}/"
    )


# ---------------------------------------------------------------------------
# K1: Databricks open range on a day layout
# ---------------------------------------------------------------------------


def test_databricks_open_start_mid_day_reads_that_day(day_volume):
    """13:00 .. 23:00 of day 1 (11 rows) plus all 24 rows of day 2.

    Before round 18 the start was floored to 12:00, after the 00:00 the day
    folder is dated at, and only day 2's 24 rows came back.
    """
    out = day_volume.fetch_data_as_dataframe(start="2024-01-01 12:30")
    assert len(out) == 11 + 24
    assert out["systime"].min() == pd.Timestamp("2024-01-01 13:00")


def test_databricks_open_start_untrimmed_keeps_the_whole_day(day_volume):
    out = day_volume.fetch_data_as_dataframe(
        start="2024-01-01 12:30", trim_to_range=False
    )
    assert len(out) == 48


def test_databricks_open_end_mid_day_reads_that_day(day_volume):
    """00:00 .. 06:00 of day 2 (7 rows) plus all 24 rows of day 1."""
    out = day_volume.fetch_data_as_dataframe(end="2024-01-02 06:00")
    assert len(out) == 24 + 7


def test_databricks_open_start_aware_bound_is_read_in_utc(day_volume):
    """01:30 in Berlin is 00:30 UTC on day 1: 23 rows of day 1, 24 of day 2."""
    start = pd.Timestamp("2024-01-01 01:30", tz="Europe/Berlin")
    out = day_volume.fetch_data_as_dataframe(start=start)
    assert len(out) == 23 + 24


def test_databricks_open_start_on_hour_layout_still_floors_to_the_hour(tmp_path):
    """With ``{H}`` the 12:00 folder overlaps a 12:30 start (12:30, 12:45)."""
    pytest.importorskip("pyarrow")
    for hour in ("2024-01-01 11:00", "2024-01-01 12:00", "2024-01-01 13:00"):
        ts = pd.Timestamp(hour)
        folder = tmp_path / ts.strftime("%Y/%m/%d/%H")
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "systime": pd.date_range(ts, periods=4, freq="15min"),
                "uuid": "t",
                "value_double": 1.0,
            }
        ).to_parquet(folder / "t.parquet")
    loader = DatabricksUnityParquetLoader(volume_path=str(tmp_path))
    out = loader.fetch_data_as_dataframe(start="2024-01-01 12:30")
    assert len(out) == 2 + 4
    untrimmed = loader.fetch_data_as_dataframe(
        start="2024-01-01 12:30", trim_to_range=False
    )
    assert len(untrimmed) == 8


def test_databricks_month_layout_floors_to_the_month(tmp_path):
    """A ``{Y}/{m}/`` layout dates a folder at the 1st; a mid-month start reads it."""
    pytest.importorskip("pyarrow")
    folder = tmp_path / "2024/01"
    folder.mkdir(parents=True)
    _day("2024-01-15").to_parquet(folder / "t.parquet")
    loader = DatabricksUnityParquetLoader(
        volume_path=str(tmp_path), hour_pattern="{Y}/{m}/"
    )
    out = loader.fetch_data_as_dataframe(start="2024-01-15 12:30")
    assert len(out) == 11


# ---------------------------------------------------------------------------
# K1: Azure parquet / flexible loaders on a day layout (already correct)
# ---------------------------------------------------------------------------

AZURE_DAYS = {
    "2024/01/01/t.parquet": "2024-01-01",
    "2024/01/02/t.parquet": "2024-01-02",
}


@pytest.fixture
def azure_parquet():
    pytest.importorskip("pyarrow")
    blobs = {name: _parquet_bytes(_day(day)) for name, day in AZURE_DAYS.items()}
    return _azure(AzureBlobParquetLoader, blobs, "{Y}/{m}/{d}/")


def test_azure_parquet_range_mid_day_start_reads_that_day(azure_parquet):
    """13:00 .. 23:00 of day 1 and 00:00 .. 06:00 of day 2: 11 + 7 rows."""
    start, end = "2024-01-01 12:30", "2024-01-02 06:00"
    assert len(azure_parquet.load_by_time_range(start, end)) == 18
    streamed = azure_parquet.stream_by_time_range(start, end)
    assert sum(len(df) for _, df in streamed) == 18
    assert (
        len(azure_parquet.load_files_by_time_range_and_uuids(start, end, ["t"])) == 18
    )
    streamed = azure_parquet.stream_files_by_time_range_and_uuids(start, end, ["t"])
    assert sum(len(df) for _, df in streamed) == 18


def test_azure_flexible_mid_day_start_lists_that_day():
    loader = _azure(
        AzureBlobFlexibleFileLoader,
        {"2024/01/01/a.json": b"{}", "2024/01/02/b.json": b"{}"},
        "{Y}/{m}/{d}/",
    )
    start, end = "2024-01-01 12:30", "2024-01-02 06:00"
    both = ["2024/01/01/a.json", "2024/01/02/b.json"]
    assert loader.list_files_by_time_range(start, end) == both
    assert list(loader.iter_file_names_by_time_range(start, end)) == both
    streamed = loader.stream_files_by_time_range(start, end)
    assert [name for name, _ in streamed] == both
    streamed = loader.stream_files_by_time_range_and_basenames(start, end, ["a.json"])
    assert [name for name, _ in streamed] == ["2024/01/01/a.json"]


# ---------------------------------------------------------------------------
# K2: Base keeps the requested signal rows, warns from the whole frame
# ---------------------------------------------------------------------------

T = pd.date_range("2025-01-01 08:00", periods=6, freq="1min")


def _frame() -> pd.DataFrame:
    """Two signals, shuffled, one NaT and one duplicate row of ``other``."""
    run = pd.DataFrame(
        {
            "systime": T,
            "uuid": "run",
            "value_bool": np.array([1, 1, 0, 1, 1, 0], dtype=bool),
        }
    )
    other = pd.DataFrame({"systime": T, "uuid": "other", "value_bool": True})
    df = pd.concat([run, other, other.iloc[[2]]], ignore_index=True)
    df["value_double"] = np.nan
    df.loc[3, "systime"] = pd.NaT
    return df.sample(frac=1.0, random_state=3)


def _record(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [(w.category, str(w.message)) for w in caught]


def test_detector_keeps_only_its_signal_rows():
    df = _frame()
    det, _ = _record(lambda: ts_shape.MachineStateEvents(df, "run"))
    assert set(det.dataframe["uuid"]) == {"run"}
    assert len(det.dataframe) == 6
    assert signal_source(det) is not None
    assert repr(det).startswith("MachineStateEvents(rows=13,")


def test_kept_rows_are_the_full_frame_rows_and_warnings_match():
    df = _frame()
    det, det_warnings = _record(lambda: ts_shape.MachineStateEvents(df, "run"))
    full, base_warnings = _record(lambda: Base(df))
    rows = full.dataframe[full.dataframe["uuid"] == "run"]
    pd.testing.assert_frame_equal(det.dataframe, rows)
    assert det_warnings == base_warnings
    texts = [text for _, text in det_warnings]
    assert any("1 missing or unparseable" in t for t in texts)
    assert any("1 duplicate timestamp" in t for t in texts)
    pd.testing.assert_frame_equal(det.get_dataframe(), full.dataframe)


def test_unknown_signal_keeps_the_full_frame_path():
    df = _frame()
    with pytest.raises(ValueError, match="not found"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_shape.MachineStateEvents(df, "missing")
    det, _ = _record(lambda: ts_shape.ThresholdMonitoringEvents(df, "missing"))
    assert signal_source(det) is None
    assert len(det.dataframe) == len(df)


def test_text_numbers_keep_the_full_frame_path():
    """Numbers stored as text are parsed (with a warning) from the whole column."""
    df = _frame()
    df["value_double"] = ["1.5"] * (len(df) - 1) + ["x"]
    det, caught = _record(lambda: ts_shape.ThresholdMonitoringEvents(df, "run"))
    assert signal_source(det) is None
    assert any("not numeric" in text for _, text in caught)


def test_unknown_method_signal_lists_every_input_uuid():
    df = _frame()
    det, _ = _record(lambda: ts_shape.LongDowntimeEvents(df, "run"))
    prod = pd.DataFrame({"systime": T, "uuid": "count", "value_integer": 1})
    _, caught = _record(
        lambda: det.count_events_between_gaps(prod, production_uuid="nope")
    )
    texts = [text for _, text in caught]
    assert any("'count', 'other', 'run'" in t for t in texts), texts


def test_pandas_frame_input_is_not_changed():
    df = _frame()
    snapshot = df.copy(deep=True)
    _record(lambda: ts_shape.MachineStateEvents(df, "run").detect_run_idle())
    pd.testing.assert_frame_equal(df, snapshot)
