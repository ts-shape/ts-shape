"""Field test 0.0.88: metadata, typing, profile noise, loading, cycle profile."""

from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import DataQualityWarning
from ts_shape.events.profiling.cycles import (
    CycleProfileEvents,
    _grid_period,
)
from ts_shape.events.profiling.runner import _per_hour, _top_row, prepare_signals
from ts_shape.signals.identify import identify_objects
from ts_shape.utils.personal import is_personal


def _long(uuid: str, times, values, column: str = "value_double") -> pd.DataFrame:
    df = pd.DataFrame({"systime": times, "uuid": uuid})
    for c in ("value_double", "value_integer", "value_bool", "value_string"):
        df[c] = None
    df[column] = values
    return df


# ---------------------------------------------------------------------------
# Metadata workbook
# ---------------------------------------------------------------------------


def _flat() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": ["t1", "t2", "s", "txt"],
            "label": ["Zone 1", "Zone 1", "Speed", "Recipe"],
            "config.nodePath[string]": ["/a", "/b", "/c", "/d"],
            "config.unit": ["°C", None, "min1", None],
            "is_enabled": [True, True, True, False],
            "is_archived": [False, True, False, False],
            "scaling_factor": [2.0, 1.0, 1.0, 1.0],
            "scaling_offset": [1.0, 0.0, 0.0, 1.0],
        }
    )


def test_flat_frame_is_nested_back() -> None:
    meta = ts_shape.MetadataJsonLoader.from_flat_frame(_flat())
    table = meta.to_df()
    assert {"nodePath", "unit", "is_enabled", "scaling_offset"} <= set(table.columns)
    assert meta.delivering().index.tolist() == ["t1", "s"]
    assert meta.units().to_dict() == {"t1": "°C", "s": "min1"}
    assert meta.duplicate_labels()["uuids"].tolist() == [["t1", "t2"]]
    assert meta.scaling().loc["t1"].tolist() == [2.0, 1.0]


def test_workbook_is_read_without_extra_packages(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "datapoints.xlsx"
    _flat().to_excel(path, index=False)
    meta = ts_shape.MetadataJsonLoader.from_workbook(str(path))
    assert meta.to_df().loc["s", "unit"] == "min1"
    assert meta.to_df().loc["t1", "nodePath"] == "/a"


def test_apply_scaling_is_opt_in_and_skips_text() -> None:
    meta = ts_shape.MetadataJsonLoader.from_flat_frame(_flat())
    t = pd.date_range("2026-10-01", periods=2, freq="1min")
    df = pd.concat(
        [_long("t1", t, [10.0, 20.0]), _long("txt", t, ["A", "B"], "value_string")],
        ignore_index=True,
    )
    with pytest.warns(DataQualityWarning, match="text or bool"):
        out = ts_shape.apply_scaling(df, meta)
    assert out.loc[out["uuid"] == "t1", "value_double"].tolist() == [21.0, 41.0]
    assert out.loc[out["uuid"] == "txt", "value_string"].tolist() == ["A", "B"]


def test_plant_unit_spellings() -> None:
    pytest.importorskip("pint")
    convert = ts_shape.UnitConverter.convert_value
    assert convert(1200, "min1", "1/s") == pytest.approx(20.0)
    assert convert(2, "Kw", "W") == pytest.approx(2000.0)
    assert convert(20, "°C", "K") == pytest.approx(293.15)


# ---------------------------------------------------------------------------
# Personal data, labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "Worker",
        "worker login",
        "Rüstblatt Bedienpersonal",
        "Mitarbeiter-Nr",
        "Operator",
    ],
)
def test_personal_labels(label: str) -> None:
    assert is_personal(label)


@pytest.mark.parametrize("label", ["Zone 1 temperature", "Production Quantity"])
def test_not_personal(label: str) -> None:
    assert not is_personal(label)


def test_profile_with_metadata_labels_and_excludes_personal_data() -> None:
    df = ts_shape.make_timeseries(["a", "b"], n_points=50)
    meta = pd.DataFrame({"uuid": ["a", "b"], "label": ["Temp", "Bedienpersonal"]})
    report = ts_shape.check_signals(df, metadata=meta)
    assert report.table[["uuid", "label"]].values.tolist() == [["a", "Temp"]]


def test_prepare_signals_keeps_two_datapoints_with_one_label() -> None:
    t = pd.date_range("2026-10-01", periods=3, freq="1s")
    raw = pd.concat([_long("u1", t, [1.0, 2, 3]), _long("u2", t, [4.0, 5, 6])])
    meta = pd.DataFrame({"uuid": ["u1", "u2"], "label": ["Zone", "Zone"]})
    out = prepare_signals(raw, meta)
    assert sorted(out["uuid"].unique()) == ["Zone [u1]", "Zone [u2]"]


def test_identify_objects_label_vetoes() -> None:
    frame = ts_shape.make_object_signals()
    frame = frame[0] if isinstance(frame, tuple) else frame
    table = identify_objects(frame).to_frame()
    ids = table.loc[table["role"] == "identifier", "uuid"].tolist()[:2]
    labels = {ids[0]: "Production Quantity", ids[1]: "WEEKDAY"}
    vetoed = identify_objects(frame, labels=labels).to_frame().set_index("uuid")
    assert vetoed.loc[ids[0], "role"] == "counter"
    assert vetoed.loc[ids[1], "role"] == "measurement"


# ---------------------------------------------------------------------------
# Typing: a running timer is not a counter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_running_timer_is_not_a_counter(scale: float) -> None:
    t = pd.date_range("2026-10-05", periods=600, freq="1s")
    timer = (np.arange(600) % 120) * scale
    spec = ts_shape.infer_specs(_long("tiCycle", t, timer))["tiCycle"]
    assert spec.kind == "continuous"
    assert "running timer" in spec.notes["kind"]


def test_part_counter_is_still_a_counter() -> None:
    t = pd.date_range("2026-10-05", periods=600, freq="1s")
    parts = np.cumsum(np.random.default_rng(0).random(600) < 0.05).astype(float)
    assert ts_shape.infer_specs(_long("parts", t, parts))["parts"].kind == "counter"


# ---------------------------------------------------------------------------
# profile() noise
# ---------------------------------------------------------------------------


def _with_hole() -> pd.DataFrame:
    t = pd.date_range("2026-10-01", periods=600, freq="10s").append(
        pd.date_range("2026-10-03 05:00", periods=600, freq="10s")
    )
    rng = np.random.default_rng(1)
    return pd.concat(
        [_long(f"s{i}", t, rng.normal(0, 1, len(t))) for i in range(6)],
        ignore_index=True,
    )


def test_asset_hole_is_one_issue() -> None:
    issues = ts_shape.check_signals(_with_hole()).issues()
    holes = issues[issues["issue"] == "data hole"]
    assert len(holes) == 1 and holes["uuid"].iloc[0] == "(asset)"
    assert not (issues["issue"] == "logging gaps").any()
    clock = issues[issues["issue"] == "shared polling clock"]
    assert "6 of 6" in clock["detail"].iloc[0]


def test_short_constant_run_is_not_stuck() -> None:
    t = pd.date_range("2026-10-01", periods=2000, freq="64ms")
    values = np.random.default_rng(2).normal(0, 1, len(t))
    values[100:170] = 1.0  # constant for ~4.5 s
    issues = ts_shape.check_signals(_long("fast", t, values)).issues()
    assert not (issues["issue"] == "possibly stuck").any()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _export(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    for hour in range(3):
        folder = src / "export" / "2026" / "10" / "02" / f"{hour:02d}"
        folder.mkdir(parents=True)
        t = pd.date_range(f"2026-10-02 {hour:02d}:00", periods=60, freq="1min")
        for uuid in ("a", "b"):
            frame = _long(uuid, t, np.arange(60.0))
            frame["plctime"] = 0
            frame["sequence_number"] = 0
            frame["value_bytes"] = None
            frame.to_parquet(folder / f"{uuid}.parquet")
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in src.rglob("*.parquet"):
            zf.write(file, file.relative_to(src))
    return archive


def test_zip_export_is_read_in_place(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    archive = _export(tmp_path)
    loader = ts_shape.ParquetLoader
    assert len(loader.load_all_files(str(archive))) == 360
    part = loader.load_by_time_range(
        str(archive), "2026-10-02 00:30", "2026-10-02 01:29"
    )
    assert len(part) == 120
    assert len(loader.load_by_uuid_list(str(archive), ["a"])) == 180


def test_stream_by_time_range_and_compact_frame(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    archive = _export(tmp_path)
    chunks = list(
        ts_shape.ParquetLoader.stream_by_time_range(
            str(archive), "2026-10-02", "2026-10-02 03:00", chunk="1h", compact=True
        )
    )
    assert [len(c) for c in chunks] == [120, 120, 120]
    assert isinstance(chunks[0]["uuid"].dtype, pd.CategoricalDtype)
    assert "plctime" not in chunks[0].columns


def test_end_before_start_raises(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    archive = _export(tmp_path)
    with pytest.raises(ValueError, match="swapped"):
        ts_shape.ParquetLoader.load_by_time_range(
            str(archive), "2026-10-03", "2026-10-02"
        )


# ---------------------------------------------------------------------------
# Cycle profile
# ---------------------------------------------------------------------------


def test_summary_helpers_survive_all_nan() -> None:
    table = pd.DataFrame({"segment": ["a", "b"], "share": [np.nan, np.nan]})
    assert _top_row(table, "share") is None
    assert _top_row(table.assign(share=[0.2, 0.5]), "share")["segment"] == "b"
    assert np.isnan(_per_hour(0.0)) and _per_hour(60.0) == 60.0


def test_grid_period() -> None:
    base = pd.Timestamp("2026-10-05").value
    on_grid = base + np.array([0, 1, 2, 5, 6, 9, 13]) * 17_600_000_000
    assert _grid_period(on_grid) == pytest.approx(17.6)
    rng = np.random.default_rng(3)
    free = base + np.cumsum(rng.integers(1_000_000_000, 60_000_000_000, 50))
    assert _grid_period(free) == 0.0


def test_coarse_polling_is_not_evaluable() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fine, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=7)
        coarse, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=7, poll_period="30s")
        good = CycleProfileEvents(fine).evaluation_window()
        bad = CycleProfileEvents(coarse).evaluation_window()
    assert good["evaluable"].all()
    assert not bad["evaluable"].any()
    assert bad["reason"].str.startswith("anchor_below_resolution").all()
    assert (bad["time_resolution_seconds"] > 0).all()


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label", ["User_ID", "Current_User", "Username", "Personal_Name", "Log-in name"]
)
def test_personal_labels_with_separators(label: str) -> None:
    assert is_personal(label)


@pytest.mark.parametrize(
    "label",
    [
        "AnalogInput_3",
        "Catalog index",
        "Benutzeroberfläche",
        "User defined 1",
        "Ventil Bediener-Taste",
    ],
)
def test_machine_labels_are_not_personal(label: str) -> None:
    assert not is_personal(label)


def test_excluded_signals_are_counted() -> None:
    t = pd.date_range("2026-10-01", periods=3, freq="1s")
    raw = pd.concat([_long("u1", t, [1.0, 2, 3]), _long("u2", t, [4.0, 5, 6])])
    meta = pd.DataFrame({"uuid": ["u1", "u2"], "label": ["Temp", "Worker"]})
    with pytest.warns(DataQualityWarning, match="1 signal"):
        out = prepare_signals(raw, meta)
    assert out["uuid"].unique().tolist() == ["Temp"]


def test_id_labels_are_never_vetoed() -> None:
    frame = ts_shape.make_object_signals()
    frame = frame[0] if isinstance(frame, tuple) else frame
    table = identify_objects(frame).to_frame()
    ids = table.loc[table["role"] == "identifier", "uuid"].tolist()[:3]
    labels = dict(zip(ids, ["Part number", "RFID Tag ID", "Werkstück-Nr"], strict=True))
    vetoed = identify_objects(frame, labels=labels).to_frame().set_index("uuid")
    assert (vetoed.loc[ids, "role"] == "identifier").all()


def test_shift_reset_fast_counter_stays_a_counter() -> None:
    t = pd.date_range("2026-10-05", periods=3 * 5760, freq="5s")
    bottles = (np.arange(len(t)) % 5760) * 50.0  # 10/s, reset every 8 h
    assert ts_shape.infer_specs(_long("b", t, bottles))["b"].kind == "counter"


def test_compact_frame_keeps_value_columns_and_text() -> None:
    df = pd.DataFrame(
        {
            "uuid": ["a", "b"],
            "value_double": [np.nan, np.nan],
            "value_integer": [1.0, "x"],
            "plctime": [0, 0],
        }
    )
    out = ts_shape.compact_frame(df)
    assert "value_double" in out.columns and "plctime" not in out.columns
    assert out["value_integer"].tolist() == [1.0, "x"]


def test_extra_metadata_field_does_not_rename_config_columns() -> None:
    meta = ts_shape.MetadataJsonLoader(
        [
            {
                "uuid": "a",
                "label": "x",
                "datatype": "Int",
                "config": {"datatype": "Real", "nodePath": "/a"},
            }
        ]
    )
    assert {"nodePath", "datatype", "top.datatype"} <= set(meta.to_df().columns)


def test_flags_in_words() -> None:
    flat = _flat().assign(
        is_enabled=["on", "aktiv", "enabled", "off"], is_archived=False
    )
    meta = ts_shape.MetadataJsonLoader.from_flat_frame(flat)
    assert meta.delivering().index.tolist() == ["t1", "t2", "s"]


def test_zip_skips_macos_members(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    archive = _export(tmp_path)
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("__MACOSX/2026/10/02/00/._a.parquet", b"junk")
    assert len(ts_shape.ParquetLoader.load_all_files(str(archive))) == 360
