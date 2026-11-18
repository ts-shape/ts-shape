"""ts_shape.cycle_profile: the two loader outputs in, a profile out.

The timeseries goes through ``AzureBlobParquetLoader`` unchanged (only its
container client is replaced by an in-memory one); the metadata through
``MetadataJsonLoader``. No mapping, no machine context.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.cycle_profile import (
    CycleProfileResult,
    load_azure_timeseries,
    main,
    prepare_signals,
    profile_cycles,
    signal_labels,
)
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

pytest.importorskip("pyarrow")


def _raw_export(frame: pd.DataFrame) -> tuple[dict[str, bytes], list[dict]]:
    """The simulated PLC as the historian stores it: opaque uuids, systime as
    Unix nanoseconds, one parquet file per signal and hour folder."""
    names = sorted(frame["uuid"].unique())
    opaque = {n: f"{i:08d}-0000-4000-8000-000000000000" for i, n in enumerate(names)}
    raw = frame.assign(
        uuid=frame["uuid"].map(opaque),
        systime=frame["systime"].astype("datetime64[ns]").astype("int64"),
        value_bool=frame["value_bool"].astype(object),
        sequence_number=0,
        plctime=0,
    )
    blobs = {}
    hours = pd.to_datetime(raw["systime"], unit="ns").dt.floor("h")
    for (hour, uuid), part in raw.groupby([hours, "uuid"]):
        buffer = io.BytesIO()
        part.to_parquet(buffer, index=False)
        blobs[f"plc/{hour:%Y/%m/%d/%H}/{uuid}.parquet"] = buffer.getvalue()
    metadata = [
        {"uuid": u, "label": n, "config": {"nodePath": f"/Machine/{n.split(':')[1]}"}}
        for n, u in opaque.items()
    ]
    return blobs, metadata


class _Container:
    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs

    def list_blobs(self, name_starts_with=None):
        return [
            SimpleNamespace(name=n)
            for n in sorted(self.blobs)
            if not name_starts_with or n.startswith(name_starts_with)
        ]

    def download_blob(self, name):
        return SimpleNamespace(readall=lambda: self.blobs[name])


@pytest.fixture(scope="module")
def export():
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=150, seed=5, start="2026-01-05 06:00:00"
    )
    blobs, metadata = _raw_export(frame)
    loader = object.__new__(AzureBlobParquetLoader)
    loader.container_client = _Container(blobs)
    loader.prefix = "plc/"
    loader.max_workers = 4
    loader.hour_pattern = "{Y}/{m}/{d}/{H}/"
    meta = MetadataJsonLoader(metadata).to_df()
    raw = load_azure_timeseries(
        loader, "2026-01-05T00:00:00", "2026-01-06T00:00:00", list(meta.index)
    )
    return frame, truth, raw, meta


def test_azure_loader_output_is_the_input(export):
    _, _, raw, meta = export
    assert pd.api.types.is_integer_dtype(raw["systime"])
    result = profile_cycles(raw, meta)
    assert result.summary["anchor_event"] == "/Machine/part_counter+1"
    assert result.summary["variant_uuid"] == "/Machine/article"
    assert result.summary["evaluable"]


def test_same_profile_with_or_without_metadata(export):
    frame, _, raw, meta = export
    labeled = profile_cycles(raw, meta).tables["cycle_records"]
    direct = profile_cycles(frame).tables["cycle_records"]
    assert labeled["cycle_class"].tolist() == direct["cycle_class"].tolist()
    np.testing.assert_allclose(labeled["duration_seconds"], direct["duration_seconds"])
    unlabeled = profile_cycles(raw).summary["anchor_event"]
    assert unlabeled.endswith("+1") and unlabeled.startswith("0000")


def test_labels_come_from_the_node_path():
    meta = pd.DataFrame(
        {"label": ["ns=2;s=x", "y"], "config.nodePath": ["/A/B", None]},
        index=pd.Index(["u1", "u2"], name="uuid"),
    )
    assert signal_labels(meta).to_dict() == {"u1": "/A/B", "u2": "y"}
    assert signal_labels(None).empty


def test_personal_data_is_excluded():
    raw = pd.DataFrame(
        {
            "uuid": ["w", "w", "c", "c"],
            "systime": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:01"] * 2, utc=True
            ),
            "value_string": ["Ann", "Bob", None, None],
            "value_double": [None, None, 1.0, 2.0],
        }
    )
    meta = pd.DataFrame({"uuid": ["w", "c"], "label": ["/JobData/Worker", "/Counter"]})
    out = prepare_signals(raw, meta)
    assert out["uuid"].unique().tolist() == ["/Counter"]
    assert prepare_signals(raw, meta, exclude=()).shape[0] == 4


def test_result_round_trips_through_a_folder(export, tmp_path):
    _, _, raw, meta = export
    result = profile_cycles(raw, meta)
    written = result.save(tmp_path)
    assert (tmp_path / "summary.json") in written
    loaded = CycleProfileResult.load(tmp_path)
    assert loaded.summary == result.summary
    assert set(loaded.tables) == set(result.tables)
    assert len(loaded.tables["cycle_records"]) == len(result.tables["cycle_records"])


def test_summary_numbers_come_from_the_tables(export):
    _, _, raw, meta = export
    result = profile_cycles(raw, meta)
    s, t = result.summary, result.tables
    rec = t["cycle_records"]
    assert s["cycle_count"] == len(rec)
    assert s["pause_count"] == len(t["pauses"]) == (rec["cycle_class"] == "pause").sum()
    stats = t["cycle_time_statistics"].set_index("variant")
    for v in s["variants"]:
        row = stats.loc[v["variant"]]
        assert v["throughput_gain_pct"] == pytest.approx(
            (row["median_seconds"] / row["target_seconds"] - 1) * 100
        )
    assert set(t["signals"]["kind"]) >= {"counter", "state", "continuous"}


def test_cli_needs_a_time_range():
    with pytest.raises(SystemExit):
        main([])


def test_cycle_events_place_every_event_in_its_cycle(export):
    _, _, raw, meta = export
    result = profile_cycles(raw, meta)
    ev = result.tables["cycle_events"]
    rec = result.tables["cycle_records"].set_index("cycle_uuid")
    assert set(ev["cycle_uuid"]) <= set(rec.index)
    duration = ev["cycle_uuid"].map(rec["duration_seconds"])
    assert (ev["offset_seconds"] >= 0).all()
    assert (ev["offset_seconds"] < duration).all()
    # the step chain enters step 10 once in every normal cycle
    normal = ev[ev["cycle_class"] == "normal"]
    steps = normal[normal["event"] == "/Machine/step=10"].groupby("cycle_uuid").size()
    assert (steps == 1).mean() > 0.95
