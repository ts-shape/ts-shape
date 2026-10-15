import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from ts_shape.loader.metadata.metadata_db_loader import DatapointDB  # noqa: E402


class DummyResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class DummyConnection:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params=None):
        return DummyResult(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyEngine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return DummyConnection(self._rows)


def test_metadata_db_loader_monkeypatched(monkeypatch, tmp_path):
    rows = [
        ("u1", "L1", {"x": 1}),
        ("u2", "L2", {"x": 2}),
    ]

    monkeypatch.setattr(
        "ts_shape.loader.metadata.metadata_db_loader.create_engine",
        lambda *args, **kwargs: DummyEngine(rows),
    )

    db = DatapointDB(
        device_names=["Device A"],
        db_user="u",
        db_pass="p",
        db_host="h",
        output_path=str(tmp_path),
        required_uuid_list=["u1"],
        filter_enabled=True,
    )

    uuids = db.get_all_uuids()
    # Device key present and filtered
    assert list(uuids.keys()) == ["Device A"]
    assert uuids["Device A"] == ["u1"]


def test_metadata_db_loader_device_without_data_points(monkeypatch, tmp_path):
    # Used to raise KeyError: 'uuid' (the empty frame had no columns).
    monkeypatch.setattr(
        "ts_shape.loader.metadata.metadata_db_loader.create_engine",
        lambda *args, **kwargs: DummyEngine([]),
    )

    db = DatapointDB(
        device_names=["Empty Device"],
        db_user="u",
        db_pass="p",
        db_host="h",
        output_path=str(tmp_path),
        required_uuid_list=["u1"],
    )

    assert db.get_all_uuids() == {"Empty Device": []}
    assert db.get_all_metadata() == {"Empty Device": []}
    frame = db.device_metadata["Empty Device"]
    assert list(frame.columns) == ["uuid", "label", "config"]
    assert len(frame) == 0
    assert (tmp_path / "Empty_Device_data_points.json").read_text() == "[]"
