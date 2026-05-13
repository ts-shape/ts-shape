import pytest

requests = pytest.importorskip("requests")

from ts_shape.loader.metadata.metadata_api_loader import DatapointAPI

BASE = "http://api"

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

DATATRONS = [
    {
        "id": 1,
        "name": "DT-1",
        "serialNumber": "SN-001",
        "deviceUUID": "dt-uuid-1",
        "model": "ModelX",
    }
]
DEVICES = [
    {"id": 10, "name": "Device A", "serialNumber": "DEV-A", "deviceUUID": "dev-uuid-a"},
    {"id": 11, "name": "Device B", "serialNumber": "DEV-B", "deviceUUID": "dev-uuid-b"},
]
DATAPOINTS = [
    {
        "uuid": "u1",
        "label": "Temperature",
        "config": {"x": 1},
        "enabled": True,
        "unit": "°C",
    },
    {
        "uuid": "u2",
        "label": "Pressure",
        "config": {"x": 2},
        "enabled": False,
        "unit": "bar",
    },
    {
        "uuid": "u3",
        "label": "Temp Extra",
        "config": {"x": 3},
        "enabled": True,
        "unit": "°C",
    },
]


class DummyResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def make_fake_get(datatrons=None, devices=None, datapoints=None):
    """Return a fake requests.get that routes by URL path."""
    dt = datatrons if datatrons is not None else DATATRONS
    dv = devices if devices is not None else DEVICES
    dp = datapoints if datapoints is not None else DATAPOINTS

    def fake_get(url, headers=None):
        if "data_points" in url:
            return DummyResp(dp)
        if (
            "/devices" in url
            and url.split("/devices")[1].lstrip("/").isdigit() is False
            and url.endswith("devices")
        ):
            return DummyResp(dv)
        if "/devices" in url:
            return DummyResp(dv)
        # /api/datatrons
        return DummyResp(dt)

    return fake_get


# ---------------------------------------------------------------------------
# Core GET methods
# ---------------------------------------------------------------------------


def test_get_datatrons(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.get_datatrons()
    assert result == DATATRONS


def test_get_devices(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.get_devices(datatron_id=1)
    assert result == DEVICES


def test_get_datapoints(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.get_datapoints(datatron_id=1, device_id=10)
    assert result == DATAPOINTS


# ---------------------------------------------------------------------------
# Search methods
# ---------------------------------------------------------------------------


def test_search_datatrons(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.search_datatrons("DT-1")
    assert len(result) == 1 and result[0]["name"] == "DT-1"


def test_search_datatrons_no_match(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    assert api.search_datatrons("zzz") == []


def test_search_devices(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.search_devices(datatron_id=1, query="Device A")
    assert len(result) == 1 and result[0]["id"] == 10


def test_search_devices_substring(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    # "device" matches both "Device A" and "Device B"
    result = api.search_devices(datatron_id=1, query="device")
    assert len(result) == 2


def test_search_datapoints(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.search_datapoints(datatron_id=1, device_id=10, query="temp")
    labels = {r["label"] for r in result}
    assert labels == {"Temperature", "Temp Extra"}


def test_search_datapoints_by_uuid(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.search_datapoints(1, 10, "u2", fields=["uuid"])
    assert len(result) == 1 and result[0]["uuid"] == "u2"


def test_search_datapoints_case_insensitive(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.search_datapoints(1, 10, "TEMPERATURE")
    assert any(r["label"] == "Temperature" for r in result)


# ---------------------------------------------------------------------------
# Cross-hierarchy find methods
# ---------------------------------------------------------------------------


def test_find_devices(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.find_devices("Device A")
    assert len(result) == 1
    assert result[0]["name"] == "Device A"
    assert result[0]["datatron_id"] == 1


def test_find_devices_includes_datatron_id(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.find_devices("device")
    assert all("datatron_id" in r for r in result)
    assert len(result) == 2


def test_find_datapoints(monkeypatch):
    # Use a single-device setup so "pressure" appears exactly once.
    monkeypatch.setattr(requests, "get", make_fake_get(devices=[DEVICES[0]]))
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    result = api.find_datapoints("pressure")
    assert len(result) == 1
    assert result[0]["label"] == "Pressure"
    assert "datatron_id" in result[0]
    assert "device_id" in result[0]


def test_find_datapoints_no_match(monkeypatch):
    monkeypatch.setattr(requests, "get", make_fake_get())
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="tok")
    assert api.find_datapoints("zzz") == []


# ---------------------------------------------------------------------------
# High-level _api_access / get_all_uuids (original behaviour preserved)
# ---------------------------------------------------------------------------


def test_metadata_api_loader_monkeypatched(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", make_fake_get())

    api = DatapointAPI(
        device_names=["Device A"],
        base_url=BASE,
        api_token="tok",
        output_path=str(tmp_path),
        required_uuid_list=["u1"],
        filter_enabled=True,
    )

    uuids = api.get_all_uuids()
    assert set(uuids.keys()) == {"Device A"}
    assert uuids["Device A"] == ["u1"]


def test_filter_enabled_false(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", make_fake_get())

    api = DatapointAPI(
        device_names=["Device A"],
        base_url=BASE,
        api_token="tok",
        output_path=str(tmp_path),
        filter_enabled=False,
    )

    uuids = api.get_all_uuids()
    # All three datapoints returned when not filtering by enabled
    assert set(uuids["Device A"]) == {"u1", "u2", "u3"}


def test_bearer_token_in_headers(monkeypatch):
    captured = {}

    def capturing_get(url, headers=None):
        captured["headers"] = headers
        return DummyResp(
            DATATRONS
            if "devices" not in url and "data_points" not in url
            else DEVICES if "data_points" not in url else DATAPOINTS
        )

    monkeypatch.setattr(requests, "get", capturing_get)
    api = DatapointAPI(device_names=[], base_url=BASE, api_token="my-jwt-token")
    # Trigger a real GET call to verify the Authorization header is set correctly.
    api.get_datatrons()
    assert captured["headers"]["Authorization"] == "Bearer my-jwt-token"


def test_get_all_metadata_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", make_fake_get())

    api = DatapointAPI(
        device_names=["Device A"],
        base_url=BASE,
        api_token="tok",
        output_path=str(tmp_path),
        filter_enabled=True,
    )

    metadata = api.get_all_metadata()
    assert "Device A" in metadata
    for record in metadata["Device A"]:
        assert set(record.keys()) >= {"uuid", "label", "config"}
