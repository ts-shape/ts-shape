import logging
import requests
import pandas as pd  # type: ignore
import json
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DatapointAPI:
    """
    Loads datatron, device, and datapoint metadata from the Datadash REST API.

    Authentication is bearer-token only — pass an externally obtained JWT token:

        api = DatapointAPI(
            base_url="https://datadash.example.com",
            api_token="eyJhbGciOiJSUzI1NiJ9...",
            device_names=["Sensor A"],
        )

    The three core GET methods mirror the API hierarchy and can be called
    independently.  ``get_all_uuids()`` aggregates them into the UUID lists
    required by ``AzureBlobParquetLoader``.
    """

    _DATATRONS_PATH = "/api/datatrons"
    _DEVICES_PATH = "/api/datatrons/{datatron_id}/devices"
    _DATAPOINTS_PATH = "/api/datatrons/{datatron_id}/devices/{device_id}/data_points"

    def __init__(
        self,
        device_names: List[str],
        base_url: str,
        api_token: str,
        output_path: str = "data",
        required_uuid_list: Optional[List[str]] = None,
        filter_enabled: bool = True,
        timeout: int = 30,
    ):
        """
        :param device_names: Device names to collect datapoints for.
        :param base_url: API host, e.g. ``"https://datadash.example.com"``.
        :param api_token: JWT bearer token for ``Authorization: Bearer <token>``.
        :param output_path: Directory to write per-device JSON exports.
        :param required_uuid_list: Optional allowlist; only matching UUIDs are kept.
        :param filter_enabled: When True, only datapoints with ``enabled=True`` are kept.
        :param timeout: Per-request timeout in seconds; prevents an unresponsive
            server from hanging the loader indefinitely.
        """
        self.device_names = device_names
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.output_path = output_path
        self.timeout = timeout
        self.required_uuid_list: List[str] = required_uuid_list or []
        self.filter_enabled = filter_enabled
        self.device_metadata: Dict[str, pd.DataFrame] = {}
        self.device_uuids: Dict[str, List[str]] = {}
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}",
        }
        self._api_access()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> list:
        url = f"{self.base_url}{path}"
        response = requests.get(url, headers=self._headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _filter_by_query(
        self, records: List[Dict], query: str, fields: List[str]
    ) -> List[Dict]:
        q = query.lower()
        return [
            r for r in records if any(q in str(r.get(f, "")).lower() for f in fields)
        ]

    # ------------------------------------------------------------------
    # Core GET methods — one per API level
    # ------------------------------------------------------------------

    def get_datatrons(self) -> List[Dict]:
        """GET /api/datatrons — return all datatrons."""
        return self._get(self._DATATRONS_PATH)

    def get_devices(self, datatron_id) -> List[Dict]:
        """GET /api/datatrons/{datatron_id}/devices — return all devices for a datatron."""
        return self._get(self._DEVICES_PATH.format(datatron_id=datatron_id))

    def get_datapoints(self, datatron_id, device_id) -> List[Dict]:
        """GET /api/datatrons/{datatron_id}/devices/{device_id}/data_points."""
        return self._get(
            self._DATAPOINTS_PATH.format(datatron_id=datatron_id, device_id=device_id)
        )

    # ------------------------------------------------------------------
    # Search methods — substring match within a single endpoint's results
    # ------------------------------------------------------------------

    def search_datatrons(
        self, query: str, fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """Filter datatrons whose fields contain *query* (case-insensitive)."""
        fields = fields or ["name", "serialNumber", "deviceUUID", "model"]
        return self._filter_by_query(self.get_datatrons(), query, fields)

    def search_devices(
        self, datatron_id, query: str, fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """Filter devices for a datatron whose fields contain *query*."""
        fields = fields or ["name", "serialNumber", "deviceUUID"]
        return self._filter_by_query(self.get_devices(datatron_id), query, fields)

    def search_datapoints(
        self, datatron_id, device_id, query: str, fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """Filter datapoints for a device whose fields contain *query*."""
        fields = fields or ["label", "uuid", "unit"]
        return self._filter_by_query(
            self.get_datapoints(datatron_id, device_id), query, fields
        )

    # ------------------------------------------------------------------
    # Cross-hierarchy find methods — no parent ID needed
    # ------------------------------------------------------------------

    def find_devices(
        self, query: str, fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search all devices across all datatrons.

        Each result dict includes an extra ``datatron_id`` key.
        """
        fields = fields or ["name", "serialNumber", "deviceUUID"]
        results = []
        for datatron in self.get_datatrons():
            for device in self.search_devices(datatron["id"], query, fields):
                results.append({**device, "datatron_id": datatron["id"]})
        return results

    def find_datapoints(
        self, query: str, fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search all datapoints across all datatrons and devices.

        Each result dict includes extra ``datatron_id`` and ``device_id`` keys.
        """
        fields = fields or ["label", "uuid", "unit"]
        results = []
        for datatron in self.get_datatrons():
            for device in self.get_devices(datatron["id"]):
                for dp in self.search_datapoints(
                    datatron["id"], device["id"], query, fields
                ):
                    results.append(
                        {**dp, "datatron_id": datatron["id"], "device_id": device["id"]}
                    )
        return results

    # ------------------------------------------------------------------
    # High-level access — builds device_metadata / device_uuids at init
    # ------------------------------------------------------------------

    def _api_access(self) -> None:
        """Navigate datatrons → devices → datapoints and populate internal state."""
        for device_name in self.device_names:
            metadata: list = []
            for datatron in self.get_datatrons():
                for device in self.get_devices(datatron["id"]):
                    if device["name"] == device_name:
                        metadata = self.get_datapoints(datatron["id"], device["id"])
                        break
                if metadata:
                    break

            metadata_df = pd.DataFrame(metadata)
            if not metadata_df.empty:
                if self.filter_enabled:
                    metadata_df = metadata_df[
                        metadata_df[  # noqa: E712 pandas-series-bool-comparison
                            "enabled"
                        ]  # noqa: E712 pandas-series-bool-comparison
                        == True  # noqa: E712 pandas-series-bool-comparison
                    ]  # noqa: E712 pandas-series-bool-comparison

                metadata_df = metadata_df[["uuid", "label", "config"]]

                if self.required_uuid_list:
                    metadata_df = metadata_df[
                        metadata_df["uuid"].isin(self.required_uuid_list)
                    ]

                self.device_metadata[device_name] = metadata_df
                self.device_uuids[device_name] = metadata_df["uuid"].tolist()
                self._export_json(metadata_df.to_dict(orient="records"), device_name)

    def _export_json(self, data_points: List[Dict], device_name: str) -> None:
        """Write data points to a JSON file for the specified device."""
        file_name = (
            f"{self.output_path}/{device_name.replace(' ', '_')}_data_points.json"
        )
        with open(file_name, "w") as f:
            json.dump(data_points, f, indent=2)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_all_uuids(self) -> Dict[str, List[str]]:
        """Return ``{device_name: [uuid, ...]}`` — ready for the Azure parquet loader."""
        return self.device_uuids

    def get_all_metadata(self) -> Dict[str, List[Dict]]:
        """Return ``{device_name: [record, ...]}`` with uuid/label/config columns."""
        return {
            device: metadata.to_dict(orient="records")
            for device, metadata in self.device_metadata.items()
        }

    def display_dataframe(self, device_name: Optional[str] = None) -> None:
        """Log the metadata DataFrame for one device or all devices."""
        if device_name:
            if device_name in self.device_metadata:
                logger.info("Metadata for device: %s", device_name)
                logger.info("\n%s", self.device_metadata[device_name])
            else:
                logger.warning("No metadata found for device: %s", device_name)
        else:
            for device, metadata in self.device_metadata.items():
                logger.info("Metadata for device: %s", device)
                logger.info("\n%s", metadata)
