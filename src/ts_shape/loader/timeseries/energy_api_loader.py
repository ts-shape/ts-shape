"""Energy API Loader

Load timeseries data from energy data REST APIs and normalize into the
standard ts_shape DataFrame schema (systime, uuid, value_double, is_delta).

Classes:
- EnergyAPILoader: Fetch energy data from REST APIs and return DataFrames.
"""

import logging
import pandas as pd  # type: ignore
import requests
from typing import Any

logger = logging.getLogger(__name__)


class EnergyAPILoader:
    """Load energy timeseries data from REST APIs.

    Fetches JSON data from energy REST endpoints and normalises the
    response into the standard ts_shape DataFrame columns.

    Example usage::

        loader = EnergyAPILoader(
            base_url="https://api.energy-provider.example/v1",
            headers={"Authorization": "Bearer <token>"},
        )

        # Fetch as DataFrame
        df = loader.fetch_data_as_dataframe(
            endpoint="/meters/readings",
            params={"meter_id": "M-001", "start": "2024-01-01", "end": "2024-01-02"},
            time_key="timestamp",
            value_key="value",
            uuid_value="meter:M-001",
        )
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialise the loader.

        Args:
            base_url: Root URL of the energy API (no trailing slash).
            headers: Optional HTTP headers (e.g. auth tokens).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_data_as_dataframe(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        time_key: str = "timestamp",
        value_key: str = "value",
        uuid_value: str = "energy:default",
        data_root: str | None = None,
    ) -> pd.DataFrame:
        """Fetch energy data from a REST endpoint and return a DataFrame.

        Args:
            endpoint: API endpoint path (e.g. ``/meters/readings``).
            params: Query parameters forwarded to ``requests.get``.
            time_key: JSON key containing the timestamp.
            value_key: JSON key containing the numeric reading.
            uuid_value: UUID string to assign to every row.
            data_root: Optional key in the JSON response that contains the
                       list of records (e.g. ``"data"`` or ``"readings"``).
                       When *None* the response is expected to be a list.

        Returns:
            DataFrame with columns: systime, uuid, value_double, is_delta.
        """
        records = self._fetch_json(endpoint, params=params)
        return self._normalize(
            records,
            time_key=time_key,
            value_key=value_key,
            uuid_value=uuid_value,
            data_root=data_root,
        )

    def fetch_multiple_meters(
        self,
        endpoint: str,
        meter_ids: list[str],
        *,
        meter_param: str = "meter_id",
        params: dict[str, Any] | None = None,
        time_key: str = "timestamp",
        value_key: str = "value",
        data_root: str | None = None,
    ) -> pd.DataFrame:
        """Fetch data for several meters and combine into one DataFrame.

        Each meter receives its own ``uuid`` equal to
        ``"energy:<meter_id>"``.

        Args:
            endpoint: API endpoint path.
            meter_ids: List of meter identifiers.
            meter_param: Query-parameter name used to select a meter.
            params: Additional query parameters (shared across calls).
            time_key: JSON key for timestamp.
            value_key: JSON key for value.
            data_root: JSON key wrapping the record list.

        Returns:
            Combined DataFrame sorted by systime.
        """
        frames: list[pd.DataFrame] = []
        base_params = dict(params) if params else {}

        for mid in meter_ids:
            call_params = {**base_params, meter_param: mid}
            df = self.fetch_data_as_dataframe(
                endpoint,
                params=call_params,
                time_key=time_key,
                value_key=value_key,
                uuid_value=f"energy:{mid}",
                data_root=data_root,
            )
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["systime", "uuid", "value_double", "is_delta"])

        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("systime").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Execute GET request and return parsed JSON."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = requests.get(
            url, headers=self.headers, params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _normalize(
        self,
        raw: Any,
        *,
        time_key: str,
        value_key: str,
        uuid_value: str,
        data_root: str | None,
    ) -> pd.DataFrame:
        """Convert raw JSON into a standard ts_shape DataFrame."""
        if data_root is not None and isinstance(raw, dict):
            records = raw.get(data_root, [])
        elif isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = [raw]
        else:
            records = []

        if not records:
            return pd.DataFrame(columns=["systime", "uuid", "value_double", "is_delta"])

        df = pd.DataFrame(records)

        # Map time column
        if time_key in df.columns:
            df["systime"] = pd.to_datetime(df[time_key])
        else:
            return pd.DataFrame(columns=["systime", "uuid", "value_double", "is_delta"])

        # Map value column
        if value_key in df.columns:
            df["value_double"] = pd.to_numeric(df[value_key], errors="coerce")
        else:
            df["value_double"] = float("nan")

        df["uuid"] = uuid_value
        df["is_delta"] = True

        result = df[["systime", "uuid", "value_double", "is_delta"]].copy()
        return result.sort_values("systime").reset_index(drop=True)
