"""Energy API Loader

Load timeseries data from energy data REST APIs and normalize into the
standard ts_shape DataFrame schema (systime, uuid, value_double, is_delta).

``systime`` is tz-aware UTC: timestamps with an offset are converted, those
without one are taken as UTC (the same convention as
:class:`~ts_shape.loader.timeseries.azure_blob_energy_loader.AzureBlobEnergyLoader`).

Classes:
- EnergyAPILoader: Fetch energy data from REST APIs and return DataFrames.
"""

import logging
import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import requests
else:
    try:
        import requests
    except ImportError:  # optional extra: ts-shape[api]
        requests = None

from ts_shape.errors import DataQualityWarning
from ts_shape.loader._utils import (
    loader_repr,
    parse_utc_times,
    redact_mapping,
    redact_url,
    require_optional,
)
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import caller_stacklevel

logger = logging.getLogger(__name__)


class EnergyAPILoader:
    """Load energy timeseries data from REST APIs.

    Fetches JSON data from energy REST endpoints and normalizes the
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
        """Initialize the loader.

        Args:
            base_url: Root URL of the energy API (no trailing slash).
            headers: Optional HTTP headers (e.g. auth tokens).
            timeout: Request timeout in seconds.
        """
        require_optional(requests, "requests", "api")
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout

    def __repr__(self) -> str:
        # Header values (tokens) and query values (API keys) are masked.
        return loader_repr(
            self,
            base_url=redact_url(self.base_url),
            headers=redact_mapping(self.headers),
            timeout=self.timeout,
        )

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
            df["systime"] = parse_utc_times(df[time_key])
        else:
            warnings.warn(
                f"{uuid_value}: none of the {len(df)} API record(s) has the time "
                f"key {time_key!r} (keys: {list(df.columns)}); they are skipped.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return pd.DataFrame(columns=["systime", "uuid", "value_double", "is_delta"])

        # Map value column
        if value_key in df.columns:
            df["value_double"] = to_numeric_checked(
                df[value_key], what=f"{uuid_value} value key {value_key!r}"
            ).astype("float64")
        else:
            warnings.warn(
                f"{uuid_value}: none of the {len(df)} API record(s) has the value "
                f"key {value_key!r} (keys: {list(df.columns)}); values are NaN.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            df["value_double"] = float("nan")

        df["uuid"] = uuid_value
        df["is_delta"] = True

        result = df[["systime", "uuid", "value_double", "is_delta"]].copy()
        return result.sort_values("systime").reset_index(drop=True)
