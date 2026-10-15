import json
import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy import URL, create_engine, text
else:
    try:
        from sqlalchemy import create_engine, text
    except ImportError:  # optional extra: ts-shape[sql]
        create_engine = text = None

from ts_shape.loader._utils import (
    POSTGRES_DRIVER,
    create_postgres_engine,
    postgres_url,
    require_optional,
)

logger = logging.getLogger(__name__)


class DatapointDB:
    """Load datapoint metadata per device from the PostgreSQL config repository.

    Needs the ``postgres`` extra (``pip install ts-shape[postgres]``).
    ``repr(db)`` masks the password.
    """

    def __init__(
        self,
        device_names: list[str],
        db_user: str,
        db_pass: str,
        db_host: str,
        output_path: str = "data",
        required_uuid_list: list[str] | None = None,
        filter_enabled: bool = True,
        *,
        db_port: int = 5432,
        db_name: str = "config_repository",
        db_driver: str = POSTGRES_DRIVER,
    ):
        """Initialize the DatapointDB class and load metadata for each device.

        Args:
            device_names: List of device names to retrieve metadata for.
            db_user: Database user.
            db_pass: Database password.
            db_host: Database host.
            output_path: Directory to save JSON files.
            required_uuid_list: List of UUIDs to filter the metadata (optional).
            filter_enabled: Whether to filter metadata by "enabled == True" and
                "archived == False" (default is True).
            db_port: Database port. Default 5432.
            db_name: Database name. Default ``"config_repository"``.
            db_driver: DB-API driver. Default ``"psycopg2"`` (the ``postgres``
                extra).

        Raises:
            ImportError: SQLAlchemy or the database driver is not installed.
        """
        require_optional(create_engine, "sqlalchemy", "sql")
        self.device_names = device_names
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_driver = db_driver
        self.output_path = output_path
        self.required_uuid_list = required_uuid_list or []
        self.filter_enabled = filter_enabled
        self.device_metadata: dict[str, pd.DataFrame] = (
            {}
        )  # Store metadata for each device
        self.device_uuids: dict[str, list[str]] = {}  # Store UUIDs for each device
        self._db_access()

    @property
    def url(self) -> "URL":
        """The connection URL; ``str(url)`` masks the password."""
        return postgres_url(
            username=self.db_user,
            password=self.db_pass,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            driver=self.db_driver,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(device_names={self.device_names!r}, "
            f"url={str(self.url)!r})"
        )

    def _db_access(self) -> None:
        """Connect to the database and retrieve metadata for each device."""
        engine = create_postgres_engine(create_engine, self.url)

        with engine.connect() as conn:
            for device_name in self.device_names:
                query = """
                    SELECT dp.uuid, dp.label, dp.config
                    FROM data_points dp
                    INNER JOIN devices dev ON dev.id = dp.device_id
                    WHERE dev.name = :device_name
                """
                if self.filter_enabled:
                    query += " AND dp.enabled = true AND dp.archived = false"

                result = conn.execute(text(query), {"device_name": device_name})
                data_points = [
                    {"uuid": r[0], "label": r[1], "config": r[2]}
                    for r in result.fetchall()
                ]

                # Convert to DataFrame and filter by required UUIDs if necessary.
                # Explicit columns: a device without data points still gets
                # an (empty) frame with uuid / label / config.
                metadata_df = pd.DataFrame(
                    data_points, columns=["uuid", "label", "config"]
                )
                if not metadata_df.empty and self.required_uuid_list:
                    metadata_df = metadata_df[
                        metadata_df["uuid"].isin(self.required_uuid_list)
                    ]

                # Store metadata and UUIDs for the device
                self.device_metadata[device_name] = metadata_df
                self.device_uuids[device_name] = metadata_df["uuid"].tolist()

                # Export to JSON file
                self._export_json(metadata_df.to_dict(orient="records"), device_name)

    def _export_json(self, data_points: list[dict[str, str]], device_name: str) -> None:
        """Export data points to a JSON file for the specified device."""
        file_name = (
            f"{self.output_path}/{device_name.replace(' ', '_')}_data_points.json"
        )
        with open(file_name, "w") as f:
            json.dump(data_points, f, indent=2)

    def get_all_uuids(self) -> dict[str, list[str]]:
        """Return a dictionary of UUIDs for each device."""
        return self.device_uuids

    def get_all_metadata(self) -> dict[str, list[dict[str, str]]]:
        """Return a dictionary of metadata for each device."""
        return {
            device: metadata.to_dict(orient="records")
            for device, metadata in self.device_metadata.items()
        }

    def display_dataframe(
        self, device_name: str | None = None, aggregate: bool = False
    ) -> None:
        """Display metadata as a DataFrame for a specific device or all devices.

        Args:
            device_name: Name of the device to display metadata for (optional).
            aggregate: If True, combine metadata from all devices into a single
                DataFrame.
        """
        if aggregate:
            combined_df = pd.concat(
                self.device_metadata.values(), keys=self.device_metadata.keys()
            )
            logger.info("Aggregated metadata for all devices:")
            logger.info("\n%s", combined_df)
        elif device_name:
            if device_name in self.device_metadata:
                logger.info("Metadata for device: %s", device_name)
                logger.info("\n%s", self.device_metadata[device_name])
            else:
                logger.warning("No metadata found for device: %s", device_name)
        else:
            for device, metadata in self.device_metadata.items():
                logger.info("Metadata for device: %s", device)
                logger.info("\n%s", metadata)
