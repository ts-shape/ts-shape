import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy import create_engine, text
else:
    try:
        from sqlalchemy import create_engine, text
    except ImportError:  # optional extra: ts-shape[sql]
        create_engine = text = None

from ts_shape.loader._utils import require_optional

logger = logging.getLogger(__name__)


class TimescaleDBDataAccess:
    """Load timeseries data for given UUIDs and a time range from TimescaleDB."""

    def __init__(
        self,
        start_timestamp: str,
        end_timestamp: str,
        uuids: list[str],
        db_config: dict[str, str],
    ):
        """Initialize the TimescaleDB loader and create the database engine.

        Args:
            start_timestamp: Start of the time range (inclusive) to query.
            end_timestamp: End of the time range (inclusive) to query.
            uuids: List of UUIDs to retrieve data for.
            db_config: Connection settings with the keys ``db_user``,
                ``db_pass``, ``db_host`` and ``db_name``.
        """
        require_optional(create_engine, "sqlalchemy", "sql")
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.uuids = uuids
        self.db_config = db_config
        self.engine = create_engine(
            f'postgresql://{db_config["db_user"]}:{db_config["db_pass"]}@{db_config["db_host"]}/{db_config["db_name"]}'
        )

    def _fetch_data(self, uuid: str) -> pd.DataFrame:
        # Bound parameters keep values out of the SQL text — this avoids
        # injection and quoting bugs (e.g. a uuid containing an apostrophe).
        query = text("""
            SELECT uuid::text, systime, value_integer, value_string,
                   value_double, value_bool, is_delta
            FROM telemetry
            WHERE uuid = :uuid
              AND systime BETWEEN :start_ts AND :end_ts
            ORDER BY systime ASC
            """)
        return pd.read_sql(
            query,
            self.engine,
            params={
                "uuid": uuid,
                "start_ts": self.start_timestamp,
                "end_ts": self.end_timestamp,
            },
            chunksize=10000,
        )

    def fetch_data_as_parquet(self, output_dir: str):
        """Retrieve timeseries data from TimescaleDB and save it as Parquet files.

        Rows are grouped by the hour of ``systime`` and written to one file per
        UUID and hour at ``<output_dir>/year/month/day/hour/<uuid>.parquet``.

        Args:
            output_dir: Base directory to save the Parquet files.
        """
        for uuid in self.uuids:
            hourly_data = {}

            for chunk in self._fetch_data(uuid):
                if not chunk.empty:
                    # Group the data by hour to accumulate rows for each hour
                    chunk["hour"] = chunk["systime"].dt.floor("h")
                    grouped = chunk.groupby("hour")

                    for hour, group in grouped:
                        if hour not in hourly_data:
                            hourly_data[hour] = group
                        else:
                            hourly_data[hour] = pd.concat(
                                [hourly_data[hour], group], ignore_index=True
                            )

            # Write each hour's data to a single Parquet file
            for hour, data in hourly_data.items():
                timeslot_dir = Path(
                    str(hour.year),
                    str(hour.month).zfill(2),
                    str(hour.day).zfill(2),
                    str(hour.hour).zfill(2),
                )
                output_path = Path(output_dir, timeslot_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                data.to_parquet(output_path / f"{uuid}.parquet", index=False)

    def fetch_data_as_dataframe(self) -> pd.DataFrame:
        """Retrieve timeseries data from TimescaleDB as a single DataFrame.

        Returns:
            A combined DataFrame with data for all specified UUIDs within the time
            range.
        """
        df_list = [chunk for uuid in self.uuids for chunk in self._fetch_data(uuid)]
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
