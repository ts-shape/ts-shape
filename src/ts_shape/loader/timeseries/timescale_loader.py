import logging
from pathlib import Path
import pandas as pd  # type: ignore
from sqlalchemy import create_engine
from typing import List, Dict

logger = logging.getLogger(__name__)


class TimescaleDBDataAccess:
    def __init__(
        self,
        start_timestamp: str,
        end_timestamp: str,
        uuids: List[str],
        db_config: Dict[str, str],
    ):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.uuids = uuids
        self.db_config = db_config
        self.engine = create_engine(
            f'postgresql://{db_config["db_user"]}:{db_config["db_pass"]}@{db_config["db_host"]}/{db_config["db_name"]}'
        )

    def _fetch_data(self, uuid: str) -> pd.DataFrame:
        query = f"""
            SELECT uuid::text, systime, value_integer, value_string, value_double, value_bool, is_delta 
            FROM telemetry 
            WHERE uuid = '{uuid}' 
              AND systime BETWEEN '{self.start_timestamp}' AND '{self.end_timestamp}' 
            ORDER BY systime ASC
        """
        return pd.read_sql(query, self.engine, chunksize=10000)

    def fetch_data_as_parquet(self, output_dir: str):
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
        """
        Retrieves timeseries data from TimescaleDB and returns it as a single DataFrame.
        :return: A combined DataFrame with data for all specified UUIDs within the time range.
        """
        df_list = [chunk for uuid in self.uuids for chunk in self._fetch_data(uuid)]
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
