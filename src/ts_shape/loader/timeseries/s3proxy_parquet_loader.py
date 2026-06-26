import logging
from collections.abc import Iterator
from pathlib import Path
import pandas as pd
import s3fs

from ts_shape.loader._utils import require_config, retry_call

logger = logging.getLogger(__name__)

_REQUIRED_S3_KEYS = [
    "endpoint_url",
    "key",
    "secret",
    "use_ssl",
    "version_aware",
    "s3_path_base",
]


class S3ProxyDataAccess:
    """
    A class to access timeseries data via an S3 proxy. This class retrieves
    data for specified UUIDs within a defined time range, with the option to
    output data as Parquet files or as a single combined DataFrame.
    """

    def __init__(
        self,
        start_timestamp: str,
        end_timestamp: str,
        uuids: list[str],
        s3_config: dict[str, str],
    ):
        """
        Initialize the S3ProxyDataAccess object.
        :param start_timestamp: Start timestamp in "Year-Month-Day Hour:Minute:Second" format.
        :param end_timestamp: End timestamp in "Year-Month-Day Hour:Minute:Second" format.
        :param uuids: List of UUIDs to retrieve data for.
        :param s3_config: Configuration dictionary for S3 connection.
        """
        require_config(s3_config, _REQUIRED_S3_KEYS, name="s3_config")
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.uuids = uuids
        self.s3_config = s3_config

        # Establish connection to S3 using provided configuration
        self.s3 = s3fs.S3FileSystem(
            endpoint_url=s3_config["endpoint_url"],
            key=s3_config["key"],
            secret=s3_config["secret"],
            use_ssl=s3_config["use_ssl"],
            version_aware=s3_config["version_aware"],
        )
        self.s3_path_base = s3_config["s3_path_base"]

    def _generate_timeslot_paths(self) -> Iterator[Path]:
        """
        Generates a sequence of time-based directory paths for each hour in the range
        between start_timestamp and end_timestamp.
        :return: A generator yielding paths in the format year/month/day/hour.
        """
        for timeslot in pd.date_range(
            start=self.start_timestamp, end=self.end_timestamp, freq="h"
        ):
            timeslot_dir = Path(
                str(timeslot.year),
                str(timeslot.month).zfill(2),
                str(timeslot.day).zfill(2),
                str(timeslot.hour).zfill(2),
            )
            yield timeslot_dir

    def _fetch_parquet(self, uuid: str, timeslot_dir: Path) -> "pd.DataFrame | None":
        """
        Fetches a Parquet file from S3 for a specific UUID and time slot.
        :param uuid: The UUID for which data is being retrieved.
        :param timeslot_dir: Directory path for the specific time slot.
        :return: DataFrame if the file is found, else None.
        """
        s3_path = f"{self.s3_path_base}{timeslot_dir}/{uuid}.parquet"

        def _read() -> pd.DataFrame:
            with self.s3.open(s3_path, "rb") as remote_file:
                return pd.read_parquet(remote_file)

        try:
            # Retry transient S3/network errors; a genuine miss (FileNotFoundError)
            # is excluded so it short-circuits to the "no data" path below.
            return retry_call(
                _read,
                exclude=(FileNotFoundError,),
                description=f"s3 read {s3_path}",
            )
        except FileNotFoundError:
            logger.debug("Data for UUID %s at %s not found.", uuid, timeslot_dir)
            return None

    def fetch_data_as_parquet(self, output_dir: str) -> None:
        """
        Retrieves timeseries data from S3 and saves it as Parquet files.
        Each file is saved in a directory structure of UUID/year/month/day/hour.
        :param output_dir: Base directory to save the Parquet files.
        """
        for timeslot_dir in self._generate_timeslot_paths():
            for uuid in set(self.uuids):
                df = self._fetch_parquet(uuid, timeslot_dir)
                if df is not None:
                    output_path = Path(output_dir, timeslot_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(output_path / f"{uuid}.parquet")

    def fetch_data_as_dataframe(self) -> pd.DataFrame:
        """
        Retrieves timeseries data from S3 and returns it as a single DataFrame.
        :return: A combined DataFrame with data for all specified UUIDs and time slots.
        """
        data_frames = [
            self._fetch_parquet(uuid, timeslot_dir)
            for timeslot_dir in self._generate_timeslot_paths()
            for uuid in set(self.uuids)
        ]
        return (
            pd.concat([df for df in data_frames if df is not None], ignore_index=True)
            if data_frames
            else pd.DataFrame()
        )
