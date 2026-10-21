import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

try:
    import s3fs
except ImportError:  # optional extra: ts-shape[s3]
    s3fs = None

from ts_shape.loader._utils import (
    empty_signal_frame,
    loader_repr,
    redact_url,
    require_config,
    require_optional,
    retry_call,
    trim_to_time_range,
    utc_folder_hours,
)
from ts_shape.utils.deprecation import deprecated_attribute, renamed_kwargs

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

    #: Deprecated names of :attr:`start` / :attr:`end`.
    start_timestamp = deprecated_attribute("start")
    end_timestamp = deprecated_attribute("end")

    @renamed_kwargs(start_timestamp="start", end_timestamp="end")
    def __init__(
        self,
        start: str,
        end: str,
        uuids: list[str],
        s3_config: dict[str, str],
    ):
        """Initialize the S3ProxyDataAccess object.

        Args:
            start: Start timestamp in "Year-Month-Day Hour:Minute:Second"
                format.
            end: End timestamp in "Year-Month-Day Hour:Minute:Second"
                format.
            uuids: List of UUIDs to retrieve data for.
            s3_config: Configuration dictionary for the S3 connection. Must contain
                ``endpoint_url``, ``key``, ``secret``, ``use_ssl``,
                ``version_aware`` and ``s3_path_base``.
        """
        require_optional(s3fs, "s3fs", "s3")
        require_config(s3_config, _REQUIRED_S3_KEYS, name="s3_config")
        self.start = start
        self.end = end
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

    def __repr__(self) -> str:
        # key and secret are left out.
        return loader_repr(
            self,
            start=self.start,
            end=self.end,
            uuid_count=len(self.uuids),
            endpoint_url=redact_url(str(self.s3_config["endpoint_url"])),
            s3_path_base=self.s3_path_base,
        )

    def _generate_timeslot_paths(self) -> Iterator[Path]:
        """Generate time-based directory paths for each hour in the range.

        The folders are UTC hours: the hour that contains ``start`` through the
        hour that contains ``end``. tz-aware bounds are converted to UTC;
        naive bounds are taken as UTC.

        Returns:
            A generator yielding paths in the format year/month/day/hour.
        """
        for timeslot in utc_folder_hours(self.start, self.end):
            timeslot_dir = Path(
                str(timeslot.year),
                str(timeslot.month).zfill(2),
                str(timeslot.day).zfill(2),
                str(timeslot.hour).zfill(2),
            )
            yield timeslot_dir

    def _fetch_parquet(self, uuid: str, timeslot_dir: Path) -> "pd.DataFrame | None":
        """Fetch a Parquet file from S3 for a specific UUID and time slot.

        Args:
            uuid: The UUID for which data is being retrieved.
            timeslot_dir: Directory path for the specific time slot.

        Returns:
            DataFrame if the file is found, else None.
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
        """Retrieve timeseries data from S3 and save it as Parquet files.

        Each file is saved as ``<output_dir>/year/month/day/hour/<uuid>.parquet``,
        with only the rows whose ``systime`` lies in ``[start, end]`` (a file
        left with no rows is not written).

        Args:
            output_dir: Base directory to save the Parquet files.
        """
        for timeslot_dir in self._generate_timeslot_paths():
            for uuid in set(self.uuids):
                df = self._fetch_parquet(uuid, timeslot_dir)
                if df is not None:
                    df = trim_to_time_range(df, self.start, self.end)
                if df is not None and not df.empty:
                    output_path = Path(output_dir, timeslot_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(output_path / f"{uuid}.parquet")

    def fetch_data_as_dataframe(self) -> pd.DataFrame:
        """Retrieve timeseries data from S3 and return it as a single DataFrame.

        Hour folders are read in full; rows whose ``systime`` falls outside
        ``[start, end]`` (both inclusive) are then dropped.

        Returns:
            A combined DataFrame with data for all specified UUIDs and time
            slots; an empty long-format frame (see
            :func:`~ts_shape.loader._utils.empty_signal_frame`) when no file
            was found.
        """
        data_frames = [
            df
            for timeslot_dir in self._generate_timeslot_paths()
            for uuid in set(self.uuids)
            if (df := self._fetch_parquet(uuid, timeslot_dir)) is not None
        ]
        if not data_frames:
            return empty_signal_frame()
        out = pd.concat(data_frames, ignore_index=True)
        return trim_to_time_range(out, self.start, self.end)
