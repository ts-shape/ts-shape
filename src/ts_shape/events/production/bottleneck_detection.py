import logging
from typing import Any, Literal, overload

import pandas as pd
from pandas.tseries.frequencies import to_offset

from ts_shape.events._compat import warn_dict_return
from ts_shape.events._windows import floor_local
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only

logger = logging.getLogger(__name__)


class BottleneckDetectionEvents(Base):
    """Production: Bottleneck Detection

    Identify which station constrains a production line by analyzing
    utilization of multiple boolean run-state signals.

    Methods:
    - station_utilization: Per-station uptime percentage per window.
    - detect_bottleneck: Identify the bottleneck station per window.
    - shifting_bottleneck: Track when the bottleneck moves between stations.
    - throughput_constraint_summary: Summary statistics across all stations.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        event_uuid: str = "prod:bottleneck",
        value_column: str = "value_bool",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.event_uuid = event_uuid
        self.value_column = value_column

    @keyword_only(2)
    def station_utilization(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = True,
    ) -> pd.DataFrame:
        """Per-station uptime percentage per time window.

        Args:
            station_uuids: List of UUID strings identifying station run-state signals.
            window: Resample window (e.g. '1h', '30m').
            time_weighted: When True (default), utilization is
                time-weighted: each sample is weighted by the gap to the next
                sample (as in ``RuntimeAccountingEvents``), so it reflects
                wall-clock time in the running state on change-encoded and
                irregularly sampled signals. The trailing sample of a station
                has no successor and carries no weight. When False,
                utilization is the mean of the boolean samples in the window,
                which equals the time share only under uniform sampling.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid (the station uuid), source_uuid (the
            station uuid), utilization_pct.
        """
        rows: list[dict[str, Any]] = []
        offset = to_offset(window)

        for uid in station_uuids:
            station = self.dataframe[self.dataframe["uuid"] == uid].copy()
            if station.empty:
                continue
            station[self.time_column] = pd.to_datetime(station[self.time_column])
            station = station.sort_values(self.time_column)
            station["state"] = station[self.value_column].fillna(False).astype(float)

            if time_weighted:
                # Weight each sample by the gap to the next sample, bucketed by
                # the sample's window. The trailing sample has no successor.
                station["dur"] = (
                    station[self.time_column].shift(-1) - station[self.time_column]
                ).dt.total_seconds()
                station = station[station["dur"].notna()]
                if station.empty:
                    continue
                station["win"] = floor_local(station[self.time_column], window)
                station["run_dur"] = station["state"] * station["dur"]
                grouped = station.groupby("win").agg(
                    run=("run_dur", "sum"), total=("dur", "sum")
                )
                series = grouped.apply(
                    lambda r: (r["run"] / r["total"]) if r["total"] > 0 else 0.0,
                    axis=1,
                )
            else:
                series = (
                    station.set_index(self.time_column)["state"].resample(window).mean()
                )

            for ts, pct in series.items():
                if pd.notna(pct):
                    rows.append(
                        {
                            "start": ts,
                            "end": ts + offset,
                            "duration_seconds": (ts + offset - ts).total_seconds(),
                            "uuid": uid,
                            "source_uuid": uid,
                            "utilization_pct": pct * 100,
                        }
                    )

        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "uuid",
                    "source_uuid",
                    "utilization_pct",
                ]
            )
        )

    @keyword_only(2)
    def detect_bottleneck(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = True,
    ) -> pd.DataFrame:
        """Identify the bottleneck station per window.

        The bottleneck is the station with the highest utilization —
        it is always running because it is the constraint.

        Args:
            station_uuids: List of station run-state UUIDs.
            window: Resample window.
            time_weighted: Forwarded to :meth:`station_utilization`
                (default True).

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid (the
            detector's event uuid), source_uuid (the bottleneck station uuid),
            bottleneck_uuid, utilization_pct.
        """
        util = self.station_utilization(
            station_uuids, window=window, time_weighted=time_weighted
        )
        if util.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "uuid",
                    "source_uuid",
                    "bottleneck_uuid",
                    "utilization_pct",
                ]
            )

        idx = util.groupby("start")["utilization_pct"].idxmax()
        bottlenecks = util.loc[idx].copy()
        window_td = pd.to_timedelta(window)

        out = pd.DataFrame(
            {
                "start": bottlenecks["start"].array,
                "end": bottlenecks["start"].array + window_td,
                "bottleneck_uuid": bottlenecks["uuid"].values,
                "utilization_pct": bottlenecks["utilization_pct"].values,
            }
        )
        out.insert(
            2, "duration_seconds", (out["end"] - out["start"]).dt.total_seconds()
        )
        out.insert(3, "uuid", self.event_uuid)
        out.insert(4, "source_uuid", out["bottleneck_uuid"])
        return out

    @keyword_only(2)
    def shifting_bottleneck(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = True,
    ) -> pd.DataFrame:
        """Track when the bottleneck identity changes between stations.

        Args:
            station_uuids: List of station run-state UUIDs.
            window: Resample window.
            time_weighted: Forwarded to :meth:`station_utilization`.

        Returns:
            DataFrame with columns: systime, uuid (the detector's event uuid),
            source_uuid (``"<from_uuid>,<to_uuid>"``), from_uuid, to_uuid,
            previous_utilization, new_utilization.
        """
        bottlenecks = self.detect_bottleneck(
            station_uuids, window=window, time_weighted=time_weighted
        )
        if bottlenecks.empty or len(bottlenecks) < 2:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "from_uuid",
                    "to_uuid",
                    "previous_utilization",
                    "new_utilization",
                ]
            )

        shifts: list[dict[str, Any]] = []
        prev_uuid = bottlenecks.iloc[0]["bottleneck_uuid"]
        prev_util = bottlenecks.iloc[0]["utilization_pct"]

        for i in range(1, len(bottlenecks)):
            curr_uuid = bottlenecks.iloc[i]["bottleneck_uuid"]
            curr_util = bottlenecks.iloc[i]["utilization_pct"]
            if curr_uuid != prev_uuid:
                shifts.append(
                    {
                        "systime": bottlenecks.iloc[i]["start"],
                        "uuid": self.event_uuid,
                        "source_uuid": f"{prev_uuid},{curr_uuid}",
                        "from_uuid": prev_uuid,
                        "to_uuid": curr_uuid,
                        "previous_utilization": prev_util,
                        "new_utilization": curr_util,
                    }
                )
            prev_uuid = curr_uuid
            prev_util = curr_util

        return (
            pd.DataFrame(shifts)
            if shifts
            else pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "from_uuid",
                    "to_uuid",
                    "previous_utilization",
                    "new_utilization",
                ]
            )
        )

    @overload
    def throughput_constraint_summary(
        self,
        station_uuids: list[str],
        window: str = ...,
        *,
        time_weighted: bool = ...,
        as_frame: Literal[True] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def throughput_constraint_summary(
        self,
        station_uuids: list[str],
        window: str = ...,
        *,
        time_weighted: bool = ...,
        as_frame: Literal[False],
    ) -> dict[str, Any]: ...

    @keyword_only(2)
    def throughput_constraint_summary(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = True,
        as_frame: bool = True,
    ) -> pd.DataFrame | dict[str, Any]:
        """Summary statistics for bottleneck analysis.

        Args:
            station_uuids: List of station run-state UUIDs.
            window: Resample window.
            time_weighted: Forwarded to :meth:`station_utilization`.
            as_frame: When True (default), return one row per station (see
                below). ``as_frame=False`` returns the dictionary form; it is
                deprecated, emits a ``DeprecationWarning`` and will be removed
                in the next major release.

        Returns:
            With ``as_frame=True`` (default), a summary DataFrame with one row
            per station that has data, in ``station_uuids`` order, columns:

            - start, end, duration_seconds: the analysed span (first window
                start to last window end), the same on every row
            - uuid, source_uuid: the station uuid
            - bottleneck_count: windows in which the station was the
                bottleneck (``bottleneck_counts`` of the dict form)
            - bottleneck_pct: share of windows, 0-100
                (``bottleneck_percentages``)
            - avg_utilization_pct: mean utilization over the windows
                (``avg_utilization_per_station``)
            - most_frequent_bottleneck: True on the station that was the
                bottleneck most often

            With ``as_frame=False`` (deprecated), a dict with:
            bottleneck_counts, bottleneck_percentages,
            most_frequent_bottleneck, avg_utilization_per_station.
        """
        if not as_frame:
            warn_dict_return("BottleneckDetectionEvents.throughput_constraint_summary")
        bottlenecks = self.detect_bottleneck(
            station_uuids, window=window, time_weighted=time_weighted
        )
        util = self.station_utilization(
            station_uuids, window=window, time_weighted=time_weighted
        )
        summary = self._constraint_summary(bottlenecks, util)
        if not as_frame:
            return summary
        return self._constraint_frame(station_uuids, bottlenecks, util, summary)

    _CONSTRAINT_COLUMNS = [
        "start",
        "end",
        "duration_seconds",
        "uuid",
        "source_uuid",
        "bottleneck_count",
        "bottleneck_pct",
        "avg_utilization_pct",
        "most_frequent_bottleneck",
    ]

    def _constraint_frame(
        self,
        station_uuids: list[str],
        bottlenecks: pd.DataFrame,
        util: pd.DataFrame,
        summary: dict[str, Any],
    ) -> pd.DataFrame:
        if util.empty:
            return pd.DataFrame(columns=self._CONSTRAINT_COLUMNS)
        start, end = util["start"].min(), util["end"].max()
        seen = set(util["uuid"])
        rows = []
        for uid in dict.fromkeys(station_uuids):
            if uid not in seen:
                continue
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "duration_seconds": (end - start).total_seconds(),
                    "uuid": uid,
                    "source_uuid": uid,
                    "bottleneck_count": int(summary["bottleneck_counts"].get(uid, 0)),
                    "bottleneck_pct": float(
                        summary["bottleneck_percentages"].get(uid, 0.0)
                    ),
                    "avg_utilization_pct": float(
                        summary["avg_utilization_per_station"].get(uid, float("nan"))
                    ),
                    "most_frequent_bottleneck": uid
                    == summary["most_frequent_bottleneck"],
                }
            )
        return pd.DataFrame(rows, columns=self._CONSTRAINT_COLUMNS)

    @staticmethod
    def _constraint_summary(
        bottlenecks: pd.DataFrame, util: pd.DataFrame
    ) -> dict[str, Any]:
        if bottlenecks.empty:
            return {
                "bottleneck_counts": {},
                "bottleneck_percentages": {},
                "most_frequent_bottleneck": None,
                "avg_utilization_per_station": {},
            }

        counts = bottlenecks["bottleneck_uuid"].value_counts().to_dict()
        total = len(bottlenecks)
        percentages = {k: (v / total * 100) for k, v in counts.items()}
        most_frequent = bottlenecks["bottleneck_uuid"].value_counts().idxmax()

        avg_util = {}
        if not util.empty:
            avg_util = util.groupby("uuid")["utilization_pct"].mean().to_dict()

        return {
            "bottleneck_counts": counts,
            "bottleneck_percentages": percentages,
            "most_frequent_bottleneck": most_frequent,
            "avg_utilization_per_station": avg_util,
        }
