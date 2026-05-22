"""Line balancing and takt analysis for assembly / production lines.

Classic industrial-engineering line-balancing metrics computed from
per-station cycle-completion signals:

- station cycle times (windowed),
- line balance efficiency, balance delay and smoothness index,
- theoretical minimum number of stations for a given takt,
- a Yamazumi (station-loading) table.

A station is identified by a boolean cycle-completion-trigger signal; the
time between consecutive rising edges is that station's cycle time.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.events._output import empty_event_df, finalize_summary_df
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

_Number = Union[int, float, str]


def _to_seconds(value: Optional[_Number]) -> Optional[float]:
    """Coerce a duration (seconds number or pandas offset string) to seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return pd.to_timedelta(value).total_seconds()


class LineBalancingEvents(Base):
    """Line balancing and takt analysis from per-station cycle signals.

    Example usage::

        lb = LineBalancingEvents(
            df,
            station_uuids={
                "uuid_s1": "Station 1",
                "uuid_s2": "Station 2",
                "uuid_s3": "Station 3",
            },
        )
        lb.station_cycle_times(window="1h")
        lb.balance_metrics(takt_time="55s", window="1h")
        lb.yamazumi(demand=480, available_time="8h")
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        station_uuids: Dict[str, str],
        *,
        event_uuid: str = "prod:line_balance",
        value_column: str = "value_bool",
        time_column: str = "systime",
    ) -> None:
        """Initialize the line-balancing analyser.

        Args:
            dataframe: Input DataFrame with timeseries data.
            station_uuids: Mapping of cycle-completion-trigger UUID -> station
                name, in line order, e.g. ``{"u1": "Station 1"}``.
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the boolean cycle trigger.
            time_column: Name of the timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.station_uuids = dict(station_uuids)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        for uuid in self.station_uuids:
            self._validate_uuid(self.dataframe, uuid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rising_edge_times(self, uuid: str) -> pd.Series:
        """Return the timestamps of rising edges of one station's trigger."""
        sig = (
            self.dataframe[self.dataframe["uuid"] == uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if sig.empty:
            return pd.Series(dtype="datetime64[ns]")
        sig[self.time_column] = pd.to_datetime(sig[self.time_column])
        state = sig[self.value_column].fillna(False).astype(bool)
        prev = state.shift(fill_value=False)
        edges = sig[(~prev) & state]
        return edges[self.time_column].reset_index(drop=True)

    def _all_cycles(self) -> pd.DataFrame:
        """Long-format cycle table for every station.

        Columns: ``uuid``, ``station_name``, ``completion``, ``cycle_time``.
        The cycle time is the gap to the previous completion at that station.
        """
        rows: List[pd.DataFrame] = []
        for uuid, name in self.station_uuids.items():
            times = self._rising_edge_times(uuid)
            if len(times) < 2:
                continue
            cycle_time = times.diff().dt.total_seconds()
            rows.append(
                pd.DataFrame(
                    {
                        "uuid": uuid,
                        "station_name": name,
                        "completion": times.iloc[1:].reset_index(drop=True),
                        "cycle_time": cycle_time.iloc[1:].reset_index(drop=True),
                    }
                )
            )
        if not rows:
            return pd.DataFrame(
                columns=["uuid", "station_name", "completion", "cycle_time"]
            )
        return pd.concat(rows, ignore_index=True)

    def _resolve_takt(
        self,
        takt_time: Optional[_Number],
        demand: Optional[float],
        available_time: Optional[_Number],
    ) -> Optional[float]:
        """Resolve takt time (seconds) from an explicit value or demand/time."""
        if takt_time is not None:
            return _to_seconds(takt_time)
        avail = _to_seconds(available_time)
        if demand is not None and avail is not None and demand > 0:
            return avail / demand
        return None

    @staticmethod
    def _window_bounds(window: str, starts: pd.Series) -> pd.Series:
        """Return the window-end timestamps for a series of window starts."""
        return starts + pd.Timedelta(window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def station_cycle_times(self, window: str = "1h") -> pd.DataFrame:
        """Per-station cycle-time statistics per time window.

        Args:
            window: Resample window (e.g. ``"1h"``, ``"30m"``).

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            uuid, station_name, cycle_time_mean, cycle_time_median,
            cycle_time_std, cycle_count.
        """
        extra = [
            "uuid",
            "station_name",
            "cycle_time_mean",
            "cycle_time_median",
            "cycle_time_std",
            "cycle_count",
        ]
        cycles = self._all_cycles()
        if cycles.empty:
            return empty_event_df("summary", extra_cols=extra)

        cycles["start"] = cycles["completion"].dt.floor(window)
        grouped = cycles.groupby(["start", "uuid", "station_name"], sort=True)
        agg = grouped["cycle_time"].agg(
            cycle_time_mean="mean",
            cycle_time_median="median",
            cycle_time_std="std",
            cycle_count="count",
        )
        out = agg.reset_index()
        out["end"] = self._window_bounds(window, out["start"])
        out["cycle_time_mean"] = out["cycle_time_mean"].round(3)
        out["cycle_time_median"] = out["cycle_time_median"].round(3)
        out["cycle_time_std"] = out["cycle_time_std"].round(3)
        return finalize_summary_df(out)

    def balance_metrics(
        self,
        *,
        takt_time: Optional[_Number] = None,
        demand: Optional[float] = None,
        available_time: Optional[_Number] = None,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Line-level balance metrics per time window.

        Balance efficiency = sum(station times) / (n_stations * bottleneck).
        Takt is resolved from ``takt_time`` or from ``demand`` plus
        ``available_time``; when neither is given, ``theoretical_min_stations``
        is ``NaN``.

        Args:
            takt_time: Takt time as seconds or an offset string (e.g. "55s").
            demand: Units required over ``available_time``.
            available_time: Available production time (seconds or offset string).
            window: Resample window.

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            n_stations, bottleneck_uuid, bottleneck_cycle_time, takt_seconds,
            balance_efficiency_pct, balance_delay_pct, smoothness_index,
            theoretical_min_stations.
        """
        extra = [
            "n_stations",
            "bottleneck_uuid",
            "bottleneck_cycle_time",
            "takt_seconds",
            "balance_efficiency_pct",
            "balance_delay_pct",
            "smoothness_index",
            "theoretical_min_stations",
        ]
        cycles = self._all_cycles()
        if cycles.empty:
            return empty_event_df("summary", extra_cols=extra)

        takt = self._resolve_takt(takt_time, demand, available_time)
        cycles["start"] = cycles["completion"].dt.floor(window)
        means = cycles.groupby(["start", "uuid"])["cycle_time"].mean().reset_index()

        rows: List[Dict[str, Any]] = []
        for start, grp in means.groupby("start", sort=True):
            times = grp["cycle_time"].to_numpy(dtype=float)
            n = len(times)
            bottleneck = float(times.max())
            total = float(times.sum())
            efficiency = total / (n * bottleneck) if bottleneck > 0 else 0.0
            smoothness = float(np.sqrt(np.sum((bottleneck - times) ** 2)))
            if takt is not None and takt > 0:
                theoretical_min = math.ceil(total / takt)
            else:
                theoretical_min = np.nan
            bottleneck_uuid = grp.loc[grp["cycle_time"].idxmax(), "uuid"]
            rows.append(
                {
                    "start": start,
                    "end": start + pd.Timedelta(window),
                    "n_stations": n,
                    "bottleneck_uuid": bottleneck_uuid,
                    "bottleneck_cycle_time": round(bottleneck, 3),
                    "takt_seconds": round(takt, 3) if takt is not None else np.nan,
                    "balance_efficiency_pct": round(efficiency * 100, 2),
                    "balance_delay_pct": round((1 - efficiency) * 100, 2),
                    "smoothness_index": round(smoothness, 3),
                    "theoretical_min_stations": theoretical_min,
                }
            )

        return finalize_summary_df(pd.DataFrame(rows))

    def yamazumi(
        self,
        *,
        takt_time: Optional[_Number] = None,
        demand: Optional[float] = None,
        available_time: Optional[_Number] = None,
    ) -> pd.DataFrame:
        """Yamazumi (station-loading) table over the whole dataset.

        Args:
            takt_time: Takt time as seconds or an offset string.
            demand: Units required over ``available_time``.
            available_time: Available production time (seconds or offset string).

        Returns:
            DataFrame with columns: uuid, station_name, cycle_time_mean,
            takt_seconds, loading_pct, idle_to_takt_seconds, is_bottleneck.
            Stations are returned in the order given to the constructor.
        """
        columns = [
            "uuid",
            "station_name",
            "cycle_time_mean",
            "takt_seconds",
            "loading_pct",
            "idle_to_takt_seconds",
            "is_bottleneck",
        ]
        cycles = self._all_cycles()
        if cycles.empty:
            return pd.DataFrame(columns=columns)

        takt = self._resolve_takt(takt_time, demand, available_time)
        means = cycles.groupby("uuid")["cycle_time"].mean()
        bottleneck_uuid = means.idxmax()

        rows: List[Dict[str, Any]] = []
        for uuid, name in self.station_uuids.items():
            if uuid not in means.index:
                continue
            mean_ct = float(means[uuid])
            if takt is not None and takt > 0:
                loading = round(mean_ct / takt * 100, 2)
                idle = round(max(takt - mean_ct, 0.0), 3)
            else:
                loading = np.nan
                idle = np.nan
            rows.append(
                {
                    "uuid": uuid,
                    "station_name": name,
                    "cycle_time_mean": round(mean_ct, 3),
                    "takt_seconds": round(takt, 3) if takt is not None else np.nan,
                    "loading_pct": loading,
                    "idle_to_takt_seconds": idle,
                    "is_bottleneck": uuid == bottleneck_uuid,
                }
            )
        return pd.DataFrame(rows, columns=columns)
