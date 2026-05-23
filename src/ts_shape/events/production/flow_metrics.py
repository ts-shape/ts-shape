"""Flow metrics and Little's Law analysis for production lines.

Treats a process as a queue between an *entry* signal and an *exit* signal
and derives the classic flow metrics an industrial engineer relies on:

- work in process (WIP) over time, time-weighted,
- throughput (units out per window),
- lead time (FIFO-matched entry -> exit),
- a flow summary tying them together via Little's Law
  (``WIP = throughput x lead_time``) plus Process Cycle Efficiency.

Entry and exit are boolean signals; each rising edge is one unit.
"""

import logging
from typing import Optional, Union

import pandas as pd  # type: ignore

from ts_shape.events._output import (
    empty_event_df,
    finalize_point_df,
    finalize_summary_df,
)
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

_Number = Union[int, float, str]


def _to_seconds(value: _Number | None) -> float | None:
    """Coerce a duration (seconds number or pandas offset string) to seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return pd.to_timedelta(value).total_seconds()


class FlowMetricsEvents(Base):
    """WIP, throughput, lead time and Little's Law metrics for a process.

    Example usage::

        flow = FlowMetricsEvents(df, entry_uuid="u_in", exit_uuid="u_out")
        flow.wip_over_time(window="1h")
        flow.throughput(window="1h")
        flow.lead_time()
        flow.flow_summary(value_add_seconds=120, window="1h")
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        entry_uuid: str,
        exit_uuid: str,
        *,
        event_uuid: str = "prod:flow",
        value_column: str = "value_bool",
        time_column: str = "systime",
    ) -> None:
        """Initialize the flow-metrics analyser.

        Args:
            dataframe: Input DataFrame with timeseries data.
            entry_uuid: UUID of the boolean unit-entry signal.
            exit_uuid: UUID of the boolean unit-exit signal.
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the boolean trigger.
            time_column: Name of the timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.entry_uuid = entry_uuid
        self.exit_uuid = exit_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, entry_uuid)
        self._validate_uuid(self.dataframe, exit_uuid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rising_edge_times(self, uuid: str) -> pd.Series:
        """Return the timestamps of rising edges of a boolean signal."""
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

    def _wip_per_window(self, window: str) -> pd.DataFrame:
        """Time-weighted WIP statistics per window.

        Columns: ``start``, ``wip_mean``, ``wip_max``, ``wip_min``.
        """
        entries = self._rising_edge_times(self.entry_uuid)
        exits = self._rising_edge_times(self.exit_uuid)
        if entries.empty and exits.empty:
            return pd.DataFrame(columns=["start", "wip_mean", "wip_max", "wip_min"])

        events = pd.concat(
            [
                pd.DataFrame({"time": entries, "delta": 1}),
                pd.DataFrame({"time": exits, "delta": -1}),
            ],
            ignore_index=True,
        ).sort_values("time")
        events["wip"] = events["delta"].cumsum()
        # Level holding from each event time onward (last value at ties).
        level = events.groupby("time")["wip"].last()

        first = level.index.min()
        last = level.index.max()
        edges = pd.date_range(
            first.floor(window), last + pd.Timedelta(window), freq=window
        )
        idx = level.index.union(pd.DatetimeIndex(edges))
        step = level.reindex(idx).ffill().fillna(0.0)

        seg = pd.DataFrame({"wip": step})
        seg["dur"] = seg.index.to_series().diff().shift(-1).dt.total_seconds()
        seg = seg[seg["dur"] > 0].copy()
        if seg.empty:
            return pd.DataFrame(columns=["start", "wip_mean", "wip_max", "wip_min"])
        seg["start"] = seg.index.floor(window)
        seg["weighted"] = seg["wip"] * seg["dur"]

        grouped = seg.groupby("start")
        out = grouped.agg(
            weighted=("weighted", "sum"),
            dur=("dur", "sum"),
            wip_max=("wip", "max"),
            wip_min=("wip", "min"),
        ).reset_index()
        out["wip_mean"] = (out["weighted"] / out["dur"]).round(3)
        return out[["start", "wip_mean", "wip_max", "wip_min"]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wip_over_time(self, window: str = "1h") -> pd.DataFrame:
        """Time-weighted work-in-process per window.

        Args:
            window: Resample window (e.g. ``"1h"``).

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            wip_mean, wip_max, wip_min.
        """
        extra = ["wip_mean", "wip_max", "wip_min"]
        wip = self._wip_per_window(window)
        if wip.empty:
            return empty_event_df("summary", extra_cols=extra)
        wip["end"] = wip["start"] + pd.Timedelta(window)
        return finalize_summary_df(wip)

    def throughput(self, window: str = "1h") -> pd.DataFrame:
        """Units completed (exit rising edges) per window.

        Args:
            window: Resample window.

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            units_out, throughput_per_hour.
        """
        extra = ["units_out", "throughput_per_hour"]
        exits = self._rising_edge_times(self.exit_uuid)
        if exits.empty:
            return empty_event_df("summary", extra_cols=extra)

        window_hours = pd.Timedelta(window).total_seconds() / 3600.0
        counts = exits.dt.floor(window).value_counts().sort_index()
        out = counts.reset_index()
        out.columns = ["start", "units_out"]
        out["end"] = out["start"] + pd.Timedelta(window)
        out["throughput_per_hour"] = (out["units_out"] / window_hours).round(3)
        return finalize_summary_df(out)

    def lead_time(self) -> pd.DataFrame:
        """FIFO-matched lead time per unit (entry -> exit).

        The nth entry is matched to the nth exit (first-in-first-out).

        Returns:
            Point-shape DataFrame with columns: systime (exit time), uuid,
            source_uuid, lead_time_seconds, unit_index.
        """
        extra = ["lead_time_seconds", "unit_index"]
        entries = self._rising_edge_times(self.entry_uuid)
        exits = self._rising_edge_times(self.exit_uuid)
        n = min(len(entries), len(exits))
        if n == 0:
            return empty_event_df("point", extra_cols=extra)

        lead = (exits.iloc[:n].to_numpy() - entries.iloc[:n].to_numpy()) / pd.Timedelta(
            seconds=1
        )
        out = pd.DataFrame(
            {
                "systime": exits.iloc[:n].to_numpy(),
                "lead_time_seconds": lead.round(3),
                "unit_index": range(n),
            }
        )
        return finalize_point_df(out, uuid=self.event_uuid, source_uuid=self.exit_uuid)

    def flow_summary(
        self,
        *,
        value_add_seconds: _Number | None = None,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Combined flow metrics with a Little's Law consistency check.

        Little's Law lead time = WIP / throughput. ``consistency_ratio`` is the
        measured FIFO lead time divided by that prediction (≈ 1 for a stable,
        FIFO process). Process Cycle Efficiency = value-add time / lead time.

        Args:
            value_add_seconds: Value-add (touch) time per unit, for PCE. Seconds
                number or offset string; omit to skip the PCE column.
            window: Resample window.

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            wip_mean, throughput_per_hour, lead_time_mean_seconds,
            littles_law_lead_time_seconds, consistency_ratio, and
            process_cycle_efficiency_pct when ``value_add_seconds`` is given.
        """
        extra = [
            "wip_mean",
            "throughput_per_hour",
            "lead_time_mean_seconds",
            "littles_law_lead_time_seconds",
            "consistency_ratio",
        ]
        if value_add_seconds is not None:
            extra.append("process_cycle_efficiency_pct")

        wip = self._wip_per_window(window)
        thru = self.throughput(window)
        if wip.empty or thru.empty:
            return empty_event_df("summary", extra_cols=extra)

        merged = wip[["start", "wip_mean"]].merge(
            thru[["start", "throughput_per_hour"]], on="start", how="outer"
        )

        # Mean FIFO lead time per window, bucketed by exit time.
        lead = self.lead_time()
        if not lead.empty:
            lead = lead.copy()
            lead["start"] = lead["systime"].dt.floor(window)
            lead_mean = lead.groupby("start")["lead_time_seconds"].mean().reset_index()
            lead_mean.columns = ["start", "lead_time_mean_seconds"]
            merged = merged.merge(lead_mean, on="start", how="left")
        else:
            merged["lead_time_mean_seconds"] = float("nan")

        merged = merged.sort_values("start").reset_index(drop=True)
        merged["wip_mean"] = merged["wip_mean"].fillna(0.0)
        merged["throughput_per_hour"] = merged["throughput_per_hour"].fillna(0.0)

        # Little's Law: L (seconds) = WIP / throughput_per_second.
        tput_per_sec = merged["throughput_per_hour"] / 3600.0
        merged["littles_law_lead_time_seconds"] = (
            (merged["wip_mean"] / tput_per_sec).where(tput_per_sec > 0).round(3)
        )
        merged["consistency_ratio"] = (
            merged["lead_time_mean_seconds"] / merged["littles_law_lead_time_seconds"]
        ).round(3)
        merged["lead_time_mean_seconds"] = merged["lead_time_mean_seconds"].round(3)

        if value_add_seconds is not None:
            va = _to_seconds(value_add_seconds)
            merged["process_cycle_efficiency_pct"] = (
                (va / merged["lead_time_mean_seconds"] * 100.0)
                .where(merged["lead_time_mean_seconds"] > 0)
                .round(2)
            )

        merged["end"] = merged["start"] + pd.Timedelta(window)
        return finalize_summary_df(merged)
